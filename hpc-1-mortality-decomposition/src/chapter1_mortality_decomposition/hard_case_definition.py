from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from chapter1_mortality_decomposition.baseline_evaluation import (
    DEFAULT_BASELINE_ARTIFACT_ROOT,
    DEFAULT_HORIZONS,
    _discover_prediction_artifacts,
    _load_prediction_frame,
)
from chapter1_mortality_decomposition.baseline_logistic import IDENTIFIER_COLUMNS
from chapter1_mortality_decomposition.utils import require_columns, write_dataframe, write_text


MODEL_NAME = "logistic_regression"
HARD_CASE_RULE = "asic_logistic_last_eligible_nonfatal_q75_v1"
DEFAULT_HARD_CASE_OUTPUT_ROOT = (
    Path("artifacts")
    / "chapter1"
    / "evaluation"
    / "asic"
    / "hard_cases"
    / "primary_medians"
)
DEFAULT_HARD_CASE_OUTPUT_DIR = (
    DEFAULT_HARD_CASE_OUTPUT_ROOT / MODEL_NAME
)
SMALL_HARD_CASE_COUNT_THRESHOLD = 20
SMALL_HARD_CASE_PERCENT_THRESHOLD = 0.05
LAST_POINT_GROUP_COLUMNS = ["hospital_id", "stay_id_global", "horizon_h"]
LAST_POINT_SORT_COLUMNS = [
    "hospital_id",
    "stay_id_global",
    "horizon_h",
    "prediction_time_h",
    "block_index",
    "instance_id",
]
REQUIRED_PREDICTION_COLUMNS = set(IDENTIFIER_COLUMNS) | {"predicted_probability", "model_name"}


@dataclass(frozen=True)
class HardCaseArtifacts:
    stay_level_path: Path
    horizon_summary_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class HardCaseRunResult:
    input_root: Path
    output_dir: Path
    horizons_processed: tuple[int, ...]
    hard_case_rule: str
    artifacts: HardCaseArtifacts
    stay_level: pd.DataFrame
    horizon_summary: pd.DataFrame


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(payload: dict[str, object], path: Path) -> Path:
    def _json_default(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if value is pd.NA or pd.isna(value):
            return None
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    return write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        path,
    )


def _output_extension(output_format: str) -> str:
    if output_format not in {"csv", "parquet"}:
        raise ValueError(f"Unsupported output format: {output_format}")
    return "csv" if output_format == "csv" else "parquet"


def _normalize_input_root(input_root: Path, *, model_name: str = MODEL_NAME) -> Path:
    root = Path(input_root)
    if (root / model_name).is_dir():
        return root
    if root.name == model_name and any(path.is_dir() for path in root.glob("horizon_*h")):
        return root.parent
    raise ValueError(
        f"Could not locate {model_name!r} baseline artifacts from input root {root}. "
        f"Expected either {root / model_name} or horizon_*h directories under {root}."
    )


def _require_no_missing_selection_keys(predictions: pd.DataFrame, *, source_name: str) -> None:
    missing_counts = {
        column: int(predictions[column].isna().sum())
        for column in ("hospital_id", "stay_id_global", "horizon_h", "prediction_time_h", "block_index")
        if column in predictions.columns and int(predictions[column].isna().sum()) > 0
    }
    if missing_counts:
        raise ValueError(
            f"{source_name} contains missing values in required last-point selection columns: "
            f"{missing_counts}"
        )


def _validate_prediction_frame(
    predictions: pd.DataFrame,
    *,
    source_name: str,
    horizon_h: int,
    expected_source_model_name: str | None = MODEL_NAME,
    probability_column: str = "predicted_probability",
    output_model_name: str | None = None,
) -> pd.DataFrame:
    required_columns = set(IDENTIFIER_COLUMNS) | {"model_name", probability_column}
    require_columns(predictions, required_columns, source_name)
    if predictions.empty:
        raise ValueError(f"No prediction rows found for horizon {horizon_h} at {source_name}.")

    validated = predictions.copy()
    validated["model_name"] = validated["model_name"].astype("string")
    validated["hospital_id"] = validated["hospital_id"].astype("string")
    validated["stay_id_global"] = validated["stay_id_global"].astype("string")
    validated["instance_id"] = validated["instance_id"].astype("string")
    validated["split"] = validated["split"].astype("string")
    validated["prediction_time_h"] = pd.to_numeric(validated["prediction_time_h"], errors="coerce")
    validated["block_index"] = pd.to_numeric(validated["block_index"], errors="coerce")
    validated["horizon_h"] = pd.to_numeric(validated["horizon_h"], errors="coerce").astype("Int64")
    validated["label_value"] = pd.to_numeric(validated["label_value"], errors="coerce").astype("Int64")
    validated["predicted_probability"] = pd.to_numeric(
        validated[probability_column],
        errors="coerce",
    )

    _require_no_missing_selection_keys(validated, source_name=source_name)

    missing_probability_count = int(validated["predicted_probability"].isna().sum())
    if missing_probability_count:
        raise ValueError(
            f"{source_name} contains {missing_probability_count} rows with missing predicted_probability."
        )
    if not validated["predicted_probability"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError(f"{source_name} contains predicted_probability values outside [0, 1].")

    if validated["label_value"].isna().any():
        raise ValueError(f"{source_name} contains missing label_value entries.")
    invalid_label_values = sorted(
        {
            int(value)
            for value in validated["label_value"].dropna().astype(int).tolist()
            if int(value) not in {0, 1}
        }
    )
    if invalid_label_values:
        raise ValueError(
            f"{source_name} contains unsupported label_value codes: {invalid_label_values}"
        )

    if validated["model_name"].nunique(dropna=True) != 1:
        raise ValueError(f"{source_name} contains multiple model_name values.")
    observed_model_name = str(validated["model_name"].dropna().iloc[0])
    if (
        expected_source_model_name is not None
        and observed_model_name != expected_source_model_name
    ):
        raise ValueError(
            f"{source_name} contained model_name={observed_model_name!r}; expected "
            f"{expected_source_model_name!r}."
        )
    validated["model_name"] = output_model_name or observed_model_name

    duplicate_instance_count = int(validated.duplicated(subset=["instance_id"]).sum())
    if duplicate_instance_count:
        raise ValueError(
            f"{source_name} contains {duplicate_instance_count} duplicated instance_id values."
        )

    return validated


def select_last_eligible_stay_points(
    predictions: pd.DataFrame,
    *,
    source_name: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    validated = predictions.copy()
    ordered = validated.sort_values(LAST_POINT_SORT_COLUMNS, kind="stable").reset_index(drop=True)

    # Saved Chapter 1 baseline predictions are already horizon-valid; within each stay/horizon
    # we only need the final eligible row before the event or censoring endpoint.
    last_points = (
        ordered.groupby(LAST_POINT_GROUP_COLUMNS, sort=False, group_keys=False)
        .tail(1)
        .reset_index(drop=True)
    )

    duplicate_last_points = int(last_points.duplicated(subset=LAST_POINT_GROUP_COLUMNS).sum())
    if duplicate_last_points:
        raise ValueError(
            f"{source_name} produced duplicated stay-level last-point selections: "
            f"{duplicate_last_points} duplicated stay/horizon rows."
        )

    selection_counts = {
        "n_prediction_rows_loaded": int(ordered.shape[0]),
        "n_selected_last_points": int(last_points.shape[0]),
        "n_rows_collapsed_before_last_point": int(ordered.shape[0] - last_points.shape[0]),
    }
    return last_points, selection_counts


def classify_hard_cases_for_horizon(
    last_points: pd.DataFrame,
    *,
    selection_counts: dict[str, int],
    model_name: str = MODEL_NAME,
    hard_case_rule: str = HARD_CASE_RULE,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if last_points.empty:
        raise ValueError("Cannot classify hard cases from an empty last-point frame.")

    horizon_values = sorted(last_points["horizon_h"].dropna().astype(int).unique().tolist())
    if len(horizon_values) != 1:
        raise ValueError(
            f"Expected one horizon per last-point frame, found {horizon_values}."
        )
    horizon_h = int(horizon_values[0])

    fatal_last_points = last_points[last_points["label_value"].astype(int).eq(1)].copy()
    nonfatal_last_points = last_points[last_points["label_value"].astype(int).eq(0)].copy()
    if fatal_last_points.empty:
        raise ValueError(
            f"Horizon {horizon_h} has no fatal stays after last-point selection."
        )
    if nonfatal_last_points.empty:
        raise ValueError(
            f"Horizon {horizon_h} has no nonfatal stays after last-point selection."
        )

    nonfatal_q75_threshold = float(nonfatal_last_points["predicted_probability"].quantile(0.75))
    if not np.isfinite(nonfatal_q75_threshold) or not 0.0 <= nonfatal_q75_threshold <= 1.0:
        raise ValueError(
            f"Horizon {horizon_h} produced an invalid nonfatal q75 threshold "
            f"{nonfatal_q75_threshold!r}."
        )

    stay_level = last_points[
        [
            "stay_id_global",
            "hospital_id",
            "horizon_h",
            "split",
            "label_value",
            "instance_id",
            "block_index",
            "prediction_time_h",
            "predicted_probability",
            "model_name",
        ]
    ].copy()
    stay_level["nonfatal_q75_threshold"] = nonfatal_q75_threshold
    stay_level["hard_case_flag"] = (
        stay_level["label_value"].astype(int).eq(1)
        & stay_level["predicted_probability"].le(nonfatal_q75_threshold)
    )
    stay_level["hard_case_rule"] = hard_case_rule

    n_hard_cases = int(stay_level["hard_case_flag"].sum())
    n_fatal_last_points = int(fatal_last_points.shape[0])
    pct_fatal_hard_cases = float(n_hard_cases / n_fatal_last_points)

    warning_reasons: list[str] = []
    if n_hard_cases < SMALL_HARD_CASE_COUNT_THRESHOLD:
        warning_reasons.append(f"n_hard_cases_lt_{SMALL_HARD_CASE_COUNT_THRESHOLD}")
    if pct_fatal_hard_cases < SMALL_HARD_CASE_PERCENT_THRESHOLD:
        warning_reasons.append(
            f"pct_fatal_hard_cases_lt_{SMALL_HARD_CASE_PERCENT_THRESHOLD:.2f}"
        )

    summary_row: dict[str, object] = {
        "horizon_h": horizon_h,
        "model_name": model_name,
        "n_prediction_rows_loaded": int(selection_counts["n_prediction_rows_loaded"]),
        "n_rows_collapsed_before_last_point": int(
            selection_counts["n_rows_collapsed_before_last_point"]
        ),
        "n_selected_last_points": int(selection_counts["n_selected_last_points"]),
        "n_nonfatal_last_points": int(nonfatal_last_points.shape[0]),
        "n_fatal_last_points": n_fatal_last_points,
        "nonfatal_q75_threshold": nonfatal_q75_threshold,
        "n_hard_cases": n_hard_cases,
        "pct_fatal_hard_cases": pct_fatal_hard_cases,
        "subgroup_size_warning": bool(warning_reasons),
        "warning_reason": "; ".join(warning_reasons) if warning_reasons else pd.NA,
    }
    return stay_level, summary_row


def build_hard_case_tables_from_prediction_frames(
    prediction_frames_by_horizon: dict[int, pd.DataFrame],
    *,
    source_names_by_horizon: dict[int, str] | None = None,
    expected_source_model_name: str | None = MODEL_NAME,
    probability_column: str = "predicted_probability",
    output_model_name: str = MODEL_NAME,
    hard_case_rule: str = HARD_CASE_RULE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    if not prediction_frames_by_horizon:
        raise ValueError("No prediction frames were supplied for hard-case classification.")

    stay_level_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    prediction_sources_by_horizon: dict[str, str] = {}

    for horizon_h in sorted(int(horizon) for horizon in prediction_frames_by_horizon):
        predictions = prediction_frames_by_horizon[int(horizon_h)]
        source_name = (
            source_names_by_horizon.get(int(horizon_h), f"horizon_{int(horizon_h)}")
            if source_names_by_horizon is not None
            else f"horizon_{int(horizon_h)}"
        )
        validated = _validate_prediction_frame(
            predictions,
            source_name=source_name,
            horizon_h=int(horizon_h),
            expected_source_model_name=expected_source_model_name,
            probability_column=probability_column,
            output_model_name=output_model_name,
        )
        last_points, selection_counts = select_last_eligible_stay_points(
            validated,
            source_name=source_name,
        )
        stay_level, summary_row = classify_hard_cases_for_horizon(
            last_points,
            selection_counts=selection_counts,
            model_name=output_model_name,
            hard_case_rule=hard_case_rule,
        )
        stay_level_frames.append(stay_level)
        summary_rows.append(summary_row)
        prediction_sources_by_horizon[str(int(horizon_h))] = source_name

    stay_level_output = pd.concat(stay_level_frames, ignore_index=True).sort_values(
        ["horizon_h", "hospital_id", "stay_id_global"],
        kind="stable",
    ).reset_index(drop=True)
    horizon_summary_output = pd.DataFrame(summary_rows).sort_values(
        ["horizon_h"],
        kind="stable",
    ).reset_index(drop=True)
    return stay_level_output, horizon_summary_output, prediction_sources_by_horizon


def run_asic_logistic_hard_case_definition(
    *,
    input_root: Path = DEFAULT_BASELINE_ARTIFACT_ROOT,
    output_dir: Path = DEFAULT_HARD_CASE_OUTPUT_DIR,
    horizons: Sequence[int] | None = None,
    output_format: str = "csv",
) -> HardCaseRunResult:
    baseline_root = _normalize_input_root(Path(input_root))
    artifacts = _discover_prediction_artifacts(
        baseline_root,
        models=[MODEL_NAME],
        horizons=horizons or DEFAULT_HORIZONS,
    )
    if not artifacts:
        raise ValueError(
            f"No {MODEL_NAME!r} prediction artifacts were discovered under {baseline_root}."
        )

    prediction_frames_by_horizon = {
        int(artifact.horizon_h): _load_prediction_frame(artifact)
        for artifact in artifacts
    }
    prediction_paths_by_horizon = {
        int(artifact.horizon_h): str(artifact.predictions_path.resolve())
        for artifact in artifacts
    }
    stay_level_output, horizon_summary_output, prediction_source_lookup = (
        build_hard_case_tables_from_prediction_frames(
            prediction_frames_by_horizon,
            source_names_by_horizon=prediction_paths_by_horizon,
            expected_source_model_name=MODEL_NAME,
            probability_column="predicted_probability",
            output_model_name=MODEL_NAME,
            hard_case_rule=HARD_CASE_RULE,
        )
    )
    horizons_processed = sorted(prediction_frames_by_horizon)

    extension = _output_extension(output_format)
    stay_level_path = write_dataframe(
        stay_level_output,
        Path(output_dir) / f"stay_level_hard_case_flags.{extension}",
        output_format=output_format,
    )
    horizon_summary_path = write_dataframe(
        horizon_summary_output,
        Path(output_dir) / f"horizon_hard_case_summary.{extension}",
        output_format=output_format,
    )
    manifest_path = _write_json(
        {
            "timestamp_utc": _utc_timestamp(),
            "model_name": MODEL_NAME,
            "hard_case_rule": HARD_CASE_RULE,
            "input_root": str(baseline_root.resolve()),
            "output_dir": str(Path(output_dir).resolve()),
            "horizons_processed": horizons_processed,
            "small_hard_case_count_threshold": SMALL_HARD_CASE_COUNT_THRESHOLD,
            "small_hard_case_percent_threshold": SMALL_HARD_CASE_PERCENT_THRESHOLD,
            "prediction_paths_by_horizon": prediction_source_lookup,
            "stay_level_artifact": str(Path(stay_level_path).resolve()),
            "horizon_summary_artifact": str(Path(horizon_summary_path).resolve()),
        },
        Path(output_dir) / "run_manifest.json",
    )

    return HardCaseRunResult(
        input_root=baseline_root,
        output_dir=Path(output_dir),
        horizons_processed=tuple(horizons_processed),
        hard_case_rule=HARD_CASE_RULE,
        artifacts=HardCaseArtifacts(
            stay_level_path=stay_level_path,
            horizon_summary_path=horizon_summary_path,
            manifest_path=manifest_path,
        ),
        stay_level=stay_level_output,
        horizon_summary=horizon_summary_output,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Define Chapter 1 ASIC low-predicted fatal cases from saved logistic-regression "
            "baseline prediction artifacts."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_BASELINE_ARTIFACT_ROOT,
        help=(
            "Root baseline artifact directory containing model subdirectories, or the "
            "logistic_regression model directory itself."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_HARD_CASE_OUTPUT_DIR,
        help="Directory where hard-case artifacts will be written.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        help="Optional subset of horizons to process.",
    )
    parser.add_argument(
        "--output-format",
        choices=("csv", "parquet"),
        default="csv",
        help="Output format for the written hard-case artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = run_asic_logistic_hard_case_definition(
        input_root=args.input_root,
        output_dir=args.output_dir,
        horizons=args.horizons,
        output_format=args.output_format,
    )

    print(f"Hard-case rule: {result.hard_case_rule}")
    print(f"Stay-level artifact: {result.artifacts.stay_level_path}")
    print(f"Horizon summary artifact: {result.artifacts.horizon_summary_path}")
    print(f"Run manifest: {result.artifacts.manifest_path}")
    for row in result.horizon_summary.itertuples(index=False):
        warning_text = row.warning_reason if isinstance(row.warning_reason, str) and row.warning_reason else "none"
        print(
            f"horizon {int(row.horizon_h)}h -> nonfatal_last={int(row.n_nonfatal_last_points)}, "
            f"fatal_last={int(row.n_fatal_last_points)}, q75={float(row.nonfatal_q75_threshold):.6f}, "
            f"hard_cases={int(row.n_hard_cases)}, pct_fatal_hard_cases={float(row.pct_fatal_hard_cases):.3f}, "
            f"warnings: {warning_text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
