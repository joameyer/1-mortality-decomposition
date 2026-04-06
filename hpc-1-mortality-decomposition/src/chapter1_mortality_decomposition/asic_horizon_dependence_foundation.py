from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from chapter1_mortality_decomposition.utils import (
    normalize_boolean_codes,
    read_dataframe,
    require_columns,
    write_dataframe,
    write_text,
)


DEFAULT_HARD_CASE_DIR = (
    Path("artifacts")
    / "chapter1"
    / "evaluation"
    / "asic"
    / "hard_cases"
    / "primary_medians"
    / "logistic_regression"
)
DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "chapter1"
    / "evaluation"
    / "asic"
    / "horizon_dependence"
    / "foundation"
)
DEFAULT_HORIZONS = (8, 16, 24, 48, 72)
REQUIRED_STAY_LEVEL_COLUMNS = {
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
    "nonfatal_q75_threshold",
    "hard_case_flag",
    "hard_case_rule",
}
REQUIRED_SAVED_SUMMARY_COLUMNS = {
    "horizon_h",
    "n_nonfatal_last_points",
    "n_fatal_last_points",
    "nonfatal_q75_threshold",
    "n_hard_cases",
    "pct_fatal_hard_cases",
}
REQUIRED_PREDICTION_COLUMNS = {
    "instance_id",
    "stay_id_global",
    "hospital_id",
    "block_index",
    "prediction_time_h",
    "horizon_h",
    "split",
    "label_value",
    "predicted_probability",
    "model_name",
}
KNOWN_TARGET_SUMMARY = {
    8: {
        "nonfatal_last_n": 4713,
        "fatal_last_n": 1639,
        "nonfatal_q75_threshold": 0.004425,
        "hard_case_n": 351,
    },
    16: {
        "nonfatal_last_n": 4713,
        "fatal_last_n": 1670,
        "nonfatal_q75_threshold": 0.009295,
        "hard_case_n": 342,
    },
    24: {
        "nonfatal_last_n": 4696,
        "fatal_last_n": 1682,
        "nonfatal_q75_threshold": 0.014598,
        "hard_case_n": 346,
    },
    48: {
        "nonfatal_last_n": 4542,
        "fatal_last_n": 1697,
        "nonfatal_q75_threshold": 0.032415,
        "hard_case_n": 352,
    },
    72: {
        "nonfatal_last_n": 4326,
        "fatal_last_n": 1704,
        "nonfatal_q75_threshold": 0.052678,
        "hard_case_n": 364,
    },
}


@dataclass(frozen=True)
class HorizonFoundationArtifacts:
    horizon_summary_csv_path: Path
    horizon_summary_markdown_path: Path
    note_path: Path


@dataclass(frozen=True)
class HorizonFoundationRunResult:
    hard_case_dir: Path
    output_dir: Path
    horizons: tuple[int, ...]
    artifacts: HorizonFoundationArtifacts
    horizon_summary: pd.DataFrame
    cross_horizon_matching_ready: bool
    schema_harmonization_required: bool
    mismatches_vs_saved_summary: tuple[str, ...]
    mismatches_vs_known_target: tuple[str, ...]


def _normalize_horizons(horizons: Sequence[int] | None) -> tuple[int, ...]:
    values = tuple(sorted({int(horizon) for horizon in (horizons or DEFAULT_HORIZONS)}))
    unsupported = sorted(set(values) - set(DEFAULT_HORIZONS))
    if unsupported:
        raise ValueError(f"Unsupported horizons requested: {unsupported}")
    return values


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def _file_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix in {"csv", "parquet", "json", "md", "txt"}:
        return suffix
    return path.suffix.lower() or "unknown"


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _read_columns(path: Path) -> tuple[str, ...]:
    if path.suffix.lower() == ".csv":
        return tuple(pd.read_csv(path, nrows=0).columns.tolist())
    return tuple(read_dataframe(path).columns.tolist())


def _format_float(value: float | int | str, digits: int = 6) -> str:
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def _markdown_table(rows: list[dict[str, object]], columns: Sequence[str]) -> str:
    if not rows:
        return ""

    rendered_rows = []
    widths = {column: len(column) for column in columns}
    for row in rows:
        rendered = {column: str(row.get(column, "")) for column in columns}
        rendered_rows.append(rendered)
        for column, value in rendered.items():
            widths[column] = max(widths[column], len(value))

    header = "| " + " | ".join(column.ljust(widths[column]) for column in columns) + " |"
    divider = "| " + " | ".join("-" * widths[column] for column in columns) + " |"
    body = [
        "| " + " | ".join(row[column].ljust(widths[column]) for column in columns) + " |"
        for row in rendered_rows
    ]
    return "\n".join([header, divider, *body])


def _load_manifest_inputs(
    hard_case_dir: Path,
    horizons: tuple[int, ...],
) -> tuple[dict[str, object], Path, Path, dict[int, Path]]:
    manifest_path = hard_case_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Hard-case manifest not found at {manifest_path}")

    manifest = _load_json(manifest_path)
    stay_level_path = Path(str(manifest["stay_level_artifact"]))
    saved_summary_path = Path(str(manifest["horizon_summary_artifact"]))
    prediction_paths = {
        int(horizon): Path(str(path))
        for horizon, path in dict(manifest["prediction_paths_by_horizon"]).items()
        if int(horizon) in horizons
    }

    missing_prediction_paths = sorted(set(horizons) - set(prediction_paths))
    if missing_prediction_paths:
        raise ValueError(
            "Manifest is missing prediction paths for requested horizons: "
            f"{missing_prediction_paths}"
        )

    return manifest, stay_level_path, saved_summary_path, prediction_paths


def _validate_stay_level_artifact(
    stay_level_path: Path,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    stay_level = read_dataframe(stay_level_path)
    require_columns(stay_level, REQUIRED_STAY_LEVEL_COLUMNS, str(stay_level_path))

    validated = stay_level.copy()
    validated["stay_id_global"] = validated["stay_id_global"].astype("string")
    validated["hospital_id"] = validated["hospital_id"].astype("string")
    validated["split"] = validated["split"].astype("string")
    validated["instance_id"] = validated["instance_id"].astype("string")
    validated["model_name"] = validated["model_name"].astype("string")
    validated["hard_case_rule"] = validated["hard_case_rule"].astype("string")
    validated["horizon_h"] = pd.to_numeric(validated["horizon_h"], errors="coerce").astype("Int64")
    validated["label_value"] = pd.to_numeric(validated["label_value"], errors="coerce").astype("Int64")
    validated["block_index"] = pd.to_numeric(validated["block_index"], errors="coerce")
    validated["prediction_time_h"] = pd.to_numeric(validated["prediction_time_h"], errors="coerce")
    validated["predicted_probability"] = pd.to_numeric(
        validated["predicted_probability"],
        errors="coerce",
    )
    validated["nonfatal_q75_threshold"] = pd.to_numeric(
        validated["nonfatal_q75_threshold"],
        errors="coerce",
    )
    validated["hard_case_flag"] = normalize_boolean_codes(validated["hard_case_flag"])

    filtered = validated[validated["horizon_h"].isin(list(horizons))].copy()
    observed_horizons = tuple(sorted(int(value) for value in filtered["horizon_h"].dropna().unique()))
    if observed_horizons != horizons:
        raise ValueError(
            f"Stay-level artifact {stay_level_path} covered horizons {observed_horizons}, "
            f"expected {horizons}."
        )

    missing_counts = {
        column: int(filtered[column].isna().sum())
        for column in (
            "stay_id_global",
            "hospital_id",
            "horizon_h",
            "label_value",
            "instance_id",
            "prediction_time_h",
            "predicted_probability",
            "nonfatal_q75_threshold",
            "hard_case_flag",
        )
        if int(filtered[column].isna().sum()) > 0
    }
    if missing_counts:
        raise ValueError(
            f"Stay-level artifact {stay_level_path} contains missing values in required fields: "
            f"{missing_counts}"
        )

    duplicate_count = int(filtered.duplicated(["stay_id_global", "horizon_h"]).sum())
    if duplicate_count:
        raise ValueError(
            f"Stay-level artifact {stay_level_path} contains {duplicate_count} duplicated "
            "stay_id_global/horizon_h rows."
        )

    hospital_conflicts = (
        filtered.groupby("stay_id_global", observed=True)["hospital_id"].nunique(dropna=False).gt(1)
    )
    if bool(hospital_conflicts.any()):
        conflicted_ids = hospital_conflicts[hospital_conflicts].index.tolist()
        raise ValueError(
            "stay_id_global was not stable across hospital_id values for: "
            f"{conflicted_ids[:5]}"
        )

    invalid_label_values = sorted(
        {
            int(value)
            for value in filtered["label_value"].dropna().astype(int).tolist()
            if int(value) not in {0, 1}
        }
    )
    if invalid_label_values:
        raise ValueError(
            f"Stay-level artifact {stay_level_path} contains unsupported label_value codes: "
            f"{invalid_label_values}"
        )

    if not filtered["predicted_probability"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError(f"{stay_level_path} contains predicted_probability values outside [0, 1].")
    if not filtered["nonfatal_q75_threshold"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError(f"{stay_level_path} contains nonfatal_q75_threshold values outside [0, 1].")

    threshold_counts = filtered.groupby("horizon_h", observed=True)["nonfatal_q75_threshold"].nunique()
    inconsistent_thresholds = threshold_counts[threshold_counts.ne(1)]
    if not inconsistent_thresholds.empty:
        raise ValueError(
            "Stay-level artifact did not keep a single nonfatal_q75_threshold per horizon: "
            f"{inconsistent_thresholds.to_dict()}"
        )

    invalid_hard_cases = filtered["hard_case_flag"].astype(bool) & filtered["label_value"].astype(int).eq(0)
    if bool(invalid_hard_cases.any()):
        raise ValueError(
            f"{stay_level_path} marked nonfatal stays as hard cases, which is inconsistent "
            "with the frozen Chapter 1 rule."
        )

    return filtered.reset_index(drop=True)


def _validate_saved_summary_artifact(
    saved_summary_path: Path,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    saved_summary = read_dataframe(saved_summary_path)
    require_columns(saved_summary, REQUIRED_SAVED_SUMMARY_COLUMNS, str(saved_summary_path))

    validated = saved_summary.copy()
    validated["horizon_h"] = pd.to_numeric(validated["horizon_h"], errors="coerce").astype("Int64")
    validated["n_nonfatal_last_points"] = pd.to_numeric(
        validated["n_nonfatal_last_points"],
        errors="coerce",
    ).astype("Int64")
    validated["n_fatal_last_points"] = pd.to_numeric(
        validated["n_fatal_last_points"],
        errors="coerce",
    ).astype("Int64")
    validated["nonfatal_q75_threshold"] = pd.to_numeric(
        validated["nonfatal_q75_threshold"],
        errors="coerce",
    )
    validated["n_hard_cases"] = pd.to_numeric(validated["n_hard_cases"], errors="coerce").astype("Int64")
    validated["pct_fatal_hard_cases"] = pd.to_numeric(
        validated["pct_fatal_hard_cases"],
        errors="coerce",
    )
    filtered = validated[validated["horizon_h"].isin(list(horizons))].copy()

    observed_horizons = tuple(sorted(int(value) for value in filtered["horizon_h"].dropna().unique()))
    if observed_horizons != horizons:
        raise ValueError(
            f"Saved horizon summary {saved_summary_path} covered horizons {observed_horizons}, "
            f"expected {horizons}."
        )
    return filtered.reset_index(drop=True)


def _inspect_prediction_artifacts(
    prediction_paths: dict[int, Path],
) -> tuple[list[dict[str, object]], bool]:
    inspections: list[dict[str, object]] = []
    observed_schemas: dict[int, tuple[str, ...]] = {}

    for horizon_h, prediction_path in sorted(prediction_paths.items()):
        columns = _read_columns(prediction_path)
        require_columns(
            pd.DataFrame(columns=columns),
            REQUIRED_PREDICTION_COLUMNS,
            str(prediction_path),
        )
        observed_schemas[horizon_h] = columns
        inspections.append(
            {
                "horizon_h": horizon_h,
                "prediction_path": prediction_path,
                "prediction_format": _file_format(prediction_path),
                "prediction_columns": columns,
                "stay_id_field": "stay_id_global",
                "label_field": "label_value",
                "predicted_risk_field": "predicted_probability",
                "hard_case_flag_field": "derived_from_stay_level_artifact",
                "threshold_field": "nonfatal_q75_threshold_saved_in_stay_level_artifact",
            }
        )

    schema_harmonization_required = len(set(observed_schemas.values())) > 1
    return inspections, schema_harmonization_required


def _build_internal_horizon_summary(stay_level: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon_h in sorted(int(value) for value in stay_level["horizon_h"].dropna().unique()):
        horizon_df = stay_level[stay_level["horizon_h"].astype(int).eq(horizon_h)].copy()
        fatal_n = int(horizon_df["label_value"].astype(int).eq(1).sum())
        hard_case_n = int(horizon_df["hard_case_flag"].astype(bool).sum())
        rows.append(
            {
                "horizon_h": horizon_h,
                "nonfatal_last_n": int(horizon_df["label_value"].astype(int).eq(0).sum()),
                "fatal_last_n": fatal_n,
                "nonfatal_q75_threshold": float(horizon_df["nonfatal_q75_threshold"].iloc[0]),
                "hard_case_n": hard_case_n,
                "hard_case_share_among_fatal": float(hard_case_n / fatal_n),
            }
        )
    return pd.DataFrame(rows).sort_values("horizon_h", kind="stable").reset_index(drop=True)


def _build_requested_horizon_summary(internal_summary: pd.DataFrame) -> pd.DataFrame:
    output = internal_summary.copy()
    output["horizon"] = output["horizon_h"].astype(int).map(lambda value: f"{value}h")
    return output[
        [
            "horizon",
            "nonfatal_last_n",
            "fatal_last_n",
            "nonfatal_q75_threshold",
            "hard_case_n",
            "hard_case_share_among_fatal",
        ]
    ].reset_index(drop=True)


def _compare_with_saved_summary(
    internal_summary: pd.DataFrame,
    saved_summary: pd.DataFrame,
) -> tuple[str, ...]:
    saved = saved_summary.rename(
        columns={
            "n_nonfatal_last_points": "nonfatal_last_n",
            "n_fatal_last_points": "fatal_last_n",
            "n_hard_cases": "hard_case_n",
            "pct_fatal_hard_cases": "hard_case_share_among_fatal",
        }
    )[
        [
            "horizon_h",
            "nonfatal_last_n",
            "fatal_last_n",
            "nonfatal_q75_threshold",
            "hard_case_n",
            "hard_case_share_among_fatal",
        ]
    ].copy()

    merged = internal_summary.merge(
        saved,
        on="horizon_h",
        suffixes=("_derived", "_saved"),
        how="inner",
    )
    mismatches: list[str] = []
    for row in merged.itertuples(index=False):
        checks = [
            (
                int(row.nonfatal_last_n_derived),
                int(row.nonfatal_last_n_saved),
                "nonfatal_last_n",
            ),
            (
                int(row.fatal_last_n_derived),
                int(row.fatal_last_n_saved),
                "fatal_last_n",
            ),
            (
                int(row.hard_case_n_derived),
                int(row.hard_case_n_saved),
                "hard_case_n",
            ),
        ]
        for derived_value, saved_value, label in checks:
            if derived_value != saved_value:
                mismatches.append(
                    f"{int(row.horizon_h)}h: derived {label}={derived_value} but saved summary has "
                    f"{saved_value}"
                )

        threshold_delta = abs(
            float(row.nonfatal_q75_threshold_derived) - float(row.nonfatal_q75_threshold_saved)
        )
        if threshold_delta > 1e-12:
            mismatches.append(
                f"{int(row.horizon_h)}h: derived nonfatal_q75_threshold="
                f"{float(row.nonfatal_q75_threshold_derived):.12f} but saved summary has "
                f"{float(row.nonfatal_q75_threshold_saved):.12f}"
            )

        share_delta = abs(
            float(row.hard_case_share_among_fatal_derived)
            - float(row.hard_case_share_among_fatal_saved)
        )
        if share_delta > 1e-12:
            mismatches.append(
                f"{int(row.horizon_h)}h: derived hard_case_share_among_fatal="
                f"{float(row.hard_case_share_among_fatal_derived):.12f} but saved summary has "
                f"{float(row.hard_case_share_among_fatal_saved):.12f}"
            )

    return tuple(mismatches)


def _compare_with_known_target(internal_summary: pd.DataFrame) -> tuple[str, ...]:
    mismatches: list[str] = []
    for row in internal_summary.itertuples(index=False):
        target = KNOWN_TARGET_SUMMARY[int(row.horizon_h)]
        if int(row.nonfatal_last_n) != target["nonfatal_last_n"]:
            mismatches.append(
                f"{int(row.horizon_h)}h nonfatal_last_n local={int(row.nonfatal_last_n)} "
                f"vs known_target={target['nonfatal_last_n']}"
            )
        if int(row.fatal_last_n) != target["fatal_last_n"]:
            mismatches.append(
                f"{int(row.horizon_h)}h fatal_last_n local={int(row.fatal_last_n)} "
                f"vs known_target={target['fatal_last_n']}"
            )
        if abs(float(row.nonfatal_q75_threshold) - target["nonfatal_q75_threshold"]) > 1e-12:
            mismatches.append(
                f"{int(row.horizon_h)}h nonfatal_q75_threshold local="
                f"{float(row.nonfatal_q75_threshold):.12f} vs known_target="
                f"{target['nonfatal_q75_threshold']:.6f}"
            )
        if int(row.hard_case_n) != target["hard_case_n"]:
            mismatches.append(
                f"{int(row.horizon_h)}h hard_case_n local={int(row.hard_case_n)} "
                f"vs known_target={target['hard_case_n']}"
            )
    return tuple(mismatches)


def _build_coverage_rows(stay_level: pd.DataFrame) -> list[dict[str, object]]:
    coverage_sets = {
        int(horizon): set(
            stay_level.loc[stay_level["horizon_h"].astype(int).eq(int(horizon)), "stay_id_global"]
            .astype(str)
            .tolist()
        )
        for horizon in sorted(int(value) for value in stay_level["horizon_h"].dropna().unique())
    }
    baseline = coverage_sets[min(coverage_sets)]
    rows: list[dict[str, object]] = []
    for horizon_h in sorted(coverage_sets):
        horizon_df = stay_level[stay_level["horizon_h"].astype(int).eq(horizon_h)].copy()
        missing_vs_baseline = sorted(baseline - coverage_sets[horizon_h])
        rows.append(
            {
                "horizon": f"{horizon_h}h",
                "stay_rows": int(horizon_df.shape[0]),
                "unique_stays": int(horizon_df["stay_id_global"].nunique()),
                "fatal_last_n": int(horizon_df["label_value"].astype(int).eq(1).sum()),
                "nonfatal_last_n": int(horizon_df["label_value"].astype(int).eq(0).sum()),
                "missing_vs_8h": ", ".join(missing_vs_baseline) if missing_vs_baseline else "none",
            }
        )
    return rows


def _build_prediction_inspection_rows(
    prediction_inspections: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for inspection in prediction_inspections:
        rows.append(
            {
                "horizon": f"{int(inspection['horizon_h'])}h",
                "prediction_path": _display_path(Path(str(inspection["prediction_path"]))),
                "format": str(inspection["prediction_format"]),
                "columns": ", ".join(str(value) for value in inspection["prediction_columns"]),
                "stay_id_field": str(inspection["stay_id_field"]),
                "label_field": str(inspection["label_field"]),
                "predicted_risk_field": str(inspection["predicted_risk_field"]),
                "hard_case_flag_field": str(inspection["hard_case_flag_field"]),
                "threshold_field": str(inspection["threshold_field"]),
            }
        )
    return rows


def _build_known_target_rows(internal_summary: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in internal_summary.itertuples(index=False):
        target = KNOWN_TARGET_SUMMARY[int(row.horizon_h)]
        rows.append(
            {
                "horizon": f"{int(row.horizon_h)}h",
                "local_nonfatal_last_n": int(row.nonfatal_last_n),
                "target_nonfatal_last_n": target["nonfatal_last_n"],
                "local_fatal_last_n": int(row.fatal_last_n),
                "target_fatal_last_n": target["fatal_last_n"],
                "local_q75": _format_float(float(row.nonfatal_q75_threshold)),
                "target_q75": _format_float(float(target["nonfatal_q75_threshold"])),
                "local_hard_case_n": int(row.hard_case_n),
                "target_hard_case_n": target["hard_case_n"],
            }
        )
    return rows


def _build_reporting_markdown(summary: pd.DataFrame) -> str:
    rows = [
        {
            "horizon": str(row.horizon),
            "nonfatal_last_n": int(row.nonfatal_last_n),
            "fatal_last_n": int(row.fatal_last_n),
            "nonfatal_q75_threshold": _format_float(float(row.nonfatal_q75_threshold)),
            "hard_case_n": int(row.hard_case_n),
            "hard_case_share_among_fatal": _format_float(
                float(row.hard_case_share_among_fatal)
            ),
        }
        for row in summary.itertuples(index=False)
    ]
    table = _markdown_table(
        rows,
        columns=[
            "horizon",
            "nonfatal_last_n",
            "fatal_last_n",
            "nonfatal_q75_threshold",
            "hard_case_n",
            "hard_case_share_among_fatal",
        ],
    )
    return "\n".join(
        [
            "# Horizon Summary",
            "",
            "Local synthetic Chapter 1 foundation summary derived from the saved ASIC logistic",
            "stay-level hard-case artifact. Values are suitable for implementation checks only and",
            "should not be used for substantive interpretation.",
            "",
            table,
            "",
        ]
    )


def _build_foundation_note(
    *,
    hard_case_dir: Path,
    stay_level_path: Path,
    saved_summary_path: Path,
    manifest: dict[str, object],
    stay_level: pd.DataFrame,
    internal_summary: pd.DataFrame,
    prediction_inspections: list[dict[str, object]],
    schema_harmonization_required: bool,
    mismatches_vs_saved_summary: tuple[str, ...],
    mismatches_vs_known_target: tuple[str, ...],
) -> str:
    coverage_rows = _build_coverage_rows(stay_level)
    prediction_rows = _build_prediction_inspection_rows(prediction_inspections)
    target_rows = _build_known_target_rows(internal_summary)

    stay_level_columns = ", ".join(stay_level.columns.tolist())
    horizons_processed = ", ".join(f"{int(horizon)}h" for horizon in internal_summary["horizon_h"])
    hospital_by_stay_unique = bool(
        stay_level.groupby("stay_id_global", observed=True)["hospital_id"].nunique(dropna=False).eq(1).all()
    )
    matching_ready = (
        stay_level["stay_id_global"].notna().all()
        and stay_level["hospital_id"].notna().all()
        and hospital_by_stay_unique
        and not schema_harmonization_required
    )

    saved_summary_status = (
        "Derived counts and thresholds exactly matched the saved `horizon_hard_case_summary.csv` artifact."
        if not mismatches_vs_saved_summary
        else "Saved summary mismatches were detected and are listed below."
    )
    target_status = (
        "All local values differ from the known HPC summary, which is expected because this repository uses small synthetic stand-in data."
        if mismatches_vs_known_target
        else "Local values happened to match the known HPC summary."
    )
    package2_status = (
        "Package 2 can proceed cleanly: stable stay matching is available via `stay_id_global`, with `hospital_id` available as a defensive secondary key. Later overlap work still needs to respect the horizon-specific coverage shrinkage at 48h and 72h."
        if matching_ready
        else "Package 2 should pause until the identifier/schema issues listed here are resolved."
    )

    note_lines = [
        "# Artifact Foundation Note",
        "",
        "Local outputs in this directory come from the repository's small synthetic stand-in data.",
        "They verify implementation and artifact contracts only; they are not scientifically interpretable.",
        "",
        "## Input Files Used",
        "",
        f"- Hard-case manifest: `{_display_path(hard_case_dir / 'run_manifest.json')}`",
        f"- Saved stay-level hard-case artifact: `{_display_path(stay_level_path)}`",
        f"- Saved horizon summary artifact: `{_display_path(saved_summary_path)}`",
        f"- Hard-case rule from manifest: `{manifest['hard_case_rule']}`",
        f"- Horizons inspected: {horizons_processed}",
        "",
        "## Shared Stay-Level Hard-Case Artifact",
        "",
        f"- Path: `{_display_path(stay_level_path)}`",
        f"- Format: `{_file_format(stay_level_path)}`",
        f"- Columns: `{stay_level_columns}`",
        "- Stable stay identifier field: `stay_id_global`",
        "- Fatal/nonfatal label field: `label_value`",
        "- Predicted-risk field: `predicted_probability`",
        "- Hard-case flag field: `hard_case_flag`",
        "- Horizon-specific threshold field: `nonfatal_q75_threshold`",
        "- Schema status: one combined harmonized stay-level file spans all five horizons via `horizon_h`.",
        "",
        "## Horizon-Specific Prediction Inputs",
        "",
        _markdown_table(
            prediction_rows,
            columns=[
                "horizon",
                "prediction_path",
                "format",
                "stay_id_field",
                "label_field",
                "predicted_risk_field",
                "hard_case_flag_field",
                "threshold_field",
            ],
        ),
        "",
        "Prediction artifact columns by horizon:",
        "",
        _markdown_table(
            prediction_rows,
            columns=["horizon", "columns"],
        ),
        "",
        "## Schema Consistency Assessment",
        "",
        f"- Prediction schemas identical across horizons: `{not schema_harmonization_required}`",
        "- Stay identifier available across all horizons: `True` (`stay_id_global` is non-missing and unique within each `stay_id_global`/`horizon_h` pair).",
        f"- `stay_id_global` maps to exactly one `hospital_id`: `{hospital_by_stay_unique}`",
        "- Any horizon missing fields needed for later overlap analysis: `False`",
        f"- Saved summary cross-check: {saved_summary_status}",
        "",
        "## Per-Horizon Stay Coverage",
        "",
        _markdown_table(
            coverage_rows,
            columns=[
                "horizon",
                "stay_rows",
                "unique_stays",
                "nonfatal_last_n",
                "fatal_last_n",
                "missing_vs_8h",
            ],
        ),
        "",
        "## Known True-Data Reference Comparison",
        "",
        target_status,
        "",
        _markdown_table(
            target_rows,
            columns=[
                "horizon",
                "local_nonfatal_last_n",
                "target_nonfatal_last_n",
                "local_fatal_last_n",
                "target_fatal_last_n",
                "local_q75",
                "target_q75",
                "local_hard_case_n",
                "target_hard_case_n",
            ],
        ),
        "",
        "## Mismatches And Ambiguities",
        "",
    ]

    if mismatches_vs_saved_summary:
        note_lines.extend(f"- Saved-summary mismatch: {item}" for item in mismatches_vs_saved_summary)
    else:
        note_lines.append("- No mismatches were found between the saved stay-level artifact and the saved horizon summary artifact.")

    if mismatches_vs_known_target:
        note_lines.extend(f"- Known-target difference: {item}" for item in mismatches_vs_known_target)
    else:
        note_lines.append("- No differences were found relative to the known true-data reference.")

    note_lines.extend(
        [
            "",
            "## Package 2 Readiness",
            "",
            package2_status,
            "",
            "Local numeric values in this note are synthetic test outputs only and should not be substantively interpreted.",
            "",
        ]
    )
    return "\n".join(note_lines)


def run_asic_horizon_dependence_foundation(
    *,
    hard_case_dir: Path = DEFAULT_HARD_CASE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    horizons: Sequence[int] | None = None,
    output_format: str = "csv",
) -> HorizonFoundationRunResult:
    normalized_horizons = _normalize_horizons(horizons)
    resolved_hard_case_dir = Path(hard_case_dir)
    manifest, stay_level_path, saved_summary_path, prediction_paths = _load_manifest_inputs(
        resolved_hard_case_dir,
        normalized_horizons,
    )
    stay_level = _validate_stay_level_artifact(stay_level_path, normalized_horizons)
    saved_summary = _validate_saved_summary_artifact(saved_summary_path, normalized_horizons)
    prediction_inspections, schema_harmonization_required = _inspect_prediction_artifacts(
        prediction_paths
    )

    internal_summary = _build_internal_horizon_summary(stay_level)
    requested_summary = _build_requested_horizon_summary(internal_summary)
    mismatches_vs_saved_summary = _compare_with_saved_summary(internal_summary, saved_summary)
    mismatches_vs_known_target = _compare_with_known_target(internal_summary)

    resolved_output_dir = Path(output_dir)
    summary_csv_path = write_dataframe(
        requested_summary,
        resolved_output_dir / "horizon_summary.csv",
        output_format=output_format,
    )
    summary_markdown_path = write_text(
        _build_reporting_markdown(requested_summary),
        resolved_output_dir / "horizon_summary.md",
    )
    note_path = write_text(
        _build_foundation_note(
            hard_case_dir=resolved_hard_case_dir,
            stay_level_path=stay_level_path,
            saved_summary_path=saved_summary_path,
            manifest=manifest,
            stay_level=stay_level,
            internal_summary=internal_summary,
            prediction_inspections=prediction_inspections,
            schema_harmonization_required=schema_harmonization_required,
            mismatches_vs_saved_summary=mismatches_vs_saved_summary,
            mismatches_vs_known_target=mismatches_vs_known_target,
        ),
        resolved_output_dir / "artifact_foundation_note.md",
    )

    cross_horizon_matching_ready = (
        stay_level["stay_id_global"].notna().all()
        and stay_level["hospital_id"].notna().all()
        and not schema_harmonization_required
        and bool(
            stay_level.groupby("stay_id_global", observed=True)["hospital_id"]
            .nunique(dropna=False)
            .eq(1)
            .all()
        )
    )

    return HorizonFoundationRunResult(
        hard_case_dir=resolved_hard_case_dir,
        output_dir=resolved_output_dir,
        horizons=normalized_horizons,
        artifacts=HorizonFoundationArtifacts(
            horizon_summary_csv_path=summary_csv_path,
            horizon_summary_markdown_path=summary_markdown_path,
            note_path=note_path,
        ),
        horizon_summary=requested_summary,
        cross_horizon_matching_ready=cross_horizon_matching_ready,
        schema_harmonization_required=schema_harmonization_required,
        mismatches_vs_saved_summary=mismatches_vs_saved_summary,
        mismatches_vs_known_target=mismatches_vs_known_target,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect frozen ASIC logistic hard-case artifacts across horizons and write the "
            "foundation summary outputs for later horizon-dependence analysis."
        )
    )
    parser.add_argument(
        "--hard-case-dir",
        type=Path,
        default=DEFAULT_HARD_CASE_DIR,
        help="Directory containing the saved logistic hard-case artifact set and run_manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where horizon_summary.csv, horizon_summary.md, and artifact_foundation_note.md will be written.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        help="Optional subset of frozen Chapter 1 horizons to inspect.",
    )
    parser.add_argument(
        "--output-format",
        choices=("csv",),
        default="csv",
        help="Output format for the machine-readable horizon summary.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = run_asic_horizon_dependence_foundation(
        hard_case_dir=args.hard_case_dir,
        output_dir=args.output_dir,
        horizons=args.horizons,
        output_format=args.output_format,
    )

    print(f"Hard-case directory: {result.hard_case_dir}")
    print(f"Output directory: {result.output_dir}")
    print(
        "Cross-horizon stay matching ready: "
        f"{'yes' if result.cross_horizon_matching_ready else 'no'}"
    )
    print(
        "Prediction schema harmonization required: "
        f"{'yes' if result.schema_harmonization_required else 'no'}"
    )
    if result.mismatches_vs_saved_summary:
        print("Saved summary mismatches:")
        for item in result.mismatches_vs_saved_summary:
            print(f"- {item}")
    else:
        print("Saved summary cross-check: exact match")

    if result.mismatches_vs_known_target:
        print("Known true-data comparison: local synthetic values differ from the HPC summary, as expected.")
    else:
        print("Known true-data comparison: local values matched the HPC summary.")

    for row in result.horizon_summary.itertuples(index=False):
        print(
            f"{row.horizon}: nonfatal_last_n={int(row.nonfatal_last_n)}, "
            f"fatal_last_n={int(row.fatal_last_n)}, "
            f"nonfatal_q75_threshold={float(row.nonfatal_q75_threshold):.6f}, "
            f"hard_case_n={int(row.hard_case_n)}, "
            f"hard_case_share_among_fatal={float(row.hard_case_share_among_fatal):.3f}"
        )
    print(f"Wrote {result.artifacts.horizon_summary_csv_path}")
    print(f"Wrote {result.artifacts.horizon_summary_markdown_path}")
    print(f"Wrote {result.artifacts.note_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
