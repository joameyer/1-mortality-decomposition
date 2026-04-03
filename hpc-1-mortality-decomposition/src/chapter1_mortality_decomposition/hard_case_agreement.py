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
from chapter1_mortality_decomposition.hard_case_definition import (
    DEFAULT_HARD_CASE_OUTPUT_ROOT,
    HARD_CASE_RULE as LOGISTIC_HARD_CASE_RULE,
    MODEL_NAME as LOGISTIC_MODEL_NAME,
    SMALL_HARD_CASE_COUNT_THRESHOLD,
    SMALL_HARD_CASE_PERCENT_THRESHOLD,
    _normalize_input_root as _normalize_hard_case_input_root,
    build_hard_case_tables_from_prediction_frames,
)
from chapter1_mortality_decomposition.utils import read_dataframe, require_columns, write_dataframe, write_text
from chapter1_mortality_decomposition.xgboost_recalibration import (
    DEFAULT_RECALIBRATION_OUTPUT_DIR,
)


DEFAULT_XGB_RECALIBRATION_METHOD = "platt"
XGB_RECALIBRATION_METHODS = ("platt", "isotonic")
SOURCE_XGBOOST_MODEL_NAME = "xgboost"
AGREEMENT_JOIN_KEYS = ["stay_id_global", "horizon_h"]
AGREEMENT_RULE = "asic_hard_case_cross_model_agreement_v1"
XGB_RECAL_HARD_CASE_RULE_TEMPLATE = "asic_{model_name}_last_eligible_nonfatal_q75_v1"
XGB_RECAL_VARIANT_MODEL_NAMES = {
    "platt": "xgboost_platt",
    "isotonic": "xgboost_isotonic",
}
DEFAULT_AGREEMENT_OUTPUT_DIR = (
    DEFAULT_HARD_CASE_OUTPUT_ROOT
    / "agreement"
    / f"{LOGISTIC_MODEL_NAME}_vs_{XGB_RECAL_VARIANT_MODEL_NAMES[DEFAULT_XGB_RECALIBRATION_METHOD]}"
)
SMALL_AGREEMENT_COUNT_THRESHOLD = 20
SMALL_AGREEMENT_PERCENT_THRESHOLD = 0.05


@dataclass(frozen=True)
class HardCaseAgreementArtifacts:
    stay_level_path: Path
    horizon_summary_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class HardCaseAgreementRunResult:
    logistic_input_root: Path
    xgb_recalibration_root: Path
    output_dir: Path
    horizons_processed: tuple[int, ...]
    logistic_model_name: str
    xgb_recal_model_name: str
    xgb_recalibration_method: str
    artifacts: HardCaseAgreementArtifacts
    stay_level_agreement: pd.DataFrame
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


def _normalize_xgb_recalibration_root(recalibration_root: Path) -> Path:
    root = Path(recalibration_root)
    if root.name == "xgboost" and any(path.is_dir() for path in root.glob("horizon_*h")):
        return root
    candidate = root / "xgboost"
    if candidate.is_dir():
        return candidate
    raise ValueError(
        f"Could not locate saved XGBoost recalibration artifacts from {root}. "
        f"Expected either {root} as the xgboost recalibration directory or {candidate}."
    )


def _load_logistic_prediction_frames(
    input_root: Path,
    horizons: Sequence[int] | None,
) -> tuple[dict[int, pd.DataFrame], dict[int, str], tuple[int, ...]]:
    baseline_root = _normalize_hard_case_input_root(Path(input_root), model_name=LOGISTIC_MODEL_NAME)
    artifacts = _discover_prediction_artifacts(
        baseline_root,
        models=[LOGISTIC_MODEL_NAME],
        horizons=horizons or DEFAULT_HORIZONS,
    )
    if not artifacts:
        raise ValueError(
            f"No {LOGISTIC_MODEL_NAME!r} prediction artifacts were discovered under {baseline_root}."
        )

    frames_by_horizon = {
        int(artifact.horizon_h): _load_prediction_frame(artifact)
        for artifact in artifacts
    }
    source_paths_by_horizon = {
        int(artifact.horizon_h): str(artifact.predictions_path.resolve())
        for artifact in artifacts
    }
    selected_horizons = tuple(sorted(frames_by_horizon))
    return frames_by_horizon, source_paths_by_horizon, selected_horizons


def _canonical_xgb_prediction_path(
    recalibration_root: Path,
    *,
    horizon_h: int,
    recalibration_method: str,
) -> Path:
    return (
        recalibration_root
        / f"horizon_{int(horizon_h)}h"
        / f"xgboost_{recalibration_method}_canonical_predictions.csv"
    )


def _method_xgb_prediction_path(
    recalibration_root: Path,
    *,
    horizon_h: int,
    recalibration_method: str,
) -> Path:
    return (
        recalibration_root
        / f"horizon_{int(horizon_h)}h"
        / f"{recalibration_method}_predictions.csv"
    )


def _standardize_recalibrated_xgboost_predictions(
    predictions_path: Path,
    *,
    horizon_h: int,
    recalibration_method: str,
) -> pd.DataFrame:
    model_variant_name = XGB_RECAL_VARIANT_MODEL_NAMES[recalibration_method]
    predictions = read_dataframe(predictions_path)

    canonical_required_columns = {
        "instance_id",
        "stay_id_global",
        "hospital_id",
        "block_index",
        "prediction_time_h",
        "horizon_h",
        "split",
        "label_value",
        "model_variant",
        "predicted_probability",
    }
    method_required_columns = {
        "instance_id",
        "stay_id_global",
        "hospital_id",
        "block_index",
        "prediction_time_h",
        "horizon_h",
        "split",
        "label_value",
        "model_name",
        "recalibrated_probability",
        "recalibration_method",
    }

    standardized = predictions.copy()
    if canonical_required_columns.issubset(standardized.columns):
        standardized["model_variant"] = standardized["model_variant"].astype("string")
        observed_variants = standardized["model_variant"].dropna().unique().tolist()
        if observed_variants != [model_variant_name]:
            raise ValueError(
                f"{predictions_path} should contain only model_variant={model_variant_name!r}, "
                f"found {observed_variants}."
            )
        standardized["model_name"] = model_variant_name
        standardized["predicted_probability"] = pd.to_numeric(
            standardized["predicted_probability"],
            errors="coerce",
        )
        return standardized

    require_columns(standardized, method_required_columns, str(predictions_path))
    standardized["model_name"] = standardized["model_name"].astype("string")
    observed_model_names = standardized["model_name"].dropna().unique().tolist()
    if observed_model_names != [SOURCE_XGBOOST_MODEL_NAME]:
        raise ValueError(
            f"{predictions_path} should contain only model_name={SOURCE_XGBOOST_MODEL_NAME!r}, "
            f"found {observed_model_names}."
        )
    standardized["recalibration_method"] = standardized["recalibration_method"].astype("string")
    observed_methods = standardized["recalibration_method"].dropna().unique().tolist()
    if observed_methods != [recalibration_method]:
        raise ValueError(
            f"{predictions_path} should contain only recalibration_method={recalibration_method!r}, "
            f"found {observed_methods}."
        )
    if "recalibration_status" in standardized.columns:
        observed_statuses = (
            standardized["recalibration_status"].astype("string").dropna().unique().tolist()
        )
        if observed_statuses != ["fit"]:
            raise ValueError(
                f"{predictions_path} should contain only fit recalibration_status values, "
                f"found {observed_statuses}."
            )
    standardized["model_name"] = model_variant_name
    standardized["predicted_probability"] = pd.to_numeric(
        standardized["recalibrated_probability"],
        errors="coerce",
    )
    observed_horizons = (
        pd.to_numeric(standardized["horizon_h"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    if observed_horizons != [int(horizon_h)]:
        raise ValueError(
            f"{predictions_path} should contain only horizon {int(horizon_h)}, found {observed_horizons}."
        )
    return standardized


def _load_recalibrated_xgboost_prediction_frames(
    recalibration_root: Path,
    horizons: Sequence[int] | None,
    *,
    recalibration_method: str,
) -> tuple[dict[int, pd.DataFrame], dict[int, str], tuple[int, ...]]:
    normalized_root = _normalize_xgb_recalibration_root(Path(recalibration_root))
    selected_horizons = tuple(int(horizon) for horizon in (horizons or DEFAULT_HORIZONS))

    frames_by_horizon: dict[int, pd.DataFrame] = {}
    source_paths_by_horizon: dict[int, str] = {}
    for horizon_h in selected_horizons:
        canonical_path = _canonical_xgb_prediction_path(
            normalized_root,
            horizon_h=horizon_h,
            recalibration_method=recalibration_method,
        )
        method_path = _method_xgb_prediction_path(
            normalized_root,
            horizon_h=horizon_h,
            recalibration_method=recalibration_method,
        )
        selected_path = canonical_path if canonical_path.exists() else method_path
        if not selected_path.exists():
            raise FileNotFoundError(
                "Expected saved recalibrated XGBoost predictions at one of: "
                f"{canonical_path}, {method_path}"
            )
        frames_by_horizon[int(horizon_h)] = _standardize_recalibrated_xgboost_predictions(
            selected_path,
            horizon_h=int(horizon_h),
            recalibration_method=recalibration_method,
        )
        source_paths_by_horizon[int(horizon_h)] = str(selected_path.resolve())

    return frames_by_horizon, source_paths_by_horizon, tuple(sorted(frames_by_horizon))


def _validate_agreement_input_stay_level(
    stay_level: pd.DataFrame,
    *,
    source_name: str,
) -> pd.DataFrame:
    require_columns(
        stay_level,
        {
            "stay_id_global",
            "hospital_id",
            "horizon_h",
            "label_value",
            "instance_id",
            "block_index",
            "prediction_time_h",
            "predicted_probability",
            "nonfatal_q75_threshold",
            "hard_case_flag",
            "hard_case_rule",
            "model_name",
        },
        source_name,
    )
    validated = stay_level.copy()
    validated["stay_id_global"] = validated["stay_id_global"].astype("string")
    validated["hospital_id"] = validated["hospital_id"].astype("string")
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
    validated["hard_case_flag"] = validated["hard_case_flag"].astype(bool)

    duplicate_count = int(validated.duplicated(subset=AGREEMENT_JOIN_KEYS).sum())
    if duplicate_count:
        raise ValueError(
            f"{source_name} contains duplicated stay-horizon rows for agreement join: {duplicate_count}."
        )
    if validated["predicted_probability"].isna().any():
        raise ValueError(f"{source_name} contains missing predicted_probability values.")
    if validated["nonfatal_q75_threshold"].isna().any():
        raise ValueError(f"{source_name} contains missing nonfatal_q75_threshold values.")
    if not validated["predicted_probability"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError(f"{source_name} contains predicted_probability values outside [0, 1].")
    if not validated["nonfatal_q75_threshold"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError(f"{source_name} contains nonfatal_q75_threshold values outside [0, 1].")
    return validated


def _prepare_fatal_subset(
    stay_level: pd.DataFrame,
    *,
    model_prefix: str,
    source_name: str,
) -> pd.DataFrame:
    validated = _validate_agreement_input_stay_level(stay_level, source_name=source_name)
    fatal_subset = validated[validated["label_value"].astype(int).eq(1)].copy()
    if fatal_subset.empty:
        raise ValueError(f"{source_name} contains no fatal stay-level rows for agreement analysis.")

    renamed = fatal_subset.rename(
        columns={
            "hospital_id": f"{model_prefix}_hospital_id",
            "label_value": f"{model_prefix}_label_value",
            "instance_id": f"{model_prefix}_instance_id",
            "block_index": f"{model_prefix}_block_index",
            "prediction_time_h": f"{model_prefix}_prediction_time_h",
            "predicted_probability": f"{model_prefix}_predicted_probability",
            "nonfatal_q75_threshold": f"{model_prefix}_nonfatal_q75_threshold",
            "hard_case_flag": f"{model_prefix}_hard_case_flag",
            "hard_case_rule": f"{model_prefix}_hard_case_rule",
            "model_name": f"{model_prefix}_model_name",
        }
    )
    return renamed


def build_hard_case_agreement_tables(
    logistic_stay_level: pd.DataFrame,
    xgb_recal_stay_level: pd.DataFrame,
    *,
    logistic_model_name: str,
    xgb_recal_model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logistic_fatal = _prepare_fatal_subset(
        logistic_stay_level,
        model_prefix="logistic",
        source_name="logistic_stay_level",
    )
    xgb_fatal = _prepare_fatal_subset(
        xgb_recal_stay_level,
        model_prefix="xgb_recal",
        source_name="xgb_recal_stay_level",
    )

    merged = logistic_fatal.merge(
        xgb_fatal,
        on=AGREEMENT_JOIN_KEYS,
        how="outer",
        indicator=True,
    )

    matched = merged[merged["_merge"].eq("both")].copy()
    if matched.empty:
        raise ValueError("No fatal stay-level overlap exists between logistic and recalibrated XGBoost.")

    hospital_mismatch = matched[
        matched["logistic_hospital_id"].astype("string").ne(
            matched["xgb_recal_hospital_id"].astype("string")
        )
    ]
    if not hospital_mismatch.empty:
        examples = hospital_mismatch[
            [
                "stay_id_global",
                "horizon_h",
                "logistic_hospital_id",
                "xgb_recal_hospital_id",
            ]
        ].head(5)
        raise ValueError(
            "Hospital ID mismatch detected while joining fatal stay-level hard-case outputs. "
            f"Examples: {examples.to_dict(orient='records')}"
        )

    label_mismatch = matched[
        matched["logistic_label_value"].astype(int).ne(
            matched["xgb_recal_label_value"].astype(int)
        )
    ]
    if not label_mismatch.empty:
        raise ValueError(
            "Label mismatch detected while joining fatal stay-level hard-case outputs."
        )

    matched["hospital_id"] = matched["logistic_hospital_id"].astype("string")
    matched["logistic_hard_case_flag"] = matched["logistic_hard_case_flag"].astype(bool)
    matched["xgb_recal_hard_case_flag"] = matched["xgb_recal_hard_case_flag"].astype(bool)
    matched["hard_case_agreement_flag"] = (
        matched["logistic_hard_case_flag"] & matched["xgb_recal_hard_case_flag"]
    )
    matched["hard_case_logistic_only_flag"] = (
        matched["logistic_hard_case_flag"] & ~matched["xgb_recal_hard_case_flag"]
    )
    matched["hard_case_xgb_only_flag"] = (
        ~matched["logistic_hard_case_flag"] & matched["xgb_recal_hard_case_flag"]
    )
    matched["agreement_rule"] = AGREEMENT_RULE

    stay_level_agreement = matched[
        [
            "stay_id_global",
            "hospital_id",
            "horizon_h",
            "logistic_predicted_probability",
            "logistic_nonfatal_q75_threshold",
            "logistic_hard_case_flag",
            "logistic_instance_id",
            "logistic_block_index",
            "logistic_prediction_time_h",
            "logistic_hard_case_rule",
            "xgb_recal_predicted_probability",
            "xgb_recal_nonfatal_q75_threshold",
            "xgb_recal_hard_case_flag",
            "xgb_recal_instance_id",
            "xgb_recal_block_index",
            "xgb_recal_prediction_time_h",
            "xgb_recal_hard_case_rule",
            "hard_case_agreement_flag",
            "hard_case_logistic_only_flag",
            "hard_case_xgb_only_flag",
            "agreement_rule",
        ]
    ].copy()
    stay_level_agreement["logistic_model_name"] = logistic_model_name
    stay_level_agreement["xgb_recal_model_name"] = xgb_recal_model_name
    stay_level_agreement = stay_level_agreement[
        [
            "stay_id_global",
            "hospital_id",
            "horizon_h",
            "logistic_model_name",
            "logistic_predicted_probability",
            "logistic_nonfatal_q75_threshold",
            "logistic_hard_case_flag",
            "logistic_instance_id",
            "logistic_block_index",
            "logistic_prediction_time_h",
            "logistic_hard_case_rule",
            "xgb_recal_model_name",
            "xgb_recal_predicted_probability",
            "xgb_recal_nonfatal_q75_threshold",
            "xgb_recal_hard_case_flag",
            "xgb_recal_instance_id",
            "xgb_recal_block_index",
            "xgb_recal_prediction_time_h",
            "xgb_recal_hard_case_rule",
            "hard_case_agreement_flag",
            "hard_case_logistic_only_flag",
            "hard_case_xgb_only_flag",
            "agreement_rule",
        ]
    ].sort_values(["horizon_h", "hospital_id", "stay_id_global"], kind="stable").reset_index(drop=True)

    summary_rows: list[dict[str, object]] = []
    unmatched_counts = (
        merged.assign(
            left_only=lambda frame: frame["_merge"].eq("left_only"),
            right_only=lambda frame: frame["_merge"].eq("right_only"),
        )
        .groupby("horizon_h", dropna=False)
        .agg(
            n_fatal_logistic_only_available=("left_only", "sum"),
            n_fatal_xgb_recal_only_available=("right_only", "sum"),
        )
        .reset_index()
    )
    unmatched_counts["n_fatal_dropped_unmatched"] = (
        unmatched_counts["n_fatal_logistic_only_available"]
        + unmatched_counts["n_fatal_xgb_recal_only_available"]
    )
    unmatched_counts = unmatched_counts.set_index("horizon_h")

    for horizon_h, horizon_df in stay_level_agreement.groupby("horizon_h", dropna=False):
        horizon_key = int(horizon_h)
        n_fatal_with_both = int(horizon_df.shape[0])
        if n_fatal_with_both == 0:
            raise ValueError(f"Horizon {horizon_key} has an empty agreement population.")

        n_logistic_hard = int(horizon_df["logistic_hard_case_flag"].sum())
        n_xgb_recal_hard = int(horizon_df["xgb_recal_hard_case_flag"].sum())
        n_both_hard = int(horizon_df["hard_case_agreement_flag"].sum())
        n_logistic_only = int(horizon_df["hard_case_logistic_only_flag"].sum())
        n_xgb_only = int(horizon_df["hard_case_xgb_only_flag"].sum())
        n_union_hard = int(
            (
                horizon_df["logistic_hard_case_flag"]
                | horizon_df["xgb_recal_hard_case_flag"]
            ).sum()
        )
        pct_logistic_hard = float(n_logistic_hard / n_fatal_with_both)
        pct_xgb_recal_hard = float(n_xgb_recal_hard / n_fatal_with_both)
        pct_both_hard_among_fatal = float(n_both_hard / n_fatal_with_both)
        jaccard = float(n_both_hard / n_union_hard) if n_union_hard else np.nan
        logistic_confirmed = float(n_both_hard / n_logistic_hard) if n_logistic_hard else np.nan
        xgb_confirmed = float(n_both_hard / n_xgb_recal_hard) if n_xgb_recal_hard else np.nan

        warning_reasons: list[str] = []
        if n_both_hard < SMALL_AGREEMENT_COUNT_THRESHOLD:
            warning_reasons.append(f"n_both_hard_lt_{SMALL_AGREEMENT_COUNT_THRESHOLD}")
        if pct_both_hard_among_fatal < SMALL_AGREEMENT_PERCENT_THRESHOLD:
            warning_reasons.append(
                f"pct_both_hard_among_fatal_lt_{SMALL_AGREEMENT_PERCENT_THRESHOLD:.2f}"
            )

        unmatched_row = unmatched_counts.loc[horizon_key]
        summary_rows.append(
            {
                "horizon_h": horizon_key,
                "logistic_model_name": logistic_model_name,
                "xgb_recal_model_name": xgb_recal_model_name,
                "n_fatal_with_both_models_available": n_fatal_with_both,
                "n_fatal_logistic_only_available": int(
                    unmatched_row["n_fatal_logistic_only_available"]
                ),
                "n_fatal_xgb_recal_only_available": int(
                    unmatched_row["n_fatal_xgb_recal_only_available"]
                ),
                "n_fatal_dropped_unmatched": int(unmatched_row["n_fatal_dropped_unmatched"]),
                "n_logistic_hard": n_logistic_hard,
                "n_xgb_recal_hard": n_xgb_recal_hard,
                "n_both_hard": n_both_hard,
                "n_logistic_only": n_logistic_only,
                "n_xgb_recal_only": n_xgb_only,
                "pct_logistic_hard": pct_logistic_hard,
                "pct_xgb_recal_hard": pct_xgb_recal_hard,
                "pct_both_hard_among_fatal": pct_both_hard_among_fatal,
                "hard_case_union_count": n_union_hard,
                "jaccard_hard_case_overlap": jaccard,
                "pct_logistic_hard_confirmed_by_xgb": logistic_confirmed,
                "pct_xgb_recal_hard_confirmed_by_logistic": xgb_confirmed,
                "agreement_subgroup_warning": bool(warning_reasons),
                "warning_reason": "; ".join(warning_reasons) if warning_reasons else pd.NA,
            }
        )

    horizon_summary = pd.DataFrame(summary_rows).sort_values(["horizon_h"], kind="stable").reset_index(drop=True)
    return stay_level_agreement, horizon_summary


def run_asic_hard_case_agreement_sensitivity(
    *,
    logistic_input_root: Path = DEFAULT_BASELINE_ARTIFACT_ROOT,
    xgb_recalibration_root: Path = DEFAULT_RECALIBRATION_OUTPUT_DIR,
    xgb_recalibration_method: str = DEFAULT_XGB_RECALIBRATION_METHOD,
    output_dir: Path | None = None,
    horizons: Sequence[int] | None = None,
    output_format: str = "csv",
) -> HardCaseAgreementRunResult:
    if xgb_recalibration_method not in XGB_RECALIBRATION_METHODS:
        raise ValueError(
            f"Unsupported XGBoost recalibration method: {xgb_recalibration_method}. "
            f"Expected one of {list(XGB_RECALIBRATION_METHODS)}."
        )

    logistic_frames, logistic_sources, logistic_horizons = _load_logistic_prediction_frames(
        logistic_input_root,
        horizons,
    )
    xgb_frames, xgb_sources, xgb_horizons = _load_recalibrated_xgboost_prediction_frames(
        xgb_recalibration_root,
        horizons,
        recalibration_method=xgb_recalibration_method,
    )

    if logistic_horizons != xgb_horizons:
        raise ValueError(
            f"Logistic horizons {list(logistic_horizons)} and recalibrated XGBoost horizons "
            f"{list(xgb_horizons)} do not match."
        )

    xgb_recal_model_name = XGB_RECAL_VARIANT_MODEL_NAMES[xgb_recalibration_method]
    xgb_recal_hard_case_rule = XGB_RECAL_HARD_CASE_RULE_TEMPLATE.format(
        model_name=xgb_recal_model_name
    )
    resolved_output_dir = Path(
        output_dir
        if output_dir is not None
        else (
            DEFAULT_HARD_CASE_OUTPUT_ROOT
            / "agreement"
            / f"{LOGISTIC_MODEL_NAME}_vs_{xgb_recal_model_name}"
        )
    )

    logistic_stay_level, logistic_summary, _ = build_hard_case_tables_from_prediction_frames(
        logistic_frames,
        source_names_by_horizon=logistic_sources,
        expected_source_model_name=LOGISTIC_MODEL_NAME,
        probability_column="predicted_probability",
        output_model_name=LOGISTIC_MODEL_NAME,
        hard_case_rule=LOGISTIC_HARD_CASE_RULE,
    )
    xgb_stay_level, xgb_summary, _ = build_hard_case_tables_from_prediction_frames(
        xgb_frames,
        source_names_by_horizon=xgb_sources,
        expected_source_model_name=xgb_recal_model_name,
        probability_column="predicted_probability",
        output_model_name=xgb_recal_model_name,
        hard_case_rule=xgb_recal_hard_case_rule,
    )

    stay_level_agreement, horizon_summary = build_hard_case_agreement_tables(
        logistic_stay_level,
        xgb_stay_level,
        logistic_model_name=LOGISTIC_MODEL_NAME,
        xgb_recal_model_name=xgb_recal_model_name,
    )

    extension = _output_extension(output_format)
    stay_level_path = write_dataframe(
        stay_level_agreement,
        resolved_output_dir / f"fatal_stay_level_hard_case_agreement.{extension}",
        output_format=output_format,
    )
    horizon_summary_path = write_dataframe(
        horizon_summary,
        resolved_output_dir / f"horizon_hard_case_agreement_summary.{extension}",
        output_format=output_format,
    )
    manifest_path = _write_json(
        {
            "timestamp_utc": _utc_timestamp(),
            "agreement_rule": AGREEMENT_RULE,
            "join_keys": AGREEMENT_JOIN_KEYS,
            "logistic_model_name": LOGISTIC_MODEL_NAME,
            "logistic_hard_case_rule": LOGISTIC_HARD_CASE_RULE,
            "xgb_recal_model_name": xgb_recal_model_name,
            "xgb_recalibration_method": xgb_recalibration_method,
            "xgb_recal_hard_case_rule": xgb_recal_hard_case_rule,
            "logistic_input_root": str(
                _normalize_hard_case_input_root(
                    Path(logistic_input_root),
                    model_name=LOGISTIC_MODEL_NAME,
                ).resolve()
            ),
            "xgb_recalibration_root": str(_normalize_xgb_recalibration_root(Path(xgb_recalibration_root)).resolve()),
            "output_dir": str(resolved_output_dir.resolve()),
            "horizons_processed": list(logistic_horizons),
            "small_hard_case_count_threshold": SMALL_HARD_CASE_COUNT_THRESHOLD,
            "small_hard_case_percent_threshold": SMALL_HARD_CASE_PERCENT_THRESHOLD,
            "small_agreement_count_threshold": SMALL_AGREEMENT_COUNT_THRESHOLD,
            "small_agreement_percent_threshold": SMALL_AGREEMENT_PERCENT_THRESHOLD,
            "logistic_prediction_sources_by_horizon": logistic_sources,
            "xgb_recal_prediction_sources_by_horizon": xgb_sources,
            "logistic_horizon_summary_rows": logistic_summary.to_dict(orient="records"),
            "xgb_recal_horizon_summary_rows": xgb_summary.to_dict(orient="records"),
            "stay_level_agreement_artifact": str(stay_level_path.resolve()),
            "horizon_summary_artifact": str(horizon_summary_path.resolve()),
        },
        resolved_output_dir / "run_manifest.json",
    )

    return HardCaseAgreementRunResult(
        logistic_input_root=_normalize_hard_case_input_root(
            Path(logistic_input_root),
            model_name=LOGISTIC_MODEL_NAME,
        ),
        xgb_recalibration_root=_normalize_xgb_recalibration_root(Path(xgb_recalibration_root)),
        output_dir=resolved_output_dir,
        horizons_processed=logistic_horizons,
        logistic_model_name=LOGISTIC_MODEL_NAME,
        xgb_recal_model_name=xgb_recal_model_name,
        xgb_recalibration_method=xgb_recalibration_method,
        artifacts=HardCaseAgreementArtifacts(
            stay_level_path=stay_level_path,
            horizon_summary_path=horizon_summary_path,
            manifest_path=manifest_path,
        ),
        stay_level_agreement=stay_level_agreement,
        horizon_summary=horizon_summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Chapter 1 ASIC hard-case agreement sensitivity analysis between logistic "
            "regression and recalibrated XGBoost."
        )
    )
    parser.add_argument(
        "--logistic-input-root",
        type=Path,
        default=DEFAULT_BASELINE_ARTIFACT_ROOT,
        help=(
            "Root baseline artifact directory containing model subdirectories, or the "
            "logistic_regression model directory itself."
        ),
    )
    parser.add_argument(
        "--xgb-recalibration-root",
        type=Path,
        default=DEFAULT_RECALIBRATION_OUTPUT_DIR,
        help="Directory containing saved XGBoost recalibration artifacts.",
    )
    parser.add_argument(
        "--xgb-recalibration-method",
        choices=XGB_RECALIBRATION_METHODS,
        default=DEFAULT_XGB_RECALIBRATION_METHOD,
        help="Saved recalibrated XGBoost variant to use for the sensitivity analysis.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where agreement artifacts will be written.",
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
        help="Output format for the written agreement artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = run_asic_hard_case_agreement_sensitivity(
        logistic_input_root=args.logistic_input_root,
        xgb_recalibration_root=args.xgb_recalibration_root,
        xgb_recalibration_method=args.xgb_recalibration_method,
        output_dir=args.output_dir,
        horizons=args.horizons,
        output_format=args.output_format,
    )

    print(f"Stay-level agreement artifact: {result.artifacts.stay_level_path}")
    print(f"Horizon summary artifact: {result.artifacts.horizon_summary_path}")
    print(f"Run manifest: {result.artifacts.manifest_path}")
    for row in result.horizon_summary.itertuples(index=False):
        warning_text = row.warning_reason if isinstance(row.warning_reason, str) and row.warning_reason else "none"
        print(
            f"horizon {int(row.horizon_h)}h -> matched_fatal={int(row.n_fatal_with_both_models_available)}, "
            f"logistic_hard={int(row.n_logistic_hard)}, xgb_recal_hard={int(row.n_xgb_recal_hard)}, "
            f"both={int(row.n_both_hard)}, logistic_only={int(row.n_logistic_only)}, "
            f"xgb_only={int(row.n_xgb_recal_only)}, jaccard="
            f"{float(row.jaccard_hard_case_overlap) if pd.notna(row.jaccard_hard_case_overlap) else float('nan'):.3f}, "
            f"warnings: {warning_text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
