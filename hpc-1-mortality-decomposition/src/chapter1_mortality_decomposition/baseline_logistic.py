from __future__ import annotations

import argparse
import json
import pickle
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chapter1_mortality_decomposition.model_ready import build_chapter1_model_ready_dataset
from chapter1_mortality_decomposition.run_config import (
    DEFAULT_CHAPTER1_RUN_CONFIG_PATH,
    load_chapter1_run_config,
)
from chapter1_mortality_decomposition.utils import (
    ensure_directory,
    read_dataframe,
    require_columns,
    write_dataframe,
    write_text,
)


DEFAULT_PRIMARY_MODEL_READY_DATASET_PATH = (
    Path("artifacts") / "chapter1" / "model_ready" / "chapter1_primary_model_ready_dataset.csv"
)
DEFAULT_FEATURE_SET_DEFINITION_PATH = (
    Path("artifacts") / "chapter1" / "feature_sets" / "chapter1_feature_set_definition.csv"
)
DEFAULT_LOGISTIC_BASELINE_OUTPUT_DIR = (
    Path("artifacts")
    / "chapter1"
    / "baselines"
    / "asic"
    / "primary_medians"
    / "logistic_regression"
)

PRIMARY_FEATURE_SET_NAME = "primary"
MODEL_NAME = "logistic_regression"
SUPPORTED_DYNAMIC_STATISTICS = {"obs_count", "mean", "median", "min", "max", "last"}
EXPECTED_BASELINE_SPLITS = ("train", "validation", "test")
IDENTIFIER_COLUMNS = [
    "instance_id",
    "stay_id_global",
    "hospital_id",
    "block_index",
    "prediction_time_h",
    "horizon_h",
    "split",
    "label_value",
]
REQUIRED_MODEL_READY_COLUMNS = set(IDENTIFIER_COLUMNS)
ALL_VALID_PREDICTION_COLUMNS = [
    "instance_id",
    "stay_id_global",
    "hospital_id",
    "block_index",
    "block_start_h",
    "block_end_h",
    "prediction_time_h",
    "horizon_h",
    "split",
    "label_value",
    "is_labelable",
    "unlabeled_reason",
    "predicted_probability",
    "model_name",
]
PRIMARY_PROXY_LABEL_BASENAME = "chapter1_proxy_horizon_labels"
PRIMARY_STAY_SPLIT_BASENAME = "chapter1_stay_split_assignments"
ALL_VALID_PREDICTIONS_FILENAME = "all_valid_predictions.csv"
ALL_VALID_PREDICTION_QC_FILENAME = "all_valid_prediction_qc.csv"


@dataclass(frozen=True)
class LogisticBaselineArtifacts:
    predictions_path: Path
    all_valid_predictions_path: Path
    all_valid_prediction_qc_path: Path
    metrics_path: Path
    metadata_path: Path
    selected_features_path: Path
    preprocessing_path: Path
    model_path: Path
    pipeline_path: Path


@dataclass(frozen=True)
class HorizonLogisticBaselineResult:
    horizon_h: int
    feature_columns: tuple[str, ...]
    warnings: tuple[str, ...]
    artifacts: LogisticBaselineArtifacts


@dataclass(frozen=True)
class LogisticBaselineRunResult:
    input_dataset_path: Path
    feature_set_definition_path: Path
    output_dir: Path
    selected_feature_columns: tuple[str, ...]
    horizons_processed: tuple[int, ...]
    horizon_results: tuple[HorizonLogisticBaselineResult, ...]
    summary_path: Path
    manifest_path: Path


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


def _write_pickle(obj: object, path: Path) -> Path:
    ensure_directory(path.parent)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)
    return path


def _artifact_format_from_path(path: Path) -> str:
    return "parquet" if path.suffix.lower() == ".parquet" else "csv"


def _resolve_preprocessing_artifact_root(
    input_dataset_path: Path,
    preprocessing_root: Path | None,
) -> Path:
    if preprocessing_root is not None:
        return Path(preprocessing_root)

    input_path = Path(input_dataset_path)
    if input_path.parent.name == "model_ready":
        return input_path.parent.parent

    raise ValueError(
        "Could not infer the Chapter 1 preprocessing artifact root from the baseline input "
        f"dataset path {input_path}. Pass preprocessing_root explicitly."
    )


def _resolve_standardized_input_settings(
    *,
    standardized_input_dir: Path | None,
    standardized_input_format: str | None,
    run_config_path: Path | None,
) -> tuple[Path, str]:
    if standardized_input_dir is not None:
        return Path(standardized_input_dir), str(standardized_input_format or "csv")

    resolved_run_config_path = (
        Path(run_config_path) if run_config_path is not None else DEFAULT_CHAPTER1_RUN_CONFIG_PATH
    )
    run_config = load_chapter1_run_config(resolved_run_config_path)
    return run_config.input_dir, str(standardized_input_format or run_config.input_format)


def _load_all_valid_scoring_inputs(
    *,
    input_dir: Path,
    input_format: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    extension = "parquet" if str(input_format).strip().lower() == "parquet" else "csv"
    blocked_dynamic_features_path = input_dir / "blocked" / f"asic_8h_blocked_dynamic_features.{extension}"
    mech_vent_episode_level_path = (
        input_dir / "qc" / f"mech_vent_ge_24h_episode_level.{extension}"
    )
    missing_paths = [
        str(path)
        for path in (blocked_dynamic_features_path, mech_vent_episode_level_path)
        if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Missing standardized ASIC inputs required for all-valid Chapter 1 scoring: "
            + ", ".join(missing_paths)
        )

    return (
        read_dataframe(blocked_dynamic_features_path),
        read_dataframe(mech_vent_episode_level_path),
    )


def build_primary_all_valid_scoring_dataset(
    *,
    input_dataset_path: Path,
    feature_set_definition: pd.DataFrame,
    preprocessing_root: Path | None = None,
    standardized_input_dir: Path | None = None,
    standardized_input_format: str | None = None,
    run_config_path: Path | None = None,
) -> pd.DataFrame:
    input_path = Path(input_dataset_path)
    artifact_root = _resolve_preprocessing_artifact_root(input_path, preprocessing_root)
    artifact_format = _artifact_format_from_path(input_path)
    labels_path = artifact_root / "labels" / f"{PRIMARY_PROXY_LABEL_BASENAME}.{artifact_format}"
    stay_splits_path = artifact_root / "splits" / f"{PRIMARY_STAY_SPLIT_BASENAME}.{artifact_format}"

    missing_paths = [
        str(path)
        for path in (labels_path, stay_splits_path)
        if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Missing Chapter 1 preprocessing artifacts required for all-valid scoring: "
            + ", ".join(missing_paths)
        )

    standardized_dir, standardized_format = _resolve_standardized_input_settings(
        standardized_input_dir=standardized_input_dir,
        standardized_input_format=standardized_input_format,
        run_config_path=run_config_path,
    )
    blocked_dynamic_features, mech_vent_episode_level = _load_all_valid_scoring_inputs(
        input_dir=standardized_dir,
        input_format=standardized_format,
    )

    proxy_labels = read_dataframe(labels_path)
    stay_split_assignments = read_dataframe(stay_splits_path)
    primary_feature_set_definition = feature_set_definition[
        feature_set_definition["feature_set_name"].astype("string").eq(PRIMARY_FEATURE_SET_NAME)
    ].reset_index(drop=True)
    if primary_feature_set_definition.empty:
        raise ValueError(
            "No primary feature-set rows were found while building the all-valid scoring dataset."
        )

    all_valid_model_ready = build_chapter1_model_ready_dataset(
        usable_labels=proxy_labels,
        blocked_dynamic_features=blocked_dynamic_features,
        feature_set_definition=primary_feature_set_definition,
        feature_set_name=PRIMARY_FEATURE_SET_NAME,
        mech_vent_episode_level=mech_vent_episode_level,
        stay_split_assignments=stay_split_assignments,
    ).table
    validate_expected_split_labels(all_valid_model_ready, dataset_name="all_valid_scoring_dataset")
    return all_valid_model_ready


def build_all_valid_prediction_frame(
    scoring_df: pd.DataFrame,
    predicted_probability: np.ndarray,
    *,
    model_name: str,
) -> pd.DataFrame:
    if scoring_df.shape[0] != int(len(predicted_probability)):
        raise ValueError(
            "All-valid prediction export received mismatched row and prediction counts: "
            f"{scoring_df.shape[0]} rows vs {len(predicted_probability)} predictions."
        )

    output = scoring_df.loc[
        :,
        [
            column
            for column in (
                "instance_id",
                "stay_id_global",
                "hospital_id",
                "block_index",
                "block_start_h",
                "block_end_h",
                "prediction_time_h",
                "horizon_h",
                "split",
                "label_value",
                "unlabeled_reason",
            )
            if column in scoring_df.columns
        ],
    ].copy()
    if "split" not in output.columns:
        output["split"] = pd.Series(pd.NA, index=output.index, dtype="string")
    else:
        output["split"] = output["split"].astype("string")
    if "label_value" not in output.columns:
        output["label_value"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    else:
        output["label_value"] = pd.to_numeric(output["label_value"], errors="coerce").astype("Int64")
    if "unlabeled_reason" not in output.columns:
        output["unlabeled_reason"] = pd.Series(pd.NA, index=output.index, dtype="string")
    else:
        output["unlabeled_reason"] = output["unlabeled_reason"].astype("string")

    if "proxy_horizon_labelable" in scoring_df.columns:
        output["is_labelable"] = scoring_df["proxy_horizon_labelable"].astype("boolean")
    elif "label_available" in scoring_df.columns:
        output["is_labelable"] = scoring_df["label_available"].astype("boolean")
    else:
        output["is_labelable"] = output["label_value"].notna().astype("boolean")

    output["predicted_probability"] = np.asarray(predicted_probability, dtype=float)
    output["model_name"] = model_name
    duplicate_count = int(output["instance_id"].astype("string").duplicated().sum())
    if duplicate_count:
        raise ValueError(
            "All-valid prediction export should contain unique instance_id values, found "
            f"{duplicate_count} duplicates."
        )
    return output.loc[:, ALL_VALID_PREDICTION_COLUMNS].copy()


def build_all_valid_prediction_qc(
    *,
    evaluation_predictions: pd.DataFrame,
    all_valid_predictions: pd.DataFrame,
    model_name: str,
    horizon_h: int,
) -> pd.DataFrame:
    evaluation_ids = evaluation_predictions["instance_id"].astype("string")
    all_valid_ids = all_valid_predictions["instance_id"].astype("string")
    duplicate_all_valid_count = int(all_valid_ids.duplicated().sum())
    if duplicate_all_valid_count:
        raise ValueError(
            "All-valid prediction QC expected unique instance_id values, found "
            f"{duplicate_all_valid_count} duplicates."
        )

    evaluation_id_set = set(evaluation_ids.tolist())
    all_valid_id_set = set(all_valid_ids.tolist())
    missing_evaluation_ids = sorted(evaluation_id_set - all_valid_id_set)
    if all_valid_predictions.shape[0] < evaluation_predictions.shape[0]:
        raise RuntimeError(
            "All-valid prediction export must contain at least as many rows as the evaluation "
            f"prediction export for horizon {horizon_h}h."
        )
    if missing_evaluation_ids:
        raise RuntimeError(
            "Evaluation predictions were not preserved in all_valid_predictions.csv for "
            f"horizon {horizon_h}h; missing instance_id values: {missing_evaluation_ids[:10]}"
        )

    labelable_mask = all_valid_predictions["is_labelable"].astype("boolean").fillna(False)
    labelable_count = int(labelable_mask.sum())
    return pd.DataFrame(
        [
            {
                "model_name": model_name,
                "horizon_h": int(horizon_h),
                "evaluation_prediction_count": int(evaluation_predictions.shape[0]),
                "all_valid_prediction_count": int(all_valid_predictions.shape[0]),
                "all_valid_labelable_count": labelable_count,
                "all_valid_unlabeled_count": int(all_valid_predictions.shape[0] - labelable_count),
                "all_valid_labelable_fraction": (
                    float(labelable_count / all_valid_predictions.shape[0])
                    if all_valid_predictions.shape[0]
                    else np.nan
                ),
                "evaluation_subset_of_all_valid": True,
                "evaluation_count_equals_all_valid_labelable_count": (
                    int(evaluation_predictions.shape[0]) == labelable_count
                ),
                "missing_evaluation_instance_count": 0,
                "duplicate_all_valid_instance_count": duplicate_all_valid_count,
            }
        ]
    )


def _normalize_horizons(
    dataset: pd.DataFrame,
    horizons: Sequence[int] | None,
) -> tuple[int, ...]:
    available = sorted(pd.to_numeric(dataset["horizon_h"], errors="coerce").dropna().astype(int).unique())
    if horizons is None:
        return tuple(available)

    requested = []
    missing = []
    for value in horizons:
        horizon = int(value)
        if horizon in available:
            requested.append(horizon)
        else:
            missing.append(horizon)
    if missing:
        raise ValueError(
            "Requested Chapter 1 logistic-regression horizons are missing from the model-ready "
            f"dataset: {missing}. Available horizons: {available}"
        )
    if not requested:
        raise ValueError("At least one Chapter 1 logistic-regression horizon must be selected.")
    return tuple(dict.fromkeys(requested))


def _ordered_unique(values: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def validate_expected_split_labels(
    dataset: pd.DataFrame,
    *,
    dataset_name: str,
) -> None:
    split_values = dataset["split"].astype("string")
    missing_split_count = int(split_values.isna().sum())
    unexpected_split_labels = sorted(
        {
            str(value)
            for value in split_values.dropna().unique().tolist()
            if str(value) not in EXPECTED_BASELINE_SPLITS
        }
    )
    if missing_split_count == 0 and not unexpected_split_labels:
        return

    problems: list[str] = []
    if missing_split_count:
        problems.append(f"missing split values={missing_split_count}")
    if unexpected_split_labels:
        problems.append(f"unexpected split labels={unexpected_split_labels}")
    raise ValueError(
        f"{dataset_name} must use only the expected Chapter 1 baseline split labels "
        f"{list(EXPECTED_BASELINE_SPLITS)}; found " + "; ".join(problems)
    )


def select_primary_logistic_feature_columns(
    model_ready: pd.DataFrame,
    feature_set_definition: pd.DataFrame,
) -> tuple[list[str], dict[str, object]]:
    require_columns(
        model_ready,
        REQUIRED_MODEL_READY_COLUMNS,
        "primary_model_ready_dataset",
    )
    require_columns(
        feature_set_definition,
        {
            "feature_set_name",
            "feature_name",
            "base_variable",
            "statistic",
            "selected_for_model",
        },
        "feature_set_definition",
    )

    feature_definition = feature_set_definition.copy()
    feature_definition["feature_set_name"] = feature_definition["feature_set_name"].astype("string")
    feature_definition["feature_name"] = feature_definition["feature_name"].astype("string")
    feature_definition["base_variable"] = feature_definition["base_variable"].astype("string")
    statistic = feature_definition["statistic"].astype("string")
    selected_mask = feature_definition["selected_for_model"].astype(bool)
    primary_selected = feature_definition[
        feature_definition["feature_set_name"].eq(PRIMARY_FEATURE_SET_NAME) & selected_mask
    ].copy()

    if primary_selected.empty:
        raise ValueError("No selected primary feature-set rows were found in the feature definition.")

    median_rows = primary_selected[statistic.loc[primary_selected.index].eq("median")].copy()
    static_context_rows = primary_selected[
        statistic.loc[primary_selected.index].isna()
        | statistic.loc[primary_selected.index].eq("")
        | ~statistic.loc[primary_selected.index].isin(SUPPORTED_DYNAMIC_STATISTICS)
    ].copy()

    expected_dynamic_median_columns = median_rows["feature_name"].tolist()
    missing_dynamic_columns = [
        column for column in expected_dynamic_median_columns if column not in model_ready.columns
    ]
    if missing_dynamic_columns:
        raise ValueError(
            "Primary feature mapping expected median columns that are absent from the primary "
            f"model-ready dataset: {missing_dynamic_columns}"
        )

    static_context_feature_columns = [
        column for column in static_context_rows["feature_name"].tolist() if column in model_ready.columns
    ]
    selected_feature_columns = _ordered_unique(
        [*expected_dynamic_median_columns, *static_context_feature_columns]
    )

    if not selected_feature_columns:
        raise ValueError("No Chapter 1 logistic-regression feature columns were selected.")

    mapping_report = {
        "feature_rule": (
            "primary feature set only; dynamic block-level variables restricted to median summary "
            "columns; configured primary static/context columns included only if explicitly "
            "represented in the primary model-ready dataset"
        ),
        "selected_feature_count": len(selected_feature_columns),
        "selected_feature_columns": selected_feature_columns,
        "primary_dynamic_median_feature_columns": expected_dynamic_median_columns,
        "primary_static_context_feature_columns": static_context_feature_columns,
        "ignored_primary_statistics": sorted(
            set(
                statistic.loc[primary_selected.index]
                .dropna()
                .astype("string")
                .tolist()
            )
            - {"median"}
        ),
        "extended_feature_columns_explicitly_excluded": sorted(
            feature_set_definition.loc[
                feature_set_definition["feature_set_name"].astype("string").eq("extended")
                & feature_set_definition["selected_for_model"].astype(bool)
                & feature_set_definition["feature_name"].astype("string").str.endswith("_median"),
                "feature_name",
            ]
            .dropna()
            .astype("string")
            .tolist()
        ),
    }
    return selected_feature_columns, mapping_report


def _build_numeric_feature_frame(
    dataset: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    feature_frame = dataset.loc[:, list(feature_columns)].copy()
    for column in feature_frame.columns:
        feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")
    return feature_frame


def _build_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("median_imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def _build_model() -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
    )


def _compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float, list[str]]:
    notes: list[str] = []
    if y_true.size == 0:
        notes.append("calibration_unavailable_empty_split")
        return np.nan, np.nan, notes

    event_count = int((y_true == 1).sum())
    non_event_count = int((y_true == 0).sum())
    if event_count == 0 or non_event_count == 0:
        notes.append("calibration_unavailable_single_class")
        return np.nan, np.nan, notes

    clipped_prob = np.clip(y_prob.astype(float), 1e-6, 1 - 1e-6)
    logit_prob = np.log(clipped_prob / (1 - clipped_prob))
    if np.unique(np.round(logit_prob, 12)).size < 2:
        notes.append("calibration_unavailable_constant_predictions")
        return np.nan, np.nan, notes

    recalibration_model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=1000,
    )
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            recalibration_model.fit(logit_prob.reshape(-1, 1), y_true)
        notes.extend(
            f"calibration_fit_warning:{warning.category.__name__}" for warning in caught
        )
        return (
            float(recalibration_model.intercept_[0]),
            float(recalibration_model.coef_[0][0]),
            notes,
        )
    except Exception as exc:  # pragma: no cover - defensive branch
        notes.append(f"calibration_fit_failed:{type(exc).__name__}")
        return np.nan, np.nan, notes


def compute_binary_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
) -> dict[str, object]:
    y_true_array = np.asarray(y_true, dtype=int)
    y_prob_array = np.asarray(y_prob, dtype=float)

    sample_count = int(y_true_array.size)
    event_count = int((y_true_array == 1).sum())
    non_event_count = int((y_true_array == 0).sum())
    notes: list[str] = []

    metrics: dict[str, object] = {
        "sample_count": sample_count,
        "event_count": event_count,
        "non_event_count": non_event_count,
        "event_rate": float(event_count / sample_count) if sample_count else np.nan,
        "auroc": np.nan,
        "auprc": np.nan,
        "calibration_intercept": np.nan,
        "calibration_slope": np.nan,
        "metric_notes": pd.NA,
    }

    if sample_count == 0:
        notes.append("empty_split")
    elif not np.isfinite(y_prob_array).all():
        notes.extend(
            [
                "auroc_unavailable_non_finite_predictions",
                "auprc_unavailable_non_finite_predictions",
                "calibration_unavailable_non_finite_predictions",
            ]
        )
    elif event_count == 0 or non_event_count == 0:
        notes.extend(
            [
                "auroc_unavailable_single_class",
                "auprc_unavailable_single_class",
            ]
        )
    else:
        try:
            metrics["auroc"] = float(roc_auc_score(y_true_array, y_prob_array))
        except Exception as exc:  # pragma: no cover - defensive branch
            notes.append(f"auroc_failed:{type(exc).__name__}")

        try:
            metrics["auprc"] = float(average_precision_score(y_true_array, y_prob_array))
        except Exception as exc:  # pragma: no cover - defensive branch
            notes.append(f"auprc_failed:{type(exc).__name__}")

    if np.isfinite(y_prob_array).all():
        calibration_intercept, calibration_slope, calibration_notes = _compute_calibration_metrics(
            y_true_array,
            y_prob_array,
        )
        metrics["calibration_intercept"] = calibration_intercept
        metrics["calibration_slope"] = calibration_slope
        notes.extend(calibration_notes)
    if notes:
        metrics["metric_notes"] = "; ".join(_ordered_unique(notes))
    return metrics


def _prediction_frame(
    split_df: pd.DataFrame,
    predicted_probability: np.ndarray,
) -> pd.DataFrame:
    prediction_df = split_df.loc[:, IDENTIFIER_COLUMNS].copy()
    prediction_df["predicted_probability"] = predicted_probability
    prediction_df["model_name"] = MODEL_NAME
    return prediction_df


def _metrics_frame(
    split_df: pd.DataFrame,
    predicted_probability: np.ndarray,
    *,
    horizon_h: int,
    warnings_for_split: list[str] | None = None,
) -> pd.DataFrame:
    metrics = compute_binary_classification_metrics(
        split_df["label_value"].astype(int).to_numpy(),
        predicted_probability,
    )
    notes = []
    if warnings_for_split:
        notes.extend(warnings_for_split)
    existing_notes = metrics.get("metric_notes")
    if isinstance(existing_notes, str):
        notes.extend(existing_notes.split("; "))
    metrics["metric_notes"] = "; ".join(_ordered_unique(notes)) if notes else pd.NA
    metrics["horizon_h"] = horizon_h
    metrics["model_name"] = MODEL_NAME
    return pd.DataFrame([metrics])


def _horizon_output_dir(root_output_dir: Path, horizon_h: int) -> Path:
    return root_output_dir / f"horizon_{int(horizon_h)}h"


def run_horizon_logistic_regression(
    horizon_dataset: pd.DataFrame,
    *,
    all_valid_horizon_dataset: pd.DataFrame,
    feature_columns: Sequence[str],
    input_dataset_path: Path,
    feature_set_definition_path: Path,
    output_dir: Path,
) -> HorizonLogisticBaselineResult:
    require_columns(
        horizon_dataset,
        REQUIRED_MODEL_READY_COLUMNS,
        "horizon_dataset",
    )
    validate_expected_split_labels(horizon_dataset, dataset_name="horizon_dataset")
    require_columns(
        all_valid_horizon_dataset,
        {
            "instance_id",
            "stay_id_global",
            "hospital_id",
            "block_index",
            "prediction_time_h",
            "horizon_h",
            "split",
        },
        "all_valid_horizon_dataset",
    )
    validate_expected_split_labels(
        all_valid_horizon_dataset,
        dataset_name="all_valid_horizon_dataset",
    )

    horizon_value = int(pd.to_numeric(horizon_dataset["horizon_h"], errors="coerce").dropna().iloc[0])
    horizon_path = _horizon_output_dir(output_dir, horizon_value)
    ensure_directory(horizon_path)

    ordered_dataset = horizon_dataset.sort_values(
        ["split", "hospital_id", "stay_id_global", "block_index", "prediction_time_h", "instance_id"],
        kind="stable",
    ).reset_index(drop=True)
    ordered_all_valid_dataset = all_valid_horizon_dataset.sort_values(
        ["split", "hospital_id", "stay_id_global", "block_index", "prediction_time_h", "instance_id"],
        kind="stable",
    ).reset_index(drop=True)
    warnings_for_horizon: list[str] = []

    train_df = ordered_dataset[ordered_dataset["split"].astype("string").eq("train")].reset_index(drop=True)
    if train_df.empty:
        warnings_for_horizon.append("training_split_empty_model_not_fit")
    elif train_df["label_value"].astype(int).nunique() < 2:
        warnings_for_horizon.append("training_split_single_class_model_not_fit")

    preprocessing = _build_preprocessor()
    model = _build_model()
    fitted_pipeline: Pipeline | None = None

    if not warnings_for_horizon:
        train_features = _build_numeric_feature_frame(train_df, feature_columns)
        train_labels = train_df["label_value"].astype(int).to_numpy()
        transformed_train_features = preprocessing.fit_transform(train_features)
        model.fit(transformed_train_features, train_labels)
        fitted_pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessing),
                ("model", model),
            ]
        )

    prediction_frames: list[pd.DataFrame] = []
    metrics_frames: list[pd.DataFrame] = []
    split_row_counts: dict[str, dict[str, int]] = {}

    for split_name in EXPECTED_BASELINE_SPLITS:
        split_df = ordered_dataset[
            ordered_dataset["split"].astype("string").eq(split_name)
        ].reset_index(drop=True)
        split_row_counts[split_name] = {
            "sample_count": int(split_df.shape[0]),
            "event_count": int(split_df["label_value"].eq(1).sum()),
            "non_event_count": int(split_df["label_value"].eq(0).sum()),
        }
        if split_df.empty:
            predicted_probability = np.array([], dtype=float)
            split_notes = ["empty_split"]
        elif fitted_pipeline is None:
            predicted_probability = np.full(split_df.shape[0], np.nan, dtype=float)
            split_notes = ["model_not_fit_for_horizon"]
        else:
            split_features = _build_numeric_feature_frame(split_df, feature_columns)
            transformed_split_features = preprocessing.transform(split_features)
            predicted_probability = model.predict_proba(transformed_split_features)[:, 1]
            split_notes = []

        prediction_frames.append(_prediction_frame(split_df, predicted_probability))
        metrics_frames.append(
            _metrics_frame(
                split_df,
                predicted_probability,
                horizon_h=horizon_value,
                warnings_for_split=split_notes,
            ).assign(split=split_name)
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    if predictions.shape[0] != ordered_dataset.shape[0]:
        raise RuntimeError(
            "Chapter 1 logistic-regression prediction export row count did not match the "
            f"input horizon dataset for horizon {horizon_value}h: expected "
            f"{ordered_dataset.shape[0]}, wrote {predictions.shape[0]}."
        )
    if fitted_pipeline is None:
        all_valid_predicted_probability = np.full(ordered_all_valid_dataset.shape[0], np.nan, dtype=float)
    else:
        all_valid_features = _build_numeric_feature_frame(ordered_all_valid_dataset, feature_columns)
        transformed_all_valid_features = preprocessing.transform(all_valid_features)
        all_valid_predicted_probability = model.predict_proba(transformed_all_valid_features)[:, 1]
    all_valid_predictions = build_all_valid_prediction_frame(
        ordered_all_valid_dataset,
        all_valid_predicted_probability,
        model_name=MODEL_NAME,
    )
    all_valid_prediction_qc = build_all_valid_prediction_qc(
        evaluation_predictions=predictions,
        all_valid_predictions=all_valid_predictions,
        model_name=MODEL_NAME,
        horizon_h=horizon_value,
    )
    metrics = pd.concat(metrics_frames, ignore_index=True)[
        [
            "horizon_h",
            "split",
            "model_name",
            "sample_count",
            "event_count",
            "non_event_count",
            "event_rate",
            "auroc",
            "auprc",
            "calibration_intercept",
            "calibration_slope",
            "metric_notes",
        ]
    ]

    selected_features_path = _write_json(
        {
            "feature_selection_rule": (
                "primary feature set only; dynamic variables restricted to median summary columns; "
                "no obs_count/mean/min/max/last columns; no observation-process columns; "
                "no extended columns; no missingness indicators"
            ),
            "selected_feature_columns": list(feature_columns),
        },
        horizon_path / "selected_feature_columns.json",
    )
    predictions_path = write_dataframe(
        predictions,
        horizon_path / "predictions.csv",
        output_format="csv",
    )
    all_valid_predictions_path = write_dataframe(
        all_valid_predictions,
        horizon_path / ALL_VALID_PREDICTIONS_FILENAME,
        output_format="csv",
    )
    all_valid_prediction_qc_path = write_dataframe(
        all_valid_prediction_qc,
        horizon_path / ALL_VALID_PREDICTION_QC_FILENAME,
        output_format="csv",
    )
    metrics_path = write_dataframe(
        metrics,
        horizon_path / "metrics.csv",
        output_format="csv",
    )

    metadata_payload = {
        "timestamp_utc": _utc_timestamp(),
        "model_name": MODEL_NAME,
        "horizon_h": horizon_value,
        "input_dataset_path": str(input_dataset_path.resolve()),
        "feature_set_definition_path": str(feature_set_definition_path.resolve()),
        "selected_feature_columns": list(feature_columns),
        "selected_feature_count": len(feature_columns),
        "prediction_exports": {
            "predictions.csv": (
                "Evaluation-only predictions on horizon-labelable rows used for formal metrics."
            ),
            ALL_VALID_PREDICTIONS_FILENAME: (
                "Predictions for all scorable valid instances at this horizon, including unlabeled "
                "rows, for descriptive trajectory analysis only."
            ),
            "formal_metrics_note": (
                "metrics.csv must continue to be interpreted on the labeled evaluation subset only."
            ),
        },
        "preprocessing": {
            "median_imputation": "fit_on_training_split_only",
            "scaling": "standard_scaler_fit_on_training_split_only",
            "missingness_indicators_in_model": False,
        },
        "model": {
            "class_name": "sklearn.linear_model.LogisticRegression",
            "penalty": "default_l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
        },
        "split_counts": split_row_counts,
        "warnings": warnings_for_horizon,
    }
    metadata_path = _write_json(metadata_payload, horizon_path / "metadata.json")

    if fitted_pipeline is not None:
        preprocessing_path = _write_pickle(preprocessing, horizon_path / "preprocessing.pkl")
        model_path = _write_pickle(model, horizon_path / "logistic_regression_model.pkl")
        pipeline_path = _write_pickle(fitted_pipeline, horizon_path / "pipeline.pkl")
    else:
        preprocessing_path = _write_json(
            {"warning": "model_not_fit_for_horizon"},
            horizon_path / "preprocessing_unavailable.json",
        )
        model_path = _write_json(
            {"warning": "model_not_fit_for_horizon"},
            horizon_path / "model_unavailable.json",
        )
        pipeline_path = _write_json(
            {"warning": "model_not_fit_for_horizon"},
            horizon_path / "pipeline_unavailable.json",
        )

    return HorizonLogisticBaselineResult(
        horizon_h=horizon_value,
        feature_columns=tuple(feature_columns),
        warnings=tuple(warnings_for_horizon),
        artifacts=LogisticBaselineArtifacts(
            predictions_path=predictions_path,
            all_valid_predictions_path=all_valid_predictions_path,
            all_valid_prediction_qc_path=all_valid_prediction_qc_path,
            metrics_path=metrics_path,
            metadata_path=metadata_path,
            selected_features_path=selected_features_path,
            preprocessing_path=Path(preprocessing_path),
            model_path=Path(model_path),
            pipeline_path=Path(pipeline_path),
        ),
    )


def run_asic_primary_logistic_regression(
    *,
    input_dataset_path: Path = DEFAULT_PRIMARY_MODEL_READY_DATASET_PATH,
    feature_set_definition_path: Path = DEFAULT_FEATURE_SET_DEFINITION_PATH,
    output_dir: Path = DEFAULT_LOGISTIC_BASELINE_OUTPUT_DIR,
    horizons: Sequence[int] | None = None,
    preprocessing_root: Path | None = None,
    standardized_input_dir: Path | None = None,
    standardized_input_format: str | None = None,
    run_config_path: Path | None = None,
) -> LogisticBaselineRunResult:
    model_ready = read_dataframe(Path(input_dataset_path))
    feature_set_definition = read_dataframe(Path(feature_set_definition_path))

    require_columns(
        model_ready,
        REQUIRED_MODEL_READY_COLUMNS,
        "primary_model_ready_dataset",
    )
    feature_columns, feature_mapping_report = select_primary_logistic_feature_columns(
        model_ready,
        feature_set_definition,
    )
    selected_horizons = _normalize_horizons(model_ready, horizons)
    all_valid_scoring_dataset = build_primary_all_valid_scoring_dataset(
        input_dataset_path=Path(input_dataset_path),
        feature_set_definition=feature_set_definition,
        preprocessing_root=preprocessing_root,
        standardized_input_dir=standardized_input_dir,
        standardized_input_format=standardized_input_format,
        run_config_path=run_config_path,
    )

    horizon_results: list[HorizonLogisticBaselineResult] = []
    summary_rows: list[dict[str, object]] = []
    for horizon_h in selected_horizons:
        horizon_dataset = model_ready[
            pd.to_numeric(model_ready["horizon_h"], errors="coerce").eq(int(horizon_h))
        ].reset_index(drop=True)
        all_valid_horizon_dataset = all_valid_scoring_dataset[
            pd.to_numeric(all_valid_scoring_dataset["horizon_h"], errors="coerce").eq(int(horizon_h))
        ].reset_index(drop=True)
        horizon_result = run_horizon_logistic_regression(
            horizon_dataset,
            all_valid_horizon_dataset=all_valid_horizon_dataset,
            feature_columns=feature_columns,
            input_dataset_path=Path(input_dataset_path),
            feature_set_definition_path=Path(feature_set_definition_path),
            output_dir=Path(output_dir),
        )
        horizon_results.append(horizon_result)

        metrics = pd.read_csv(horizon_result.artifacts.metrics_path)
        metrics_by_split = metrics.set_index("split")
        summary_rows.append(
            {
                "horizon_h": horizon_h,
                "feature_count": len(feature_columns),
                "train_sample_count": int(metrics_by_split.at["train", "sample_count"])
                if "train" in metrics_by_split.index
                else 0,
                "train_event_count": int(metrics_by_split.at["train", "event_count"])
                if "train" in metrics_by_split.index
                else 0,
                "validation_sample_count": int(metrics_by_split.at["validation", "sample_count"])
                if "validation" in metrics_by_split.index
                else 0,
                "validation_event_count": int(metrics_by_split.at["validation", "event_count"])
                if "validation" in metrics_by_split.index
                else 0,
                "test_sample_count": int(metrics_by_split.at["test", "sample_count"])
                if "test" in metrics_by_split.index
                else 0,
                "test_event_count": int(metrics_by_split.at["test", "event_count"])
                if "test" in metrics_by_split.index
                else 0,
                "train_metric_notes": metrics_by_split.at["train", "metric_notes"]
                if "train" in metrics_by_split.index
                else pd.NA,
                "validation_metric_notes": metrics_by_split.at["validation", "metric_notes"]
                if "validation" in metrics_by_split.index
                else pd.NA,
                "test_metric_notes": metrics_by_split.at["test", "metric_notes"]
                if "test" in metrics_by_split.index
                else pd.NA,
                "warnings": "; ".join(horizon_result.warnings) if horizon_result.warnings else pd.NA,
                "output_dir": str(_horizon_output_dir(Path(output_dir), horizon_h)),
            }
        )

    ensure_directory(Path(output_dir))
    summary_path = write_dataframe(
        pd.DataFrame(summary_rows),
        Path(output_dir) / "horizon_run_summary.csv",
        output_format="csv",
    )
    manifest_payload = {
        "timestamp_utc": _utc_timestamp(),
        "model_name": MODEL_NAME,
        "input_dataset_path": str(Path(input_dataset_path).resolve()),
        "feature_set_definition_path": str(Path(feature_set_definition_path).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "horizons_processed": list(selected_horizons),
        "selected_feature_columns": feature_columns,
        "selected_feature_count": len(feature_columns),
        "feature_mapping_report": feature_mapping_report,
        "horizon_output_dirs": {
            str(horizon_result.horizon_h): str(
                _horizon_output_dir(Path(output_dir), horizon_result.horizon_h).resolve()
            )
            for horizon_result in horizon_results
        },
    }
    manifest_path = _write_json(manifest_payload, Path(output_dir) / "run_manifest.json")

    return LogisticBaselineRunResult(
        input_dataset_path=Path(input_dataset_path),
        feature_set_definition_path=Path(feature_set_definition_path),
        output_dir=Path(output_dir),
        selected_feature_columns=tuple(feature_columns),
        horizons_processed=selected_horizons,
        horizon_results=tuple(horizon_results),
        summary_path=summary_path,
        manifest_path=manifest_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Chapter 1 ASIC logistic-regression baseline on the frozen primary "
            "model-ready dataset using primary median feature columns only."
        )
    )
    parser.add_argument(
        "--input-dataset",
        type=Path,
        default=DEFAULT_PRIMARY_MODEL_READY_DATASET_PATH,
        help="Path to the Chapter 1 primary model-ready dataset.",
    )
    parser.add_argument(
        "--feature-set-definition",
        type=Path,
        default=DEFAULT_FEATURE_SET_DEFINITION_PATH,
        help="Path to the Chapter 1 feature-set definition artifact.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_LOGISTIC_BASELINE_OUTPUT_DIR,
        help="Root output directory for ASIC logistic-regression baseline artifacts.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        help="Optional subset of horizons to process.",
    )
    parser.add_argument(
        "--preprocessing-root",
        type=Path,
        help=(
            "Optional root directory containing the Chapter 1 preprocessing artifacts "
            "(instances/, labels/, splits/). If omitted, it is inferred from --input-dataset."
        ),
    )
    parser.add_argument(
        "--standardized-input-dir",
        type=Path,
        help=(
            "Optional standardized ASIC input directory used to rebuild all-valid scoring rows. "
            "Defaults to the Chapter 1 run config input_dir."
        ),
    )
    parser.add_argument(
        "--standardized-input-format",
        choices=("csv", "parquet"),
        help=(
            "Format of the standardized ASIC input directory used for all-valid scoring. "
            "Defaults to the Chapter 1 run config input_format."
        ),
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        help=(
            "Optional Chapter 1 run config used to resolve the default standardized input "
            "directory for all-valid scoring."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = run_asic_primary_logistic_regression(
        input_dataset_path=args.input_dataset,
        feature_set_definition_path=args.feature_set_definition,
        output_dir=args.output_dir,
        horizons=args.horizons,
        preprocessing_root=args.preprocessing_root,
        standardized_input_dir=args.standardized_input_dir,
        standardized_input_format=args.standardized_input_format,
        run_config_path=args.run_config,
    )

    print("Selected feature columns:")
    for feature_column in result.selected_feature_columns:
        print(f"- {feature_column}")
    print()
    print(f"Horizons processed: {', '.join(str(horizon) for horizon in result.horizons_processed)}")
    print(f"Feature count: {len(result.selected_feature_columns)}")
    print(f"Run summary: {result.summary_path}")
    print(f"Run manifest: {result.manifest_path}")
    for horizon_result in result.horizon_results:
        metrics = pd.read_csv(horizon_result.artifacts.metrics_path)
        split_notes = {
            row.split: row.metric_notes
            for row in metrics.itertuples(index=False)
            if isinstance(row.metric_notes, str) and row.metric_notes
        }
        warnings_text = "; ".join(horizon_result.warnings) if horizon_result.warnings else "none"
        metric_notes_text = (
            "; ".join(f"{split}={note}" for split, note in split_notes.items())
            if split_notes
            else "none"
        )
        print(
            f"horizon {horizon_result.horizon_h}h -> {horizon_result.artifacts.predictions_path.parent} "
            f"(warnings: {warnings_text}; metric notes: {metric_notes_text}; "
            f"all-valid: {horizon_result.artifacts.all_valid_predictions_path.name})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
