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


@dataclass(frozen=True)
class LogisticBaselineArtifacts:
    predictions_path: Path
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

    horizon_value = int(pd.to_numeric(horizon_dataset["horizon_h"], errors="coerce").dropna().iloc[0])
    horizon_path = _horizon_output_dir(output_dir, horizon_value)
    ensure_directory(horizon_path)

    ordered_dataset = horizon_dataset.sort_values(
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

    horizon_results: list[HorizonLogisticBaselineResult] = []
    summary_rows: list[dict[str, object]] = []
    for horizon_h in selected_horizons:
        horizon_dataset = model_ready[
            pd.to_numeric(model_ready["horizon_h"], errors="coerce").eq(int(horizon_h))
        ].reset_index(drop=True)
        horizon_result = run_horizon_logistic_regression(
            horizon_dataset,
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = run_asic_primary_logistic_regression(
        input_dataset_path=args.input_dataset,
        feature_set_definition_path=args.feature_set_definition,
        output_dir=args.output_dir,
        horizons=args.horizons,
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
            f"(warnings: {warnings_text}; metric notes: {metric_notes_text})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
