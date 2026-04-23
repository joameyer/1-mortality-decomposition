from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"
if "OPENBLAS_NUM_THREADS" not in os.environ:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
if "MKL_NUM_THREADS" not in os.environ:
    os.environ["MKL_NUM_THREADS"] = "1"
if "NUMEXPR_NUM_THREADS" not in os.environ:
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chapter1_mortality_decomposition.baseline_logistic import (
    ALL_VALID_PREDICTIONS_FILENAME,
    ALL_VALID_PREDICTION_QC_FILENAME,
    DEFAULT_FEATURE_SET_DEFINITION_PATH,
    DEFAULT_PRIMARY_MODEL_READY_DATASET_PATH,
    EXPECTED_BASELINE_SPLITS,
    IDENTIFIER_COLUMNS,
    REQUIRED_MODEL_READY_COLUMNS,
    _build_numeric_feature_frame,
    _horizon_output_dir,
    _normalize_horizons,
    _ordered_unique,
    build_all_valid_prediction_frame,
    build_all_valid_prediction_qc,
    build_primary_all_valid_scoring_dataset,
    compute_binary_classification_metrics,
    select_primary_logistic_feature_columns,
    validate_expected_split_labels,
)
from chapter1_mortality_decomposition.utils import (
    ensure_directory,
    read_dataframe,
    require_columns,
    write_dataframe,
    write_text,
)


DEFAULT_FEEDFORWARD_NN_BASELINE_OUTPUT_DIR = (
    Path("artifacts")
    / "chapter1"
    / "baselines"
    / "asic"
    / "primary_medians"
    / "feedforward_nn"
)

MODEL_NAME = "feedforward_nn"
DEFAULT_FEEDFORWARD_NN_RANDOM_STATE = 20260422
DEFAULT_FEEDFORWARD_NN_PARAMETERS = {
    "hidden_layer_sizes": (16,),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-4,
    "learning_rate_init": 1e-3,
    "batch_size": 64,
    "max_epochs": 80,
    "patience": 8,
    "improvement_tol": 1e-4,
    "prediction_batch_size": 4096,
}


@dataclass(frozen=True)
class FeedforwardNNBaselineArtifacts:
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
class HorizonFeedforwardNNBaselineResult:
    horizon_h: int
    feature_columns: tuple[str, ...]
    warnings: tuple[str, ...]
    artifacts: FeedforwardNNBaselineArtifacts


@dataclass(frozen=True)
class FeedforwardNNBaselineRunResult:
    input_dataset_path: Path
    feature_set_definition_path: Path
    output_dir: Path
    selected_feature_columns: tuple[str, ...]
    horizons_processed: tuple[int, ...]
    horizon_results: tuple[HorizonFeedforwardNNBaselineResult, ...]
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


def _build_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("median_imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def _build_model(*, batch_size: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=DEFAULT_FEEDFORWARD_NN_PARAMETERS["hidden_layer_sizes"],
        activation=str(DEFAULT_FEEDFORWARD_NN_PARAMETERS["activation"]),
        solver=str(DEFAULT_FEEDFORWARD_NN_PARAMETERS["solver"]),
        alpha=float(DEFAULT_FEEDFORWARD_NN_PARAMETERS["alpha"]),
        learning_rate_init=float(DEFAULT_FEEDFORWARD_NN_PARAMETERS["learning_rate_init"]),
        batch_size=int(batch_size),
        max_iter=1,
        warm_start=True,
        shuffle=True,
        early_stopping=False,
        random_state=DEFAULT_FEEDFORWARD_NN_RANDOM_STATE,
    )


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


def _monitor_log_loss(
    model: MLPClassifier,
    features: np.ndarray,
    labels: np.ndarray,
) -> float:
    probabilities = np.clip(model.predict_proba(features)[:, 1], 1e-6, 1 - 1e-6)
    return float(log_loss(labels, probabilities, labels=[0, 1]))


def _as_float32(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float32, order="C")


def _predict_probabilities_in_batches(
    *,
    preprocessing: Pipeline,
    model: MLPClassifier,
    dataset: pd.DataFrame,
    feature_columns: Sequence[str],
    batch_size: int = int(DEFAULT_FEEDFORWARD_NN_PARAMETERS["prediction_batch_size"]),
) -> np.ndarray:
    if dataset.empty:
        return np.array([], dtype=float)

    batch_outputs: list[np.ndarray] = []
    effective_batch_size = max(1, int(batch_size))
    for start in range(0, int(dataset.shape[0]), effective_batch_size):
        stop = min(start + effective_batch_size, int(dataset.shape[0]))
        batch_features = _build_numeric_feature_frame(
            dataset.iloc[start:stop].reset_index(drop=True),
            feature_columns,
        )
        transformed_batch = _as_float32(preprocessing.transform(batch_features))
        batch_outputs.append(model.predict_proba(transformed_batch)[:, 1].astype(float, copy=False))
    return np.concatenate(batch_outputs)


def _fit_feedforward_nn(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
) -> tuple[Pipeline, MLPClassifier, dict[str, object], list[str]]:
    fit_notes: list[str] = []
    train_features = _build_numeric_feature_frame(train_df, feature_columns)
    train_labels = train_df["label_value"].astype(int).to_numpy()
    if np.unique(train_labels).size < 2:
        raise ValueError("training_split_single_class_model_not_fit")

    validation_features = _build_numeric_feature_frame(validation_df, feature_columns)
    validation_labels = validation_df["label_value"].astype(int).to_numpy()

    preprocessing = _build_preprocessor()
    transformed_train_features = _as_float32(preprocessing.fit_transform(train_features))
    transformed_validation_features = (
        _as_float32(preprocessing.transform(validation_features))
        if not validation_df.empty
        else np.empty((0, transformed_train_features.shape[1]), dtype=float)
    )

    batch_size = max(
        1,
        min(
            int(DEFAULT_FEEDFORWARD_NN_PARAMETERS["batch_size"]),
            int(transformed_train_features.shape[0]),
        ),
    )
    model = _build_model(batch_size=batch_size)

    monitor_features = transformed_validation_features
    monitor_labels = validation_labels
    monitor_split = "validation"
    if validation_df.empty:
        monitor_features = transformed_train_features
        monitor_labels = train_labels
        monitor_split = "train"
        fit_notes.append("validation_split_empty_early_stopping_monitored_on_train")

    max_epochs = int(DEFAULT_FEEDFORWARD_NN_PARAMETERS["max_epochs"])
    patience = int(DEFAULT_FEEDFORWARD_NN_PARAMETERS["patience"])
    improvement_tol = float(DEFAULT_FEEDFORWARD_NN_PARAMETERS["improvement_tol"])

    best_model: MLPClassifier | None = None
    best_monitor_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_completed = 0
    warning_messages: list[str] = []

    for epoch in range(1, max_epochs + 1):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(transformed_train_features, train_labels)

        epochs_completed = epoch
        for warning in caught:
            if issubclass(warning.category, ConvergenceWarning):
                continue
            warning_messages.append(f"training_warning:{warning.category.__name__}")

        monitor_loss = _monitor_log_loss(model, monitor_features, monitor_labels)
        if monitor_loss + improvement_tol < best_monitor_loss:
            best_monitor_loss = monitor_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                fit_notes.append("early_stopping_triggered")
                break

    if best_model is None:
        best_model = copy.deepcopy(model)
        best_epoch = epochs_completed
        best_monitor_loss = _monitor_log_loss(best_model, monitor_features, monitor_labels)

    fit_notes.extend(_ordered_unique(warning_messages))
    fitted_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", best_model),
        ]
    )
    fit_metadata = {
        "random_state": DEFAULT_FEEDFORWARD_NN_RANDOM_STATE,
        "monitor_split": monitor_split,
        "monitor_sample_count": int(monitor_labels.size),
        "epochs_completed": int(epochs_completed),
        "best_epoch": int(best_epoch),
        "best_monitor_log_loss": float(best_monitor_loss),
        "stopped_early": bool(epochs_completed < max_epochs),
        "max_epochs": max_epochs,
        "patience": patience,
        "improvement_tol": improvement_tol,
        "batch_size": batch_size,
        "fit_notes": _ordered_unique(fit_notes),
    }
    return fitted_pipeline, best_model, fit_metadata, _ordered_unique(fit_notes)


def run_horizon_feedforward_nn(
    horizon_dataset: pd.DataFrame,
    *,
    all_valid_horizon_dataset: pd.DataFrame,
    feature_columns: Sequence[str],
    input_dataset_path: Path,
    feature_set_definition_path: Path,
    output_dir: Path,
) -> HorizonFeedforwardNNBaselineResult:
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
    validation_df = ordered_dataset[
        ordered_dataset["split"].astype("string").eq("validation")
    ].reset_index(drop=True)
    if train_df.empty:
        warnings_for_horizon.append("training_split_empty_model_not_fit")

    fitted_pipeline: Pipeline | None = None
    model: MLPClassifier | None = None
    fit_metadata: dict[str, object] = {}

    if not warnings_for_horizon:
        try:
            fitted_pipeline, model, fit_metadata, fit_notes = _fit_feedforward_nn(
                train_df,
                validation_df,
                feature_columns=feature_columns,
            )
            warnings_for_horizon.extend(fit_notes)
        except ValueError as exc:
            warnings_for_horizon.append(str(exc))

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
        elif fitted_pipeline is None or model is None:
            predicted_probability = np.full(split_df.shape[0], np.nan, dtype=float)
            split_notes = ["model_not_fit_for_horizon"]
        else:
            predicted_probability = _predict_probabilities_in_batches(
                preprocessing=fitted_pipeline.named_steps["preprocessing"],
                model=model,
                dataset=split_df,
                feature_columns=feature_columns,
            )
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
            "Chapter 1 feedforward-NN prediction export row count did not match the input "
            f"horizon dataset for horizon {horizon_value}h: expected "
            f"{ordered_dataset.shape[0]}, wrote {predictions.shape[0]}."
        )

    if fitted_pipeline is None or model is None:
        all_valid_predicted_probability = np.full(ordered_all_valid_dataset.shape[0], np.nan, dtype=float)
    else:
        all_valid_predicted_probability = _predict_probabilities_in_batches(
            preprocessing=fitted_pipeline.named_steps["preprocessing"],
            model=model,
            dataset=ordered_all_valid_dataset,
            feature_columns=feature_columns,
        )

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
            "class_name": "sklearn.neural_network.MLPClassifier",
            **DEFAULT_FEEDFORWARD_NN_PARAMETERS,
            "random_state": DEFAULT_FEEDFORWARD_NN_RANDOM_STATE,
            "architecture_scope": "single_minimal_feedforward_baseline_no_architecture_search",
            "fit_metadata": fit_metadata,
        },
        "split_counts": split_row_counts,
        "warnings": warnings_for_horizon,
    }
    metadata_path = _write_json(metadata_payload, horizon_path / "metadata.json")

    if fitted_pipeline is not None and model is not None:
        preprocessing_path = _write_pickle(
            fitted_pipeline.named_steps["preprocessing"],
            horizon_path / "preprocessing.pkl",
        )
        model_path = _write_pickle(model, horizon_path / "feedforward_nn_model.pkl")
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

    return HorizonFeedforwardNNBaselineResult(
        horizon_h=horizon_value,
        feature_columns=tuple(feature_columns),
        warnings=tuple(warnings_for_horizon),
        artifacts=FeedforwardNNBaselineArtifacts(
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


def run_asic_primary_feedforward_nn(
    *,
    input_dataset_path: Path = DEFAULT_PRIMARY_MODEL_READY_DATASET_PATH,
    feature_set_definition_path: Path = DEFAULT_FEATURE_SET_DEFINITION_PATH,
    output_dir: Path = DEFAULT_FEEDFORWARD_NN_BASELINE_OUTPUT_DIR,
    horizons: Sequence[int] | None = None,
    preprocessing_root: Path | None = None,
    standardized_input_dir: Path | None = None,
    standardized_input_format: str | None = None,
    run_config_path: Path | None = None,
) -> FeedforwardNNBaselineRunResult:
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

    horizon_results: list[HorizonFeedforwardNNBaselineResult] = []
    summary_rows: list[dict[str, object]] = []
    for horizon_h in selected_horizons:
        horizon_dataset = model_ready[
            pd.to_numeric(model_ready["horizon_h"], errors="coerce").eq(int(horizon_h))
        ].reset_index(drop=True)
        all_valid_horizon_dataset = all_valid_scoring_dataset[
            pd.to_numeric(all_valid_scoring_dataset["horizon_h"], errors="coerce").eq(int(horizon_h))
        ].reset_index(drop=True)
        horizon_result = run_horizon_feedforward_nn(
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
        "feedforward_nn_parameters": {
            **DEFAULT_FEEDFORWARD_NN_PARAMETERS,
            "random_state": DEFAULT_FEEDFORWARD_NN_RANDOM_STATE,
        },
        "horizon_output_dirs": {
            str(horizon_result.horizon_h): str(
                _horizon_output_dir(Path(output_dir), horizon_result.horizon_h).resolve()
            )
            for horizon_result in horizon_results
        },
    }
    manifest_path = _write_json(manifest_payload, Path(output_dir) / "run_manifest.json")

    return FeedforwardNNBaselineRunResult(
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
            "Run the Chapter 1 ASIC feedforward neural-network baseline on the frozen primary "
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
        default=DEFAULT_FEEDFORWARD_NN_BASELINE_OUTPUT_DIR,
        help="Root output directory for ASIC feedforward-NN baseline artifacts.",
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

    result = run_asic_primary_feedforward_nn(
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
