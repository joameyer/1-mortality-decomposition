from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

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

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - exercised in runtime environments without xgboost
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = exc
else:  # pragma: no cover - trivial assignment
    XGBOOST_IMPORT_ERROR = None


DEFAULT_XGBOOST_BASELINE_OUTPUT_DIR = (
    Path("artifacts")
    / "chapter1"
    / "baselines"
    / "asic"
    / "primary_medians"
    / "xgboost"
)

MODEL_NAME = "xgboost"
DEFAULT_XGBOOST_RANDOM_STATE = 20260327
DEFAULT_XGBOOST_PARAMETERS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": DEFAULT_XGBOOST_RANDOM_STATE,
    "n_jobs": 1,
    "verbosity": 0,
}


@dataclass(frozen=True)
class XGBoostBaselineArtifacts:
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
class HorizonXGBoostBaselineResult:
    horizon_h: int
    feature_columns: tuple[str, ...]
    warnings: tuple[str, ...]
    artifacts: XGBoostBaselineArtifacts


@dataclass(frozen=True)
class XGBoostBaselineRunResult:
    input_dataset_path: Path
    feature_set_definition_path: Path
    output_dir: Path
    selected_feature_columns: tuple[str, ...]
    horizons_processed: tuple[int, ...]
    horizon_results: tuple[HorizonXGBoostBaselineResult, ...]
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
        ]
    )


def _resolve_scale_pos_weight(train_labels: np.ndarray) -> tuple[float | None, list[str]]:
    event_count = int((train_labels == 1).sum())
    non_event_count = int((train_labels == 0).sum())
    if event_count == 0 or non_event_count == 0:
        return None, ["training_split_single_class_model_not_fit"]
    return float(non_event_count / event_count), []


def _build_model(scale_pos_weight: float | None) -> XGBClassifier:
    if XGBClassifier is None:
        raise ImportError(
            "xgboost is required for the Chapter 1 ASIC XGBoost baseline. "
            "Install the project dependencies including xgboost before running this baseline."
        ) from XGBOOST_IMPORT_ERROR

    model_parameters = dict(DEFAULT_XGBOOST_PARAMETERS)
    if scale_pos_weight is not None:
        model_parameters["scale_pos_weight"] = float(scale_pos_weight)
    return XGBClassifier(**model_parameters)


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


def run_horizon_xgboost(
    horizon_dataset: pd.DataFrame,
    *,
    all_valid_horizon_dataset: pd.DataFrame,
    feature_columns: Sequence[str],
    input_dataset_path: Path,
    feature_set_definition_path: Path,
    output_dir: Path,
) -> HorizonXGBoostBaselineResult:
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

    preprocessing = _build_preprocessor()
    model: XGBClassifier | None = None
    fitted_pipeline: Pipeline | None = None
    scale_pos_weight: float | None = None

    if not warnings_for_horizon:
        train_features = _build_numeric_feature_frame(train_df, feature_columns)
        train_labels = train_df["label_value"].astype(int).to_numpy()
        scale_pos_weight, imbalance_notes = _resolve_scale_pos_weight(train_labels)
        warnings_for_horizon.extend(imbalance_notes)

        if not warnings_for_horizon:
            transformed_train_features = preprocessing.fit_transform(train_features)
            model = _build_model(scale_pos_weight)
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
        elif fitted_pipeline is None or model is None:
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
            "Chapter 1 XGBoost prediction export row count did not match the input "
            f"horizon dataset for horizon {horizon_value}h: expected "
            f"{ordered_dataset.shape[0]}, wrote {predictions.shape[0]}."
        )
    if fitted_pipeline is None or model is None:
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
            "scaling": "not_used_for_xgboost_baseline",
            "missingness_indicators_in_model": False,
        },
        "model": {
            "class_name": "xgboost.XGBClassifier" if XGBClassifier is not None else "unavailable",
            **DEFAULT_XGBOOST_PARAMETERS,
            "scale_pos_weight": scale_pos_weight,
            "class_imbalance_handling": (
                "training_non_event_count / training_event_count"
                if scale_pos_weight is not None
                else "not_applied"
            ),
        },
        "split_counts": split_row_counts,
        "warnings": warnings_for_horizon,
    }
    metadata_path = _write_json(metadata_payload, horizon_path / "metadata.json")

    if fitted_pipeline is not None and model is not None:
        preprocessing_path = _write_pickle(preprocessing, horizon_path / "preprocessing.pkl")
        model_path = _write_pickle(model, horizon_path / "xgboost_model.pkl")
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

    return HorizonXGBoostBaselineResult(
        horizon_h=horizon_value,
        feature_columns=tuple(feature_columns),
        warnings=tuple(warnings_for_horizon),
        artifacts=XGBoostBaselineArtifacts(
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


def run_asic_primary_xgboost(
    *,
    input_dataset_path: Path = DEFAULT_PRIMARY_MODEL_READY_DATASET_PATH,
    feature_set_definition_path: Path = DEFAULT_FEATURE_SET_DEFINITION_PATH,
    output_dir: Path = DEFAULT_XGBOOST_BASELINE_OUTPUT_DIR,
    horizons: Sequence[int] | None = None,
    preprocessing_root: Path | None = None,
    standardized_input_dir: Path | None = None,
    standardized_input_format: str | None = None,
    run_config_path: Path | None = None,
) -> XGBoostBaselineRunResult:
    if XGBClassifier is None:
        raise ImportError(
            "xgboost is required for the Chapter 1 ASIC XGBoost baseline. "
            "Install the project dependencies including xgboost before running this baseline."
        ) from XGBOOST_IMPORT_ERROR

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

    horizon_results: list[HorizonXGBoostBaselineResult] = []
    summary_rows: list[dict[str, object]] = []
    for horizon_h in selected_horizons:
        horizon_dataset = model_ready[
            pd.to_numeric(model_ready["horizon_h"], errors="coerce").eq(int(horizon_h))
        ].reset_index(drop=True)
        all_valid_horizon_dataset = all_valid_scoring_dataset[
            pd.to_numeric(all_valid_scoring_dataset["horizon_h"], errors="coerce").eq(int(horizon_h))
        ].reset_index(drop=True)
        horizon_result = run_horizon_xgboost(
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
        "xgboost_parameters": DEFAULT_XGBOOST_PARAMETERS,
        "horizon_output_dirs": {
            str(horizon_result.horizon_h): str(
                _horizon_output_dir(Path(output_dir), horizon_result.horizon_h).resolve()
            )
            for horizon_result in horizon_results
        },
    }
    manifest_path = _write_json(manifest_payload, Path(output_dir) / "run_manifest.json")

    return XGBoostBaselineRunResult(
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
            "Run the Chapter 1 ASIC XGBoost baseline on the frozen primary "
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
        default=DEFAULT_XGBOOST_BASELINE_OUTPUT_DIR,
        help="Root output directory for ASIC XGBoost baseline artifacts.",
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

    result = run_asic_primary_xgboost(
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
