from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

try:
    if "MPLCONFIGDIR" not in os.environ:
        matplotlib_cache_dir = Path("/tmp") / "chapter1_mortality_decomposition_matplotlib"
        matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(matplotlib_cache_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - environment dependency branch
    plt = None
    MATPLOTLIB_IMPORT_ERROR = exc
else:  # pragma: no cover - trivial assignment
    MATPLOTLIB_IMPORT_ERROR = None

from chapter1_mortality_decomposition.baseline_evaluation import (
    DEFAULT_HORIZONS,
    PREDICTION_REQUIRED_COLUMNS,
    _dynamic_probability_axis_limit,
    _empty_risk_bin_summary,
    _ranked_quantile_risk_summary,
    compute_evaluation_metrics,
)
from chapter1_mortality_decomposition.baseline_logistic import IDENTIFIER_COLUMNS
from chapter1_mortality_decomposition.utils import (
    ensure_directory,
    read_dataframe,
    require_columns,
    write_dataframe,
    write_text,
)


DEFAULT_BASELINE_ARTIFACT_ROOT = (
    Path("artifacts") / "chapter1" / "baselines" / "asic" / "primary_medians"
)
DEFAULT_RECALIBRATION_OUTPUT_DIR = (
    Path("artifacts")
    / "chapter1"
    / "recalibration"
    / "asic"
    / "primary_medians"
    / "xgboost"
)
DEFAULT_REFERENCE_MODEL_NAME = "logistic_regression"
DEFAULT_XGBOOST_MODEL_NAME = "xgboost"
DEFAULT_RECALIBRATION_METHODS = ("platt", "isotonic")
DEFAULT_RELIABILITY_BIN_COUNT = 10
FIT_SPLIT_NAME = "validation"
PRIMARY_EVALUATION_SPLIT = "test"
VARIANT_ORDER = (
    "logistic_regression",
    "xgboost_raw",
    "xgboost_platt",
    "xgboost_isotonic",
)
VARIANT_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "xgboost_raw": "XGBoost Raw",
    "xgboost_platt": "XGBoost + Platt",
    "xgboost_isotonic": "XGBoost + Isotonic",
}
VARIANT_COLORS = {
    "logistic_regression": "#4c78a8",
    "xgboost_raw": "#f58518",
    "xgboost_platt": "#54a24b",
    "xgboost_isotonic": "#e45756",
}
RELIABILITY_SUMMARY_COLUMNS = [
    "model_variant",
    "horizon_h",
    "split",
    "sample_scope",
    "group_id",
    "bin_index",
    "bin_label",
    "sample_count",
    "event_count",
    "non_event_count",
    "sample_fraction",
    "event_fraction_of_events",
    "predicted_probability_mean",
    "predicted_probability_min",
    "predicted_probability_max",
    "observed_mortality",
]
CANONICAL_VARIANT_PREDICTION_COLUMNS = [
    *IDENTIFIER_COLUMNS,
    "model_variant",
    "predicted_probability",
]


@dataclass(frozen=True)
class HorizonRecalibrationArtifacts:
    horizon_h: int
    output_dir: Path
    comparison_metrics_path: Path
    reliability_summary_path: Path
    reliability_plot_path: Path
    probability_distribution_plot_path: Path
    metadata_path: Path
    method_prediction_paths: dict[str, Path]
    method_metrics_paths: dict[str, Path]
    method_statuses: dict[str, str]
    canonical_prediction_paths: dict[str, Path]
    combined_canonical_predictions_path: Path


@dataclass(frozen=True)
class XGBoostRecalibrationRunResult:
    input_root: Path
    output_dir: Path
    horizons_processed: tuple[int, ...]
    combined_comparison_metrics_path: Path
    combined_test_reliability_summary_path: Path
    summary_figure_path: Path
    interpretation_note_path: Path
    manifest_path: Path
    horizon_results: tuple[HorizonRecalibrationArtifacts, ...]


def _require_matplotlib() -> None:
    if plt is None:
        raise ImportError(
            "matplotlib is required for the Chapter 1 XGBoost recalibration package."
        ) from MATPLOTLIB_IMPORT_ERROR


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ordered_unique(values: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


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


def _write_json(payload: dict[str, object], path: Path) -> Path:
    return write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        path,
    )


def _normalize_horizons(
    input_root: Path,
    horizons: Sequence[int] | None,
) -> tuple[int, ...]:
    available_horizons = sorted(
        int(path.name.removeprefix("horizon_").removesuffix("h"))
        for path in (input_root / DEFAULT_XGBOOST_MODEL_NAME).glob("horizon_*h")
        if path.is_dir()
    )
    selected_horizons = tuple(int(horizon) for horizon in (horizons or DEFAULT_HORIZONS))
    missing_horizons = [horizon for horizon in selected_horizons if horizon not in available_horizons]
    if missing_horizons:
        raise ValueError(
            f"XGBoost prediction artifacts are missing requested horizons {missing_horizons}. "
            f"Available horizons under {input_root / DEFAULT_XGBOOST_MODEL_NAME}: {available_horizons}"
        )
    return selected_horizons


def _prediction_artifact_path(input_root: Path, model_name: str, horizon_h: int) -> Path:
    predictions_path = input_root / model_name / f"horizon_{int(horizon_h)}h" / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Expected saved predictions at {predictions_path}")
    return predictions_path


def _load_prediction_frame(
    predictions_path: Path,
    *,
    expected_model_name: str,
    expected_horizon_h: int,
) -> pd.DataFrame:
    predictions = read_dataframe(predictions_path)
    require_columns(predictions, PREDICTION_REQUIRED_COLUMNS, str(predictions_path))
    predictions = predictions.copy()
    predictions["model_name"] = predictions["model_name"].astype("string")
    predictions["split"] = predictions["split"].astype("string")
    predictions["hospital_id"] = predictions["hospital_id"].astype("string")
    predictions["horizon_h"] = pd.to_numeric(predictions["horizon_h"], errors="coerce").astype("Int64")
    predictions["label_value"] = pd.to_numeric(predictions["label_value"], errors="coerce").astype("Int64")
    predictions["predicted_probability"] = pd.to_numeric(
        predictions["predicted_probability"],
        errors="coerce",
    )
    if predictions["model_name"].nunique(dropna=True) != 1:
        raise ValueError(f"{predictions_path} contains multiple model_name values.")
    observed_model_name = str(predictions["model_name"].dropna().iloc[0])
    if observed_model_name != expected_model_name:
        raise ValueError(
            f"{predictions_path} reported model_name {observed_model_name!r}, "
            f"expected {expected_model_name!r}."
        )
    observed_horizons = predictions["horizon_h"].dropna().astype(int).unique().tolist()
    if observed_horizons != [int(expected_horizon_h)]:
        raise ValueError(
            f"{predictions_path} should contain only horizon {expected_horizon_h}, "
            f"found {observed_horizons}."
        )
    return predictions.sort_values(
        ["split", "hospital_id", "stay_id_global", "block_index", "prediction_time_h", "instance_id"],
        kind="stable",
    ).reset_index(drop=True)


def _verify_reference_alignment(
    logistic_predictions: pd.DataFrame,
    xgboost_predictions: pd.DataFrame,
    *,
    horizon_h: int,
) -> None:
    logistic_index = logistic_predictions.loc[:, IDENTIFIER_COLUMNS].copy()
    xgboost_index = xgboost_predictions.loc[:, IDENTIFIER_COLUMNS].copy()
    if not logistic_index.equals(xgboost_index):
        raise ValueError(
            "Logistic-regression and XGBoost saved predictions do not align row-for-row "
            f"for horizon {horizon_h}h."
        )


def _combine_notes(*note_groups: Sequence[str] | str | None) -> object:
    notes: list[str] = []
    for group in note_groups:
        if group is None:
            continue
        if isinstance(group, str):
            notes.extend([note for note in group.split("; ") if note])
            continue
        if isinstance(group, Sequence):
            notes.extend([str(note) for note in group if note])
            continue
        if pd.isna(group):
            continue
        notes.append(str(group))
    combined = _ordered_unique(notes)
    return "; ".join(combined) if combined else pd.NA


def _finite_probability_stats(y_prob: np.ndarray) -> dict[str, object]:
    finite = y_prob[np.isfinite(y_prob)]
    if finite.size == 0:
        return {
            "mean_predicted_risk": np.nan,
            "predicted_probability_min": np.nan,
            "predicted_probability_median": np.nan,
            "predicted_probability_p95": np.nan,
            "predicted_probability_max": np.nan,
        }
    return {
        "mean_predicted_risk": float(finite.mean()),
        "predicted_probability_min": float(finite.min()),
        "predicted_probability_median": float(np.median(finite)),
        "predicted_probability_p95": float(np.quantile(finite, 0.95)),
        "predicted_probability_max": float(finite.max()),
    }


def _variant_metrics_frame(
    predictions: pd.DataFrame,
    *,
    model_variant: str,
    probability_column: str,
    base_notes: Sequence[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name in ("train", FIT_SPLIT_NAME, PRIMARY_EVALUATION_SPLIT):
        split_df = predictions[predictions["split"].eq(split_name)].reset_index(drop=True)
        y_true = split_df["label_value"].astype(int).to_numpy()
        y_prob = pd.to_numeric(split_df[probability_column], errors="coerce").to_numpy(dtype=float)
        metrics = compute_evaluation_metrics(y_true, y_prob)
        probability_stats = _finite_probability_stats(y_prob)
        rows.append(
            {
                "model_variant": model_variant,
                "horizon_h": int(split_df["horizon_h"].iloc[0]) if not split_df.empty else pd.NA,
                "split": split_name,
                **probability_stats,
                **metrics,
                "notes": _combine_notes(metrics.get("metric_notes"), base_notes),
            }
        )
    return pd.DataFrame(rows)


def _save_placeholder_figure(
    output_path: Path,
    *,
    title: str,
    message: str,
    figsize: tuple[float, float],
) -> Path:
    _require_matplotlib()
    ensure_directory(output_path.parent)
    figure, axis = plt.subplots(figsize=figsize)
    axis.axis("off")
    axis.text(0.5, 0.6, title, ha="center", va="center", fontsize=13, fontweight="bold")
    axis.text(0.5, 0.4, message, ha="center", va="center", fontsize=11, wrap=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _fit_platt_recalibrator(
    validation_predictions: pd.DataFrame,
) -> tuple[Callable[[np.ndarray], np.ndarray] | None, dict[str, object]]:
    notes: list[str] = []
    finite_validation = validation_predictions[
        validation_predictions["predicted_probability"].map(np.isfinite)
    ].copy()
    fit_summary = {
        "method_name": "platt",
        "fit_split": FIT_SPLIT_NAME,
        "fit_sample_count": int(finite_validation.shape[0]),
        "fit_event_count": int(finite_validation["label_value"].eq(1).sum()),
        "fit_non_event_count": int(finite_validation["label_value"].eq(0).sum()),
        "status": "unavailable",
        "notes": [],
    }
    if finite_validation.empty:
        notes.append("platt_fit_unavailable_no_finite_validation_predictions")
        fit_summary["notes"] = notes
        return None, fit_summary

    y_true = finite_validation["label_value"].astype(int).to_numpy()
    y_prob = finite_validation["predicted_probability"].to_numpy(dtype=float)
    if np.unique(y_true).size < 2:
        notes.append("platt_fit_unavailable_single_class_validation_split")
        fit_summary["notes"] = notes
        return None, fit_summary

    clipped_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    logit_prob = np.log(clipped_prob / (1 - clipped_prob))
    if np.unique(np.round(logit_prob, 12)).size < 2:
        notes.append("platt_fit_constant_validation_margin")

    recalibration_model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=1000,
    )
    try:
        recalibration_model.fit(logit_prob.reshape(-1, 1), y_true)
    except Exception as exc:  # pragma: no cover - defensive branch
        notes.append(f"platt_fit_failed:{type(exc).__name__}")
        fit_summary["notes"] = notes
        return None, fit_summary

    def transform(probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=float)
        transformed = np.full(probabilities.shape, np.nan, dtype=float)
        finite_mask = np.isfinite(probabilities)
        if finite_mask.any():
            clipped = np.clip(probabilities[finite_mask], 1e-6, 1 - 1e-6)
            logits = np.log(clipped / (1 - clipped))
            transformed[finite_mask] = recalibration_model.predict_proba(logits.reshape(-1, 1))[:, 1]
        return transformed

    fit_summary.update(
        {
            "status": "fit",
            "notes": notes,
            "intercept": float(recalibration_model.intercept_[0]),
            "slope": float(recalibration_model.coef_[0][0]),
        }
    )
    return transform, fit_summary


def _fit_isotonic_recalibrator(
    validation_predictions: pd.DataFrame,
) -> tuple[Callable[[np.ndarray], np.ndarray] | None, dict[str, object]]:
    notes: list[str] = []
    finite_validation = validation_predictions[
        validation_predictions["predicted_probability"].map(np.isfinite)
    ].copy()
    fit_summary = {
        "method_name": "isotonic",
        "fit_split": FIT_SPLIT_NAME,
        "fit_sample_count": int(finite_validation.shape[0]),
        "fit_event_count": int(finite_validation["label_value"].eq(1).sum()),
        "fit_non_event_count": int(finite_validation["label_value"].eq(0).sum()),
        "status": "unavailable",
        "notes": [],
    }
    if finite_validation.empty:
        notes.append("isotonic_fit_unavailable_no_finite_validation_predictions")
        fit_summary["notes"] = notes
        return None, fit_summary

    y_true = finite_validation["label_value"].astype(int).to_numpy()
    y_prob = np.clip(
        finite_validation["predicted_probability"].to_numpy(dtype=float),
        0.0,
        1.0,
    )
    if np.unique(y_true).size < 2:
        notes.append("isotonic_fit_unavailable_single_class_validation_split")
        fit_summary["notes"] = notes
        return None, fit_summary
    if np.unique(np.round(y_prob, 12)).size < 2:
        notes.append("isotonic_fit_constant_validation_predictions")

    recalibration_model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    try:
        recalibration_model.fit(y_prob, y_true)
    except Exception as exc:  # pragma: no cover - defensive branch
        notes.append(f"isotonic_fit_failed:{type(exc).__name__}")
        fit_summary["notes"] = notes
        return None, fit_summary

    def transform(probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=float)
        transformed = np.full(probabilities.shape, np.nan, dtype=float)
        finite_mask = np.isfinite(probabilities)
        if finite_mask.any():
            transformed[finite_mask] = recalibration_model.predict(
                np.clip(probabilities[finite_mask], 0.0, 1.0)
            )
        return transformed

    fit_summary.update(
        {
            "status": "fit",
            "notes": notes,
            "threshold_count": int(len(recalibration_model.X_thresholds_)),
        }
    )
    return transform, fit_summary


def _build_recalibrated_predictions(
    raw_xgboost_predictions: pd.DataFrame,
    *,
    recalibration_method: str,
    transform: Callable[[np.ndarray], np.ndarray] | None,
    fit_summary: dict[str, object],
) -> pd.DataFrame:
    recalibrated_predictions = raw_xgboost_predictions.loc[:, IDENTIFIER_COLUMNS].copy()
    recalibrated_predictions["model_name"] = raw_xgboost_predictions["model_name"].astype("string")
    recalibrated_predictions["raw_predicted_probability"] = pd.to_numeric(
        raw_xgboost_predictions["predicted_probability"],
        errors="coerce",
    )
    raw_probabilities = recalibrated_predictions["raw_predicted_probability"].to_numpy(dtype=float)
    recalibrated_predictions["recalibrated_probability"] = (
        transform(raw_probabilities) if transform is not None else np.full(raw_probabilities.shape, np.nan)
    )
    recalibrated_predictions["recalibration_method"] = recalibration_method
    recalibrated_predictions["recalibration_status"] = str(fit_summary["status"])
    recalibrated_predictions["recalibration_notes"] = _combine_notes(
        fit_summary.get("notes", []),
        "fitted_on_validation_split_only" if fit_summary.get("status") == "fit" else None,
    )
    return recalibrated_predictions


def _build_canonical_variant_predictions(
    predictions: pd.DataFrame,
    *,
    model_variant: str,
    probability_column: str,
) -> pd.DataFrame:
    canonical_predictions = predictions.loc[:, IDENTIFIER_COLUMNS].copy()
    canonical_predictions["model_variant"] = model_variant
    canonical_predictions["predicted_probability"] = pd.to_numeric(
        predictions[probability_column],
        errors="coerce",
    )
    return canonical_predictions.loc[:, CANONICAL_VARIANT_PREDICTION_COLUMNS].copy()


def _variant_reliability_summary(
    predictions: pd.DataFrame,
    *,
    model_variant: str,
    probability_column: str,
    split_name: str,
    horizon_h: int,
    n_bins: int,
) -> pd.DataFrame:
    split_predictions = predictions[predictions["split"].eq(split_name)].reset_index(drop=True)
    if split_predictions.empty:
        return pd.DataFrame(columns=RELIABILITY_SUMMARY_COLUMNS)

    summary_input = split_predictions.loc[:, IDENTIFIER_COLUMNS].copy()
    summary_input["predicted_probability"] = pd.to_numeric(
        split_predictions[probability_column],
        errors="coerce",
    )
    summary = _ranked_quantile_risk_summary(
        summary_input,
        model_name=model_variant,
        horizon_h=horizon_h,
        split_name=split_name,
        sample_scope="overall",
        group_id="overall",
        n_bins=n_bins,
    )
    if summary.empty:
        return pd.DataFrame(columns=RELIABILITY_SUMMARY_COLUMNS)
    summary = summary.rename(columns={"model_name": "model_variant"})
    return summary.loc[:, RELIABILITY_SUMMARY_COLUMNS].copy()


def _plot_reliability_comparison(
    combined_summary: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
    subtitle: str,
) -> Path:
    if combined_summary.empty:
        return _save_placeholder_figure(
            output_path,
            title=title,
            message=f"{subtitle}\nNo finite prediction bins were available.",
            figsize=(8.0, 6.0),
        )

    _require_matplotlib()
    ensure_directory(output_path.parent)
    axis_limit = _dynamic_probability_axis_limit(
        combined_summary.rename(columns={"model_variant": "model_name"})
    )
    figure, axis = plt.subplots(figsize=(8.0, 6.0))
    axis.plot(
        [0.0, axis_limit],
        [0.0, axis_limit],
        linestyle="--",
        color="black",
        linewidth=1.0,
        label="Perfect calibration",
    )
    for model_variant in VARIANT_ORDER:
        subset = combined_summary[combined_summary["model_variant"].eq(model_variant)].sort_values(
            "predicted_probability_mean"
        )
        if subset.empty:
            continue
        axis.plot(
            subset["predicted_probability_mean"],
            subset["observed_mortality"],
            marker="o",
            linewidth=2.0,
            color=VARIANT_COLORS[model_variant],
            label=VARIANT_DISPLAY_NAMES[model_variant],
        )

    axis.set_xlim(0.0, axis_limit)
    axis.set_ylim(0.0, axis_limit)
    axis.set_xlabel("Mean predicted risk")
    axis.set_ylabel("Observed mortality")
    axis.set_title(title)
    axis.grid(alpha=0.25, linewidth=0.6)
    axis.legend(loc="upper left")
    axis.text(
        0.02,
        0.98,
        subtitle,
        ha="left",
        va="top",
        transform=axis.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _plot_probability_distribution(
    prediction_variants: dict[str, pd.DataFrame],
    *,
    output_path: Path,
    split_name: str,
    title: str,
    subtitle: str,
) -> Path:
    variant_arrays: dict[str, np.ndarray] = {}
    for model_variant, frame in prediction_variants.items():
        split_frame = frame[frame["split"].eq(split_name)].reset_index(drop=True)
        probability_column = (
            "recalibrated_probability" if "recalibrated_probability" in split_frame.columns else "predicted_probability"
        )
        values = pd.to_numeric(split_frame[probability_column], errors="coerce").to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size:
            variant_arrays[model_variant] = finite_values

    if not variant_arrays:
        return _save_placeholder_figure(
            output_path,
            title=title,
            message=f"{subtitle}\nNo finite predicted probabilities were available.",
            figsize=(8.0, 5.0),
        )

    _require_matplotlib()
    ensure_directory(output_path.parent)
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    bins = np.linspace(0.0, 1.0, 31)
    for model_variant in VARIANT_ORDER:
        values = variant_arrays.get(model_variant)
        if values is None:
            continue
        axis.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=2.0,
            color=VARIANT_COLORS[model_variant],
            label=VARIANT_DISPLAY_NAMES[model_variant],
        )
    axis.set_xlabel("Predicted probability")
    axis.set_ylabel("Count")
    axis.set_title(title)
    axis.grid(alpha=0.25, linewidth=0.6)
    axis.legend(loc="upper right")
    axis.text(
        0.02,
        0.98,
        subtitle,
        ha="left",
        va="top",
        transform=axis.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _plot_horizon_summary(
    test_metrics: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
) -> Path:
    if test_metrics.empty:
        return _save_placeholder_figure(
            output_path,
            title=title,
            message="No test-split metrics were available.",
            figsize=(12.0, 4.5),
        )

    _require_matplotlib()
    ensure_directory(output_path.parent)
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.5), sharex=True)
    metric_panels = [
        ("brier_score", "Brier score"),
        ("calibration_slope", "Calibration slope"),
        ("calibration_intercept", "Calibration intercept"),
    ]
    horizons = sorted(test_metrics["horizon_h"].dropna().astype(int).unique().tolist())

    for axis, (metric_name, label) in zip(axes, metric_panels):
        plotted = False
        for model_variant in VARIANT_ORDER:
            subset = test_metrics[test_metrics["model_variant"].eq(model_variant)].sort_values("horizon_h")
            metric_values = pd.to_numeric(subset[metric_name], errors="coerce")
            if metric_values.notna().any():
                axis.plot(
                    subset["horizon_h"],
                    metric_values,
                    marker="o",
                    linewidth=2.0,
                    color=VARIANT_COLORS[model_variant],
                    label=VARIANT_DISPLAY_NAMES[model_variant],
                )
                plotted = True
        axis.set_title(label)
        axis.set_xlabel("Horizon (h)")
        axis.set_xticks(horizons)
        axis.grid(alpha=0.25, linewidth=0.6)
        if metric_name == "brier_score":
            axis.set_ylim(bottom=0.0)
        if not plotted:
            axis.text(
                0.5,
                0.5,
                "Metric not binary-evaluable\non current test artifacts",
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontsize=10,
            )
    axes[0].legend(loc="upper left")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _horizon_output_dir(output_dir: Path, horizon_h: int) -> Path:
    return output_dir / f"horizon_{int(horizon_h)}h"


def _format_metric_value(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


def _build_interpretation_note(combined_metrics: pd.DataFrame) -> str:
    test_metrics = combined_metrics[combined_metrics["split"].eq(PRIMARY_EVALUATION_SPLIT)].copy()
    lines = [
        "# Chapter 1 ASIC XGBoost Recalibration Review",
        "",
        f"Recalibration methods fit on `{FIT_SPLIT_NAME}` only and applied unchanged to `{PRIMARY_EVALUATION_SPLIT}`.",
        "Reference comparisons include Logistic Regression and raw XGBoost from the saved baseline prediction artifacts.",
        "",
    ]

    if test_metrics.empty:
        lines.append("No test-split rows were available, so no recalibration interpretation was possible.")
        return "\n".join(lines)

    test_evaluable = bool(test_metrics["binary_metrics_evaluable"].astype(bool).any())
    if not test_evaluable:
        lines.extend(
            [
                "The currently available test artifacts are not binary-evaluable for any horizon.",
                "On this local artifact bundle, the test split has no events, so AUROC, AUPRC, calibration slope, and calibration intercept are unavailable on test.",
                "Brier score and reliability-style plots were still written, but full probability interpretation should wait for a binary-evaluable frozen test artifact set.",
                "",
            ]
        )

    for horizon_h in sorted(test_metrics["horizon_h"].dropna().astype(int).unique().tolist()):
        horizon_metrics = test_metrics[test_metrics["horizon_h"].astype(int).eq(horizon_h)].copy()
        by_variant = horizon_metrics.set_index("model_variant")
        raw_row = by_variant.loc["xgboost_raw"]
        logistic_row = by_variant.loc["logistic_regression"]
        platt_row = by_variant.loc["xgboost_platt"]
        isotonic_row = by_variant.loc["xgboost_isotonic"]
        lines.append(
            f"- {horizon_h}h test: logistic Brier {_format_metric_value(logistic_row.brier_score)}, "
            f"raw XGBoost Brier {_format_metric_value(raw_row.brier_score)}, "
            f"Platt Brier {_format_metric_value(platt_row.brier_score)}, "
            f"isotonic Brier {_format_metric_value(isotonic_row.brier_score)}."
        )
        if bool(horizon_metrics["binary_metrics_evaluable"].astype(bool).any()):
            lines.append(
                f"  Raw XGBoost calibration slope/intercept: "
                f"{_format_metric_value(raw_row.calibration_slope)} / {_format_metric_value(raw_row.calibration_intercept)}; "
                f"Platt: {_format_metric_value(platt_row.calibration_slope)} / {_format_metric_value(platt_row.calibration_intercept)}; "
                f"isotonic: {_format_metric_value(isotonic_row.calibration_slope)} / {_format_metric_value(isotonic_row.calibration_intercept)}."
            )

    lines.extend(
        [
            "",
            "Preliminary interpretation:",
            "- Recalibration can improve XGBoost probability outputs without changing the underlying ranking model, but the current local bundle is too sparse on test to make a strong chapter-level claim.",
            "- Logistic regression should remain the primary Chapter 1 anchor unless the full evaluable ASIC test bundle shows recalibrated XGBoost delivers stable probability improvement without losing the expected discrimination advantage.",
        ]
    )
    return "\n".join(lines)


def run_asic_xgboost_recalibration(
    *,
    input_root: Path = DEFAULT_BASELINE_ARTIFACT_ROOT,
    output_dir: Path = DEFAULT_RECALIBRATION_OUTPUT_DIR,
    horizons: Sequence[int] | None = None,
    reliability_bin_count: int = DEFAULT_RELIABILITY_BIN_COUNT,
) -> XGBoostRecalibrationRunResult:
    input_root = Path(input_root)
    output_dir = Path(output_dir)
    selected_horizons = _normalize_horizons(input_root, horizons)

    horizon_results: list[HorizonRecalibrationArtifacts] = []
    combined_metrics_frames: list[pd.DataFrame] = []
    combined_reliability_summaries: list[pd.DataFrame] = []
    consumed_prediction_artifacts: list[dict[str, object]] = []

    for horizon_h in selected_horizons:
        logistic_predictions_path = _prediction_artifact_path(
            input_root,
            DEFAULT_REFERENCE_MODEL_NAME,
            horizon_h,
        )
        xgboost_predictions_path = _prediction_artifact_path(
            input_root,
            DEFAULT_XGBOOST_MODEL_NAME,
            horizon_h,
        )
        logistic_predictions = _load_prediction_frame(
            logistic_predictions_path,
            expected_model_name=DEFAULT_REFERENCE_MODEL_NAME,
            expected_horizon_h=horizon_h,
        )
        xgboost_predictions = _load_prediction_frame(
            xgboost_predictions_path,
            expected_model_name=DEFAULT_XGBOOST_MODEL_NAME,
            expected_horizon_h=horizon_h,
        )
        _verify_reference_alignment(logistic_predictions, xgboost_predictions, horizon_h=horizon_h)

        consumed_prediction_artifacts.extend(
            [
                {
                    "model_name": DEFAULT_REFERENCE_MODEL_NAME,
                    "horizon_h": horizon_h,
                    "predictions_path": str(logistic_predictions_path.resolve()),
                },
                {
                    "model_name": DEFAULT_XGBOOST_MODEL_NAME,
                    "horizon_h": horizon_h,
                    "predictions_path": str(xgboost_predictions_path.resolve()),
                },
            ]
        )

        horizon_output_dir = _horizon_output_dir(output_dir, horizon_h)
        ensure_directory(horizon_output_dir)

        validation_predictions = xgboost_predictions[
            xgboost_predictions["split"].eq(FIT_SPLIT_NAME)
        ].reset_index(drop=True)
        fitters = {
            "platt": _fit_platt_recalibrator,
            "isotonic": _fit_isotonic_recalibrator,
        }
        recalibrated_prediction_frames: dict[str, pd.DataFrame] = {}
        recalibrated_metrics_frames: dict[str, pd.DataFrame] = {}
        method_prediction_paths: dict[str, Path] = {}
        method_metrics_paths: dict[str, Path] = {}
        method_fit_summaries: dict[str, dict[str, object]] = {}
        method_statuses: dict[str, str] = {}
        canonical_prediction_paths: dict[str, Path] = {}
        canonical_prediction_frames: dict[str, pd.DataFrame] = {}

        logistic_metrics = _variant_metrics_frame(
            logistic_predictions,
            model_variant="logistic_regression",
            probability_column="predicted_probability",
            base_notes=["reference_saved_logistic_predictions"],
        )
        raw_xgboost_metrics = _variant_metrics_frame(
            xgboost_predictions,
            model_variant="xgboost_raw",
            probability_column="predicted_probability",
            base_notes=["reference_saved_xgboost_predictions"],
        )
        canonical_prediction_frames["xgboost_raw"] = _build_canonical_variant_predictions(
            xgboost_predictions,
            model_variant="xgboost_raw",
            probability_column="predicted_probability",
        )
        canonical_prediction_paths["xgboost_raw"] = write_dataframe(
            canonical_prediction_frames["xgboost_raw"],
            horizon_output_dir / "xgboost_raw_canonical_predictions.csv",
            output_format="csv",
        )

        logistic_metrics_path = write_dataframe(
            logistic_metrics,
            horizon_output_dir / "logistic_reference_metrics_by_split.csv",
            output_format="csv",
        )
        raw_xgboost_metrics_path = write_dataframe(
            raw_xgboost_metrics,
            horizon_output_dir / "xgboost_raw_metrics_by_split.csv",
            output_format="csv",
        )

        for method_name in DEFAULT_RECALIBRATION_METHODS:
            transform, fit_summary = fitters[method_name](validation_predictions)
            method_fit_summaries[method_name] = fit_summary
            method_statuses[method_name] = str(fit_summary["status"])
            recalibrated_predictions = _build_recalibrated_predictions(
                xgboost_predictions,
                recalibration_method=method_name,
                transform=transform,
                fit_summary=fit_summary,
            )
            variant_name = f"xgboost_{method_name}"
            recalibrated_metrics = _variant_metrics_frame(
                recalibrated_predictions,
                model_variant=variant_name,
                probability_column="recalibrated_probability",
                base_notes=[
                    "recalibrated_saved_xgboost_predictions",
                    f"fit_split={FIT_SPLIT_NAME}",
                    f"recalibration_method={method_name}",
                    *[str(note) for note in fit_summary.get("notes", [])],
                ],
            )
            predictions_path = write_dataframe(
                recalibrated_predictions,
                horizon_output_dir / f"{method_name}_predictions.csv",
                output_format="csv",
            )
            metrics_path = write_dataframe(
                recalibrated_metrics,
                horizon_output_dir / f"{method_name}_metrics_by_split.csv",
                output_format="csv",
            )
            _write_json(
                fit_summary,
                horizon_output_dir / f"{method_name}_fit_summary.json",
            )
            recalibrated_prediction_frames[method_name] = recalibrated_predictions
            recalibrated_metrics_frames[method_name] = recalibrated_metrics
            method_prediction_paths[method_name] = predictions_path
            method_metrics_paths[method_name] = metrics_path
            canonical_variant_name = f"xgboost_{method_name}"
            canonical_prediction_frames[canonical_variant_name] = _build_canonical_variant_predictions(
                recalibrated_predictions,
                model_variant=canonical_variant_name,
                probability_column="recalibrated_probability",
            )
            canonical_prediction_paths[canonical_variant_name] = write_dataframe(
                canonical_prediction_frames[canonical_variant_name],
                horizon_output_dir / f"{canonical_variant_name}_canonical_predictions.csv",
                output_format="csv",
            )

        combined_canonical_predictions = pd.concat(
            [canonical_prediction_frames[model_variant] for model_variant in VARIANT_ORDER[1:]],
            ignore_index=True,
        ).sort_values(
            [*IDENTIFIER_COLUMNS, "model_variant"],
            kind="stable",
        ).reset_index(drop=True)
        combined_canonical_predictions_path = write_dataframe(
            combined_canonical_predictions,
            horizon_output_dir / "xgboost_canonical_variant_predictions.csv",
            output_format="csv",
        )

        comparison_metrics = pd.concat(
            [
                logistic_metrics,
                raw_xgboost_metrics,
                *[recalibrated_metrics_frames[method_name] for method_name in DEFAULT_RECALIBRATION_METHODS],
            ],
            ignore_index=True,
        ).sort_values(["split", "model_variant"]).reset_index(drop=True)
        comparison_metrics_path = write_dataframe(
            comparison_metrics,
            horizon_output_dir / "comparison_metrics.csv",
            output_format="csv",
        )

        test_reliability_summary = pd.concat(
            [
                _variant_reliability_summary(
                    logistic_predictions,
                    model_variant="logistic_regression",
                    probability_column="predicted_probability",
                    split_name=PRIMARY_EVALUATION_SPLIT,
                    horizon_h=horizon_h,
                    n_bins=reliability_bin_count,
                ),
                _variant_reliability_summary(
                    xgboost_predictions,
                    model_variant="xgboost_raw",
                    probability_column="predicted_probability",
                    split_name=PRIMARY_EVALUATION_SPLIT,
                    horizon_h=horizon_h,
                    n_bins=reliability_bin_count,
                ),
                *[
                    _variant_reliability_summary(
                        recalibrated_prediction_frames[method_name],
                        model_variant=f"xgboost_{method_name}",
                        probability_column="recalibrated_probability",
                        split_name=PRIMARY_EVALUATION_SPLIT,
                        horizon_h=horizon_h,
                        n_bins=reliability_bin_count,
                    )
                    for method_name in DEFAULT_RECALIBRATION_METHODS
                ],
            ],
            ignore_index=True,
        )
        reliability_summary_path = write_dataframe(
            test_reliability_summary,
            horizon_output_dir / "test_reliability_binned_summary.csv",
            output_format="csv",
        )
        split_metrics_row = raw_xgboost_metrics.set_index("split").loc[PRIMARY_EVALUATION_SPLIT]
        subtitle = (
            f"Fit split: {FIT_SPLIT_NAME}; plotted split: {PRIMARY_EVALUATION_SPLIT}; "
            f"n={int(split_metrics_row['sample_count'])}; events={int(split_metrics_row['event_count'])}"
        )
        reliability_plot_path = _plot_reliability_comparison(
            test_reliability_summary,
            output_path=horizon_output_dir / "test_reliability_comparison.png",
            title=f"Chapter 1 ASIC {horizon_h}h recalibration comparison",
            subtitle=subtitle,
        )
        probability_distribution_plot_path = _plot_probability_distribution(
            {
                "logistic_regression": logistic_predictions,
                "xgboost_raw": xgboost_predictions,
                "xgboost_platt": recalibrated_prediction_frames["platt"],
                "xgboost_isotonic": recalibrated_prediction_frames["isotonic"],
            },
            output_path=horizon_output_dir / "test_probability_distribution.png",
            split_name=PRIMARY_EVALUATION_SPLIT,
            title=f"Chapter 1 ASIC {horizon_h}h predicted-probability distribution",
            subtitle=subtitle,
        )

        metadata_path = _write_json(
            {
                "timestamp_utc": _utc_timestamp(),
                "horizon_h": horizon_h,
                "fit_split": FIT_SPLIT_NAME,
                "primary_evaluation_split": PRIMARY_EVALUATION_SPLIT,
                "source_prediction_artifacts": {
                    DEFAULT_REFERENCE_MODEL_NAME: str(logistic_predictions_path.resolve()),
                    DEFAULT_XGBOOST_MODEL_NAME: str(xgboost_predictions_path.resolve()),
                },
                "reference_metric_paths": {
                    "logistic_regression": str(logistic_metrics_path.resolve()),
                    "xgboost_raw": str(raw_xgboost_metrics_path.resolve()),
                },
                "method_fit_summaries": method_fit_summaries,
                "method_prediction_paths": {
                    key: str(value.resolve()) for key, value in method_prediction_paths.items()
                },
                "canonical_prediction_paths": {
                    key: str(value.resolve()) for key, value in canonical_prediction_paths.items()
                },
                "combined_canonical_predictions_path": str(
                    combined_canonical_predictions_path.resolve()
                ),
                "method_metrics_paths": {
                    key: str(value.resolve()) for key, value in method_metrics_paths.items()
                },
                "comparison_metrics_path": str(comparison_metrics_path.resolve()),
                "test_reliability_summary_path": str(reliability_summary_path.resolve()),
                "figure_paths": {
                    "test_reliability_comparison": str(reliability_plot_path.resolve()),
                    "test_probability_distribution": str(
                        probability_distribution_plot_path.resolve()
                    ),
                },
            },
            horizon_output_dir / "metadata.json",
        )

        combined_metrics_frames.append(comparison_metrics)
        combined_reliability_summaries.append(test_reliability_summary)
        horizon_results.append(
            HorizonRecalibrationArtifacts(
                horizon_h=horizon_h,
                output_dir=horizon_output_dir,
                comparison_metrics_path=comparison_metrics_path,
                reliability_summary_path=reliability_summary_path,
                reliability_plot_path=reliability_plot_path,
                probability_distribution_plot_path=probability_distribution_plot_path,
                metadata_path=metadata_path,
                method_prediction_paths=method_prediction_paths,
                method_metrics_paths=method_metrics_paths,
                method_statuses=method_statuses,
                canonical_prediction_paths=canonical_prediction_paths,
                combined_canonical_predictions_path=combined_canonical_predictions_path,
            )
        )

    ensure_directory(output_dir)
    combined_comparison_metrics = pd.concat(combined_metrics_frames, ignore_index=True).sort_values(
        ["horizon_h", "split", "model_variant"]
    ).reset_index(drop=True)
    combined_comparison_metrics_path = write_dataframe(
        combined_comparison_metrics,
        output_dir / "combined_comparison_metrics.csv",
        output_format="csv",
    )
    combined_test_reliability_summary = pd.concat(
        combined_reliability_summaries,
        ignore_index=True,
    ) if combined_reliability_summaries else pd.DataFrame(columns=RELIABILITY_SUMMARY_COLUMNS)
    combined_test_reliability_summary_path = write_dataframe(
        combined_test_reliability_summary,
        output_dir / "combined_test_reliability_binned_summary.csv",
        output_format="csv",
    )
    summary_figure_path = _plot_horizon_summary(
        combined_comparison_metrics[
            combined_comparison_metrics["split"].eq(PRIMARY_EVALUATION_SPLIT)
        ].reset_index(drop=True),
        output_path=output_dir / "test_horizon_calibration_summary.png",
        title="Chapter 1 ASIC XGBoost recalibration summary on test split",
    )
    interpretation_note = _build_interpretation_note(combined_comparison_metrics)
    interpretation_note_path = write_text(
        interpretation_note,
        output_dir / "interpretation_note.md",
    )
    manifest_path = _write_json(
        {
            "timestamp_utc": _utc_timestamp(),
            "input_root": str(input_root.resolve()),
            "output_dir": str(output_dir.resolve()),
            "horizons_processed": list(selected_horizons),
            "fit_split": FIT_SPLIT_NAME,
            "primary_evaluation_split": PRIMARY_EVALUATION_SPLIT,
            "recalibration_methods": list(DEFAULT_RECALIBRATION_METHODS),
            "consumed_prediction_artifacts": consumed_prediction_artifacts,
            "horizon_artifacts": [
                {
                    "horizon_h": horizon_result.horizon_h,
                    "output_dir": str(horizon_result.output_dir.resolve()),
                    "comparison_metrics_path": str(
                        horizon_result.comparison_metrics_path.resolve()
                    ),
                    "reliability_summary_path": str(
                        horizon_result.reliability_summary_path.resolve()
                    ),
                    "reliability_plot_path": str(
                        horizon_result.reliability_plot_path.resolve()
                    ),
                    "probability_distribution_plot_path": str(
                        horizon_result.probability_distribution_plot_path.resolve()
                    ),
                    "metadata_path": str(horizon_result.metadata_path.resolve()),
                    "method_prediction_paths": {
                        key: str(value.resolve())
                        for key, value in horizon_result.method_prediction_paths.items()
                    },
                    "canonical_prediction_paths": {
                        key: str(value.resolve())
                        for key, value in horizon_result.canonical_prediction_paths.items()
                    },
                    "combined_canonical_predictions_path": str(
                        horizon_result.combined_canonical_predictions_path.resolve()
                    ),
                    "method_metrics_paths": {
                        key: str(value.resolve())
                        for key, value in horizon_result.method_metrics_paths.items()
                    },
                    "method_statuses": dict(horizon_result.method_statuses),
                }
                for horizon_result in horizon_results
            ],
            "output_files": {
                "combined_comparison_metrics": str(combined_comparison_metrics_path.resolve()),
                "combined_test_reliability_binned_summary": str(
                    combined_test_reliability_summary_path.resolve()
                ),
                "summary_figure": str(summary_figure_path.resolve()),
                "interpretation_note": str(interpretation_note_path.resolve()),
            },
        },
        output_dir / "run_manifest.json",
    )
    return XGBoostRecalibrationRunResult(
        input_root=input_root,
        output_dir=output_dir,
        horizons_processed=selected_horizons,
        combined_comparison_metrics_path=combined_comparison_metrics_path,
        combined_test_reliability_summary_path=combined_test_reliability_summary_path,
        summary_figure_path=summary_figure_path,
        interpretation_note_path=interpretation_note_path,
        manifest_path=manifest_path,
        horizon_results=tuple(horizon_results),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit validation-only post hoc recalibration on saved Chapter 1 ASIC XGBoost "
            "baseline predictions and compare raw versus recalibrated probabilities."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_BASELINE_ARTIFACT_ROOT,
        help="Root directory containing saved baseline prediction artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RECALIBRATION_OUTPUT_DIR,
        help="Root directory for saved recalibration outputs.",
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

    result = run_asic_xgboost_recalibration(
        input_root=args.input_root,
        output_dir=args.output_dir,
        horizons=args.horizons,
    )

    print(f"Input root: {result.input_root}")
    print(f"Output directory: {result.output_dir}")
    print(f"Combined comparison metrics: {result.combined_comparison_metrics_path}")
    print(f"Combined test reliability summary: {result.combined_test_reliability_summary_path}")
    print(f"Summary figure: {result.summary_figure_path}")
    print(f"Interpretation note: {result.interpretation_note_path}")
    print(f"Run manifest: {result.manifest_path}")
    print()
    for horizon_result in result.horizon_results:
        method_status_text = ", ".join(
            f"{method_name}={status}"
            for method_name, status in sorted(horizon_result.method_statuses.items())
        )
        print(
            f"horizon {horizon_result.horizon_h}h -> {horizon_result.output_dir} "
            f"(method_statuses: {method_status_text})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
