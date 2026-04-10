from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

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

from chapter1_mortality_decomposition.baseline_logistic import (
    IDENTIFIER_COLUMNS,
    compute_binary_classification_metrics,
)
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
DEFAULT_EVALUATION_OUTPUT_DIR = (
    Path("artifacts") / "chapter1" / "evaluation" / "asic" / "baselines" / "primary_medians"
)
DEFAULT_MODEL_NAMES = ("logistic_regression", "xgboost")
DEFAULT_HORIZONS = (8, 16, 24, 48, 72)
DEFAULT_REPORTING_SPLIT_ORDER = ("test", "validation", "train")
PRIMARY_HORIZON = 24
SECONDARY_HORIZON = 48
DEFAULT_RISK_BIN_COUNT = 10
DEFAULT_SITE_RISK_BIN_COUNT = 5
PRIMARY_SITE_CHECK_HORIZONS = (24,)
PREDICTION_REQUIRED_COLUMNS = set(IDENTIFIER_COLUMNS) | {"predicted_probability", "model_name"}
RISK_BIN_SUMMARY_COLUMNS = [
    "model_name",
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


@dataclass(frozen=True)
class PredictionArtifact:
    model_name: str
    horizon_h: int
    predictions_path: Path


@dataclass(frozen=True)
class EvaluationRunResult:
    input_root: Path
    output_dir: Path
    combined_metrics_path: Path
    combined_risk_binned_summary_path: Path
    reporting_split_summary_path: Path
    interpretation_note_path: Path
    manifest_path: Path


def _require_matplotlib() -> None:
    if plt is None:
        raise ImportError(
            "matplotlib is required for the Chapter 1 ASIC evaluation package."
        ) from MATPLOTLIB_IMPORT_ERROR


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ordered_unique(values: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


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


def _display_model_name(model_name: str) -> str:
    return {
        "logistic_regression": "Logistic Regression",
        "xgboost": "XGBoost",
    }.get(model_name, model_name.replace("_", " ").title())


def _normalize_models(
    input_root: Path,
    models: Sequence[str] | None,
) -> tuple[str, ...]:
    available_models = sorted(
        path.name for path in input_root.iterdir() if path.is_dir()
    ) if input_root.exists() else []
    if models is None:
        selected_models = tuple(DEFAULT_MODEL_NAMES)
    else:
        selected_models = tuple(dict.fromkeys(str(model) for model in models))
    missing = [model for model in selected_models if model not in available_models]
    if missing:
        raise ValueError(
            f"Missing baseline artifact directories for models: {missing}. "
            f"Available under {input_root}: {available_models}"
        )
    return selected_models


def _discover_prediction_artifacts(
    input_root: Path,
    models: Sequence[str],
    horizons: Sequence[int] | None,
) -> tuple[PredictionArtifact, ...]:
    selected_horizons = tuple(int(horizon) for horizon in (horizons or DEFAULT_HORIZONS))
    artifacts: list[PredictionArtifact] = []
    for model_name in models:
        model_root = input_root / model_name
        available_horizons = sorted(
            int(path.name.removeprefix("horizon_").removesuffix("h"))
            for path in model_root.glob("horizon_*h")
            if path.is_dir()
        )
        missing_horizons = [horizon for horizon in selected_horizons if horizon not in available_horizons]
        if missing_horizons:
            raise ValueError(
                f"Model {model_name} is missing requested horizons {missing_horizons}. "
                f"Available horizons: {available_horizons}"
            )

        for horizon_h in selected_horizons:
            # Formal Chapter 1 baseline metrics continue to run on the saved evaluation-only
            # prediction export. The parallel all_valid_predictions.csv artifact is reserved for
            # descriptive trajectory analysis and must not replace predictions.csv here.
            predictions_path = model_root / f"horizon_{horizon_h}h" / "predictions.csv"
            if not predictions_path.exists():
                raise FileNotFoundError(
                    f"Expected saved baseline predictions at {predictions_path}"
                )
            artifacts.append(
                PredictionArtifact(
                    model_name=model_name,
                    horizon_h=int(horizon_h),
                    predictions_path=predictions_path,
                )
            )
    return tuple(artifacts)


def _load_prediction_frame(artifact: PredictionArtifact) -> pd.DataFrame:
    predictions = read_dataframe(artifact.predictions_path)
    require_columns(predictions, PREDICTION_REQUIRED_COLUMNS, str(artifact.predictions_path))
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
        raise ValueError(f"{artifact.predictions_path} contains multiple model_name values.")
    if predictions["model_name"].dropna().iloc[0] != artifact.model_name:
        raise ValueError(
            f"{artifact.predictions_path} reported model_name "
            f"{predictions['model_name'].dropna().iloc[0]!r}, expected {artifact.model_name!r}."
        )
    observed_horizons = (
        predictions["horizon_h"].dropna().astype(int).unique().tolist()
    )
    if observed_horizons != [artifact.horizon_h]:
        raise ValueError(
            f"{artifact.predictions_path} should contain only horizon {artifact.horizon_h}, "
            f"found {observed_horizons}."
        )
    return predictions


def compute_evaluation_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
) -> dict[str, object]:
    y_true_array = np.asarray(y_true, dtype=int)
    y_prob_array = np.asarray(y_prob, dtype=float)

    metrics = compute_binary_classification_metrics(y_true_array, y_prob_array)
    notes: list[str] = []
    existing_notes = metrics.get("metric_notes")
    if isinstance(existing_notes, str):
        notes.extend(existing_notes.split("; "))

    metrics["brier_score"] = np.nan
    sample_count = int(metrics["sample_count"])
    if sample_count == 0:
        notes.append("brier_unavailable_empty_split")
    elif not np.isfinite(y_prob_array).all():
        notes.append("brier_unavailable_non_finite_predictions")
    else:
        try:
            metrics["brier_score"] = float(brier_score_loss(y_true_array, y_prob_array))
        except Exception as exc:  # pragma: no cover - defensive branch
            notes.append(f"brier_failed:{type(exc).__name__}")

    metrics["binary_metrics_evaluable"] = bool(
        sample_count > 0
        and int(metrics["event_count"]) > 0
        and int(metrics["non_event_count"]) > 0
        and np.isfinite(y_prob_array).all()
    )
    metrics["finite_prediction_count"] = int(np.isfinite(y_prob_array).sum())
    metrics["metric_notes"] = "; ".join(_ordered_unique(notes)) if notes else pd.NA
    return metrics


def _metrics_by_split(
    predictions: pd.DataFrame,
    *,
    model_name: str,
    horizon_h: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name in DEFAULT_REPORTING_SPLIT_ORDER:
        split_df = predictions[predictions["split"].eq(split_name)].reset_index(drop=True)
        metrics = compute_evaluation_metrics(
            split_df["label_value"].astype(int).to_numpy(),
            split_df["predicted_probability"].to_numpy(),
        )
        rows.append(
            {
                "model_name": model_name,
                "horizon_h": int(horizon_h),
                "split": split_name,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _select_reporting_split(metrics_by_split: pd.DataFrame) -> dict[str, object]:
    ordered = metrics_by_split.set_index("split")
    for split_name in DEFAULT_REPORTING_SPLIT_ORDER:
        if split_name not in ordered.index:
            continue
        row = ordered.loc[split_name]
        if bool(row["binary_metrics_evaluable"]):
            return {
                "selected_split": split_name,
                "selected_split_evaluable": True,
                "selection_reason": "first_binary_evaluable_split_in_priority_order",
                "sample_count": int(row["sample_count"]),
                "event_count": int(row["event_count"]),
                "non_event_count": int(row["non_event_count"]),
            }
    for split_name in DEFAULT_REPORTING_SPLIT_ORDER:
        if split_name not in ordered.index:
            continue
        row = ordered.loc[split_name]
        if int(row["sample_count"]) > 0 and int(row["finite_prediction_count"]) > 0:
            return {
                "selected_split": split_name,
                "selected_split_evaluable": False,
                "selection_reason": "fallback_to_first_nonempty_split_with_predictions",
                "sample_count": int(row["sample_count"]),
                "event_count": int(row["event_count"]),
                "non_event_count": int(row["non_event_count"]),
            }
    return {
        "selected_split": DEFAULT_REPORTING_SPLIT_ORDER[0],
        "selected_split_evaluable": False,
        "selection_reason": "all_splits_empty",
        "sample_count": 0,
        "event_count": 0,
        "non_event_count": 0,
    }


def _empty_risk_bin_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=RISK_BIN_SUMMARY_COLUMNS)


def _ranked_quantile_risk_summary(
    predictions: pd.DataFrame,
    *,
    model_name: str,
    horizon_h: int,
    split_name: str,
    sample_scope: str,
    group_id: str,
    n_bins: int,
) -> pd.DataFrame:
    if predictions.empty:
        return _empty_risk_bin_summary()

    finite_predictions = predictions[np.isfinite(predictions["predicted_probability"])].copy()
    if finite_predictions.empty:
        return _empty_risk_bin_summary()

    actual_bin_count = max(1, min(int(n_bins), int(finite_predictions.shape[0])))
    ranked_predictions = finite_predictions["predicted_probability"].rank(method="first")
    bin_codes = pd.qcut(ranked_predictions, q=actual_bin_count, labels=False, duplicates="drop")
    if bin_codes.isna().all():
        bin_codes = pd.Series(np.zeros(finite_predictions.shape[0], dtype=int), index=finite_predictions.index)
    finite_predictions["bin_index"] = pd.Series(bin_codes, index=finite_predictions.index).astype(int) + 1

    grouped = (
        finite_predictions.groupby("bin_index", sort=True)
        .agg(
            sample_count=("label_value", "size"),
            event_count=("label_value", lambda values: int(pd.Series(values).eq(1).sum())),
            predicted_probability_mean=("predicted_probability", "mean"),
            predicted_probability_min=("predicted_probability", "min"),
            predicted_probability_max=("predicted_probability", "max"),
        )
        .reset_index()
    )
    grouped["non_event_count"] = grouped["sample_count"] - grouped["event_count"]
    grouped["observed_mortality"] = grouped["event_count"] / grouped["sample_count"]
    grouped["sample_fraction"] = grouped["sample_count"] / grouped["sample_count"].sum()
    total_events = int(grouped["event_count"].sum())
    grouped["event_fraction_of_events"] = (
        grouped["event_count"] / total_events if total_events > 0 else np.nan
    )
    grouped["bin_label"] = grouped["bin_index"].map(lambda value: f"Q{int(value):02d}")
    grouped["model_name"] = model_name
    grouped["horizon_h"] = int(horizon_h)
    grouped["split"] = split_name
    grouped["sample_scope"] = sample_scope
    grouped["group_id"] = group_id
    return grouped.loc[:, RISK_BIN_SUMMARY_COLUMNS].copy()


def _dynamic_probability_axis_limit(summary: pd.DataFrame) -> float:
    if summary.empty:
        return 1.0
    observed_max = float(summary["observed_mortality"].max())
    predicted_max = float(summary["predicted_probability_max"].max())
    upper = max(0.10, observed_max, predicted_max) * 1.10
    return min(max(upper, 0.05), 1.0)


def _save_placeholder_figure(
    output_path: Path,
    *,
    title: str,
    message: str,
    figsize: tuple[float, float] = (8.0, 4.5),
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


def _plot_reliability(
    summary: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
    subtitle: str,
) -> Path:
    if summary.empty:
        return _save_placeholder_figure(
            output_path,
            title=title,
            message=f"{subtitle}\nNo finite prediction bins were available.",
        )

    _require_matplotlib()
    ensure_directory(output_path.parent)
    axis_limit = _dynamic_probability_axis_limit(summary)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(8.0, 7.0),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=False,
    )
    calibration_axis, count_axis = axes

    calibration_axis.plot(
        [0.0, axis_limit],
        [0.0, axis_limit],
        linestyle="--",
        color="black",
        linewidth=1.0,
        label="Perfect calibration",
    )
    calibration_axis.plot(
        summary["predicted_probability_mean"],
        summary["observed_mortality"],
        marker="o",
        linewidth=2.0,
        color="#1f77b4",
        label="Binned calibration",
    )
    calibration_axis.set_xlim(0.0, axis_limit)
    calibration_axis.set_ylim(0.0, axis_limit)
    calibration_axis.set_xlabel("Mean predicted risk")
    calibration_axis.set_ylabel("Observed mortality")
    calibration_axis.set_title(title)
    calibration_axis.grid(alpha=0.25, linewidth=0.6)
    calibration_axis.legend(loc="upper left")
    calibration_axis.text(
        0.02,
        0.98,
        subtitle,
        ha="left",
        va="top",
        transform=calibration_axis.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    x_positions = np.arange(summary.shape[0])
    count_axis.bar(x_positions, summary["sample_count"], color="#d9d9d9", edgecolor="#666666")
    count_axis.set_xticks(x_positions)
    count_axis.set_xticklabels(summary["bin_label"].tolist())
    count_axis.set_ylabel("Samples")
    count_axis.set_xlabel("Risk quantile bin")
    count_axis.grid(axis="y", alpha=0.25, linewidth=0.6)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _plot_mortality_vs_risk(
    summary: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
    subtitle: str,
) -> Path:
    if summary.empty:
        return _save_placeholder_figure(
            output_path,
            title=title,
            message=f"{subtitle}\nNo finite prediction bins were available.",
        )

    _require_matplotlib()
    ensure_directory(output_path.parent)
    axis_limit = _dynamic_probability_axis_limit(summary)
    x_positions = np.arange(summary.shape[0])
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(8.0, 7.5),
        gridspec_kw={"height_ratios": [1.2, 2.0]},
        sharex=True,
    )
    count_axis, risk_axis = axes

    count_axis.bar(
        x_positions,
        summary["sample_count"],
        color="#c7d4e0",
        edgecolor="#5c7080",
        label="All prediction instances",
    )
    count_axis.bar(
        x_positions,
        summary["event_count"],
        color="#c44e52",
        edgecolor="#8b2f31",
        label="Fatal cases",
    )
    count_axis.set_ylabel("Count")
    count_axis.set_title(title)
    count_axis.legend(loc="upper left")
    count_axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    count_axis.text(
        0.02,
        0.98,
        subtitle,
        ha="left",
        va="top",
        transform=count_axis.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    risk_axis.plot(
        x_positions,
        summary["predicted_probability_mean"],
        marker="o",
        linewidth=2.0,
        color="#1f77b4",
        label="Mean predicted risk",
    )
    risk_axis.plot(
        x_positions,
        summary["observed_mortality"],
        marker="s",
        linewidth=2.0,
        color="#c44e52",
        label="Observed mortality",
    )
    risk_axis.set_ylabel("Risk / mortality")
    risk_axis.set_xlabel("Risk quantile bin")
    risk_axis.set_ylim(0.0, axis_limit)
    risk_axis.set_xticks(x_positions)
    risk_axis.set_xticklabels(summary["bin_label"].tolist())
    risk_axis.grid(alpha=0.25, linewidth=0.6)
    risk_axis.legend(loc="upper left")

    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _plot_horizon_comparison(
    reporting_metrics: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
) -> Path:
    _require_matplotlib()
    ensure_directory(output_path.parent)

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=True)
    metric_panels = [
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("calibration_slope", "Calibration slope"),
        ("calibration_intercept", "Calibration intercept"),
    ]
    for axis, (metric_name, label) in zip(axes.flat, metric_panels):
        axis.plot(
            reporting_metrics["horizon_h"],
            reporting_metrics[metric_name],
            marker="o",
            linewidth=2.0,
            color="#1f77b4",
        )
        for row in reporting_metrics.itertuples(index=False):
            value = getattr(row, metric_name)
            if pd.notna(value):
                axis.annotate(
                    row.selected_split[:3],
                    (row.horizon_h, value),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=8,
                )
        axis.set_title(label)
        axis.set_xlabel("Horizon (h)")
        axis.grid(alpha=0.25, linewidth=0.6)
        if metric_name in {"auroc", "auprc"}:
            axis.set_ylim(0.0, 1.0)
        axis.set_xticks(reporting_metrics["horizon_h"].tolist())
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _plot_horizon_risk_structure_grid(
    risk_summaries: dict[int, pd.DataFrame],
    reporting_splits: dict[int, str],
    evaluable_by_horizon: dict[int, bool],
    *,
    output_path: Path,
    title: str,
) -> Path:
    _require_matplotlib()
    ensure_directory(output_path.parent)

    ordered_horizons = sorted(risk_summaries)
    figure, axes = plt.subplots(2, 3, figsize=(13.0, 8.0))
    axes_flat = list(axes.flat)
    for axis in axes_flat[len(ordered_horizons):]:
        axis.axis("off")

    for axis, horizon_h in zip(axes_flat, ordered_horizons):
        summary = risk_summaries[horizon_h]
        split_name = reporting_splits[horizon_h]
        axis.set_title(f"{horizon_h}h ({split_name})")
        if summary.empty or not evaluable_by_horizon.get(horizon_h, False):
            axis.axis("off")
            axis.text(
                0.5,
                0.5,
                "No evaluable holdout curve\nfor this horizon",
                ha="center",
                va="center",
                fontsize=10,
            )
            continue

        axis_limit = _dynamic_probability_axis_limit(summary)
        axis.plot(
            [0.0, axis_limit],
            [0.0, axis_limit],
            linestyle="--",
            color="black",
            linewidth=1.0,
        )
        axis.plot(
            summary["predicted_probability_mean"],
            summary["observed_mortality"],
            marker="o",
            linewidth=2.0,
            color="#1f77b4",
        )
        axis.set_xlim(0.0, axis_limit)
        axis.set_ylim(0.0, axis_limit)
        axis.set_xlabel("Predicted risk")
        axis.set_ylabel("Observed mortality")
        axis.grid(alpha=0.25, linewidth=0.6)

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _plot_site_overview(
    site_summary: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
    subtitle: str,
) -> Path:
    if site_summary.empty:
        return _save_placeholder_figure(
            output_path,
            title=title,
            message=f"{subtitle}\nNo site-level data were available.",
        )

    _require_matplotlib()
    ensure_directory(output_path.parent)
    ordered = site_summary.sort_values("sample_count", ascending=False).reset_index(drop=True)
    x_positions = np.arange(ordered.shape[0])

    figure, axes = plt.subplots(2, 1, figsize=(10.0, 8.0), sharex=True)
    count_axis, auc_axis = axes
    count_axis.bar(x_positions, ordered["sample_count"], color="#c7d4e0", label="Samples")
    count_axis.bar(x_positions, ordered["event_count"], color="#c44e52", label="Events")
    count_axis.set_ylabel("Count")
    count_axis.set_title(title)
    count_axis.legend(loc="upper right")
    count_axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    count_axis.text(
        0.02,
        0.98,
        subtitle,
        ha="left",
        va="top",
        transform=count_axis.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    auc_axis.scatter(
        x_positions,
        ordered["auroc"],
        color="#1f77b4",
        s=60,
        label="AUROC",
    )
    auc_axis.scatter(
        x_positions,
        ordered["event_rate"],
        color="#c44e52",
        s=60,
        label="Event rate",
    )
    auc_axis.set_ylabel("Metric value")
    auc_axis.set_xlabel("Hospital")
    auc_axis.set_ylim(0.0, 1.0)
    auc_axis.set_xticks(x_positions)
    auc_axis.set_xticklabels(ordered["hospital_id"].tolist(), rotation=30, ha="right")
    auc_axis.grid(alpha=0.25, linewidth=0.6)
    auc_axis.legend(loc="upper right")

    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _plot_site_risk_structure(
    site_summaries: dict[str, pd.DataFrame],
    site_metric_lookup: dict[str, dict[str, object]],
    *,
    output_path: Path,
    title: str,
) -> Path:
    if not site_metric_lookup:
        return _save_placeholder_figure(
            output_path,
            title=title,
            message="No site-level data were available.",
        )

    _require_matplotlib()
    ensure_directory(output_path.parent)
    ordered_sites = list(site_metric_lookup)
    n_sites = len(ordered_sites)
    n_columns = 2
    n_rows = int(np.ceil(n_sites / n_columns))
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(12.0, max(4.0, 4.0 * n_rows)))
    axes_flat = np.atleast_1d(axes).reshape(-1)

    for axis in axes_flat[n_sites:]:
        axis.axis("off")

    for axis, hospital_id in zip(axes_flat, ordered_sites):
        summary = site_summaries.get(hospital_id, _empty_risk_bin_summary())
        metrics = site_metric_lookup[hospital_id]
        axis.set_title(str(hospital_id))
        if summary.empty or not bool(metrics["binary_metrics_evaluable"]):
            axis.axis("off")
            axis.text(
                0.5,
                0.55,
                "No evaluable site-level\nrisk-structure curve",
                ha="center",
                va="center",
                fontsize=10,
            )
            axis.text(
                0.5,
                0.35,
                f"notes: {metrics['metric_notes']}",
                ha="center",
                va="center",
                fontsize=8,
                wrap=True,
            )
            continue

        axis_limit = _dynamic_probability_axis_limit(summary)
        axis.plot(
            [0.0, axis_limit],
            [0.0, axis_limit],
            linestyle="--",
            color="black",
            linewidth=1.0,
        )
        axis.plot(
            summary["predicted_probability_mean"],
            summary["observed_mortality"],
            marker="o",
            linewidth=2.0,
            color="#1f77b4",
        )
        axis.set_xlim(0.0, axis_limit)
        axis.set_ylim(0.0, axis_limit)
        axis.set_xlabel("Predicted risk")
        axis.set_ylabel("Observed mortality")
        axis.grid(alpha=0.25, linewidth=0.6)

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _build_interpretation_note(
    combined_metrics: pd.DataFrame,
    reporting_split_summary: pd.DataFrame,
    site_summary: pd.DataFrame,
) -> str:
    model_names = sorted(reporting_split_summary["model_name"].astype("string").unique().tolist())
    horizon_names = sorted(reporting_split_summary["horizon_h"].astype(int).unique().tolist())
    selected_splits = (
        reporting_split_summary.groupby("selected_split").size().sort_values(ascending=False).to_dict()
        if not reporting_split_summary.empty
        else {}
    )
    primary_rows = reporting_split_summary[
        reporting_split_summary["horizon_h"].astype(int).eq(PRIMARY_HORIZON)
    ].copy()
    primary_rows = primary_rows.merge(
        combined_metrics[
            [
                "model_name",
                "horizon_h",
                "split",
                "auroc",
                "auprc",
                "calibration_slope",
                "calibration_intercept",
                "brier_score",
                "metric_notes",
            ]
        ],
        left_on=["model_name", "horizon_h", "selected_split"],
        right_on=["model_name", "horizon_h", "split"],
        how="left",
    )

    primary_lines = []
    for row in primary_rows.itertuples(index=False):
        auroc_text = "NA" if pd.isna(row.auroc) else f"{row.auroc:.3f}"
        auprc_text = "NA" if pd.isna(row.auprc) else f"{row.auprc:.3f}"
        slope_text = "NA" if pd.isna(row.calibration_slope) else f"{row.calibration_slope:.3f}"
        primary_lines.append(
            f"- {_display_model_name(row.model_name)} 24h used the `{row.selected_split}` split "
            f"(AUROC {auroc_text}, AUPRC {auprc_text}, calibration slope {slope_text})."
        )

    split_summary_text = ", ".join(f"{split}={count}" for split, count in selected_splits.items()) or "none"
    all_test_selected = bool(
        not reporting_split_summary.empty
        and reporting_split_summary["selected_split"].astype("string").eq("test").all()
    )
    calibration_interpretable = bool(
        not primary_rows.empty
        and primary_rows["selected_split_evaluable"].astype(bool).all()
        and primary_rows["selected_split"].astype("string").ne("train").all()
    )

    if all_test_selected:
        holdout_text = (
            "The preferred reporting split remained the frozen test split for all evaluated horizons."
        )
    else:
        holdout_text = (
            "At least one horizon had to fall back from the frozen test split to another existing split "
            "because the local sample test partition was not binary-evaluable."
        )

    if calibration_interpretable:
        calibration_text = (
            "Calibration curves are descriptive enough to inspect risk ordering on the currently selected "
            "holdout split, but they should still be treated as preliminary until the full ASIC test set is run."
        )
    else:
        calibration_text = (
            "Calibration is not stable enough on the current sample to support strong hard-case interpretation; "
            "the outputs are best treated as smoke-test evidence that the evaluation stack works."
        )

    if site_summary.empty:
        site_text = "No site-stratified summary was available."
    else:
        site_sample_share = float(site_summary["sample_count"].max() / site_summary["sample_count"].sum())
        if site_sample_share >= 0.60:
            site_text = (
                "The site-stratified check suggests one hospital contributes a large share of the currently "
                "selected primary-horizon evaluation rows, so site-level pattern stability is limited."
            )
        else:
            site_text = (
                "No single hospital obviously dominates the primary-horizon evaluation rows in the current "
                "sample, although several site-level metrics remain sparse."
            )

    note_lines = [
        "# Chapter 1 ASIC Baseline Evaluation: First-Pass Interpretation",
        "",
        f"Models evaluated: {', '.join(_display_model_name(model) for model in model_names)}.",
        f"Horizons evaluated: {', '.join(str(horizon) + 'h' for horizon in horizon_names)}.",
        f"Reporting split usage across model-horizon pairs: {split_summary_text}.",
        "",
        holdout_text,
        calibration_text,
        site_text,
        "",
        "Primary 24h summary:",
        *primary_lines,
        "",
        "Caveats:",
        "- This run is still sample-limited, especially on the frozen test split where local smoke-test artifacts can contain no events.",
        "- Discrimination, calibration, and site-level plots should therefore be interpreted as pipeline-validation outputs first, and scientific evidence second.",
        "- Later hard-case analysis should wait for the full ASIC evaluation run on a binary-evaluable holdout split.",
    ]
    return "\n".join(note_lines)


def run_asic_baseline_evaluation(
    *,
    input_root: Path = DEFAULT_BASELINE_ARTIFACT_ROOT,
    output_dir: Path = DEFAULT_EVALUATION_OUTPUT_DIR,
    models: Sequence[str] | None = None,
    horizons: Sequence[int] | None = None,
    primary_horizon: int = PRIMARY_HORIZON,
    risk_bin_count: int = DEFAULT_RISK_BIN_COUNT,
    site_risk_bin_count: int = DEFAULT_SITE_RISK_BIN_COUNT,
) -> EvaluationRunResult:
    _require_matplotlib()
    input_root = Path(input_root)
    output_dir = Path(output_dir)
    selected_models = _normalize_models(input_root, models)
    prediction_artifacts = _discover_prediction_artifacts(input_root, selected_models, horizons)

    combined_metrics_frames: list[pd.DataFrame] = []
    combined_risk_summaries: list[pd.DataFrame] = []
    reporting_split_rows: list[dict[str, object]] = []
    baseline_artifact_manifest: list[dict[str, object]] = []
    site_summary_frames: list[pd.DataFrame] = []
    site_risk_summaries: list[pd.DataFrame] = []

    per_model_reporting_metrics: dict[str, list[pd.DataFrame]] = {model_name: [] for model_name in selected_models}
    per_model_risk_summaries: dict[str, dict[int, pd.DataFrame]] = {model_name: {} for model_name in selected_models}
    per_model_selected_splits: dict[str, dict[int, str]] = {model_name: {} for model_name in selected_models}
    per_model_evaluable: dict[str, dict[int, bool]] = {model_name: {} for model_name in selected_models}

    for artifact in prediction_artifacts:
        predictions = _load_prediction_frame(artifact)
        baseline_artifact_manifest.append(
            {
                "model_name": artifact.model_name,
                "horizon_h": artifact.horizon_h,
                "predictions_path": str(artifact.predictions_path.resolve()),
            }
        )

        metrics_by_split = _metrics_by_split(
            predictions,
            model_name=artifact.model_name,
            horizon_h=artifact.horizon_h,
        )
        combined_metrics_frames.append(metrics_by_split)

        selection = _select_reporting_split(metrics_by_split)
        reporting_split_rows.append(
            {
                "model_name": artifact.model_name,
                "horizon_h": artifact.horizon_h,
                **selection,
            }
        )
        selected_split = str(selection["selected_split"])
        selected_evaluable = bool(selection["selected_split_evaluable"])
        selected_split_predictions = predictions[predictions["split"].eq(selected_split)].reset_index(drop=True)
        risk_summary = _ranked_quantile_risk_summary(
            selected_split_predictions,
            model_name=artifact.model_name,
            horizon_h=artifact.horizon_h,
            split_name=selected_split,
            sample_scope="overall",
            group_id="overall",
            n_bins=risk_bin_count,
        )
        combined_risk_summaries.append(risk_summary)
        per_model_risk_summaries[artifact.model_name][artifact.horizon_h] = risk_summary
        per_model_selected_splits[artifact.model_name][artifact.horizon_h] = selected_split
        per_model_evaluable[artifact.model_name][artifact.horizon_h] = selected_evaluable

        model_output_dir = output_dir / artifact.model_name / f"horizon_{artifact.horizon_h}h"
        ensure_directory(model_output_dir)
        metrics_path = write_dataframe(
            metrics_by_split,
            model_output_dir / "metrics_by_split.csv",
            output_format="csv",
        )
        risk_summary_path = write_dataframe(
            risk_summary,
            model_output_dir / "risk_binned_summary.csv",
            output_format="csv",
        )

        split_metrics_row = metrics_by_split.set_index("split").loc[selected_split]
        subtitle = (
            f"Reporting split: {selected_split}; n={int(split_metrics_row['sample_count'])}; "
            f"events={int(split_metrics_row['event_count'])}; "
            f"selection={selection['selection_reason']}"
        )
        reliability_plot_path = _plot_reliability(
            risk_summary if selected_evaluable else _empty_risk_bin_summary(),
            output_path=model_output_dir / "reliability_plot.png",
            title=f"{_display_model_name(artifact.model_name)} {artifact.horizon_h}h reliability",
            subtitle=subtitle,
        )
        mortality_vs_risk_plot_path = _plot_mortality_vs_risk(
            risk_summary if selected_evaluable else _empty_risk_bin_summary(),
            output_path=model_output_dir / "mortality_vs_risk_plot.png",
            title=f"{_display_model_name(artifact.model_name)} {artifact.horizon_h}h mortality vs risk",
            subtitle=subtitle,
        )

        evaluation_metadata_path = _write_json(
            {
                "timestamp_utc": _utc_timestamp(),
                "model_name": artifact.model_name,
                "horizon_h": artifact.horizon_h,
                "source_predictions_path": str(artifact.predictions_path.resolve()),
                "selected_reporting_split": selected_split,
                "selected_split_evaluable": selected_evaluable,
                "selection_reason": selection["selection_reason"],
                "metrics_output_path": str(metrics_path.resolve()),
                "risk_binned_summary_path": str(risk_summary_path.resolve()),
                "figure_paths": {
                    "reliability_plot": str(reliability_plot_path.resolve()),
                    "mortality_vs_risk_plot": str(mortality_vs_risk_plot_path.resolve()),
                },
                "risk_binning_strategy": {
                    "type": "ranked_quantile_bins",
                    "bin_count": risk_bin_count,
                    "description": (
                        "Predicted probabilities are ranked with method='first' and then divided into "
                        "up to 10 quantile bins per selected reporting split."
                    ),
                },
            },
            model_output_dir / "evaluation_metadata.json",
        )
        per_model_reporting_metrics[artifact.model_name].append(
            metrics_by_split[metrics_by_split["split"].eq(selected_split)].assign(
                selected_split=selected_split,
                selected_split_evaluable=selected_evaluable,
                selection_reason=selection["selection_reason"],
                evaluation_metadata_path=str(evaluation_metadata_path.resolve()),
            )
        )

        if artifact.horizon_h == primary_horizon:
            site_rows: list[dict[str, object]] = []
            site_summary_lookup: dict[str, pd.DataFrame] = {}
            selected_site_predictions = selected_split_predictions.copy()
            for hospital_id, site_df in selected_site_predictions.groupby("hospital_id", sort=True):
                site_metrics = compute_evaluation_metrics(
                    site_df["label_value"].astype(int).to_numpy(),
                    site_df["predicted_probability"].to_numpy(),
                )
                site_rows.append(
                    {
                        "model_name": artifact.model_name,
                        "horizon_h": artifact.horizon_h,
                        "split": selected_split,
                        "hospital_id": str(hospital_id),
                        **site_metrics,
                    }
                )
                site_summary = _ranked_quantile_risk_summary(
                    site_df.reset_index(drop=True),
                    model_name=artifact.model_name,
                    horizon_h=artifact.horizon_h,
                    split_name=selected_split,
                    sample_scope="site",
                    group_id=str(hospital_id),
                    n_bins=site_risk_bin_count,
                )
                site_summary_lookup[str(hospital_id)] = site_summary
                site_risk_summaries.append(site_summary)

            site_summary_df = pd.DataFrame(site_rows).sort_values("hospital_id").reset_index(drop=True)
            site_summary_frames.append(site_summary_df)
            site_output_dir = output_dir / artifact.model_name
            site_summary_path = write_dataframe(
                site_summary_df,
                site_output_dir / f"primary_{primary_horizon}h_site_summary.csv",
                output_format="csv",
            )
            site_risk_binned_summary_path = write_dataframe(
                pd.concat(site_summary_lookup.values(), ignore_index=True)
                if site_summary_lookup
                else _empty_risk_bin_summary(),
                site_output_dir / f"primary_{primary_horizon}h_site_risk_binned_summary.csv",
                output_format="csv",
            )
            _plot_site_overview(
                site_summary_df,
                output_path=site_output_dir / f"primary_{primary_horizon}h_site_overview.png",
                title=f"{_display_model_name(artifact.model_name)} {primary_horizon}h site overview",
                subtitle=(
                    f"Reporting split: {selected_split}; site-level metrics may still be degenerate "
                    f"on the local sample."
                ),
            )
            site_metric_lookup = {
                str(row.hospital_id): {
                    "binary_metrics_evaluable": row.binary_metrics_evaluable,
                    "metric_notes": row.metric_notes,
                }
                for row in site_summary_df.itertuples(index=False)
            }
            _plot_site_risk_structure(
                site_summary_lookup,
                site_metric_lookup,
                output_path=site_output_dir / f"primary_{primary_horizon}h_site_risk_structure.png",
                title=f"{_display_model_name(artifact.model_name)} {primary_horizon}h site risk structure",
            )
            _write_json(
                {
                    "timestamp_utc": _utc_timestamp(),
                    "model_name": artifact.model_name,
                    "primary_horizon_h": primary_horizon,
                    "selected_reporting_split": selected_split,
                    "source_predictions_path": str(artifact.predictions_path.resolve()),
                    "site_summary_path": str(site_summary_path.resolve()),
                    "site_risk_binned_summary_path": str(site_risk_binned_summary_path.resolve()),
                    "site_risk_binning_strategy": {
                        "type": "ranked_quantile_bins",
                        "bin_count": site_risk_bin_count,
                    },
                },
                site_output_dir / f"primary_{primary_horizon}h_site_metadata.json",
            )

    combined_metrics = pd.concat(combined_metrics_frames, ignore_index=True).sort_values(
        ["model_name", "horizon_h", "split"]
    ).reset_index(drop=True)
    reporting_split_summary = pd.DataFrame(reporting_split_rows).sort_values(
        ["model_name", "horizon_h"]
    ).reset_index(drop=True)
    combined_risk_binned_summary = (
        pd.concat(combined_risk_summaries, ignore_index=True)
        if combined_risk_summaries
        else _empty_risk_bin_summary()
    )
    combined_site_summary = (
        pd.concat(site_summary_frames, ignore_index=True).sort_values(["model_name", "hospital_id"])
        if site_summary_frames
        else pd.DataFrame()
    )
    combined_site_risk_binned_summary = (
        pd.concat(site_risk_summaries, ignore_index=True)
        if site_risk_summaries
        else _empty_risk_bin_summary()
    )

    ensure_directory(output_dir)
    combined_metrics_path = write_dataframe(
        combined_metrics,
        output_dir / "combined_metrics.csv",
        output_format="csv",
    )
    reporting_split_summary_path = write_dataframe(
        reporting_split_summary,
        output_dir / "reporting_split_summary.csv",
        output_format="csv",
    )
    combined_risk_binned_summary_path = write_dataframe(
        combined_risk_binned_summary,
        output_dir / "combined_risk_binned_summary.csv",
        output_format="csv",
    )
    write_dataframe(
        combined_site_summary,
        output_dir / "combined_primary_site_summary.csv",
        output_format="csv",
    )
    write_dataframe(
        combined_site_risk_binned_summary,
        output_dir / "combined_primary_site_risk_binned_summary.csv",
        output_format="csv",
    )

    for model_name in selected_models:
        model_output_dir = output_dir / model_name
        ensure_directory(model_output_dir)
        model_reporting_metrics = (
            pd.concat(per_model_reporting_metrics[model_name], ignore_index=True)
            .sort_values("horizon_h")
            .reset_index(drop=True)
        )
        horizon_comparison_metrics_path = write_dataframe(
            model_reporting_metrics,
            model_output_dir / "horizon_comparison_metrics.csv",
            output_format="csv",
        )
        _plot_horizon_comparison(
            model_reporting_metrics,
            output_path=model_output_dir / "horizon_comparison_plot.png",
            title=(
                f"{_display_model_name(model_name)} horizon comparison "
                "(selected reporting split per horizon)"
            ),
        )
        _plot_horizon_risk_structure_grid(
            per_model_risk_summaries[model_name],
            per_model_selected_splits[model_name],
            per_model_evaluable[model_name],
            output_path=model_output_dir / "horizon_risk_structure_grid.png",
            title=f"{_display_model_name(model_name)} mortality-vs-risk across horizons",
        )
        _write_json(
            {
                "timestamp_utc": _utc_timestamp(),
                "model_name": model_name,
                "horizon_comparison_metrics_path": str(horizon_comparison_metrics_path.resolve()),
                "horizon_risk_structure_grid_path": str(
                    (model_output_dir / "horizon_risk_structure_grid.png").resolve()
                ),
            },
            model_output_dir / "model_level_evaluation_metadata.json",
        )

    interpretation_note = _build_interpretation_note(
        combined_metrics,
        reporting_split_summary,
        combined_site_summary,
    )
    interpretation_note_path = write_text(
        interpretation_note,
        output_dir / "interpretation_note.md",
    )

    manifest_path = _write_json(
        {
            "timestamp_utc": _utc_timestamp(),
            "input_root": str(input_root.resolve()),
            "output_dir": str(output_dir.resolve()),
            "models": list(selected_models),
            "horizons": sorted(int(value) for value in reporting_split_summary["horizon_h"].unique().tolist()),
            "primary_horizon": int(primary_horizon),
            "risk_binning_strategy": {
                "type": "ranked_quantile_bins",
                "overall_bin_count": risk_bin_count,
                "site_bin_count": site_risk_bin_count,
                "reporting_split_order": list(DEFAULT_REPORTING_SPLIT_ORDER),
            },
            "consumed_prediction_artifacts": baseline_artifact_manifest,
            "output_files": {
                "combined_metrics": str(combined_metrics_path.resolve()),
                "combined_risk_binned_summary": str(combined_risk_binned_summary_path.resolve()),
                "reporting_split_summary": str(reporting_split_summary_path.resolve()),
                "interpretation_note": str(interpretation_note_path.resolve()),
            },
        },
        output_dir / "run_manifest.json",
    )

    return EvaluationRunResult(
        input_root=input_root,
        output_dir=output_dir,
        combined_metrics_path=combined_metrics_path,
        combined_risk_binned_summary_path=combined_risk_binned_summary_path,
        reporting_split_summary_path=reporting_split_summary_path,
        interpretation_note_path=interpretation_note_path,
        manifest_path=manifest_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved Chapter 1 ASIC baseline model predictions for discrimination, "
            "calibration, and mortality-vs-risk structure."
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
        default=DEFAULT_EVALUATION_OUTPUT_DIR,
        help="Root directory for evaluation metrics, figures, and notes.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional subset of model directories to evaluate.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        help="Optional subset of horizons to evaluate.",
    )
    parser.add_argument(
        "--primary-horizon",
        type=int,
        default=PRIMARY_HORIZON,
        help="Primary horizon for the site-stratified sanity check.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = run_asic_baseline_evaluation(
        input_root=args.input_root,
        output_dir=args.output_dir,
        models=args.models,
        horizons=args.horizons,
        primary_horizon=args.primary_horizon,
    )

    reporting_summary = pd.read_csv(result.reporting_split_summary_path)
    print(f"Evaluation output directory: {result.output_dir}")
    print(f"Combined metrics: {result.combined_metrics_path}")
    print(f"Combined risk-binned summary: {result.combined_risk_binned_summary_path}")
    print(f"Interpretation note: {result.interpretation_note_path}")
    print(f"Run manifest: {result.manifest_path}")
    print()
    print("Selected reporting split by model and horizon:")
    for row in reporting_summary.itertuples(index=False):
        print(
            f"- {row.model_name} {int(row.horizon_h)}h -> {row.selected_split} "
            f"(evaluable={bool(row.selected_split_evaluable)}; "
            f"n={int(row.sample_count)}; events={int(row.event_count)})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
