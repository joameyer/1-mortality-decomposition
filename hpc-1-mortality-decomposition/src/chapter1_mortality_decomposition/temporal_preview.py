from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

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
    run_asic_baseline_evaluation,
)
from chapter1_mortality_decomposition.baseline_logistic import (
    run_asic_primary_logistic_regression,
)
from chapter1_mortality_decomposition.baseline_xgboost import run_asic_primary_xgboost
from chapter1_mortality_decomposition.cohort import build_chapter1_cohort
from chapter1_mortality_decomposition.config import (
    build_chapter1_feature_set_definition,
    default_chapter1_config,
    updated_chapter1_config,
)
from chapter1_mortality_decomposition.instances import build_chapter1_valid_instances
from chapter1_mortality_decomposition.labels import build_chapter1_proxy_horizon_labels
from chapter1_mortality_decomposition.model_ready import build_chapter1_model_ready_dataset
from chapter1_mortality_decomposition.pipeline import (
    _build_chapter1_cohort_summary,
    _build_chapter1_verification_summary,
)
from chapter1_mortality_decomposition.run_config import load_chapter1_run_config
from chapter1_mortality_decomposition.temporal_blocks import (
    build_asic_temporal_block_artifacts,
    write_asic_temporal_block_artifacts,
)
from chapter1_mortality_decomposition.utils import (
    ensure_directory,
    read_dataframe,
    require_columns,
    write_dataframe,
    write_text,
)


DEFAULT_BLOCK_HOURS = 16
DEFAULT_HORIZONS = (8, 16, 24, 48, 72)
DEFAULT_FROZEN_CHAPTER1_DIR = Path("artifacts") / "chapter1"
DEFAULT_PREVIEW_OUTPUT_ROOT = (
    Path("artifacts") / "chapter1" / "temporal_preview" / "asic" / "aggregation_16h"
)
DEFAULT_EIGHT_HOUR_EVALUATION_ROOT = (
    Path("artifacts") / "chapter1" / "evaluation" / "asic" / "baselines" / "primary_medians"
)
DEFAULT_NOTEBOOK_PATH = Path("notebooks") / "ch1_asic_temporal_aggregation_preview_16h.ipynb"
PRIMARY_FEATURE_SET_NAME = "primary"
PRIMARY_HORIZON_HOURS = 24
COMPARISON_AGGREGATIONS = ("8h", "16h")
COMPARISON_MODELS = ("logistic_regression", "xgboost")


@dataclass(frozen=True)
class TemporalAggregationComparisonResult:
    comparison_table_path: Path
    note_path: Path
    notebook_path: Path
    figure_paths: tuple[Path, ...]


@dataclass(frozen=True)
class TemporalAggregationPreviewRunResult:
    output_root: Path
    block_hours: int
    block_paths: dict[str, Path]
    preprocessing_paths: dict[str, Path]
    baseline_root: Path
    evaluation_root: Path
    comparison: TemporalAggregationComparisonResult


def _require_matplotlib() -> None:
    if plt is None:
        raise ImportError(
            "matplotlib is required for the ASIC temporal aggregation preview."
        ) from MATPLOTLIB_IMPORT_ERROR


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


def _display_model_name(model_name: str) -> str:
    return {
        "logistic_regression": "Logistic Regression",
        "xgboost": "XGBoost",
    }.get(model_name, model_name.replace("_", " ").title())


def _metric_text(value: object, *, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _load_standardized_asic_inputs(
    *,
    input_dir: Path,
    input_format: str,
) -> dict[str, pd.DataFrame]:
    extension = "csv" if input_format == "csv" else "parquet"
    required_paths = {
        "static_harmonized": input_dir / "static" / f"harmonized.{extension}",
        "dynamic_harmonized": input_dir / "dynamic" / f"harmonized.{extension}",
        "reference_stay_block_counts": input_dir / "blocked" / f"asic_8h_stay_block_counts.{extension}",
        "mech_vent_stay_level_qc": input_dir / "qc" / f"mech_vent_ge_24h_stay_level.{extension}",
        "mech_vent_episode_level": input_dir / "qc" / f"mech_vent_ge_24h_episode_level.{extension}",
    }

    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing standardized ASIC input artifacts required for the temporal preview: "
            + ", ".join(missing)
        )

    return {
        name: read_dataframe(path)
        for name, path in required_paths.items()
    }


def _normalize_frozen_split_assignments(
    frozen_split_assignments: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        frozen_split_assignments,
        {"stay_id_global", "hospital_id", "split"},
        "frozen_split_assignments",
    )
    normalized = frozen_split_assignments.copy()
    normalized["stay_id_global"] = normalized["stay_id_global"].astype("string")
    normalized["hospital_id"] = normalized["hospital_id"].astype("string")
    normalized["split"] = normalized["split"].astype("string")
    return normalized


def _build_frozen_split_alignment_summary(
    retained_cohort: pd.DataFrame,
    frozen_split_assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    retained = retained_cohort[["stay_id_global", "hospital_id"]].copy()
    retained["stay_id_global"] = retained["stay_id_global"].astype("string")
    retained["hospital_id"] = retained["hospital_id"].astype("string")

    frozen = _normalize_frozen_split_assignments(frozen_split_assignments)
    retained_pairs = set(map(tuple, retained.itertuples(index=False, name=None)))
    frozen_pairs = set(
        map(tuple, frozen[["stay_id_global", "hospital_id"]].itertuples(index=False, name=None))
    )

    duplicate_assignment_count = int(
        frozen[["stay_id_global", "hospital_id"]].duplicated(keep=False).sum()
    )
    missing_pairs = sorted(retained_pairs - frozen_pairs)
    extra_pairs = sorted(frozen_pairs - retained_pairs)

    summary = pd.DataFrame(
        [
            {
                "check_id": "frozen_split_assignments_have_no_duplicate_stays",
                "passed": duplicate_assignment_count == 0,
                "detail": (
                    "The frozen split table should contain exactly one row per stay/hospital pair."
                ),
                "count": duplicate_assignment_count,
            },
            {
                "check_id": "all_preview_cohort_stays_have_frozen_split_assignment",
                "passed": len(missing_pairs) == 0,
                "detail": (
                    "Every retained 16h preview cohort stay should appear in the frozen split table."
                ),
                "count": len(missing_pairs),
            },
            {
                "check_id": "frozen_split_assignments_do_not_add_extra_preview_stays",
                "passed": len(extra_pairs) == 0,
                "detail": (
                    "The preview should reuse the frozen stay split table without introducing a "
                    "different retained stay set."
                ),
                "count": len(extra_pairs),
            },
        ]
    )

    if duplicate_assignment_count > 0 or missing_pairs or extra_pairs:
        details: list[str] = []
        if duplicate_assignment_count > 0:
            details.append(f"duplicate assignment rows={duplicate_assignment_count}")
        if missing_pairs:
            details.append(f"missing preview stays={missing_pairs[:5]}")
        if extra_pairs:
            details.append(f"extra frozen stays={extra_pairs[:5]}")
        raise ValueError(
            "Frozen 8h stay split assignments could not be reused exactly for the 16h preview: "
            + "; ".join(details)
        )

    aligned = frozen.merge(retained, on=["stay_id_global", "hospital_id"], how="inner")
    return summary, aligned.sort_values(
        ["hospital_id", "split", "stay_id_global"],
        kind="stable",
    ).reset_index(drop=True)


def _preview_generation_note(
    *,
    block_hours: int,
    input_dir: Path,
    frozen_chapter1_dir: Path,
    frozen_split_assignments_path: Path,
) -> str:
    return "\n".join(
        [
            "# ASIC Temporal Aggregation Preview: 16h Generation Note",
            "",
            f"- Alternative aggregation: `{block_hours}h` completed blocks only.",
            (
                "- Blocked 16h artifacts were derived locally from the standardized ASIC "
                "harmonized dynamic table using the same generic upstream blocking contract as the "
                "existing 8h artifacts: block membership uses `time_h // block_hours`, "
                "`prediction_time_h == block_end_h`, and only structurally completed blocks are kept."
            ),
            (
                "- Cohort logic, proxy within-horizon mortality labels, feature-set boundary, "
                "bounded LOCF preprocessing, and baseline model definitions were left unchanged."
            ),
            (
                f"- Frozen stay-level split assignments were reused from "
                f"`{frozen_split_assignments_path.resolve()}` rather than being regenerated."
            ),
            (
                "- The preview keeps only the primary feature set for modeling, with dynamic "
                "features restricted to median summaries through the existing baseline-selection rule."
            ),
            (
                f"- Standardized ASIC source directory: `{input_dir.resolve()}`. "
                f"Frozen 8h Chapter 1 artifact root: `{frozen_chapter1_dir.resolve()}`."
            ),
            (
                "- This is a narrow preview only. It is not a formal temporal-sensitivity analysis "
                "and should not be used by itself to refreeze Chapter 1."
            ),
        ]
    )


def _dynamic_probability_axis_limit(summary_frames: Sequence[pd.DataFrame]) -> float:
    observed_max = 0.0
    predicted_max = 0.0
    for summary in summary_frames:
        if summary.empty:
            continue
        observed_max = max(observed_max, float(summary["observed_mortality"].max()))
        predicted_max = max(predicted_max, float(summary["predicted_probability_max"].max()))
    upper = max(0.10, observed_max, predicted_max) * 1.10
    return min(max(upper, 0.05), 1.0)


def _save_placeholder_figure(
    output_path: Path,
    *,
    title: str,
    message: str,
) -> Path:
    _require_matplotlib()
    ensure_directory(output_path.parent)
    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    axis.axis("off")
    axis.text(0.5, 0.60, title, ha="center", va="center", fontsize=13, fontweight="bold")
    axis.text(0.5, 0.38, message, ha="center", va="center", fontsize=11, wrap=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _selected_split_metric_rows(
    evaluation_root: Path,
    *,
    aggregation_label: str,
) -> pd.DataFrame:
    reporting_summary = read_dataframe(evaluation_root / "reporting_split_summary.csv")
    combined_metrics = read_dataframe(evaluation_root / "combined_metrics.csv")
    metric_columns = [
        "model_name",
        "horizon_h",
        "split",
        "sample_count",
        "event_count",
        "non_event_count",
        "event_rate",
        "auroc",
        "auprc",
        "calibration_intercept",
        "calibration_slope",
        "brier_score",
        "binary_metrics_evaluable",
        "finite_prediction_count",
        "metric_notes",
    ]
    merged = reporting_summary.merge(
        combined_metrics[metric_columns],
        left_on=["model_name", "horizon_h", "selected_split"],
        right_on=["model_name", "horizon_h", "split"],
        how="left",
        suffixes=("_selected", ""),
    )
    merged["aggregation"] = aggregation_label
    return merged[
        [
            "model_name",
            "horizon_h",
            "aggregation",
            "selected_split",
            "selected_split_evaluable",
            "selection_reason",
            "sample_count",
            "event_count",
            "non_event_count",
            "event_rate",
            "auroc",
            "auprc",
            "calibration_intercept",
            "calibration_slope",
            "brier_score",
            "binary_metrics_evaluable",
            "finite_prediction_count",
            "metric_notes",
        ]
    ].copy()


def _selected_risk_summary(
    evaluation_root: Path,
    *,
    model_name: str,
    horizon_h: int,
) -> pd.DataFrame:
    path = evaluation_root / model_name / f"horizon_{int(horizon_h)}h" / "risk_binned_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return read_dataframe(path)


def _plot_reliability_comparison(
    summaries_by_aggregation: dict[str, pd.DataFrame],
    metrics_by_aggregation: dict[str, pd.Series],
    *,
    model_name: str,
    horizon_h: int,
    output_path: Path,
) -> Path:
    _require_matplotlib()
    ordered_summaries = [summaries_by_aggregation.get(label, pd.DataFrame()) for label in COMPARISON_AGGREGATIONS]
    if all(summary.empty for summary in ordered_summaries):
        return _save_placeholder_figure(
            output_path,
            title=f"{_display_model_name(model_name)} {horizon_h}h reliability comparison",
            message="No finite risk-bin summaries were available for either aggregation.",
        )

    ensure_directory(output_path.parent)
    axis_limit = _dynamic_probability_axis_limit(ordered_summaries)
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=True)
    for axis, aggregation_label in zip(axes, COMPARISON_AGGREGATIONS):
        summary = summaries_by_aggregation.get(aggregation_label, pd.DataFrame())
        metrics = metrics_by_aggregation.get(aggregation_label)
        axis.plot([0.0, axis_limit], [0.0, axis_limit], linestyle="--", color="black", linewidth=1.0)
        axis.set_title(f"{aggregation_label} aggregation")
        axis.set_xlim(0.0, axis_limit)
        axis.set_ylim(0.0, axis_limit)
        axis.set_xlabel("Mean predicted risk")
        axis.grid(alpha=0.25, linewidth=0.6)
        if axis is axes[0]:
            axis.set_ylabel("Observed mortality")

        if summary.empty:
            axis.text(0.5, 0.5, "No summary available", ha="center", va="center", transform=axis.transAxes)
            continue

        axis.plot(
            summary["predicted_probability_mean"],
            summary["observed_mortality"],
            marker="o",
            linewidth=2.0,
            color="#1f77b4",
        )
        if metrics is not None:
            subtitle = (
                f"split={metrics['selected_split']}, n={int(metrics['sample_count'])}, "
                f"events={int(metrics['event_count'])}, AUROC={_metric_text(metrics['auroc'])}, "
                f"slope={_metric_text(metrics['calibration_slope'])}"
            )
            axis.text(
                0.02,
                0.98,
                subtitle,
                ha="left",
                va="top",
                transform=axis.transAxes,
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "none"},
            )

    figure.suptitle(f"{_display_model_name(model_name)} {horizon_h}h reliability: 8h vs 16h")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _plot_mortality_vs_risk_comparison(
    summaries_by_aggregation: dict[str, pd.DataFrame],
    metrics_by_aggregation: dict[str, pd.Series],
    *,
    model_name: str,
    horizon_h: int,
    output_path: Path,
) -> Path:
    _require_matplotlib()
    ordered_summaries = [summaries_by_aggregation.get(label, pd.DataFrame()) for label in COMPARISON_AGGREGATIONS]
    if all(summary.empty for summary in ordered_summaries):
        return _save_placeholder_figure(
            output_path,
            title=f"{_display_model_name(model_name)} {horizon_h}h mortality-vs-risk comparison",
            message="No finite risk-bin summaries were available for either aggregation.",
        )

    ensure_directory(output_path.parent)
    axis_limit = _dynamic_probability_axis_limit(ordered_summaries)
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.5), sharey=False)
    for axis, aggregation_label in zip(axes, COMPARISON_AGGREGATIONS):
        summary = summaries_by_aggregation.get(aggregation_label, pd.DataFrame())
        metrics = metrics_by_aggregation.get(aggregation_label)
        axis.set_title(f"{aggregation_label} aggregation")
        axis.set_xlabel("Risk quantile bin")
        axis.set_ylabel("Sample count")
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)

        if summary.empty:
            axis.text(0.5, 0.5, "No summary available", ha="center", va="center", transform=axis.transAxes)
            continue

        x_positions = np.arange(summary.shape[0])
        axis.bar(
            x_positions,
            summary["sample_count"],
            color="#d9d9d9",
            edgecolor="#666666",
            label="Samples",
        )
        twin_axis = axis.twinx()
        twin_axis.plot(
            x_positions,
            summary["predicted_probability_mean"],
            marker="o",
            linewidth=2.0,
            color="#1f77b4",
            label="Predicted risk",
        )
        twin_axis.plot(
            x_positions,
            summary["observed_mortality"],
            marker="s",
            linewidth=2.0,
            color="#c44e52",
            label="Observed mortality",
        )
        twin_axis.set_ylim(0.0, axis_limit)
        twin_axis.set_ylabel("Risk / mortality")
        axis.set_xticks(x_positions)
        axis.set_xticklabels(summary["bin_label"].tolist())

        if metrics is not None:
            subtitle = (
                f"split={metrics['selected_split']}, n={int(metrics['sample_count'])}, "
                f"events={int(metrics['event_count'])}, AUPRC={_metric_text(metrics['auprc'])}"
            )
            axis.text(
                0.02,
                0.98,
                subtitle,
                ha="left",
                va="top",
                transform=axis.transAxes,
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "none"},
            )
        lines_1, labels_1 = axis.get_legend_handles_labels()
        lines_2, labels_2 = twin_axis.get_legend_handles_labels()
        twin_axis.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    figure.suptitle(f"{_display_model_name(model_name)} {horizon_h}h mortality vs risk: 8h vs 16h")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _risk_structure_signature(summary: pd.DataFrame) -> dict[str, float] | None:
    if summary.empty:
        return None
    total_samples = int(summary["sample_count"].sum())
    total_events = int(summary["event_count"].sum())
    if total_samples == 0:
        return None

    mid_point = int(np.ceil(summary["bin_index"].max() / 2.0))
    lower = summary[summary["bin_index"].le(mid_point)].copy()
    upper = summary[summary["bin_index"].gt(mid_point)].copy()

    lower_sample_count = int(lower["sample_count"].sum())
    upper_sample_count = int(upper["sample_count"].sum())
    lower_event_count = int(lower["event_count"].sum())
    upper_event_count = int(upper["event_count"].sum())
    return {
        "lower_half_event_rate": (lower_event_count / lower_sample_count) if lower_sample_count else np.nan,
        "upper_half_event_rate": (upper_event_count / upper_sample_count) if upper_sample_count else np.nan,
        "upper_half_event_share": (upper_event_count / total_events) if total_events else np.nan,
        "top_bin_observed_mortality": float(summary["observed_mortality"].iloc[-1]),
        "bottom_bin_observed_mortality": float(summary["observed_mortality"].iloc[0]),
    }


def _build_preview_note(
    comparison_table: pd.DataFrame,
    *,
    eight_hour_evaluation_root: Path,
    sixteen_hour_evaluation_root: Path,
) -> str:
    pivot = comparison_table.pivot_table(
        index=["model_name", "horizon_h"],
        columns="aggregation",
        values=[
            "selected_split",
            "selected_split_evaluable",
            "sample_count",
            "event_count",
            "auroc",
            "auprc",
            "calibration_intercept",
            "calibration_slope",
            "brier_score",
            "binary_metrics_evaluable",
        ],
        aggfunc="first",
    )

    comparable_pairs: list[str] = []
    non_comparable_pairs: list[str] = []
    max_auroc_delta = 0.0
    max_auprc_delta = 0.0
    max_brier_delta = 0.0
    max_slope_delta = 0.0
    evaluable_drop_pairs: list[str] = []
    primary_lines: list[str] = []
    for model_name in COMPARISON_MODELS:
        primary_key = (model_name, PRIMARY_HORIZON_HOURS)
        if primary_key in pivot.index:
            row = pivot.loc[primary_key]
            primary_lines.append(
                (
                    f"- {_display_model_name(model_name)} {PRIMARY_HORIZON_HOURS}h: "
                    f"AUROC {_metric_text(row.get(('auroc', '8h')))} -> {_metric_text(row.get(('auroc', '16h')))}, "
                    f"AUPRC {_metric_text(row.get(('auprc', '8h')))} -> {_metric_text(row.get(('auprc', '16h')))}, "
                    f"slope {_metric_text(row.get(('calibration_slope', '8h')))} -> "
                    f"{_metric_text(row.get(('calibration_slope', '16h')))}."
                )
            )

    for (model_name, horizon_h), row in pivot.iterrows():
        auroc_8h = row.get(("auroc", "8h"))
        auroc_16h = row.get(("auroc", "16h"))
        auprc_8h = row.get(("auprc", "8h"))
        auprc_16h = row.get(("auprc", "16h"))
        brier_8h = row.get(("brier_score", "8h"))
        brier_16h = row.get(("brier_score", "16h"))
        slope_8h = row.get(("calibration_slope", "8h"))
        slope_16h = row.get(("calibration_slope", "16h"))
        split_8h = row.get(("selected_split", "8h"))
        split_16h = row.get(("selected_split", "16h"))
        pair_label = f"{_display_model_name(model_name)} {int(horizon_h)}h"

        same_holdout_split = (
            isinstance(split_8h, str)
            and isinstance(split_16h, str)
            and split_8h == split_16h
            and split_8h != "train"
        )
        if same_holdout_split:
            comparable_pairs.append(pair_label)
        else:
            non_comparable_pairs.append(pair_label)

        if same_holdout_split and pd.notna(auroc_8h) and pd.notna(auroc_16h):
            max_auroc_delta = max(max_auroc_delta, abs(float(auroc_16h) - float(auroc_8h)))
        if same_holdout_split and pd.notna(auprc_8h) and pd.notna(auprc_16h):
            max_auprc_delta = max(max_auprc_delta, abs(float(auprc_16h) - float(auprc_8h)))
        if same_holdout_split and pd.notna(brier_8h) and pd.notna(brier_16h):
            max_brier_delta = max(max_brier_delta, abs(float(brier_16h) - float(brier_8h)))
        if same_holdout_split and pd.notna(slope_8h) and pd.notna(slope_16h):
            max_slope_delta = max(max_slope_delta, abs(float(slope_16h) - float(slope_8h)))

        if bool(row.get(("binary_metrics_evaluable", "8h"), False)) and not bool(
            row.get(("binary_metrics_evaluable", "16h"), False)
        ):
            evaluable_drop_pairs.append(pair_label)

    risk_structure_lines: list[str] = []
    all_structures_ordered = True
    for model_name in COMPARISON_MODELS:
        summary_8h = _selected_risk_summary(
            eight_hour_evaluation_root,
            model_name=model_name,
            horizon_h=PRIMARY_HORIZON_HOURS,
        )
        summary_16h = _selected_risk_summary(
            sixteen_hour_evaluation_root,
            model_name=model_name,
            horizon_h=PRIMARY_HORIZON_HOURS,
        )
        signature_8h = _risk_structure_signature(summary_8h)
        signature_16h = _risk_structure_signature(summary_16h)
        if signature_8h is None or signature_16h is None:
            risk_structure_lines.append(
                f"- {_display_model_name(model_name)} {PRIMARY_HORIZON_HOURS}h risk-structure curve was too sparse for a stronger qualitative statement."
            )
            all_structures_ordered = False
            continue

        ordered_8h = (
            signature_8h["upper_half_event_rate"] > signature_8h["lower_half_event_rate"]
            and signature_8h["top_bin_observed_mortality"] >= signature_8h["bottom_bin_observed_mortality"]
        )
        ordered_16h = (
            signature_16h["upper_half_event_rate"] > signature_16h["lower_half_event_rate"]
            and signature_16h["top_bin_observed_mortality"] >= signature_16h["bottom_bin_observed_mortality"]
        )
        all_structures_ordered = all_structures_ordered and ordered_8h and ordered_16h
        risk_structure_lines.append(
            (
                f"- {_display_model_name(model_name)} {PRIMARY_HORIZON_HOURS}h upper-half event share: "
                f"{_metric_text(signature_8h['upper_half_event_share'])} at 8h vs "
                f"{_metric_text(signature_16h['upper_half_event_share'])} at 16h."
            )
        )

    if comparable_pairs and max_auroc_delta < 0.10 and max_auprc_delta < 0.05 and max_slope_delta < 0.25:
        stability_text = (
            "Across the comparable non-train holdout pairs, the 16h preview looks broadly similar "
            "rather than obviously unstable. On its own, it does not suggest that the Chapter 1 "
            "signal is an artifact of the 8h aggregation choice."
        )
    else:
        stability_text = (
            "The preview shows enough movement to justify formal Sprint 4 follow-up, but it is "
            "still too narrow to support any Chapter 1 refreeze decision by itself."
        )

    calibration_text = (
        "Calibration did not show an obvious across-the-board collapse."
        if max_brier_delta < 0.05
        else "Calibration shifted enough to warrant caution when reading the 16h preview."
    )
    risk_structure_text = (
        "The 24h mortality-vs-risk curves remain qualitatively ordered in both models."
        if all_structures_ordered
        else "The 24h mortality-vs-risk comparison should be read cautiously because at least one curve is sparse or visibly reshaped."
    )

    caveat_lines = [
        "- Only one alternative aggregation was tested, and it changes the prediction-time grid to completed 16h blocks.",
        "- The preview reuses the frozen stay-level split assignments, but reporting may still fall back from test to validation when a split is not binary-evaluable.",
        "- This is not a formal temporal-sensitivity analysis and should not be used to choose an optimal aggregation.",
    ]
    if non_comparable_pairs:
        caveat_lines.append(
            "- Split selection was not directly comparable for: "
            + ", ".join(non_comparable_pairs)
            + "."
        )
    if evaluable_drop_pairs:
        caveat_lines.append(
            "- Binary-evaluable status dropped for: " + ", ".join(evaluable_drop_pairs) + "."
        )

    note_lines = [
        "# ASIC Temporal Aggregation Preview: 8h vs 16h",
        "",
        stability_text,
        calibration_text,
        risk_structure_text,
        "",
        f"Comparable holdout pairs used for the compact stability summary: {', '.join(comparable_pairs) or 'none'}.",
        f"Maximum absolute AUROC delta across comparable holdout pairs: {max_auroc_delta:.3f}.",
        f"Maximum absolute AUPRC delta across comparable holdout pairs: {max_auprc_delta:.3f}.",
        f"Maximum absolute Brier-score delta across comparable holdout pairs: {max_brier_delta:.3f}.",
        "",
        "Primary 24h comparison:",
        *primary_lines,
        "",
        "24h mortality-vs-risk structure:",
        *risk_structure_lines,
        "",
        "Caveats:",
        *caveat_lines,
    ]
    return "\n".join(note_lines)


def _notebook_payload(
    *,
    eight_hour_evaluation_root: Path,
    sixteen_hour_evaluation_root: Path,
    comparison_table_path: Path,
    note_path: Path,
    figure_paths: Sequence[Path],
) -> dict[str, object]:
    figure_path_strings = [str(path.resolve()) for path in figure_paths]
    eight_hour_evaluation_root = eight_hour_evaluation_root.resolve()
    sixteen_hour_evaluation_root = sixteen_hour_evaluation_root.resolve()
    comparison_table_path = comparison_table_path.resolve()
    note_path = note_path.resolve()
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# ASIC temporal aggregation preview: 8h vs 16h\n",
                "\n",
                "This notebook reads saved artifacts only. It does not retrain models.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                f"EIGHT_HOUR_EVAL_ROOT = Path({eight_hour_evaluation_root.as_posix()!r})\n",
                f"SIXTEEN_HOUR_EVAL_ROOT = Path({sixteen_hour_evaluation_root.as_posix()!r})\n",
                f"COMPARISON_TABLE_PATH = Path({comparison_table_path.as_posix()!r})\n",
                f"NOTE_PATH = Path({note_path.as_posix()!r})\n",
                f"FIGURE_PATHS = [Path(path) for path in {figure_path_strings!r}]\n",
                "\n",
                "for path in [EIGHT_HOUR_EVAL_ROOT, SIXTEEN_HOUR_EVAL_ROOT, COMPARISON_TABLE_PATH, NOTE_PATH]:\n",
                "    print(path, 'exists=', path.exists())\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def load_reporting_metrics(evaluation_root: Path, aggregation: str) -> pd.DataFrame:\n",
                "    reporting = pd.read_csv(evaluation_root / 'reporting_split_summary.csv')\n",
                "    combined = pd.read_csv(evaluation_root / 'combined_metrics.csv')\n",
                "    metric_columns = [\n",
                "        'model_name', 'horizon_h', 'split', 'sample_count', 'event_count', 'non_event_count',\n",
                "        'event_rate', 'auroc', 'auprc', 'calibration_intercept', 'calibration_slope',\n",
                "        'brier_score'\n",
                "    ]\n",
                "    merged = reporting.merge(\n",
                "        combined[metric_columns],\n",
                "        left_on=['model_name', 'horizon_h', 'selected_split'],\n",
                "        right_on=['model_name', 'horizon_h', 'split'],\n",
                "        how='left',\n",
                "        suffixes=('_selected', ''),\n",
                "    )\n",
                "    merged['aggregation'] = aggregation\n",
                "    return merged[['model_name', 'horizon_h', 'aggregation', 'selected_split', 'sample_count', 'event_count', 'auroc', 'auprc', 'calibration_intercept', 'calibration_slope', 'brier_score']]\n",
                "\n",
                "comparison = pd.concat(\n",
                "    [\n",
                "        load_reporting_metrics(EIGHT_HOUR_EVAL_ROOT, '8h'),\n",
                "        load_reporting_metrics(SIXTEEN_HOUR_EVAL_ROOT, '16h'),\n",
                "    ],\n",
                "    ignore_index=True,\n",
                ")\n",
                "comparison = comparison.sort_values(['model_name', 'horizon_h', 'aggregation']).reset_index(drop=True)\n",
                "comparison\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "saved_comparison = pd.read_csv(COMPARISON_TABLE_PATH)\n",
                "saved_comparison\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(NOTE_PATH.read_text())\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "for figure_path in FIGURE_PATHS:\n",
                "    print(f'\\n## {figure_path.name}')\n",
                "    image = plt.imread(figure_path)\n",
                "    plt.figure(figsize=(12, 6))\n",
                "    plt.imshow(image)\n",
                "    plt.axis('off')\n",
                "    plt.show()\n",
            ],
        },
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _write_preview_notebook(
    *,
    eight_hour_evaluation_root: Path,
    sixteen_hour_evaluation_root: Path,
    comparison_table_path: Path,
    note_path: Path,
    figure_paths: Sequence[Path],
    notebook_path: Path,
) -> Path:
    payload = _notebook_payload(
        eight_hour_evaluation_root=eight_hour_evaluation_root,
        sixteen_hour_evaluation_root=sixteen_hour_evaluation_root,
        comparison_table_path=comparison_table_path,
        note_path=note_path,
        figure_paths=figure_paths,
    )
    return write_text(json.dumps(payload, indent=2), notebook_path)


def _build_comparison_package(
    *,
    eight_hour_evaluation_root: Path,
    sixteen_hour_evaluation_root: Path,
    comparison_output_dir: Path,
    notebook_path: Path,
) -> TemporalAggregationComparisonResult:
    comparison_output_dir = Path(comparison_output_dir)
    ensure_directory(comparison_output_dir)
    eight_hour_evaluation_root = Path(eight_hour_evaluation_root).resolve()
    sixteen_hour_evaluation_root = Path(sixteen_hour_evaluation_root).resolve()
    notebook_path = Path(notebook_path).resolve()

    metrics_8h = _selected_split_metric_rows(
        eight_hour_evaluation_root,
        aggregation_label="8h",
    )
    metrics_16h = _selected_split_metric_rows(
        sixteen_hour_evaluation_root,
        aggregation_label="16h",
    )
    comparison_table = pd.concat([metrics_8h, metrics_16h], ignore_index=True).sort_values(
        ["model_name", "horizon_h", "aggregation"],
        kind="stable",
    ).reset_index(drop=True)
    comparison_table_path = write_dataframe(
        comparison_table,
        comparison_output_dir / "aggregation_comparison_metrics.csv",
        output_format="csv",
    )

    figure_paths: list[Path] = []
    for model_name in COMPARISON_MODELS:
        model_rows = comparison_table[
            comparison_table["model_name"].astype("string").eq(model_name)
            & comparison_table["horizon_h"].astype(int).eq(PRIMARY_HORIZON_HOURS)
        ].copy()
        metrics_by_aggregation = model_rows.set_index("aggregation").to_dict(orient="index")
        summaries_by_aggregation = {
            "8h": _selected_risk_summary(
                eight_hour_evaluation_root,
                model_name=model_name,
                horizon_h=PRIMARY_HORIZON_HOURS,
            ),
            "16h": _selected_risk_summary(
                sixteen_hour_evaluation_root,
                model_name=model_name,
                horizon_h=PRIMARY_HORIZON_HOURS,
            ),
        }
        reliability_path = _plot_reliability_comparison(
            summaries_by_aggregation,
            metrics_by_aggregation,
            model_name=model_name,
            horizon_h=PRIMARY_HORIZON_HOURS,
            output_path=comparison_output_dir / f"{model_name}_24h_reliability_8h_vs_16h.png",
        )
        mortality_vs_risk_path = _plot_mortality_vs_risk_comparison(
            summaries_by_aggregation,
            metrics_by_aggregation,
            model_name=model_name,
            horizon_h=PRIMARY_HORIZON_HOURS,
            output_path=comparison_output_dir / f"{model_name}_24h_mortality_vs_risk_8h_vs_16h.png",
        )
        figure_paths.extend([reliability_path, mortality_vs_risk_path])

    note_path = write_text(
        _build_preview_note(
            comparison_table,
            eight_hour_evaluation_root=eight_hour_evaluation_root,
            sixteen_hour_evaluation_root=sixteen_hour_evaluation_root,
        ),
        comparison_output_dir / "preview_note.md",
    )
    notebook_path = _write_preview_notebook(
        eight_hour_evaluation_root=eight_hour_evaluation_root,
        sixteen_hour_evaluation_root=sixteen_hour_evaluation_root,
        comparison_table_path=comparison_table_path,
        note_path=note_path,
        figure_paths=figure_paths,
        notebook_path=notebook_path,
    )
    return TemporalAggregationComparisonResult(
        comparison_table_path=comparison_table_path,
        note_path=note_path,
        notebook_path=notebook_path,
        figure_paths=tuple(figure_paths),
    )


def run_asic_temporal_aggregation_preview(
    *,
    run_config_path: Path | None = None,
    input_dir: Path | None = None,
    input_format: str | None = None,
    output_root: Path = DEFAULT_PREVIEW_OUTPUT_ROOT,
    output_format: str = "csv",
    frozen_chapter1_dir: Path = DEFAULT_FROZEN_CHAPTER1_DIR,
    eight_hour_evaluation_root: Path = DEFAULT_EIGHT_HOUR_EVALUATION_ROOT,
    notebook_path: Path = DEFAULT_NOTEBOOK_PATH,
    block_hours: int = DEFAULT_BLOCK_HOURS,
    horizons: Sequence[int] | None = None,
) -> TemporalAggregationPreviewRunResult:
    _require_matplotlib()
    run_config = load_chapter1_run_config(run_config_path) if run_config_path else load_chapter1_run_config()
    standardized_input_dir = Path(input_dir or run_config.input_dir)
    standardized_input_format = input_format or run_config.input_format
    selected_horizons = tuple(int(horizon) for horizon in (horizons or DEFAULT_HORIZONS))
    block_hours = int(block_hours)

    config = updated_chapter1_config(
        default_chapter1_config(),
        horizons_hours=selected_horizons,
        min_required_core_groups=run_config.min_required_core_groups,
        split_random_seed=run_config.split_random_seed,
        feature_set_config_path=run_config.feature_set_config_path,
    )

    standardized_inputs = _load_standardized_asic_inputs(
        input_dir=standardized_input_dir,
        input_format=standardized_input_format,
    )

    output_root = Path(output_root)
    preprocessing_root = output_root / "preprocessing"
    baseline_root = output_root / "baselines" / "asic" / "primary_medians"
    evaluation_root = output_root / "evaluation" / "asic" / "baselines" / "primary_medians"
    comparison_root = output_root / "comparison"
    ensure_directory(preprocessing_root)

    block_artifacts = build_asic_temporal_block_artifacts(
        dynamic_harmonized=standardized_inputs["dynamic_harmonized"],
        reference_stay_block_counts=standardized_inputs["reference_stay_block_counts"],
        block_hours=block_hours,
    )
    block_paths = write_asic_temporal_block_artifacts(
        block_artifacts,
        output_dir=preprocessing_root / "blocked",
        output_format=output_format,
    )

    cohort = build_chapter1_cohort(
        static_harmonized=standardized_inputs["static_harmonized"],
        dynamic_harmonized=standardized_inputs["dynamic_harmonized"],
        stay_block_counts=block_artifacts.stay_block_counts,
        mech_vent_stay_level_qc=standardized_inputs["mech_vent_stay_level_qc"],
        config=config,
    )
    valid_instances = build_chapter1_valid_instances(
        retained_cohort=cohort.table,
        block_index=block_artifacts.block_index,
        blocked_dynamic_features=block_artifacts.blocked_dynamic_features,
        config=config,
    )
    labels = build_chapter1_proxy_horizon_labels(
        valid_instances=valid_instances.valid_instances,
        retained_cohort=cohort.table,
    )
    cohort_summary = _build_chapter1_cohort_summary(cohort, valid_instances, labels)
    verification_summary = _build_chapter1_verification_summary(cohort, valid_instances, labels)

    frozen_split_assignments_path = (
        Path(frozen_chapter1_dir) / "splits" / "chapter1_stay_split_assignments.csv"
    )
    if not frozen_split_assignments_path.exists():
        raise FileNotFoundError(
            f"Missing frozen 8h stay split assignments at {frozen_split_assignments_path}"
        )
    split_alignment_summary, frozen_split_assignments_used = _build_frozen_split_alignment_summary(
        cohort.table,
        read_dataframe(frozen_split_assignments_path),
    )

    feature_set_definition, feature_set_validation_summary = build_chapter1_feature_set_definition(
        block_artifacts.blocked_dynamic_features,
        retained_stays=cohort.retained_stays,
        config=config,
    )
    primary_feature_set_definition = feature_set_definition[
        feature_set_definition["feature_set_name"].astype("string").eq(PRIMARY_FEATURE_SET_NAME)
    ].reset_index(drop=True)
    model_ready = build_chapter1_model_ready_dataset(
        usable_labels=labels.usable_labels,
        blocked_dynamic_features=block_artifacts.blocked_dynamic_features,
        feature_set_definition=primary_feature_set_definition,
        feature_set_name=PRIMARY_FEATURE_SET_NAME,
        mech_vent_episode_level=standardized_inputs["mech_vent_episode_level"],
        stay_split_assignments=frozen_split_assignments_used,
        config=config,
    )

    extension = "csv" if output_format == "csv" else "parquet"
    preprocessing_paths = {
        "cohort_summary": write_dataframe(
            cohort_summary,
            preprocessing_root / "cohort" / f"chapter1_cohort_summary.{extension}",
            output_format=output_format,
        ),
        "cohort_verification_summary": write_dataframe(
            verification_summary,
            preprocessing_root / "cohort" / f"chapter1_verification_summary.{extension}",
            output_format=output_format,
        ),
        "retained_stay_table": write_dataframe(
            cohort.table,
            preprocessing_root / "cohort" / f"chapter1_retained_stay_table.{extension}",
            output_format=output_format,
        ),
        "valid_instances": write_dataframe(
            valid_instances.valid_instances,
            preprocessing_root / "instances" / f"chapter1_valid_instances.{extension}",
            output_format=output_format,
        ),
        "instance_counts_by_horizon": write_dataframe(
            valid_instances.counts_by_horizon,
            preprocessing_root / "instances" / f"chapter1_instance_counts_by_horizon.{extension}",
            output_format=output_format,
        ),
        "proxy_label_summary_by_horizon": write_dataframe(
            labels.summary_by_horizon,
            preprocessing_root / "labels" / f"chapter1_proxy_label_summary_by_horizon.{extension}",
            output_format=output_format,
        ),
        "usable_proxy_horizon_labels": write_dataframe(
            labels.usable_labels,
            preprocessing_root / "labels" / f"chapter1_usable_proxy_horizon_labels.{extension}",
            output_format=output_format,
        ),
        "feature_set_definition": write_dataframe(
            feature_set_definition,
            preprocessing_root / "feature_sets" / f"chapter1_feature_set_definition.{extension}",
            output_format=output_format,
        ),
        "feature_set_validation_summary": write_dataframe(
            feature_set_validation_summary,
            preprocessing_root / "feature_sets" / f"chapter1_feature_set_validation_summary.{extension}",
            output_format=output_format,
        ),
        "stay_split_assignments": write_dataframe(
            frozen_split_assignments_used,
            preprocessing_root / "splits" / f"chapter1_stay_split_assignments.{extension}",
            output_format=output_format,
        ),
        "split_alignment_summary": write_dataframe(
            split_alignment_summary,
            preprocessing_root / "splits" / f"chapter1_temporal_preview_split_alignment_summary.{extension}",
            output_format=output_format,
        ),
        "primary_model_ready_dataset": write_dataframe(
            model_ready.table,
            preprocessing_root / "model_ready" / f"chapter1_primary_model_ready_dataset.{extension}",
            output_format=output_format,
        ),
        "primary_readiness_summary": write_dataframe(
            model_ready.readiness_summary,
            preprocessing_root / "model_ready" / f"chapter1_primary_readiness_summary.{extension}",
            output_format=output_format,
        ),
        "primary_feature_availability_by_horizon": write_dataframe(
            model_ready.feature_availability_by_horizon,
            preprocessing_root / "model_ready" / f"chapter1_primary_feature_availability_by_horizon.{extension}",
            output_format=output_format,
        ),
        "primary_split_summary": write_dataframe(
            model_ready.split_summary,
            preprocessing_root / "splits" / f"chapter1_primary_split_summary.{extension}",
            output_format=output_format,
        ),
        "primary_split_verification_summary": write_dataframe(
            model_ready.split_verification_summary,
            preprocessing_root / "splits" / f"chapter1_primary_split_verification_summary.{extension}",
            output_format=output_format,
        ),
        "primary_locf_feature_summary": write_dataframe(
            model_ready.locf_feature_summary,
            preprocessing_root / "carry_forward" / f"chapter1_primary_locf_feature_summary.{extension}",
            output_format=output_format,
        ),
        "primary_ventilator_locf_summary": write_dataframe(
            model_ready.ventilator_locf_summary,
            preprocessing_root / "carry_forward" / f"chapter1_primary_ventilator_locf_summary.{extension}",
            output_format=output_format,
        ),
        "primary_missingness_by_hospital_and_family": write_dataframe(
            model_ready.missingness_by_hospital_and_family,
            preprocessing_root / "carry_forward" / f"chapter1_primary_missingness_by_hospital_and_family.{extension}",
            output_format=output_format,
        ),
        "primary_carry_forward_verification_summary": write_dataframe(
            model_ready.carry_forward_verification_summary,
            preprocessing_root / "carry_forward" / f"chapter1_primary_carry_forward_verification_summary.{extension}",
            output_format=output_format,
        ),
        "preview_generation_note": write_text(
            _preview_generation_note(
                block_hours=block_hours,
                input_dir=standardized_input_dir,
                frozen_chapter1_dir=Path(frozen_chapter1_dir),
                frozen_split_assignments_path=frozen_split_assignments_path,
            ),
            preprocessing_root / "preview_generation_note.md",
        ),
    }
    preprocessing_paths["preview_generation_manifest"] = _write_json(
        {
            "timestamp_utc": _utc_timestamp(),
            "block_hours": block_hours,
            "horizons_hours": list(selected_horizons),
            "input_dir": str(standardized_input_dir.resolve()),
            "frozen_chapter1_dir": str(Path(frozen_chapter1_dir).resolve()),
            "frozen_split_assignments_path": str(frozen_split_assignments_path.resolve()),
            "block_paths": {key: str(path.resolve()) for key, path in block_paths.items()},
            "primary_model_ready_dataset_path": str(
                preprocessing_paths["primary_model_ready_dataset"].resolve()
            ),
            "feature_set_definition_path": str(
                preprocessing_paths["feature_set_definition"].resolve()
            ),
            "retained_stay_count": int(cohort.table.shape[0]),
            "usable_label_count": int(labels.usable_labels.shape[0]),
        },
        preprocessing_root / "preview_generation_manifest.json",
    )

    logistic_result = run_asic_primary_logistic_regression(
        input_dataset_path=preprocessing_paths["primary_model_ready_dataset"],
        feature_set_definition_path=preprocessing_paths["feature_set_definition"],
        output_dir=baseline_root / "logistic_regression",
        horizons=selected_horizons,
    )
    xgboost_result = run_asic_primary_xgboost(
        input_dataset_path=preprocessing_paths["primary_model_ready_dataset"],
        feature_set_definition_path=preprocessing_paths["feature_set_definition"],
        output_dir=baseline_root / "xgboost",
        horizons=selected_horizons,
    )
    _ = logistic_result, xgboost_result

    run_asic_baseline_evaluation(
        input_root=baseline_root,
        output_dir=evaluation_root,
        models=COMPARISON_MODELS,
        horizons=selected_horizons,
        primary_horizon=PRIMARY_HORIZON_HOURS,
    )

    comparison = _build_comparison_package(
        eight_hour_evaluation_root=Path(eight_hour_evaluation_root),
        sixteen_hour_evaluation_root=evaluation_root,
        comparison_output_dir=comparison_root,
        notebook_path=notebook_path,
    )
    return TemporalAggregationPreviewRunResult(
        output_root=output_root,
        block_hours=block_hours,
        block_paths=block_paths,
        preprocessing_paths=preprocessing_paths,
        baseline_root=baseline_root,
        evaluation_root=evaluation_root,
        comparison=comparison,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the narrow ASIC 16h temporal aggregation preview using the frozen Chapter 1 "
            "split assignments and the existing logistic-regression/XGBoost baselines."
        )
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        help="Optional Chapter 1 run config. Defaults to config/ch1_run_config.json.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Override the standardized ASIC input directory.",
    )
    parser.add_argument(
        "--input-format",
        choices=("csv", "parquet"),
        help="Override the standardized ASIC input format.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_PREVIEW_OUTPUT_ROOT,
        help="Preview-only root output directory.",
    )
    parser.add_argument(
        "--output-format",
        choices=("csv", "parquet"),
        default="csv",
        help="Output artifact format for preview preprocessing tables.",
    )
    parser.add_argument(
        "--frozen-chapter1-dir",
        type=Path,
        default=DEFAULT_FROZEN_CHAPTER1_DIR,
        help="Frozen 8h Chapter 1 artifact root containing the stay split assignments.",
    )
    parser.add_argument(
        "--eight-hour-evaluation-root",
        type=Path,
        default=DEFAULT_EIGHT_HOUR_EVALUATION_ROOT,
        help="Saved frozen 8h evaluation root used for the comparison package.",
    )
    parser.add_argument(
        "--notebook-path",
        type=Path,
        default=DEFAULT_NOTEBOOK_PATH,
        help="Path for the compact review notebook.",
    )
    parser.add_argument(
        "--block-hours",
        type=int,
        default=DEFAULT_BLOCK_HOURS,
        help="Alternative aggregation block size in hours. Preview default is 16.",
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

    result = run_asic_temporal_aggregation_preview(
        run_config_path=args.run_config,
        input_dir=args.input_dir,
        input_format=args.input_format,
        output_root=args.output_root,
        output_format=args.output_format,
        frozen_chapter1_dir=args.frozen_chapter1_dir,
        eight_hour_evaluation_root=args.eight_hour_evaluation_root,
        notebook_path=args.notebook_path,
        block_hours=args.block_hours,
        horizons=args.horizons,
    )

    print(f"Preview output root: {result.output_root}")
    print(f"16h block artifacts: {result.block_paths}")
    print(f"Preprocessing dataset: {result.preprocessing_paths['primary_model_ready_dataset']}")
    print(f"Baseline root: {result.baseline_root}")
    print(f"Evaluation root: {result.evaluation_root}")
    print(f"Comparison table: {result.comparison.comparison_table_path}")
    print(f"Preview note: {result.comparison.note_path}")
    print(f"Notebook: {result.comparison.notebook_path}")
    for figure_path in result.comparison.figure_paths:
        print(f"Figure: {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
