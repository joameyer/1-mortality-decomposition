from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

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
from chapter1_mortality_decomposition.hard_case_definition import (
    HARD_CASE_RULE as LOGISTIC_HARD_CASE_RULE,
    run_asic_logistic_hard_case_definition,
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
    normalize_boolean_codes,
    read_dataframe,
    require_columns,
    write_dataframe,
    write_text,
)


DEFAULT_REFERENCE_AGGREGATION_HOURS = 8
DEFAULT_SENSITIVITY_BLOCK_HOURS = (16, 24)
DEFAULT_HORIZONS = (8, 16, 24, 48, 72)
DEFAULT_MODELS = ("logistic_regression", "xgboost")
PRIMARY_MODEL_NAME = "logistic_regression"
SECONDARY_MODEL_NAME = "xgboost"
PRIMARY_FEATURE_SET_NAME = "primary"
PRIMARY_HORIZON_HOURS = 24
DEFAULT_FROZEN_CHAPTER1_DIR = Path("artifacts") / "chapter1"
DEFAULT_REFERENCE_EVALUATION_ROOT = (
    Path("artifacts") / "chapter1" / "evaluation" / "asic" / "baselines" / "primary_medians"
)
DEFAULT_REFERENCE_HARD_CASE_DIR = (
    Path("artifacts")
    / "chapter1"
    / "evaluation"
    / "asic"
    / "hard_cases"
    / "primary_medians"
    / "logistic_regression"
)
DEFAULT_OUTPUT_ROOT = Path("artifacts") / "chapter1" / "temporal_sensitivity" / "asic"
SPLIT_ALIGNMENT_FILENAME = "chapter1_temporal_sensitivity_split_alignment_summary"
REQUIRED_HARD_CASE_STAY_LEVEL_COLUMNS = {
    "stay_id_global",
    "hospital_id",
    "horizon_h",
    "label_value",
    "hard_case_flag",
}


@dataclass(frozen=True)
class TemporalSensitivityAggregationRunResult:
    aggregation_hours: int
    aggregation_label: str
    output_root: Path
    preprocessing_root: Path
    baseline_root: Path
    evaluation_root: Path
    hard_case_root: Path
    block_paths: dict[str, Path]
    preprocessing_paths: dict[str, Path]


@dataclass(frozen=True)
class TemporalSensitivityComparisonResult:
    comparison_root: Path
    artifact_paths: dict[str, Path]


@dataclass(frozen=True)
class TemporalSensitivityRunResult:
    output_root: Path
    reference_aggregation_label: str
    sensitivity_aggregations: tuple[int, ...]
    models: tuple[str, ...]
    horizons: tuple[int, ...]
    aggregation_results: tuple[TemporalSensitivityAggregationRunResult, ...]
    comparison: TemporalSensitivityComparisonResult


def _require_matplotlib() -> None:
    if plt is None:
        raise ImportError(
            "matplotlib is required for the Chapter 1 ASIC temporal sensitivity analysis."
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


def _aggregation_label(block_hours: int) -> str:
    return f"{int(block_hours)}h"


def _normalize_models(models: Sequence[str] | None) -> tuple[str, ...]:
    values = tuple(str(model) for model in (models or DEFAULT_MODELS))
    ordered = tuple(dict.fromkeys(values))
    unsupported = sorted(set(ordered) - set(DEFAULT_MODELS))
    if unsupported:
        raise ValueError(f"Unsupported model names requested: {unsupported}")
    if PRIMARY_MODEL_NAME not in ordered:
        raise ValueError(
            "The Chapter 1 temporal sensitivity issue requires logistic_regression as the primary anchor."
        )
    return ordered


def _normalize_sensitivity_block_hours(block_hours: Sequence[int] | None) -> tuple[int, ...]:
    values = tuple(sorted({int(value) for value in (block_hours or DEFAULT_SENSITIVITY_BLOCK_HOURS)}))
    if not values:
        raise ValueError("At least one coarser aggregation must be requested.")
    if any(value <= DEFAULT_REFERENCE_AGGREGATION_HOURS for value in values):
        raise ValueError(
            "Temporal sensitivity block hours must all be coarser than the frozen 8h reference."
        )
    return values


def _output_extension(output_format: str) -> str:
    if output_format not in {"csv", "parquet"}:
        raise ValueError(f"Unsupported output format: {output_format}")
    return "csv" if output_format == "csv" else "parquet"


def _aggregation_output_root(output_root: Path, aggregation_hours: int) -> Path:
    return Path(output_root) / f"aggregation_{int(aggregation_hours)}h"


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
            "Missing standardized ASIC input artifacts required for temporal sensitivity: "
            + ", ".join(missing)
        )

    return {name: read_dataframe(path) for name, path in required_paths.items()}


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
    *,
    aggregation_label: str,
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
    unused_frozen_pairs = sorted(frozen_pairs - retained_pairs)

    summary = pd.DataFrame(
        [
            {
                "aggregation": aggregation_label,
                "check_id": "frozen_split_assignments_have_no_duplicate_stays",
                "passed": duplicate_assignment_count == 0,
                "count": duplicate_assignment_count,
                "detail": "Frozen split assignments should contain exactly one row per retained stay.",
            },
            {
                "aggregation": aggregation_label,
                "check_id": "all_retained_stays_have_frozen_split_assignment",
                "passed": len(missing_pairs) == 0,
                "count": len(missing_pairs),
                "detail": (
                    "Every retained stay in the coarsened aggregation should have a frozen 8h "
                    "stay-level split assignment."
                ),
            },
            {
                "aggregation": aggregation_label,
                "check_id": "unused_frozen_split_assignments_after_alignment",
                "passed": len(unused_frozen_pairs) == 0,
                "count": len(unused_frozen_pairs),
                "detail": (
                    "Frozen split assignments present in the 8h reference but not reused after "
                    "alignment to the coarsened aggregation."
                ),
            },
            {
                "aggregation": aggregation_label,
                "check_id": "retained_stay_count",
                "passed": True,
                "count": int(retained.shape[0]),
                "detail": "Retained stays in the coarsened aggregation before split alignment.",
            },
            {
                "aggregation": aggregation_label,
                "check_id": "frozen_assignment_count",
                "passed": True,
                "count": int(frozen.shape[0]),
                "detail": "Rows in the frozen 8h stay-level split table.",
            },
        ]
    )

    if duplicate_assignment_count > 0 or missing_pairs:
        details: list[str] = []
        if duplicate_assignment_count > 0:
            details.append(f"duplicate assignment rows={duplicate_assignment_count}")
        if missing_pairs:
            details.append(f"retained stays missing frozen split assignment={missing_pairs[:5]}")
        raise ValueError(
            "Frozen 8h stay split assignments could not be reused for temporal sensitivity: "
            + "; ".join(details)
        )

    aligned = frozen.merge(retained, on=["stay_id_global", "hospital_id"], how="inner")
    return summary, aligned.sort_values(
        ["hospital_id", "split", "stay_id_global"],
        kind="stable",
    ).reset_index(drop=True)


def _coarsened_generation_note(
    *,
    aggregation_hours: int,
    input_dir: Path,
    frozen_chapter1_dir: Path,
    frozen_split_assignments_path: Path,
) -> str:
    return "\n".join(
        [
            f"# ASIC Temporal Sensitivity Generation Note: {_aggregation_label(aggregation_hours)}",
            "",
            f"- Frozen reference aggregation: `{DEFAULT_REFERENCE_AGGREGATION_HOURS}h`.",
            f"- Coarsened sensitivity aggregation: `{aggregation_hours}h` completed blocks only.",
            (
                "- The coarsened blocked artifacts were rebuilt inside the Chapter 1 repo from the "
                "standardized harmonized ASIC dynamic table, using the saved stay-level timing proxy "
                "table from the standardized blocked inputs."
            ),
            (
                "- Block membership uses `time_h // block_hours`, saved prediction times remain "
                "`prediction_time_h == block_end_h`, and only structurally completed blocks are kept."
            ),
            (
                "- Cohort intent, proxy within-horizon mortality labels, feature-set logic, bounded "
                "LOCF preprocessing, and baseline model classes were kept consistent with the frozen "
                "Chapter 1 workflow."
            ),
            (
                f"- Frozen stay-level split assignments were reused from "
                f"`{frozen_split_assignments_path.resolve()}` whenever the retained stay remained "
                "present in the coarsened aggregation."
            ),
            (
                "- This issue is a bounded coarsening sensitivity only. It does not search for an "
                "optimal aggregation and it does not introduce finer-than-8h representations."
            ),
            (
                f"- Standardized ASIC source directory: `{input_dir.resolve()}`. "
                f"Frozen 8h Chapter 1 artifact root: `{frozen_chapter1_dir.resolve()}`."
            ),
        ]
    )


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


def _plot_reliability_triptych(
    summaries_by_aggregation: dict[str, pd.DataFrame],
    metrics_by_aggregation: dict[str, dict[str, object]],
    *,
    aggregation_labels: Sequence[str],
    model_name: str,
    horizon_h: int,
    output_path: Path,
) -> Path:
    _require_matplotlib()
    ordered_summaries = [
        summaries_by_aggregation.get(aggregation_label, pd.DataFrame())
        for aggregation_label in aggregation_labels
    ]
    if all(summary.empty for summary in ordered_summaries):
        return _save_placeholder_figure(
            output_path,
            title=f"{_display_model_name(model_name)} {horizon_h}h reliability comparison",
            message="No finite risk-bin summaries were available across the selected aggregations.",
        )

    ensure_directory(output_path.parent)
    axis_limit = _dynamic_probability_axis_limit(ordered_summaries)
    figure, axes = plt.subplots(1, len(aggregation_labels), figsize=(5.0 * len(aggregation_labels), 5.0), sharey=True)
    axes_array = np.atleast_1d(axes)
    for axis, aggregation_label in zip(axes_array, aggregation_labels):
        summary = summaries_by_aggregation.get(aggregation_label, pd.DataFrame())
        metrics = metrics_by_aggregation.get(aggregation_label)
        axis.plot([0.0, axis_limit], [0.0, axis_limit], linestyle="--", color="black", linewidth=1.0)
        axis.set_title(f"{aggregation_label} aggregation")
        axis.set_xlim(0.0, axis_limit)
        axis.set_ylim(0.0, axis_limit)
        axis.set_xlabel("Mean predicted risk")
        axis.grid(alpha=0.25, linewidth=0.6)
        if axis is axes_array[0]:
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

    figure.suptitle(
        f"{_display_model_name(model_name)} {horizon_h}h reliability: "
        + " vs ".join(aggregation_labels)
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _plot_mortality_vs_risk_triptych(
    summaries_by_aggregation: dict[str, pd.DataFrame],
    metrics_by_aggregation: dict[str, dict[str, object]],
    *,
    aggregation_labels: Sequence[str],
    model_name: str,
    horizon_h: int,
    output_path: Path,
) -> Path:
    _require_matplotlib()
    ordered_summaries = [
        summaries_by_aggregation.get(aggregation_label, pd.DataFrame())
        for aggregation_label in aggregation_labels
    ]
    if all(summary.empty for summary in ordered_summaries):
        return _save_placeholder_figure(
            output_path,
            title=f"{_display_model_name(model_name)} {horizon_h}h mortality-vs-risk comparison",
            message="No finite risk-bin summaries were available across the selected aggregations.",
        )

    ensure_directory(output_path.parent)
    axis_limit = _dynamic_probability_axis_limit(ordered_summaries)
    figure, axes = plt.subplots(1, len(aggregation_labels), figsize=(5.4 * len(aggregation_labels), 5.6), sharey=False)
    axes_array = np.atleast_1d(axes)
    for axis, aggregation_label in zip(axes_array, aggregation_labels):
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

    figure.suptitle(
        f"{_display_model_name(model_name)} {horizon_h}h mortality vs risk: "
        + " vs ".join(aggregation_labels)
    )
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


def _append_reference_deltas(
    frame: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    reference_aggregation_label: str,
    numeric_columns: Sequence[str],
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    reference = frame[frame["aggregation"].astype("string").eq(reference_aggregation_label)][
        list(key_columns) + list(numeric_columns)
    ].copy()
    rename_map = {column: f"{column}_reference_{reference_aggregation_label}" for column in numeric_columns}
    reference = reference.rename(columns=rename_map)
    merged = frame.merge(reference, on=list(key_columns), how="left")
    for column in numeric_columns:
        reference_column = f"{column}_reference_{reference_aggregation_label}"
        delta_column = f"{column}_delta_vs_{reference_aggregation_label}"
        merged[delta_column] = (
            pd.to_numeric(merged[column], errors="coerce")
            - pd.to_numeric(merged[reference_column], errors="coerce")
        )
    return merged


def _load_preprocessing_root_tables(
    preprocessing_root: Path,
    *,
    output_format: str,
) -> dict[str, pd.DataFrame]:
    extension = _output_extension(output_format)
    return {
        "cohort_summary": read_dataframe(
            preprocessing_root / "cohort" / f"chapter1_cohort_summary.{extension}"
        ),
        "instance_counts_by_horizon": read_dataframe(
            preprocessing_root / "instances" / f"chapter1_instance_counts_by_horizon.{extension}"
        ),
        "proxy_label_summary_by_horizon": read_dataframe(
            preprocessing_root / "labels" / f"chapter1_proxy_label_summary_by_horizon.{extension}"
        ),
    }


def _build_preprocessing_count_comparison(
    *,
    preprocessing_roots_by_aggregation: dict[str, Path],
    output_format: str,
    reference_aggregation_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for aggregation_label, preprocessing_root in preprocessing_roots_by_aggregation.items():
        tables = _load_preprocessing_root_tables(preprocessing_root, output_format=output_format)
        cohort_summary = tables["cohort_summary"]
        instance_counts = tables["instance_counts_by_horizon"]
        label_summary = tables["proxy_label_summary_by_horizon"]

        retained_stays = cohort_summary[
            cohort_summary["metric"].astype("string").eq("retained_stays")
        ]
        if not retained_stays.empty:
            rows.append(
                {
                    "aggregation": aggregation_label,
                    "metric_group": "cohort",
                    "metric": "retained_stays",
                    "horizon_h": pd.NA,
                    "value": int(retained_stays["value"].iloc[0]),
                }
            )

        total_valid_instances = cohort_summary[
            cohort_summary["metric"].astype("string").eq("valid_prediction_instances_total")
        ]
        if not total_valid_instances.empty:
            rows.append(
                {
                    "aggregation": aggregation_label,
                    "metric_group": "instances",
                    "metric": "valid_prediction_instances_total",
                    "horizon_h": pd.NA,
                    "value": int(total_valid_instances["value"].iloc[0]),
                }
            )

        for row in instance_counts.itertuples(index=False):
            rows.append(
                {
                    "aggregation": aggregation_label,
                    "metric_group": "instances",
                    "metric": "valid_prediction_instances",
                    "horizon_h": int(row.horizon_h),
                    "value": int(row.valid_instances),
                }
            )

        for row in label_summary.itertuples(index=False):
            for metric_name in (
                "labelable_instances",
                "positive_labels",
                "negative_labels",
                "unlabeled_instances",
            ):
                rows.append(
                    {
                        "aggregation": aggregation_label,
                        "metric_group": "labels",
                        "metric": metric_name,
                        "horizon_h": int(row.horizon_h),
                        "value": int(getattr(row, metric_name)),
                    }
                )

    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison

    reference = comparison[comparison["aggregation"].astype("string").eq(reference_aggregation_label)][
        ["metric_group", "metric", "horizon_h", "value"]
    ].rename(columns={"value": f"value_reference_{reference_aggregation_label}"})
    merged = comparison.merge(
        reference,
        on=["metric_group", "metric", "horizon_h"],
        how="left",
    )
    merged[f"value_delta_vs_{reference_aggregation_label}"] = (
        pd.to_numeric(merged["value"], errors="coerce")
        - pd.to_numeric(merged[f"value_reference_{reference_aggregation_label}"], errors="coerce")
    )
    return merged.sort_values(
        ["metric_group", "metric", "horizon_h", "aggregation"],
        kind="stable",
    ).reset_index(drop=True)


def _load_hard_case_summary(hard_case_root: Path) -> pd.DataFrame:
    summary_path = Path(hard_case_root) / "horizon_hard_case_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing hard-case summary at {summary_path}")
    return read_dataframe(summary_path)


def _build_hard_case_prevalence_summary(
    *,
    hard_case_roots_by_aggregation: dict[str, Path],
    reference_aggregation_label: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for aggregation_label, hard_case_root in hard_case_roots_by_aggregation.items():
        frame = _load_hard_case_summary(hard_case_root).copy()
        frame["aggregation"] = aggregation_label
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["horizon_h", "aggregation"],
        kind="stable",
    ).reset_index(drop=True)
    return _append_reference_deltas(
        combined,
        key_columns=["model_name", "horizon_h"],
        reference_aggregation_label=reference_aggregation_label,
        numeric_columns=[
            "n_fatal_last_points",
            "n_hard_cases",
            "pct_fatal_hard_cases",
            "nonfatal_q75_threshold",
        ],
    )


def _load_hard_case_stay_level_for_aggregation(
    hard_case_root: Path,
    *,
    aggregation_label: str,
    horizon_h: int,
) -> pd.DataFrame:
    stay_level_path = Path(hard_case_root) / "stay_level_hard_case_flags.csv"
    if not stay_level_path.exists():
        raise FileNotFoundError(f"Missing hard-case stay-level flags at {stay_level_path}")
    stay_level = read_dataframe(stay_level_path)
    require_columns(stay_level, REQUIRED_HARD_CASE_STAY_LEVEL_COLUMNS, str(stay_level_path))

    output = stay_level.copy()
    output["stay_id"] = output["stay_id_global"].astype("string")
    output["hospital_id"] = output["hospital_id"].astype("string")
    output["horizon_h"] = pd.to_numeric(output["horizon_h"], errors="coerce").astype("Int64")
    output["fatal_flag"] = pd.to_numeric(output["label_value"], errors="coerce").astype("Int64").eq(1)
    output["hard_case_flag"] = normalize_boolean_codes(output["hard_case_flag"]).fillna(False).astype(bool)
    output["available_flag"] = True
    output["aggregation"] = aggregation_label
    filtered = output[output["horizon_h"].eq(int(horizon_h))].copy()
    if filtered.empty:
        raise ValueError(
            f"No hard-case stay-level rows were available for aggregation {aggregation_label} at {horizon_h}h."
        )
    duplicate_count = int(filtered.duplicated(subset=["stay_id"]).sum())
    if duplicate_count:
        raise ValueError(
            f"Aggregation {aggregation_label} 24h hard-case stay-level artifact contains "
            f"{duplicate_count} duplicated stay_id rows."
        )
    invalid_hard_cases = filtered["hard_case_flag"] & ~filtered["fatal_flag"]
    if bool(invalid_hard_cases.any()):
        examples = filtered.loc[invalid_hard_cases, ["stay_id", "hospital_id"]].head(5)
        raise ValueError(
            "Found hard_case_flag=True rows outside the fatal population while building "
            f"aggregation overlap inputs. Examples: {examples.to_dict(orient='records')}"
        )
    return filtered[
        ["aggregation", "stay_id", "hospital_id", "fatal_flag", "hard_case_flag", "available_flag"]
    ].sort_values(["hospital_id", "stay_id"], kind="stable").reset_index(drop=True)


def _build_aggregation_pairwise_tables(
    harmonized_stay_level: pd.DataFrame,
    *,
    aggregation_labels: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairwise_denominator_rows: list[dict[str, object]] = []
    pairwise_overlap_rows: list[dict[str, object]] = []

    fatal_by_aggregation = {
        aggregation_label: harmonized_stay_level[
            harmonized_stay_level["aggregation"].astype("string").eq(aggregation_label)
            & harmonized_stay_level["fatal_flag"].astype(bool)
        ][["stay_id", "hospital_id", "hard_case_flag"]].copy()
        for aggregation_label in aggregation_labels
    }

    for aggregation_a, aggregation_b in itertools.combinations(aggregation_labels, 2):
        fatal_a = fatal_by_aggregation[aggregation_a].rename(
            columns={"hospital_id": "hospital_id_a", "hard_case_flag": "hard_case_flag_a"}
        )
        fatal_b = fatal_by_aggregation[aggregation_b].rename(
            columns={"hospital_id": "hospital_id_b", "hard_case_flag": "hard_case_flag_b"}
        )
        matched = fatal_a.merge(fatal_b, on="stay_id", how="inner")

        hospital_mismatch = matched[
            matched["hospital_id_a"].astype("string").ne(matched["hospital_id_b"].astype("string"))
        ]
        if not hospital_mismatch.empty:
            raise ValueError(
                "Hospital ID mismatch detected while matching fatal stays across aggregations. "
                f"Examples: {hospital_mismatch[['stay_id', 'hospital_id_a', 'hospital_id_b']].head(5).to_dict(orient='records')}"
            )

        matched_fatal_n = int(matched.shape[0])
        fatal_n_a = int(fatal_a.shape[0])
        fatal_n_b = int(fatal_b.shape[0])
        dropped_fatal_a_unmatched = int(fatal_n_a - matched_fatal_n)
        dropped_fatal_b_unmatched = int(fatal_n_b - matched_fatal_n)

        pairwise_denominator_rows.append(
            {
                "aggregation_a": aggregation_a,
                "aggregation_b": aggregation_b,
                "fatal_n_aggregation_a": fatal_n_a,
                "fatal_n_aggregation_b": fatal_n_b,
                "matched_fatal_n": matched_fatal_n,
                "dropped_fatal_a_unmatched": dropped_fatal_a_unmatched,
                "dropped_fatal_b_unmatched": dropped_fatal_b_unmatched,
                "matched_share_of_aggregation_a_fatal": (
                    float(matched_fatal_n / fatal_n_a) if fatal_n_a else np.nan
                ),
                "matched_share_of_aggregation_b_fatal": (
                    float(matched_fatal_n / fatal_n_b) if fatal_n_b else np.nan
                ),
            }
        )

        hard_case_flag_a = matched["hard_case_flag_a"].astype(bool)
        hard_case_flag_b = matched["hard_case_flag_b"].astype(bool)
        hard_n_a = int(hard_case_flag_a.sum())
        hard_n_b = int(hard_case_flag_b.sum())
        intersection_n = int((hard_case_flag_a & hard_case_flag_b).sum())
        union_n = int((hard_case_flag_a | hard_case_flag_b).sum())
        jaccard_index = float(intersection_n / union_n) if union_n else np.nan

        pairwise_overlap_rows.append(
            {
                "aggregation_a": aggregation_a,
                "aggregation_b": aggregation_b,
                "matched_fatal_n": matched_fatal_n,
                "hard_n_aggregation_a": hard_n_a,
                "hard_n_aggregation_b": hard_n_b,
                "intersection_n": intersection_n,
                "union_n": union_n,
                "jaccard_index": jaccard_index,
            }
        )

    return pd.DataFrame(pairwise_denominator_rows), pd.DataFrame(pairwise_overlap_rows)


def _build_aggregation_directional_overlap(
    pairwise_overlap: pd.DataFrame,
    *,
    aggregation_labels: Sequence[str],
) -> pd.DataFrame:
    directional_rows: list[dict[str, object]] = []
    for row in pairwise_overlap.itertuples(index=False):
        overlap_a_to_b = (
            float(row.intersection_n / row.hard_n_aggregation_a)
            if int(row.hard_n_aggregation_a) > 0
            else np.nan
        )
        overlap_b_to_a = (
            float(row.intersection_n / row.hard_n_aggregation_b)
            if int(row.hard_n_aggregation_b) > 0
            else np.nan
        )
        directional_rows.append(
            {
                "aggregation_from": str(row.aggregation_a),
                "aggregation_to": str(row.aggregation_b),
                "matched_fatal_n": int(row.matched_fatal_n),
                "hard_n_from": int(row.hard_n_aggregation_a),
                "hard_n_to": int(row.hard_n_aggregation_b),
                "intersection_n": int(row.intersection_n),
                "overlap_from_A_to_B": overlap_a_to_b,
            }
        )
        directional_rows.append(
            {
                "aggregation_from": str(row.aggregation_b),
                "aggregation_to": str(row.aggregation_a),
                "matched_fatal_n": int(row.matched_fatal_n),
                "hard_n_from": int(row.hard_n_aggregation_b),
                "hard_n_to": int(row.hard_n_aggregation_a),
                "intersection_n": int(row.intersection_n),
                "overlap_from_A_to_B": overlap_b_to_a,
            }
        )
    directional_overlap = pd.DataFrame(directional_rows)
    aggregation_order = {label: position for position, label in enumerate(aggregation_labels)}
    directional_overlap["_from_order"] = directional_overlap["aggregation_from"].map(aggregation_order)
    directional_overlap["_to_order"] = directional_overlap["aggregation_to"].map(aggregation_order)
    directional_overlap = directional_overlap.sort_values(
        ["_from_order", "_to_order"],
        kind="stable",
    ).drop(columns=["_from_order", "_to_order"])
    return directional_overlap.reset_index(drop=True)


def _build_aggregation_persistence_tables(
    harmonized_stay_level: pd.DataFrame,
    *,
    aggregation_labels: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fatal_any_ids = (
        harmonized_stay_level.loc[harmonized_stay_level["fatal_flag"].astype(bool), "stay_id"]
        .astype("string")
        .drop_duplicates()
        .sort_values(kind="stable")
        .tolist()
    )
    if not fatal_any_ids:
        raise ValueError("No fatal stays were available for aggregation-level persistence analysis.")

    hospital_lookup = (
        harmonized_stay_level[["stay_id", "hospital_id"]]
        .drop_duplicates(subset=["stay_id"], keep="first")
        .set_index("stay_id")["hospital_id"]
    )
    persistence = pd.DataFrame({"stay_id": fatal_any_ids})
    persistence["hospital_id"] = persistence["stay_id"].map(hospital_lookup).astype("string")

    for aggregation_label in aggregation_labels:
        aggregation_df = harmonized_stay_level[
            harmonized_stay_level["aggregation"].astype("string").eq(aggregation_label)
        ][["stay_id", "available_flag", "fatal_flag", "hard_case_flag"]].copy()
        aggregation_df = aggregation_df.set_index("stay_id")
        persistence[f"available_{aggregation_label}"] = (
            persistence["stay_id"].map(aggregation_df["available_flag"]).fillna(False).astype(bool)
        )
        persistence[f"fatal_{aggregation_label}"] = (
            persistence["stay_id"].map(aggregation_df["fatal_flag"]).fillna(False).astype(bool)
        )
        persistence[f"hard_case_{aggregation_label}"] = (
            persistence["stay_id"].map(aggregation_df["hard_case_flag"]).fillna(False).astype(bool)
        )

    available_columns = [f"available_{aggregation_label}" for aggregation_label in aggregation_labels]
    fatal_columns = [f"fatal_{aggregation_label}" for aggregation_label in aggregation_labels]
    hard_case_columns = [f"hard_case_{aggregation_label}" for aggregation_label in aggregation_labels]
    persistence["available_aggregation_n"] = persistence[available_columns].sum(axis=1).astype(int)
    persistence["fatal_aggregation_n"] = persistence[fatal_columns].sum(axis=1).astype(int)
    persistence["hard_case_aggregation_n"] = persistence[hard_case_columns].sum(axis=1).astype(int)
    persistence["hard_case_share_among_fatal_aggregations"] = (
        persistence["hard_case_aggregation_n"] / persistence["fatal_aggregation_n"]
    ).astype(float)

    distribution = (
        persistence["hard_case_aggregation_n"]
        .value_counts(dropna=False)
        .reindex(range(0, len(aggregation_labels) + 1), fill_value=0)
        .rename_axis("hard_case_aggregation_n")
        .reset_index(name="fatal_stay_count")
    )
    distribution["fatal_stay_share"] = distribution["fatal_stay_count"] / int(persistence.shape[0])
    return persistence, distribution


def _build_heatmap_matrix(
    pairwise_overlap: pd.DataFrame,
    directional_overlap: pd.DataFrame,
    *,
    aggregation_labels: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = list(aggregation_labels)
    jaccard_matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    directional_matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)

    for label in labels:
        jaccard_matrix.loc[label, label] = 1.0
        directional_matrix.loc[label, label] = 1.0

    for row in pairwise_overlap.itertuples(index=False):
        jaccard_matrix.loc[row.aggregation_a, row.aggregation_b] = row.jaccard_index
        jaccard_matrix.loc[row.aggregation_b, row.aggregation_a] = row.jaccard_index

    for row in directional_overlap.itertuples(index=False):
        directional_matrix.loc[row.aggregation_from, row.aggregation_to] = row.overlap_from_A_to_B

    return jaccard_matrix, directional_matrix


def _annotate_heatmap(matrix: pd.DataFrame, axis: plt.Axes) -> None:
    for row_index, row_label in enumerate(matrix.index):
        for column_index, column_label in enumerate(matrix.columns):
            value = matrix.loc[row_label, column_label]
            if pd.isna(value):
                axis.text(column_index, row_index, "NA", ha="center", va="center", color="#555555", fontsize=9)
            else:
                color = "white" if float(value) >= 0.5 else "#222222"
                axis.text(
                    column_index,
                    row_index,
                    f"{float(value):.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                )


def _plot_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str,
    x_label: str,
    y_label: str,
    colorbar_label: str,
    output_path: Path,
) -> Path:
    _require_matplotlib()
    ensure_directory(output_path.parent)
    figure, axis = plt.subplots(figsize=(6.4, 5.6))
    image = axis.imshow(matrix.to_numpy(dtype=float), cmap="Blues", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    _annotate_heatmap(matrix, axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _plot_persistence_barplot(
    persistence_distribution: pd.DataFrame,
    *,
    output_path: Path,
) -> Path:
    _require_matplotlib()
    ensure_directory(output_path.parent)
    figure, axis = plt.subplots(figsize=(7.0, 4.6))
    x_positions = persistence_distribution["hard_case_aggregation_n"].astype(int).tolist()
    counts = persistence_distribution["fatal_stay_count"].astype(int).tolist()
    axis.bar(x_positions, counts, color="#4c78a8", edgecolor="#2f4a6d", width=0.75)
    axis.set_xlabel("Number of aggregations labeled hard")
    axis.set_ylabel("Fatal stays")
    axis.set_title("Hard-Case Persistence Across Aggregations")
    axis.set_xticks(x_positions)
    axis.set_xlim(-0.5, max(x_positions) + 0.5)
    ymax = max(counts) if counts else 0
    axis.set_ylim(0, max(1, ymax * 1.15))
    for x_position, count in zip(x_positions, counts):
        axis.text(x_position, count + max(0.02, ymax * 0.02), str(count), ha="center", va="bottom", fontsize=9)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


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


def _build_aggregation_overlap_note(
    *,
    horizon_h: int,
    pairwise_denominators: pd.DataFrame,
    pairwise_overlap: pd.DataFrame,
    directional_overlap: pd.DataFrame,
    persistence_distribution: pd.DataFrame,
) -> str:
    jaccard_nonmissing = pairwise_overlap["jaccard_index"].dropna()
    directional_nonmissing = directional_overlap["overlap_from_A_to_B"].dropna()
    all_pairs_fully_matched = bool(
        pairwise_denominators["dropped_fatal_a_unmatched"].eq(0).all()
        and pairwise_denominators["dropped_fatal_b_unmatched"].eq(0).all()
    )
    strongest_pair = pairwise_overlap.sort_values(
        ["jaccard_index", "matched_fatal_n", "aggregation_a", "aggregation_b"],
        ascending=[False, False, True, True],
        kind="stable",
    ).iloc[0]
    weakest_pair = pairwise_overlap.sort_values(
        ["jaccard_index", "matched_fatal_n", "aggregation_a", "aggregation_b"],
        ascending=[True, False, True, True],
        kind="stable",
    ).iloc[0]

    denominator_rows = [
        {
            "aggregation_a": row.aggregation_a,
            "aggregation_b": row.aggregation_b,
            "fatal_n_aggregation_a": int(row.fatal_n_aggregation_a),
            "fatal_n_aggregation_b": int(row.fatal_n_aggregation_b),
            "matched_fatal_n": int(row.matched_fatal_n),
        }
        for row in pairwise_denominators.itertuples(index=False)
    ]
    overlap_rows = [
        {
            "aggregation_a": row.aggregation_a,
            "aggregation_b": row.aggregation_b,
            "matched_fatal_n": int(row.matched_fatal_n),
            "hard_n_aggregation_a": int(row.hard_n_aggregation_a),
            "hard_n_aggregation_b": int(row.hard_n_aggregation_b),
            "intersection_n": int(row.intersection_n),
            "union_n": int(row.union_n),
            "jaccard_index": "NA" if pd.isna(row.jaccard_index) else f"{float(row.jaccard_index):.3f}",
        }
        for row in pairwise_overlap.itertuples(index=False)
    ]
    persistence_rows = [
        {
            "hard_case_aggregation_n": int(row.hard_case_aggregation_n),
            "fatal_stay_count": int(row.fatal_stay_count),
            "fatal_stay_share": f"{float(row.fatal_stay_share):.3f}",
        }
        for row in persistence_distribution.itertuples(index=False)
    ]

    lines = [
        f"# Logistic {horizon_h}h Aggregation Overlap Note",
        "",
        f"- Hard-case rule: `{LOGISTIC_HARD_CASE_RULE}`.",
        f"- Compared population: fatal stays with saved logistic {horizon_h}h hard-case outputs under each aggregation.",
        "- Matching key: `stay_id_global` with `hospital_id` checked for consistency after matching.",
        "- Pairwise overlap uses matched fatal stays only; it does not divide by raw fatal totals from one aggregation alone.",
        "- Directional overlap uses the same matched fatal set, then divides the intersection by the hard-case count in the source aggregation.",
        (
            "- All aggregation pairs retained the same fatal stay set after matching."
            if all_pairs_fully_matched
            else "- Some aggregation pairs lost fatal stays during matching; see the denominator table."
        ),
        "",
        _markdown_table(
            denominator_rows,
            columns=[
                "aggregation_a",
                "aggregation_b",
                "fatal_n_aggregation_a",
                "fatal_n_aggregation_b",
                "matched_fatal_n",
            ],
        ),
        "",
        f"- Mean Jaccard across aggregation pairs: `{float(jaccard_nonmissing.mean()):.3f}`",
        f"- Strongest pair: `{strongest_pair.aggregation_a}` vs `{strongest_pair.aggregation_b}` with Jaccard `{float(strongest_pair.jaccard_index):.3f}`.",
        f"- Weakest pair: `{weakest_pair.aggregation_a}` vs `{weakest_pair.aggregation_b}` with Jaccard `{float(weakest_pair.jaccard_index):.3f}`.",
        f"- Mean directional overlap across ordered aggregation pairs: `{float(directional_nonmissing.mean()):.3f}`",
        "",
        _markdown_table(
            overlap_rows,
            columns=[
                "aggregation_a",
                "aggregation_b",
                "matched_fatal_n",
                "hard_n_aggregation_a",
                "hard_n_aggregation_b",
                "intersection_n",
                "union_n",
                "jaccard_index",
            ],
        ),
        "",
        _markdown_table(
            persistence_rows,
            columns=[
                "hard_case_aggregation_n",
                "fatal_stay_count",
                "fatal_stay_share",
            ],
        ),
        "",
        "This is a coarsening sensitivity output only; it should be interpreted together with the calibration, prevalence, and mortality-vs-risk summaries after the real cluster run.",
        "",
    ]
    return "\n".join(lines)


def _build_provenance_and_limitations_note(
    *,
    standardized_input_dir: Path,
    frozen_chapter1_dir: Path,
    reference_evaluation_root: Path,
    reference_hard_case_dir: Path,
    aggregation_results: Sequence[TemporalSensitivityAggregationRunResult],
    selected_split_summary: pd.DataFrame,
    split_alignment_overview: pd.DataFrame,
) -> str:
    lines = [
        "# ASIC Temporal Sensitivity Provenance And Limitations",
        "",
        "This package is a bounded coarsening sensitivity for Chapter 1, not an aggregation search and not a full temporal-resolution study.",
        "",
        "## Provenance",
        "",
        f"- Frozen primary reference: `{DEFAULT_REFERENCE_AGGREGATION_HOURS}h` Chapter 1 artifacts under `{frozen_chapter1_dir.resolve()}`.",
        f"- Frozen 8h evaluation root used for comparison: `{reference_evaluation_root.resolve()}`.",
        f"- Frozen 8h logistic hard-case root used for comparison: `{reference_hard_case_dir.resolve()}`.",
        f"- Standardized ASIC input directory used to build the coarsened aggregations: `{standardized_input_dir.resolve()}`.",
        "- The coarsened 16h and 24h blocked artifacts were rebuilt inside this repo from the standardized harmonized dynamic table plus the saved stay-level timing proxy table, not from a new upstream raw-time-series rebuild.",
        "- Cohort intent, split intent, label intent, primary feature-set logic, logistic baseline, and hard-case rule were held as constant as the current Chapter 1 repo permits.",
        "- The frozen hard-case concept was preserved exactly as the logistic last-eligible fatal-below-nonfatal-q75 rule, rather than being redefined per aggregation.",
        "",
        "## Split Reuse",
        "",
    ]
    for row in split_alignment_overview.itertuples(index=False):
        lines.append(
            f"- {row.aggregation}: `{row.check_id}` -> count `{int(row.count)}`; passed=`{bool(row.passed)}`."
        )

    fallback_rows = selected_split_summary[
        selected_split_summary["selected_split"].astype("string").ne("test")
        | ~selected_split_summary["selected_split_evaluable"].astype(bool)
    ].copy()
    lines.extend(
        [
            "",
            "## Reporting-Split Behavior",
            "",
            (
                "- All saved reporting comparisons used the preferred binary-evaluable test split."
                if fallback_rows.empty
                else "- At least one saved comparison fell back away from the preferred binary-evaluable test split."
            ),
        ]
    )
    for row in fallback_rows.itertuples(index=False):
        lines.append(
            f"- {row.model_name} {int(row.horizon_h)}h at {row.aggregation}: "
            f"selected `{row.selected_split}` (evaluable=`{bool(row.selected_split_evaluable)}`; "
            f"reason=`{row.selection_reason}`)."
        )

    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- Coarsened aggregations were generated inside the Chapter 1 repo, so they inherit the existing standardized ASIC timing proxies rather than a fresh upstream resampling pass.",
            "- Finer-than-8h aggregation was intentionally not implemented because it would require cross-repo upstream work.",
            "- Hard-case overlap was computed on matched fatal stays at the logistic 24h horizon; nonfatal stays are not part of the overlap denominator.",
            "- The exploratory `artifacts/chapter1/temporal_preview/asic/aggregation_16h/` package becomes superseded for interpretation once this formal temporal-sensitivity run exists on the real data.",
        ]
    )
    return "\n".join(lines)


def _build_supersession_note() -> str:
    return "\n".join(
        [
            "# Temporal Preview Supersession Note",
            "",
            "The earlier `artifacts/chapter1/temporal_preview/asic/aggregation_16h/` package was a narrow precursor only.",
            "Once the formal temporal sensitivity analysis has been run on the cluster and written under `artifacts/chapter1/temporal_sensitivity/asic/`, the preview should no longer be treated as the authoritative sensitivity artifact tree.",
            "",
        ]
    )


def _build_interpretation_memo_template(
    *,
    comparison_root: Path,
    aggregation_labels: Sequence[str],
) -> str:
    return "\n".join(
        [
            "# ASIC Temporal Aggregation Sensitivity Memo Template",
            "",
            "This memo interprets the formal Chapter 1 coarsening sensitivity after the real cluster outputs have been reviewed.",
            "",
            "## Decision Label",
            "",
            "- Classification: `<stable under coarsening | partially weakened under coarsening | materially aggregation-sensitive>`",
            "- Reference aggregation: `8h`.",
            f"- Coarsened sensitivities reviewed: `{', '.join(aggregation_labels[1:])}`.",
            "",
            "## Primary Anchor",
            "",
            "- Model: `logistic_regression`.",
            "- Horizon: `24h`.",
            "- Read calibration and risk-structure first; use XGBoost only as a compact robustness check.",
            "",
            "## Evidence To Cite",
            "",
            f"- Reporting metrics: `{(comparison_root / 'reporting_metric_summary.csv').name}`",
            f"- Calibration summary: `{(comparison_root / 'calibration_summary.csv').name}`",
            f"- Mortality-vs-risk summary: `{(comparison_root / 'mortality_risk_structure_summary.csv').name}`",
            f"- Hard-case prevalence summary: `{(comparison_root / 'hard_case_prevalence_summary.csv').name}`",
            f"- Logistic 24h hard-case overlap: `{(comparison_root / 'logistic_24h_hard_case_pairwise_overlap.csv').name}` and `{(comparison_root / 'logistic_24h_hard_case_directional_overlap.csv').name}`",
            "",
            "## Prompted Write-Up",
            "",
            "- Calibration: `<describe intercept/slope/Brier movement across 8h, 16h, 24h>`",
            "- Mortality-vs-risk structure: `<describe whether ordering and concentration of mortality remain qualitatively intact>`",
            "- Low-predicted fatal-case prevalence: `<describe whether the logistic 24h hard-case share is stable, attenuated, or materially shifted>`",
            "- Hard-case overlap/stability: `<describe pairwise Jaccard, directional overlap, and persistence across aggregations>`",
            "- Final interpretation: `<state stable / partially weakened / materially aggregation-sensitive and justify briefly>`",
            "",
        ]
    )


def _load_required_comparison_table(
    comparison_root: Path,
    filename: str,
) -> pd.DataFrame:
    path = Path(comparison_root) / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing comparison artifact: {path}")
    return read_dataframe(path)


def _select_single_row(
    frame: pd.DataFrame,
    *,
    source_name: str,
    **filters: object,
) -> pd.Series:
    filtered = frame.copy()
    for column, expected_value in filters.items():
        if column not in filtered.columns:
            raise KeyError(f"{source_name} is missing required column {column!r}.")
        if pd.isna(expected_value):
            filtered = filtered[filtered[column].isna()]
        else:
            filtered = filtered[filtered[column].astype("string").eq(str(expected_value))]
    if filtered.shape[0] != 1:
        raise ValueError(
            f"Expected exactly one row in {source_name} for filters {filters}, "
            f"found {filtered.shape[0]}."
        )
    return filtered.iloc[0]


def _classify_temporal_sensitivity(
    *,
    logistic_24h_metrics: pd.DataFrame,
    logistic_24h_risk: pd.DataFrame,
    logistic_24h_hard_case: pd.DataFrame,
    pairwise_overlap: pd.DataFrame,
) -> tuple[str, list[str]]:
    non_reference_metrics = logistic_24h_metrics[
        ~logistic_24h_metrics["aggregation"].astype("string").eq("8h")
    ].copy()
    non_reference_hard_case = logistic_24h_hard_case[
        ~logistic_24h_hard_case["aggregation"].astype("string").eq("8h")
    ].copy()

    max_abs_auroc_delta = float(
        non_reference_metrics["auroc_delta_vs_8h"].abs().max()
    ) if not non_reference_metrics.empty else 0.0
    max_abs_auprc_delta = float(
        non_reference_metrics["auprc_delta_vs_8h"].abs().max()
    ) if not non_reference_metrics.empty else 0.0
    max_abs_slope_delta = float(
        non_reference_metrics["calibration_slope_delta_vs_8h"].abs().max()
    ) if not non_reference_metrics.empty else 0.0
    max_abs_hard_case_share_delta = float(
        non_reference_hard_case["pct_fatal_hard_cases_delta_vs_8h"].abs().max()
    ) if not non_reference_hard_case.empty else 0.0
    all_structures_ordered = bool(
        logistic_24h_risk["structure_ordered"].astype(bool).all()
    ) if not logistic_24h_risk.empty else False
    min_jaccard = float(pairwise_overlap["jaccard_index"].min()) if not pairwise_overlap.empty else 0.0

    reasons = [
        f"max |AUROC delta|={max_abs_auroc_delta:.3f}",
        f"max |AUPRC delta|={max_abs_auprc_delta:.3f}",
        f"max |slope delta|={max_abs_slope_delta:.3f}",
        f"max |hard-case share delta|={max_abs_hard_case_share_delta:.3f}",
        f"min pairwise Jaccard={min_jaccard:.3f}",
        f"risk curves ordered across aggregations={all_structures_ordered}",
    ]

    if (
        all_structures_ordered
        and max_abs_auroc_delta <= 0.010
        and max_abs_auprc_delta <= 0.040
        and max_abs_slope_delta <= 0.050
        and max_abs_hard_case_share_delta <= 0.040
        and min_jaccard >= 0.60
    ):
        return "stable under coarsening", reasons

    if (
        all_structures_ordered
        and max_abs_auroc_delta <= 0.030
        and max_abs_auprc_delta <= 0.100
        and max_abs_slope_delta <= 0.100
        and max_abs_hard_case_share_delta <= 0.080
        and min_jaccard >= 0.40
    ):
        return "partially weakened under coarsening", reasons

    return "materially aggregation-sensitive", reasons


def _build_interpretation_memo(
    *,
    comparison_root: Path,
    reporting_metric_summary: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    mortality_risk_structure_summary: pd.DataFrame,
    hard_case_prevalence_summary: pd.DataFrame,
    pairwise_overlap: pd.DataFrame,
    directional_overlap: pd.DataFrame,
    persistence_distribution: pd.DataFrame,
) -> str:
    aggregation_labels = sorted(
        reporting_metric_summary["aggregation"].astype("string").unique().tolist(),
        key=lambda value: int(str(value).removesuffix("h")),
    )

    logistic_24h_metrics = reporting_metric_summary[
        reporting_metric_summary["model_name"].astype("string").eq(PRIMARY_MODEL_NAME)
        & reporting_metric_summary["horizon_h"].astype(int).eq(PRIMARY_HORIZON_HOURS)
    ].sort_values("aggregation", key=lambda series: series.map({label: i for i, label in enumerate(aggregation_labels)}))
    xgboost_24h_metrics = reporting_metric_summary[
        reporting_metric_summary["model_name"].astype("string").eq(SECONDARY_MODEL_NAME)
        & reporting_metric_summary["horizon_h"].astype(int).eq(PRIMARY_HORIZON_HOURS)
    ].sort_values("aggregation", key=lambda series: series.map({label: i for i, label in enumerate(aggregation_labels)}))
    logistic_24h_calibration = calibration_summary[
        calibration_summary["model_name"].astype("string").eq(PRIMARY_MODEL_NAME)
        & calibration_summary["horizon_h"].astype(int).eq(PRIMARY_HORIZON_HOURS)
    ].sort_values("aggregation", key=lambda series: series.map({label: i for i, label in enumerate(aggregation_labels)}))
    logistic_24h_risk = mortality_risk_structure_summary[
        mortality_risk_structure_summary["model_name"].astype("string").eq(PRIMARY_MODEL_NAME)
        & mortality_risk_structure_summary["horizon_h"].astype(int).eq(PRIMARY_HORIZON_HOURS)
    ].sort_values("aggregation", key=lambda series: series.map({label: i for i, label in enumerate(aggregation_labels)}))
    xgboost_24h_risk = mortality_risk_structure_summary[
        mortality_risk_structure_summary["model_name"].astype("string").eq(SECONDARY_MODEL_NAME)
        & mortality_risk_structure_summary["horizon_h"].astype(int).eq(PRIMARY_HORIZON_HOURS)
    ].sort_values("aggregation", key=lambda series: series.map({label: i for i, label in enumerate(aggregation_labels)}))
    logistic_24h_hard_case = hard_case_prevalence_summary[
        hard_case_prevalence_summary["model_name"].astype("string").eq(PRIMARY_MODEL_NAME)
        & hard_case_prevalence_summary["horizon_h"].astype(int).eq(PRIMARY_HORIZON_HOURS)
    ].sort_values("aggregation", key=lambda series: series.map({label: i for i, label in enumerate(aggregation_labels)}))

    classification, classification_reasons = _classify_temporal_sensitivity(
        logistic_24h_metrics=logistic_24h_metrics,
        logistic_24h_risk=logistic_24h_risk,
        logistic_24h_hard_case=logistic_24h_hard_case,
        pairwise_overlap=pairwise_overlap,
    )

    logistic_24h_summary = " -> ".join(
        f"{row.aggregation}: AUROC {_metric_text(row.auroc)}, AUPRC {_metric_text(row.auprc)}"
        for row in logistic_24h_metrics.itertuples(index=False)
    )
    logistic_24h_calibration_summary = " -> ".join(
        f"{row.aggregation}: intercept {_metric_text(row.calibration_intercept)}, slope {_metric_text(row.calibration_slope)}, Brier {_metric_text(row.brier_score)}"
        for row in logistic_24h_calibration.itertuples(index=False)
    )
    logistic_24h_risk_summary = " -> ".join(
        f"{row.aggregation}: upper-half event share {_metric_text(row.upper_half_event_share)}, top-bin mortality {_metric_text(row.top_bin_observed_mortality)}"
        for row in logistic_24h_risk.itertuples(index=False)
    )
    logistic_24h_hard_case_summary = " -> ".join(
        f"{row.aggregation}: {int(row.n_hard_cases)}/{int(row.n_fatal_last_points)} = {_metric_text(row.pct_fatal_hard_cases)}"
        for row in logistic_24h_hard_case.itertuples(index=False)
    )
    xgboost_24h_summary = " -> ".join(
        f"{row.aggregation}: AUROC {_metric_text(row.auroc)}, AUPRC {_metric_text(row.auprc)}"
        for row in xgboost_24h_metrics.itertuples(index=False)
    )
    xgboost_24h_risk_summary = " -> ".join(
        f"{row.aggregation}: upper-half event share {_metric_text(row.upper_half_event_share)}"
        for row in xgboost_24h_risk.itertuples(index=False)
    )

    overlap_lines = [
        f"- {row.aggregation_a} vs {row.aggregation_b}: Jaccard {_metric_text(row.jaccard_index)}, matched fatal denominator {int(row.matched_fatal_n)}, intersection {int(row.intersection_n)}."
        for row in pairwise_overlap.itertuples(index=False)
    ]
    directional_lines = [
        f"- {row.aggregation_from} -> {row.aggregation_to}: {_metric_text(row.overlap_from_A_to_B)}"
        for row in directional_overlap.itertuples(index=False)
    ]
    persistence_lines = [
        f"- hard in {int(row.hard_case_aggregation_n)} aggregations: {int(row.fatal_stay_count)} fatal stays ({_metric_text(row.fatal_stay_share)})"
        for row in persistence_distribution.itertuples(index=False)
    ]

    all_test = bool(
        logistic_24h_metrics["selected_split"].astype("string").eq("test").all()
        and logistic_24h_metrics["selected_split_evaluable"].astype(bool).all()
    )
    all_structures_ordered = bool(logistic_24h_risk["structure_ordered"].astype(bool).all())

    lines = [
        "# ASIC Temporal Aggregation Sensitivity Interpretation",
        "",
        "This memo is generated deterministically from the saved temporal sensitivity comparison artifacts.",
        "",
        "## Decision Label",
        "",
        f"- Classification: `{classification}`",
        "- Reference aggregation: `8h`.",
        f"- Coarsened sensitivities reviewed: `{', '.join(label for label in aggregation_labels if label != '8h')}`.",
        "",
        "## Primary Anchor",
        "",
        "- Model: `logistic_regression`.",
        "- Horizon: `24h`.",
        (
            "- Reporting used the binary-evaluable test split for all primary 24h comparisons."
            if all_test
            else "- At least one primary 24h comparison did not use the binary-evaluable test split."
        ),
        "",
        "## Evidence Base",
        "",
        f"- Reporting metrics: `{(comparison_root / 'reporting_metric_summary.csv').name}`",
        f"- Calibration summary: `{(comparison_root / 'calibration_summary.csv').name}`",
        f"- Mortality-vs-risk summary: `{(comparison_root / 'mortality_risk_structure_summary.csv').name}`",
        f"- Hard-case prevalence summary: `{(comparison_root / 'hard_case_prevalence_summary.csv').name}`",
        f"- Logistic 24h hard-case overlap: `{(comparison_root / 'logistic_24h_hard_case_pairwise_overlap.csv').name}` and `{(comparison_root / 'logistic_24h_hard_case_directional_overlap.csv').name}`",
        "",
        "## Findings",
        "",
        f"- Logistic 24h discrimination: {logistic_24h_summary}.",
        f"- Logistic 24h calibration: {logistic_24h_calibration_summary}.",
        (
            f"- Logistic 24h mortality-vs-risk structure remained qualitatively ordered across all aggregations: {logistic_24h_risk_summary}."
            if all_structures_ordered
            else f"- Logistic 24h mortality-vs-risk structure did not remain cleanly ordered across all aggregations: {logistic_24h_risk_summary}."
        ),
        f"- Logistic 24h low-predicted fatal-case prevalence: {logistic_24h_hard_case_summary}.",
        f"- XGBoost 24h compact robustness check: {xgboost_24h_summary}.",
        f"- XGBoost 24h risk concentration: {xgboost_24h_risk_summary}.",
        "",
        "## Hard-Case Stability",
        "",
        *overlap_lines,
        "",
        "Directional overlap:",
        *directional_lines,
        "",
        "Persistence distribution across aggregations:",
        *persistence_lines,
        "",
        "## Interpretation",
        "",
        (
            "Calibration and 24h mortality-vs-risk structure remain broadly intact under coarsening, but the primary logistic 24h signal is not perfectly invariant. "
            "Compared with 8h, AUPRC declines under both coarser aggregations and the logistic 24h hard-case share rises, while hard-case membership overlap remains substantial but incomplete."
            if classification == "partially weakened under coarsening"
            else "The primary logistic 24h pattern remains broadly stable under coarsening across the saved metrics, risk-structure summaries, and hard-case overlap outputs."
            if classification == "stable under coarsening"
            else "The primary logistic 24h pattern changes enough under coarsening that the Chapter 1 interpretation should be treated as materially aggregation-sensitive."
        ),
        "This should be interpreted as a bounded coarsening sensitivity only, not as evidence for an optimal aggregation choice.",
        "",
        "## Classification Rule Triggered",
        "",
        *[f"- {reason}" for reason in classification_reasons],
        "",
    ]
    return "\n".join(lines)


def write_temporal_sensitivity_interpretation_memo(
    *,
    comparison_root: Path,
    output_path: Path | None = None,
) -> Path:
    comparison_root = Path(comparison_root)
    reporting_metric_summary = _load_required_comparison_table(
        comparison_root,
        "reporting_metric_summary.csv",
    )
    calibration_summary = _load_required_comparison_table(
        comparison_root,
        "calibration_summary.csv",
    )
    mortality_risk_structure_summary = _load_required_comparison_table(
        comparison_root,
        "mortality_risk_structure_summary.csv",
    )
    hard_case_prevalence_summary = _load_required_comparison_table(
        comparison_root,
        "hard_case_prevalence_summary.csv",
    )
    pairwise_overlap = _load_required_comparison_table(
        comparison_root,
        "logistic_24h_hard_case_pairwise_overlap.csv",
    )
    directional_overlap = _load_required_comparison_table(
        comparison_root,
        "logistic_24h_hard_case_directional_overlap.csv",
    )
    persistence_distribution = _load_required_comparison_table(
        comparison_root,
        "logistic_24h_hard_case_persistence_distribution.csv",
    )

    resolved_output_path = (
        Path(output_path)
        if output_path is not None
        else comparison_root / "temporal_aggregation_sensitivity_interpretation.md"
    )
    written_path = write_text(
        _build_interpretation_memo(
            comparison_root=comparison_root,
            reporting_metric_summary=reporting_metric_summary,
            calibration_summary=calibration_summary,
            mortality_risk_structure_summary=mortality_risk_structure_summary,
            hard_case_prevalence_summary=hard_case_prevalence_summary,
            pairwise_overlap=pairwise_overlap,
            directional_overlap=directional_overlap,
            persistence_distribution=persistence_distribution,
        ),
        resolved_output_path,
    )
    manifest_path = comparison_root / "run_manifest.json"
    if manifest_path.exists():
        manifest_payload = json.loads(manifest_path.read_text())
        artifact_paths = manifest_payload.get("artifact_paths")
        if not isinstance(artifact_paths, dict):
            artifact_paths = {}
            manifest_payload["artifact_paths"] = artifact_paths
        artifact_paths["temporal_aggregation_sensitivity_interpretation"] = str(
            written_path.resolve()
        )
        _write_json(manifest_payload, manifest_path)
    return written_path


def _run_single_coarsened_aggregation(
    *,
    aggregation_hours: int,
    standardized_inputs: dict[str, pd.DataFrame],
    standardized_input_dir: Path,
    standardized_input_format: str,
    output_root: Path,
    output_format: str,
    frozen_chapter1_dir: Path,
    selected_horizons: tuple[int, ...],
    models: tuple[str, ...],
    config,
) -> TemporalSensitivityAggregationRunResult:
    aggregation_label = _aggregation_label(aggregation_hours)
    aggregation_root = _aggregation_output_root(output_root, aggregation_hours)
    preprocessing_root = aggregation_root / "preprocessing"
    baseline_root = aggregation_root / "baselines" / "asic" / "primary_medians"
    evaluation_root = aggregation_root / "evaluation" / "asic" / "baselines" / "primary_medians"
    hard_case_root = (
        aggregation_root
        / "evaluation"
        / "asic"
        / "hard_cases"
        / "primary_medians"
        / "logistic_regression"
    )
    ensure_directory(preprocessing_root)

    block_artifacts = build_asic_temporal_block_artifacts(
        dynamic_harmonized=standardized_inputs["dynamic_harmonized"],
        reference_stay_block_counts=standardized_inputs["reference_stay_block_counts"],
        block_hours=aggregation_hours,
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
        aggregation_label=aggregation_label,
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

    extension = _output_extension(output_format)
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
        "proxy_horizon_labels": write_dataframe(
            labels.labels,
            preprocessing_root / "labels" / f"chapter1_proxy_horizon_labels.{extension}",
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
            preprocessing_root / "splits" / f"{SPLIT_ALIGNMENT_FILENAME}.{extension}",
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
        "mech_vent_stay_level_qc": write_dataframe(
            standardized_inputs["mech_vent_stay_level_qc"],
            preprocessing_root / "qc" / f"mech_vent_ge_24h_stay_level.{extension}",
            output_format=output_format,
        ),
        "mech_vent_episode_level": write_dataframe(
            standardized_inputs["mech_vent_episode_level"],
            preprocessing_root / "qc" / f"mech_vent_ge_24h_episode_level.{extension}",
            output_format=output_format,
        ),
        "generation_note": write_text(
            _coarsened_generation_note(
                aggregation_hours=aggregation_hours,
                input_dir=standardized_input_dir,
                frozen_chapter1_dir=Path(frozen_chapter1_dir),
                frozen_split_assignments_path=frozen_split_assignments_path,
            ),
            preprocessing_root / "generation_note.md",
        ),
    }
    preprocessing_paths["generation_manifest"] = _write_json(
        {
            "timestamp_utc": _utc_timestamp(),
            "aggregation_hours": aggregation_hours,
            "aggregation_label": aggregation_label,
            "reference_aggregation_hours": DEFAULT_REFERENCE_AGGREGATION_HOURS,
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
        preprocessing_root / "generation_manifest.json",
    )

    if PRIMARY_MODEL_NAME in models:
        run_asic_primary_logistic_regression(
            input_dataset_path=preprocessing_paths["primary_model_ready_dataset"],
            feature_set_definition_path=preprocessing_paths["feature_set_definition"],
            output_dir=baseline_root / PRIMARY_MODEL_NAME,
            horizons=selected_horizons,
            preprocessing_root=preprocessing_root,
            standardized_input_dir=standardized_input_dir,
            standardized_input_format=standardized_input_format,
        )
    if SECONDARY_MODEL_NAME in models:
        run_asic_primary_xgboost(
            input_dataset_path=preprocessing_paths["primary_model_ready_dataset"],
            feature_set_definition_path=preprocessing_paths["feature_set_definition"],
            output_dir=baseline_root / SECONDARY_MODEL_NAME,
            horizons=selected_horizons,
            preprocessing_root=preprocessing_root,
            standardized_input_dir=standardized_input_dir,
            standardized_input_format=standardized_input_format,
        )

    run_asic_baseline_evaluation(
        input_root=baseline_root,
        output_dir=evaluation_root,
        models=models,
        horizons=selected_horizons,
        primary_horizon=PRIMARY_HORIZON_HOURS,
    )
    run_asic_logistic_hard_case_definition(
        input_root=baseline_root,
        output_dir=hard_case_root,
        horizons=selected_horizons,
        output_format="csv",
    )

    return TemporalSensitivityAggregationRunResult(
        aggregation_hours=aggregation_hours,
        aggregation_label=aggregation_label,
        output_root=aggregation_root,
        preprocessing_root=preprocessing_root,
        baseline_root=baseline_root,
        evaluation_root=evaluation_root,
        hard_case_root=hard_case_root,
        block_paths=block_paths,
        preprocessing_paths=preprocessing_paths,
    )


def _build_comparison_package(
    *,
    output_root: Path,
    output_format: str,
    standardized_input_dir: Path,
    reference_aggregation_label: str,
    reference_preprocessing_root: Path,
    reference_evaluation_root: Path,
    reference_hard_case_root: Path,
    aggregation_results: Sequence[TemporalSensitivityAggregationRunResult],
    models: tuple[str, ...],
) -> TemporalSensitivityComparisonResult:
    comparison_root = Path(output_root) / "comparison"
    ensure_directory(comparison_root)
    aggregation_labels = [reference_aggregation_label, *[result.aggregation_label for result in aggregation_results]]
    preprocessing_roots_by_aggregation = {
        reference_aggregation_label: Path(reference_preprocessing_root),
        **{result.aggregation_label: result.preprocessing_root for result in aggregation_results},
    }
    evaluation_roots_by_aggregation = {
        reference_aggregation_label: Path(reference_evaluation_root),
        **{result.aggregation_label: result.evaluation_root for result in aggregation_results},
    }
    hard_case_roots_by_aggregation = {
        reference_aggregation_label: Path(reference_hard_case_root),
        **{result.aggregation_label: result.hard_case_root for result in aggregation_results},
    }

    preprocessing_count_comparison = _build_preprocessing_count_comparison(
        preprocessing_roots_by_aggregation=preprocessing_roots_by_aggregation,
        output_format=output_format,
        reference_aggregation_label=reference_aggregation_label,
    )
    preprocessing_count_comparison_path = write_dataframe(
        preprocessing_count_comparison,
        comparison_root / "preprocessing_count_comparison.csv",
        output_format="csv",
    )

    reporting_metric_frames = [
        _selected_split_metric_rows(
            evaluation_root,
            aggregation_label=aggregation_label,
        )
        for aggregation_label, evaluation_root in evaluation_roots_by_aggregation.items()
    ]
    reporting_metric_summary = pd.concat(reporting_metric_frames, ignore_index=True).sort_values(
        ["model_name", "horizon_h", "aggregation"],
        kind="stable",
    ).reset_index(drop=True)
    reporting_metric_summary = _append_reference_deltas(
        reporting_metric_summary,
        key_columns=["model_name", "horizon_h"],
        reference_aggregation_label=reference_aggregation_label,
        numeric_columns=[
            "sample_count",
            "event_count",
            "non_event_count",
            "event_rate",
            "auroc",
            "auprc",
            "calibration_intercept",
            "calibration_slope",
            "brier_score",
        ],
    )
    reporting_metric_summary_path = write_dataframe(
        reporting_metric_summary,
        comparison_root / "reporting_metric_summary.csv",
        output_format="csv",
    )

    selected_split_summary = reporting_metric_summary[
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
        ]
    ].copy()
    selected_split_summary_path = write_dataframe(
        selected_split_summary,
        comparison_root / "selected_split_summary.csv",
        output_format="csv",
    )

    calibration_summary = reporting_metric_summary[
        [
            "model_name",
            "horizon_h",
            "aggregation",
            "selected_split",
            "selected_split_evaluable",
            "calibration_intercept",
            "calibration_slope",
            "brier_score",
            f"calibration_intercept_delta_vs_{reference_aggregation_label}",
            f"calibration_slope_delta_vs_{reference_aggregation_label}",
            f"brier_score_delta_vs_{reference_aggregation_label}",
            "binary_metrics_evaluable",
            "metric_notes",
        ]
    ].copy()
    calibration_summary_path = write_dataframe(
        calibration_summary,
        comparison_root / "calibration_summary.csv",
        output_format="csv",
    )

    risk_structure_rows: list[dict[str, object]] = []
    figure_paths: dict[str, Path] = {}
    for model_name in models:
        summaries_by_aggregation = {
            aggregation_label: _selected_risk_summary(
                evaluation_roots_by_aggregation[aggregation_label],
                model_name=model_name,
                horizon_h=PRIMARY_HORIZON_HOURS,
            )
            for aggregation_label in aggregation_labels
        }
        metrics_by_aggregation = {
            aggregation_label: reporting_metric_summary[
                reporting_metric_summary["aggregation"].astype("string").eq(aggregation_label)
                & reporting_metric_summary["model_name"].astype("string").eq(model_name)
                & reporting_metric_summary["horizon_h"].astype(int).eq(PRIMARY_HORIZON_HOURS)
            ]
            .iloc[0]
            .to_dict()
            for aggregation_label in aggregation_labels
            if not reporting_metric_summary[
                reporting_metric_summary["aggregation"].astype("string").eq(aggregation_label)
                & reporting_metric_summary["model_name"].astype("string").eq(model_name)
                & reporting_metric_summary["horizon_h"].astype(int).eq(PRIMARY_HORIZON_HOURS)
            ].empty
        }
        for aggregation_label in aggregation_labels:
            signature = _risk_structure_signature(summaries_by_aggregation[aggregation_label])
            metrics = metrics_by_aggregation.get(aggregation_label, {})
            risk_structure_rows.append(
                {
                    "aggregation": aggregation_label,
                    "model_name": model_name,
                    "horizon_h": PRIMARY_HORIZON_HOURS,
                    "selected_split": metrics.get("selected_split"),
                    "sample_count": metrics.get("sample_count"),
                    "event_count": metrics.get("event_count"),
                    "lower_half_event_rate": None if signature is None else signature["lower_half_event_rate"],
                    "upper_half_event_rate": None if signature is None else signature["upper_half_event_rate"],
                    "upper_half_event_share": None if signature is None else signature["upper_half_event_share"],
                    "top_bin_observed_mortality": None if signature is None else signature["top_bin_observed_mortality"],
                    "bottom_bin_observed_mortality": None if signature is None else signature["bottom_bin_observed_mortality"],
                    "structure_ordered": (
                        False
                        if signature is None
                        else bool(
                            signature["upper_half_event_rate"] > signature["lower_half_event_rate"]
                            and signature["top_bin_observed_mortality"] >= signature["bottom_bin_observed_mortality"]
                        )
                    ),
                }
            )

        reliability_path = _plot_reliability_triptych(
            summaries_by_aggregation,
            metrics_by_aggregation,
            aggregation_labels=aggregation_labels,
            model_name=model_name,
            horizon_h=PRIMARY_HORIZON_HOURS,
            output_path=comparison_root
            / f"{model_name}_24h_reliability_{'_vs_'.join(aggregation_labels)}.png",
        )
        figure_paths[f"{model_name}_reliability"] = reliability_path
        mortality_vs_risk_path = _plot_mortality_vs_risk_triptych(
            summaries_by_aggregation,
            metrics_by_aggregation,
            aggregation_labels=aggregation_labels,
            model_name=model_name,
            horizon_h=PRIMARY_HORIZON_HOURS,
            output_path=comparison_root
            / f"{model_name}_24h_mortality_vs_risk_{'_vs_'.join(aggregation_labels)}.png",
        )
        figure_paths[f"{model_name}_mortality_vs_risk"] = mortality_vs_risk_path

    mortality_risk_structure_summary = pd.DataFrame(risk_structure_rows).sort_values(
        ["model_name", "aggregation"],
        kind="stable",
    ).reset_index(drop=True)
    mortality_risk_structure_summary = _append_reference_deltas(
        mortality_risk_structure_summary,
        key_columns=["model_name", "horizon_h"],
        reference_aggregation_label=reference_aggregation_label,
        numeric_columns=[
            "sample_count",
            "event_count",
            "lower_half_event_rate",
            "upper_half_event_rate",
            "upper_half_event_share",
            "top_bin_observed_mortality",
            "bottom_bin_observed_mortality",
        ],
    )
    mortality_risk_structure_summary_path = write_dataframe(
        mortality_risk_structure_summary,
        comparison_root / "mortality_risk_structure_summary.csv",
        output_format="csv",
    )

    hard_case_prevalence_summary = _build_hard_case_prevalence_summary(
        hard_case_roots_by_aggregation=hard_case_roots_by_aggregation,
        reference_aggregation_label=reference_aggregation_label,
    )
    hard_case_prevalence_summary_path = write_dataframe(
        hard_case_prevalence_summary,
        comparison_root / "hard_case_prevalence_summary.csv",
        output_format="csv",
    )

    hard_case_overlap_inputs = pd.concat(
        [
            _load_hard_case_stay_level_for_aggregation(
                hard_case_root,
                aggregation_label=aggregation_label,
                horizon_h=PRIMARY_HORIZON_HOURS,
            )
            for aggregation_label, hard_case_root in hard_case_roots_by_aggregation.items()
        ],
        ignore_index=True,
    )
    pairwise_denominators, pairwise_overlap = _build_aggregation_pairwise_tables(
        hard_case_overlap_inputs,
        aggregation_labels=aggregation_labels,
    )
    directional_overlap = _build_aggregation_directional_overlap(
        pairwise_overlap,
        aggregation_labels=aggregation_labels,
    )
    hard_case_persistence, persistence_distribution = _build_aggregation_persistence_tables(
        hard_case_overlap_inputs,
        aggregation_labels=aggregation_labels,
    )

    pairwise_denominators_path = write_dataframe(
        pairwise_denominators,
        comparison_root / "logistic_24h_hard_case_pairwise_denominators.csv",
        output_format="csv",
    )
    pairwise_overlap_path = write_dataframe(
        pairwise_overlap,
        comparison_root / "logistic_24h_hard_case_pairwise_overlap.csv",
        output_format="csv",
    )
    directional_overlap_path = write_dataframe(
        directional_overlap,
        comparison_root / "logistic_24h_hard_case_directional_overlap.csv",
        output_format="csv",
    )
    hard_case_persistence_path = write_dataframe(
        hard_case_persistence,
        comparison_root / "logistic_24h_hard_case_persistence.csv",
        output_format="csv",
    )
    persistence_distribution_path = write_dataframe(
        persistence_distribution,
        comparison_root / "logistic_24h_hard_case_persistence_distribution.csv",
        output_format="csv",
    )

    jaccard_matrix, directional_matrix = _build_heatmap_matrix(
        pairwise_overlap,
        directional_overlap,
        aggregation_labels=aggregation_labels,
    )
    jaccard_heatmap_path = _plot_heatmap(
        jaccard_matrix,
        title="Logistic 24h Hard-Case Jaccard Across Aggregations",
        x_label="Aggregation",
        y_label="Aggregation",
        colorbar_label="Jaccard index",
        output_path=comparison_root / "logistic_24h_hard_case_jaccard_heatmap.png",
    )
    directional_overlap_heatmap_path = _plot_heatmap(
        directional_matrix,
        title="Logistic 24h Hard-Case Directional Overlap Across Aggregations",
        x_label="Aggregation to",
        y_label="Aggregation from",
        colorbar_label="Directional overlap",
        output_path=comparison_root / "logistic_24h_hard_case_directional_overlap_heatmap.png",
    )
    persistence_barplot_path = _plot_persistence_barplot(
        persistence_distribution,
        output_path=comparison_root / "logistic_24h_hard_case_persistence_barplot.png",
    )
    hard_case_overlap_note_path = write_text(
        _build_aggregation_overlap_note(
            horizon_h=PRIMARY_HORIZON_HOURS,
            pairwise_denominators=pairwise_denominators,
            pairwise_overlap=pairwise_overlap,
            directional_overlap=directional_overlap,
            persistence_distribution=persistence_distribution,
        ),
        comparison_root / "logistic_24h_hard_case_overlap_note.md",
    )

    split_alignment_overview_frames = []
    for result in aggregation_results:
        split_alignment_overview_frames.append(
            read_dataframe(result.preprocessing_paths["split_alignment_summary"])
        )
    split_alignment_overview = (
        pd.concat(split_alignment_overview_frames, ignore_index=True)
        if split_alignment_overview_frames
        else pd.DataFrame(columns=["aggregation", "check_id", "passed", "count", "detail"])
    )
    split_alignment_overview_path = write_dataframe(
        split_alignment_overview,
        comparison_root / "split_alignment_overview.csv",
        output_format="csv",
    )

    provenance_note_path = write_text(
        _build_provenance_and_limitations_note(
            standardized_input_dir=standardized_input_dir,
            frozen_chapter1_dir=reference_preprocessing_root,
            reference_evaluation_root=reference_evaluation_root,
            reference_hard_case_dir=reference_hard_case_root,
            aggregation_results=aggregation_results,
            selected_split_summary=selected_split_summary,
            split_alignment_overview=split_alignment_overview,
        ),
        comparison_root / "provenance_and_limitations.md",
    )
    supersession_note_path = write_text(
        _build_supersession_note(),
        comparison_root / "supersession_note.md",
    )
    memo_template_path = write_text(
        _build_interpretation_memo_template(
            comparison_root=comparison_root,
            aggregation_labels=aggregation_labels,
        ),
        comparison_root / "interpretation_memo_template.md",
    )
    interpretation_memo_path = write_temporal_sensitivity_interpretation_memo(
        comparison_root=comparison_root,
    )

    manifest_path = _write_json(
        {
            "timestamp_utc": _utc_timestamp(),
            "comparison_root": str(comparison_root.resolve()),
            "reference_aggregation_label": reference_aggregation_label,
            "aggregation_labels": aggregation_labels,
            "models": list(models),
            "primary_horizon_hours": PRIMARY_HORIZON_HOURS,
            "reference_preprocessing_root": str(Path(reference_preprocessing_root).resolve()),
            "reference_evaluation_root": str(Path(reference_evaluation_root).resolve()),
            "reference_hard_case_root": str(Path(reference_hard_case_root).resolve()),
            "aggregation_outputs": {
                result.aggregation_label: {
                    "preprocessing_root": str(result.preprocessing_root.resolve()),
                    "evaluation_root": str(result.evaluation_root.resolve()),
                    "hard_case_root": str(result.hard_case_root.resolve()),
                }
                for result in aggregation_results
            },
            "artifact_paths": {
                "preprocessing_count_comparison": str(preprocessing_count_comparison_path.resolve()),
                "reporting_metric_summary": str(reporting_metric_summary_path.resolve()),
                "selected_split_summary": str(selected_split_summary_path.resolve()),
                "calibration_summary": str(calibration_summary_path.resolve()),
                "mortality_risk_structure_summary": str(mortality_risk_structure_summary_path.resolve()),
                "hard_case_prevalence_summary": str(hard_case_prevalence_summary_path.resolve()),
                "logistic_24h_hard_case_pairwise_denominators": str(pairwise_denominators_path.resolve()),
                "logistic_24h_hard_case_pairwise_overlap": str(pairwise_overlap_path.resolve()),
                "logistic_24h_hard_case_directional_overlap": str(directional_overlap_path.resolve()),
                "logistic_24h_hard_case_persistence": str(hard_case_persistence_path.resolve()),
                "logistic_24h_hard_case_persistence_distribution": str(persistence_distribution_path.resolve()),
                "logistic_24h_hard_case_jaccard_heatmap": str(jaccard_heatmap_path.resolve()),
                "logistic_24h_hard_case_directional_overlap_heatmap": str(
                    directional_overlap_heatmap_path.resolve()
                ),
                "logistic_24h_hard_case_persistence_barplot": str(persistence_barplot_path.resolve()),
                "logistic_24h_hard_case_overlap_note": str(hard_case_overlap_note_path.resolve()),
                "split_alignment_overview": str(split_alignment_overview_path.resolve()),
                "provenance_and_limitations": str(provenance_note_path.resolve()),
                "supersession_note": str(supersession_note_path.resolve()),
                "interpretation_memo_template": str(memo_template_path.resolve()),
                "temporal_aggregation_sensitivity_interpretation": str(
                    interpretation_memo_path.resolve()
                ),
                **{key: str(path.resolve()) for key, path in figure_paths.items()},
            },
        },
        comparison_root / "run_manifest.json",
    )

    artifact_paths = {
        "preprocessing_count_comparison": preprocessing_count_comparison_path,
        "reporting_metric_summary": reporting_metric_summary_path,
        "selected_split_summary": selected_split_summary_path,
        "calibration_summary": calibration_summary_path,
        "mortality_risk_structure_summary": mortality_risk_structure_summary_path,
        "hard_case_prevalence_summary": hard_case_prevalence_summary_path,
        "logistic_24h_hard_case_pairwise_denominators": pairwise_denominators_path,
        "logistic_24h_hard_case_pairwise_overlap": pairwise_overlap_path,
        "logistic_24h_hard_case_directional_overlap": directional_overlap_path,
        "logistic_24h_hard_case_persistence": hard_case_persistence_path,
        "logistic_24h_hard_case_persistence_distribution": persistence_distribution_path,
        "logistic_24h_hard_case_jaccard_heatmap": jaccard_heatmap_path,
        "logistic_24h_hard_case_directional_overlap_heatmap": directional_overlap_heatmap_path,
        "logistic_24h_hard_case_persistence_barplot": persistence_barplot_path,
        "logistic_24h_hard_case_overlap_note": hard_case_overlap_note_path,
        "split_alignment_overview": split_alignment_overview_path,
        "provenance_and_limitations": provenance_note_path,
        "supersession_note": supersession_note_path,
        "interpretation_memo_template": memo_template_path,
        "temporal_aggregation_sensitivity_interpretation": interpretation_memo_path,
        "run_manifest": manifest_path,
        **figure_paths,
    }
    return TemporalSensitivityComparisonResult(
        comparison_root=comparison_root,
        artifact_paths=artifact_paths,
    )


def run_asic_temporal_aggregation_sensitivity(
    *,
    run_config_path: Path | None = None,
    input_dir: Path | None = None,
    input_format: str | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    output_format: str = "csv",
    frozen_chapter1_dir: Path = DEFAULT_FROZEN_CHAPTER1_DIR,
    reference_evaluation_root: Path = DEFAULT_REFERENCE_EVALUATION_ROOT,
    reference_hard_case_root: Path = DEFAULT_REFERENCE_HARD_CASE_DIR,
    sensitivity_block_hours: Sequence[int] | None = None,
    horizons: Sequence[int] | None = None,
    models: Sequence[str] | None = None,
) -> TemporalSensitivityRunResult:
    _require_matplotlib()
    run_config = load_chapter1_run_config(run_config_path) if run_config_path else load_chapter1_run_config()
    standardized_input_dir = Path(input_dir or run_config.input_dir)
    standardized_input_format = input_format or run_config.input_format
    selected_horizons = tuple(int(horizon) for horizon in (horizons or DEFAULT_HORIZONS))
    normalized_models = _normalize_models(models)
    normalized_block_hours = _normalize_sensitivity_block_hours(sensitivity_block_hours)
    output_root = Path(output_root)
    reference_label = _aggregation_label(DEFAULT_REFERENCE_AGGREGATION_HOURS)

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

    aggregation_results = tuple(
        _run_single_coarsened_aggregation(
            aggregation_hours=aggregation_hours,
            standardized_inputs=standardized_inputs,
            standardized_input_dir=standardized_input_dir,
            standardized_input_format=standardized_input_format,
            output_root=output_root,
            output_format=output_format,
            frozen_chapter1_dir=Path(frozen_chapter1_dir),
            selected_horizons=selected_horizons,
            models=normalized_models,
            config=config,
        )
        for aggregation_hours in normalized_block_hours
    )
    comparison = _build_comparison_package(
        output_root=output_root,
        output_format=output_format,
        standardized_input_dir=standardized_input_dir,
        reference_aggregation_label=reference_label,
        reference_preprocessing_root=Path(frozen_chapter1_dir),
        reference_evaluation_root=Path(reference_evaluation_root),
        reference_hard_case_root=Path(reference_hard_case_root),
        aggregation_results=aggregation_results,
        models=normalized_models,
    )
    return TemporalSensitivityRunResult(
        output_root=output_root,
        reference_aggregation_label=reference_label,
        sensitivity_aggregations=normalized_block_hours,
        models=normalized_models,
        horizons=selected_horizons,
        aggregation_results=aggregation_results,
        comparison=comparison,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the formal ASIC Chapter 1 temporal aggregation coarsening sensitivity "
            "analysis comparing the frozen 8h reference against 16h and 24h alternatives."
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
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output directory for the formal temporal sensitivity package.",
    )
    parser.add_argument(
        "--output-format",
        choices=("csv", "parquet"),
        default="csv",
        help="Output artifact format for preprocessing tables.",
    )
    parser.add_argument(
        "--frozen-chapter1-dir",
        type=Path,
        default=DEFAULT_FROZEN_CHAPTER1_DIR,
        help="Frozen 8h Chapter 1 artifact root containing the stay split assignments and preprocessing summaries.",
    )
    parser.add_argument(
        "--reference-evaluation-root",
        type=Path,
        default=DEFAULT_REFERENCE_EVALUATION_ROOT,
        help="Saved frozen 8h evaluation root used as the primary comparison reference.",
    )
    parser.add_argument(
        "--reference-hard-case-root",
        type=Path,
        default=DEFAULT_REFERENCE_HARD_CASE_DIR,
        help="Saved frozen 8h logistic hard-case root used as the primary comparison reference.",
    )
    parser.add_argument(
        "--sensitivity-block-hours",
        nargs="+",
        type=int,
        help="Coarsened aggregation block sizes in hours. Defaults to 16 24.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        help="Optional subset of horizons to process.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=DEFAULT_MODELS,
        help="Baseline models to run for the coarsened aggregations. Defaults to logistic_regression xgboost.",
    )
    parser.add_argument(
        "--refresh-interpretation-memo-only",
        action="store_true",
        help=(
            "Do not rerun the temporal sensitivity analysis. Regenerate only the written "
            "interpretation memo from an existing comparison directory."
        ),
    )
    parser.add_argument(
        "--comparison-root",
        type=Path,
        help=(
            "Existing temporal sensitivity comparison directory. Required with "
            "--refresh-interpretation-memo-only."
        ),
    )
    parser.add_argument(
        "--memo-output-path",
        type=Path,
        help=(
            "Optional output path for the generated interpretation memo when using "
            "--refresh-interpretation-memo-only."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.refresh_interpretation_memo_only:
        if args.comparison_root is None:
            parser.error("--comparison-root is required with --refresh-interpretation-memo-only.")
        memo_path = write_temporal_sensitivity_interpretation_memo(
            comparison_root=args.comparison_root,
            output_path=args.memo_output_path,
        )
        print(f"Interpretation memo: {memo_path}")
        return 0

    result = run_asic_temporal_aggregation_sensitivity(
        run_config_path=args.run_config,
        input_dir=args.input_dir,
        input_format=args.input_format,
        output_root=args.output_root,
        output_format=args.output_format,
        frozen_chapter1_dir=args.frozen_chapter1_dir,
        reference_evaluation_root=args.reference_evaluation_root,
        reference_hard_case_root=args.reference_hard_case_root,
        sensitivity_block_hours=args.sensitivity_block_hours,
        horizons=args.horizons,
        models=args.models,
    )

    print(f"Temporal sensitivity output root: {result.output_root}")
    print(f"Reference aggregation: {result.reference_aggregation_label}")
    print(f"Coarsened aggregations: {result.sensitivity_aggregations}")
    for aggregation_result in result.aggregation_results:
        print(f"{aggregation_result.aggregation_label} preprocessing root: {aggregation_result.preprocessing_root}")
        print(f"{aggregation_result.aggregation_label} baseline root: {aggregation_result.baseline_root}")
        print(f"{aggregation_result.aggregation_label} evaluation root: {aggregation_result.evaluation_root}")
        print(f"{aggregation_result.aggregation_label} hard-case root: {aggregation_result.hard_case_root}")
    print(f"Comparison root: {result.comparison.comparison_root}")
    for name, path in sorted(result.comparison.artifact_paths.items()):
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
