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
    DEFAULT_BASELINE_ARTIFACT_ROOT,
    DEFAULT_EVALUATION_OUTPUT_DIR,
    DEFAULT_HORIZONS,
    _discover_prediction_artifacts,
    _load_prediction_frame,
    run_asic_baseline_evaluation,
)
from chapter1_mortality_decomposition.baseline_feedforward_nn import (
    DEFAULT_FEEDFORWARD_NN_BASELINE_OUTPUT_DIR,
    DEFAULT_FEEDFORWARD_NN_PARAMETERS,
    DEFAULT_FEEDFORWARD_NN_RANDOM_STATE,
    MODEL_NAME as FEEDFORWARD_NN_MODEL_NAME,
    run_asic_primary_feedforward_nn,
)
from chapter1_mortality_decomposition.baseline_logistic import (
    DEFAULT_FEATURE_SET_DEFINITION_PATH,
    DEFAULT_PRIMARY_MODEL_READY_DATASET_PATH,
)
from chapter1_mortality_decomposition.hard_case_definition import (
    DEFAULT_HARD_CASE_OUTPUT_ROOT,
    build_hard_case_tables_from_prediction_frames,
)
from chapter1_mortality_decomposition.hard_case_agreement import (
    AGREEMENT_JOIN_KEYS,
    AGREEMENT_RULE,
    _validate_agreement_input_stay_level,
)
from chapter1_mortality_decomposition.utils import ensure_directory, read_dataframe, write_dataframe, write_text


SOURCE_LOGISTIC_MODEL_NAME = "logistic_regression"
SOURCE_XGBOOST_MODEL_NAME = "xgboost"
PRIMARY_HORIZON = 24
DEFAULT_BOUNDARY_MARGIN = 0.05
DEFAULT_FEEDFORWARD_NN_HARD_CASE_OUTPUT_DIR = DEFAULT_HARD_CASE_OUTPUT_ROOT / FEEDFORWARD_NN_MODEL_NAME
DEFAULT_FEEDFORWARD_NN_SENSITIVITY_OUTPUT_DIR = (
    DEFAULT_HARD_CASE_OUTPUT_ROOT / "agreement" / "feedforward_nn_sensitivity"
)
FEEDFORWARD_NN_HARD_CASE_RULE = "asic_feedforward_nn_last_eligible_nonfatal_q75_v1"
LOGISTIC_HARD_CASE_RULE = "asic_logistic_last_eligible_nonfatal_q75_v1"
XGBOOST_HARD_CASE_RULE = "asic_xgboost_last_eligible_nonfatal_q75_v1"


@dataclass(frozen=True)
class FeedforwardNNHardCaseSensitivityArtifacts:
    nn_hard_case_stay_level_path: Path
    nn_hard_case_summary_path: Path
    performance_summary_path: Path
    overlap_summary_path: Path
    boundary_summary_path: Path
    three_way_overlap_24h_path: Path | None
    figure_path: Path
    memo_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class FeedforwardNNHardCaseSensitivityRunResult:
    baseline_input_root: Path
    nn_output_dir: Path
    evaluation_output_dir: Path
    hard_case_output_dir: Path
    sensitivity_output_dir: Path
    horizons_processed: tuple[int, ...]
    artifacts: FeedforwardNNHardCaseSensitivityArtifacts
    performance_summary: pd.DataFrame
    overlap_summary: pd.DataFrame
    boundary_summary: pd.DataFrame
    three_way_overlap_24h: pd.DataFrame | None


def _require_matplotlib() -> None:
    if plt is None:
        raise ImportError(
            "matplotlib is required for the Chapter 1 feedforward-NN hard-case sensitivity package."
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
        SOURCE_LOGISTIC_MODEL_NAME: "Logistic Regression",
        SOURCE_XGBOOST_MODEL_NAME: "XGBoost",
        FEEDFORWARD_NN_MODEL_NAME: "Feedforward NN",
    }.get(model_name, model_name.replace("_", " ").title())


def _baseline_predictions_complete(
    baseline_root: Path,
    *,
    model_name: str,
    horizons: Sequence[int],
) -> bool:
    try:
        artifacts = _discover_prediction_artifacts(
            Path(baseline_root),
            models=[model_name],
            horizons=horizons,
        )
    except Exception:
        return False
    return len(artifacts) == len(tuple(horizons))


def _nn_evaluation_complete(
    evaluation_output_dir: Path,
    *,
    horizons: Sequence[int],
) -> bool:
    combined_metrics_path = Path(evaluation_output_dir) / "combined_metrics.csv"
    reporting_summary_path = Path(evaluation_output_dir) / "reporting_split_summary.csv"
    if not combined_metrics_path.exists() or not reporting_summary_path.exists():
        return False

    combined_metrics = read_dataframe(combined_metrics_path)
    reporting_summary = read_dataframe(reporting_summary_path)
    required_horizons = {int(horizon) for horizon in horizons}

    combined_horizons = set(
        pd.to_numeric(
            combined_metrics.loc[
                combined_metrics["model_name"].astype("string").eq(FEEDFORWARD_NN_MODEL_NAME),
                "horizon_h",
            ],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )
    reporting_horizons = set(
        pd.to_numeric(
            reporting_summary.loc[
                reporting_summary["model_name"].astype("string").eq(FEEDFORWARD_NN_MODEL_NAME),
                "horizon_h",
            ],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )
    return required_horizons.issubset(combined_horizons) and required_horizons.issubset(reporting_horizons)


def _load_prediction_frames(
    baseline_root: Path,
    *,
    model_name: str,
    horizons: Sequence[int],
) -> tuple[dict[int, pd.DataFrame], dict[int, str]]:
    artifacts = _discover_prediction_artifacts(
        Path(baseline_root),
        models=[model_name],
        horizons=horizons,
    )
    frames_by_horizon = {
        int(artifact.horizon_h): _load_prediction_frame(artifact)
        for artifact in artifacts
    }
    sources_by_horizon = {
        int(artifact.horizon_h): str(artifact.predictions_path.resolve())
        for artifact in artifacts
    }
    return frames_by_horizon, sources_by_horizon


def _derive_hard_cases(
    prediction_frames_by_horizon: dict[int, pd.DataFrame],
    source_names_by_horizon: dict[int, str],
    *,
    model_name: str,
    hard_case_rule: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stay_level, horizon_summary, _ = build_hard_case_tables_from_prediction_frames(
        prediction_frames_by_horizon,
        source_names_by_horizon=source_names_by_horizon,
        expected_source_model_name=model_name,
        probability_column="predicted_probability",
        output_model_name=model_name,
        hard_case_rule=hard_case_rule,
    )
    return stay_level, horizon_summary


def _prepare_pairwise_fatal_subset(
    stay_level: pd.DataFrame,
    *,
    prefix: str,
) -> pd.DataFrame:
    validated = _validate_agreement_input_stay_level(stay_level, source_name=f"{prefix}_stay_level")
    fatal_subset = validated[validated["label_value"].astype(int).eq(1)].copy()
    renamed = fatal_subset.rename(
        columns={
            "hospital_id": f"{prefix}_hospital_id",
            "label_value": f"{prefix}_label_value",
            "instance_id": f"{prefix}_instance_id",
            "block_index": f"{prefix}_block_index",
            "prediction_time_h": f"{prefix}_prediction_time_h",
            "predicted_probability": f"{prefix}_predicted_probability",
            "nonfatal_q75_threshold": f"{prefix}_nonfatal_q75_threshold",
            "hard_case_flag": f"{prefix}_hard_case_flag",
            "hard_case_rule": f"{prefix}_hard_case_rule",
            "model_name": f"{prefix}_model_name",
        }
    )
    return renamed


def build_pairwise_hard_case_overlap(
    left_stay_level: pd.DataFrame,
    right_stay_level: pd.DataFrame,
    *,
    left_model_name: str,
    right_model_name: str,
    left_prefix: str,
    right_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left_fatal = _prepare_pairwise_fatal_subset(left_stay_level, prefix=left_prefix)
    right_fatal = _prepare_pairwise_fatal_subset(right_stay_level, prefix=right_prefix)
    merged = left_fatal.merge(
        right_fatal,
        on=AGREEMENT_JOIN_KEYS,
        how="outer",
        indicator=True,
    )

    matched = merged[merged["_merge"].eq("both")].copy()
    if matched.empty:
        raise ValueError(
            f"No fatal stay-level overlap exists between {left_model_name} and {right_model_name}."
        )

    hospital_mismatch = matched[
        matched[f"{left_prefix}_hospital_id"].astype("string").ne(
            matched[f"{right_prefix}_hospital_id"].astype("string")
        )
    ]
    if not hospital_mismatch.empty:
        raise ValueError("Hospital ID mismatch detected while joining pairwise hard-case outputs.")

    label_mismatch = matched[
        matched[f"{left_prefix}_label_value"].astype(int).ne(
            matched[f"{right_prefix}_label_value"].astype(int)
        )
    ]
    if not label_mismatch.empty:
        raise ValueError("Label mismatch detected while joining pairwise hard-case outputs.")

    matched["hospital_id"] = matched[f"{left_prefix}_hospital_id"].astype("string")
    matched[f"{left_prefix}_hard_case_flag"] = matched[f"{left_prefix}_hard_case_flag"].astype(bool)
    matched[f"{right_prefix}_hard_case_flag"] = matched[f"{right_prefix}_hard_case_flag"].astype(bool)
    matched["intersection_hard_case_flag"] = (
        matched[f"{left_prefix}_hard_case_flag"] & matched[f"{right_prefix}_hard_case_flag"]
    )
    matched[f"{left_prefix}_only_flag"] = (
        matched[f"{left_prefix}_hard_case_flag"] & ~matched[f"{right_prefix}_hard_case_flag"]
    )
    matched[f"{right_prefix}_only_flag"] = (
        ~matched[f"{left_prefix}_hard_case_flag"] & matched[f"{right_prefix}_hard_case_flag"]
    )
    matched["pairwise_agreement_rule"] = AGREEMENT_RULE

    stay_level_columns = [
        "stay_id_global",
        "hospital_id",
        "horizon_h",
        f"{left_prefix}_model_name",
        f"{left_prefix}_predicted_probability",
        f"{left_prefix}_nonfatal_q75_threshold",
        f"{left_prefix}_hard_case_flag",
        f"{left_prefix}_instance_id",
        f"{left_prefix}_block_index",
        f"{left_prefix}_prediction_time_h",
        f"{left_prefix}_hard_case_rule",
        f"{right_prefix}_model_name",
        f"{right_prefix}_predicted_probability",
        f"{right_prefix}_nonfatal_q75_threshold",
        f"{right_prefix}_hard_case_flag",
        f"{right_prefix}_instance_id",
        f"{right_prefix}_block_index",
        f"{right_prefix}_prediction_time_h",
        f"{right_prefix}_hard_case_rule",
        "intersection_hard_case_flag",
        f"{left_prefix}_only_flag",
        f"{right_prefix}_only_flag",
        "pairwise_agreement_rule",
    ]
    stay_level_overlap = matched[stay_level_columns].sort_values(
        ["horizon_h", "hospital_id", "stay_id_global"],
        kind="stable",
    ).reset_index(drop=True)

    unmatched_counts = (
        merged.assign(
            left_only=lambda frame: frame["_merge"].eq("left_only"),
            right_only=lambda frame: frame["_merge"].eq("right_only"),
        )
        .groupby("horizon_h", dropna=False)
        .agg(
            n_fatal_left_only_available=("left_only", "sum"),
            n_fatal_right_only_available=("right_only", "sum"),
        )
        .reset_index()
        .set_index("horizon_h")
    )
    unmatched_counts["n_fatal_dropped_unmatched"] = (
        unmatched_counts["n_fatal_left_only_available"]
        + unmatched_counts["n_fatal_right_only_available"]
    )

    summary_rows: list[dict[str, object]] = []
    for horizon_h, horizon_df in stay_level_overlap.groupby("horizon_h", dropna=False):
        horizon_key = int(horizon_h)
        n_fatal_with_both = int(horizon_df.shape[0])
        n_left_hard = int(horizon_df[f"{left_prefix}_hard_case_flag"].sum())
        n_right_hard = int(horizon_df[f"{right_prefix}_hard_case_flag"].sum())
        intersection_count = int(horizon_df["intersection_hard_case_flag"].sum())
        union_count = int(
            (
                horizon_df[f"{left_prefix}_hard_case_flag"]
                | horizon_df[f"{right_prefix}_hard_case_flag"]
            ).sum()
        )
        summary_rows.append(
            {
                "pair_name": f"{left_model_name}_vs_{right_model_name}",
                "left_model_name": left_model_name,
                "right_model_name": right_model_name,
                "horizon_h": horizon_key,
                "n_fatal_with_both_models_available": n_fatal_with_both,
                "n_fatal_left_only_available": int(unmatched_counts.loc[horizon_key, "n_fatal_left_only_available"]),
                "n_fatal_right_only_available": int(unmatched_counts.loc[horizon_key, "n_fatal_right_only_available"]),
                "n_fatal_dropped_unmatched": int(unmatched_counts.loc[horizon_key, "n_fatal_dropped_unmatched"]),
                "n_left_hard": n_left_hard,
                "n_right_hard": n_right_hard,
                "intersection_count": intersection_count,
                "union_count": union_count,
                "n_left_only": int(horizon_df[f"{left_prefix}_only_flag"].sum()),
                "n_right_only": int(horizon_df[f"{right_prefix}_only_flag"].sum()),
                "jaccard_agreement": float(intersection_count / union_count) if union_count else np.nan,
                "overlap_share_of_left_hard": (
                    float(intersection_count / n_left_hard) if n_left_hard else np.nan
                ),
                "overlap_share_of_right_hard": (
                    float(intersection_count / n_right_hard) if n_right_hard else np.nan
                ),
            }
        )

    horizon_summary = pd.DataFrame(summary_rows).sort_values(["pair_name", "horizon_h"], kind="stable").reset_index(drop=True)
    return stay_level_overlap, horizon_summary


def summarize_boundary_disagreement(
    stay_level_overlap: pd.DataFrame,
    *,
    left_model_name: str,
    right_model_name: str,
    left_prefix: str,
    right_prefix: str,
    boundary_margin: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    working = stay_level_overlap.copy()
    working["left_threshold_distance"] = (
        working[f"{left_prefix}_predicted_probability"] - working[f"{left_prefix}_nonfatal_q75_threshold"]
    ).abs()
    working["right_threshold_distance"] = (
        working[f"{right_prefix}_predicted_probability"] - working[f"{right_prefix}_nonfatal_q75_threshold"]
    ).abs()
    working["disagreement_flag"] = working[f"{left_prefix}_only_flag"] | working[f"{right_prefix}_only_flag"]

    for horizon_h, horizon_df in working.groupby("horizon_h", dropna=False):
        disagreement_df = horizon_df[horizon_df["disagreement_flag"]].copy()
        n_disagreement = int(disagreement_df.shape[0])
        n_boundary_near_left = int(disagreement_df["left_threshold_distance"].le(boundary_margin).sum())
        n_boundary_near_right = int(disagreement_df["right_threshold_distance"].le(boundary_margin).sum())
        n_boundary_near_either = int(
            (
                disagreement_df["left_threshold_distance"].le(boundary_margin)
                | disagreement_df["right_threshold_distance"].le(boundary_margin)
            ).sum()
        )
        rows.append(
            {
                "pair_name": f"{left_model_name}_vs_{right_model_name}",
                "left_model_name": left_model_name,
                "right_model_name": right_model_name,
                "horizon_h": int(horizon_h),
                "boundary_margin": float(boundary_margin),
                "n_disagreement": n_disagreement,
                "n_boundary_near_left": n_boundary_near_left,
                "n_boundary_near_right": n_boundary_near_right,
                "n_boundary_near_either": n_boundary_near_either,
                "boundary_near_share_of_disagreement": (
                    float(n_boundary_near_either / n_disagreement) if n_disagreement else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["pair_name", "horizon_h"], kind="stable").reset_index(drop=True)


def build_three_way_overlap_24h_summary(
    logistic_stay_level: pd.DataFrame,
    xgboost_stay_level: pd.DataFrame,
    nn_stay_level: pd.DataFrame,
    *,
    primary_horizon: int,
) -> pd.DataFrame:
    def _fatal_flag_frame(stay_level: pd.DataFrame, model_name: str, flag_name: str) -> pd.DataFrame:
        validated = _validate_agreement_input_stay_level(stay_level, source_name=model_name)
        fatal = validated[
            validated["label_value"].astype(int).eq(1)
            & validated["horizon_h"].astype(int).eq(int(primary_horizon))
        ].copy()
        return fatal.loc[:, ["stay_id_global", "horizon_h", "hard_case_flag"]].rename(
            columns={"hard_case_flag": flag_name}
        )

    merged = _fatal_flag_frame(logistic_stay_level, SOURCE_LOGISTIC_MODEL_NAME, "logistic_hard").merge(
        _fatal_flag_frame(xgboost_stay_level, SOURCE_XGBOOST_MODEL_NAME, "xgboost_hard"),
        on=["stay_id_global", "horizon_h"],
        how="inner",
    ).merge(
        _fatal_flag_frame(nn_stay_level, FEEDFORWARD_NN_MODEL_NAME, "nn_hard"),
        on=["stay_id_global", "horizon_h"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    logistic_hard = merged["logistic_hard"].astype(bool)
    xgboost_hard = merged["xgboost_hard"].astype(bool)
    nn_hard = merged["nn_hard"].astype(bool)
    return pd.DataFrame(
        [
            {
                "horizon_h": int(primary_horizon),
                "n_fatal_with_all_three_models_available": int(merged.shape[0]),
                "n_logistic_hard": int(logistic_hard.sum()),
                "n_xgboost_hard": int(xgboost_hard.sum()),
                "n_feedforward_nn_hard": int(nn_hard.sum()),
                "n_all_three_hard": int((logistic_hard & xgboost_hard & nn_hard).sum()),
                "n_logistic_and_nn_only": int((logistic_hard & nn_hard & ~xgboost_hard).sum()),
                "n_nn_and_xgboost_only": int((nn_hard & xgboost_hard & ~logistic_hard).sum()),
                "n_logistic_and_xgboost_only": int((logistic_hard & xgboost_hard & ~nn_hard).sum()),
                "n_logistic_only": int((logistic_hard & ~xgboost_hard & ~nn_hard).sum()),
                "n_xgboost_only": int((xgboost_hard & ~logistic_hard & ~nn_hard).sum()),
                "n_feedforward_nn_only": int((nn_hard & ~logistic_hard & ~xgboost_hard).sum()),
            }
        ]
    )


def _load_nn_performance_summary(
    evaluation_output_dir: Path,
    *,
    horizons: Sequence[int],
) -> pd.DataFrame:
    reporting_summary = read_dataframe(Path(evaluation_output_dir) / "reporting_split_summary.csv")
    combined_metrics = read_dataframe(Path(evaluation_output_dir) / "combined_metrics.csv")
    reporting_summary = reporting_summary[
        reporting_summary["model_name"].astype("string").eq(FEEDFORWARD_NN_MODEL_NAME)
    ][
        [
            "model_name",
            "horizon_h",
            "selected_split",
            "selected_split_evaluable",
            "selection_reason",
        ]
    ].copy()
    combined_metrics = combined_metrics[
        combined_metrics["model_name"].astype("string").eq(FEEDFORWARD_NN_MODEL_NAME)
    ][
        [
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
            "metric_notes",
        ]
    ].copy()
    merged = reporting_summary.merge(
        combined_metrics,
        left_on=["model_name", "horizon_h", "selected_split"],
        right_on=["model_name", "horizon_h", "split"],
        how="left",
    )
    merged = merged[merged["horizon_h"].astype(int).isin([int(h) for h in horizons])].copy()
    merged["reliability_plot_path"] = merged["horizon_h"].map(
        lambda horizon: str(
            (
                Path(evaluation_output_dir)
                / FEEDFORWARD_NN_MODEL_NAME
                / f"horizon_{int(horizon)}h"
                / "reliability_plot.png"
            ).resolve()
        )
    )
    merged["mortality_vs_risk_plot_path"] = merged["horizon_h"].map(
        lambda horizon: str(
            (
                Path(evaluation_output_dir)
                / FEEDFORWARD_NN_MODEL_NAME
                / f"horizon_{int(horizon)}h"
                / "mortality_vs_risk_plot.png"
            ).resolve()
        )
    )
    return merged[
        [
            "model_name",
            "horizon_h",
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
            "metric_notes",
            "reliability_plot_path",
            "mortality_vs_risk_plot_path",
        ]
    ].sort_values(["horizon_h"], kind="stable").reset_index(drop=True)


def _plot_sensitivity_overview(
    overlap_summary: pd.DataFrame,
    *,
    output_path: Path,
) -> Path:
    _require_matplotlib()
    ensure_directory(output_path.parent)

    ordered = overlap_summary.sort_values(["horizon_h", "pair_name"], kind="stable").copy()
    counts = []
    for horizon_h, horizon_df in ordered.groupby("horizon_h", sort=True):
        nn_rows = horizon_df[horizon_df["left_model_name"].astype("string").eq(FEEDFORWARD_NN_MODEL_NAME)]
        if nn_rows.empty:
            continue
        logistic_row = nn_rows[nn_rows["right_model_name"].astype("string").eq(SOURCE_LOGISTIC_MODEL_NAME)]
        xgboost_row = nn_rows[nn_rows["right_model_name"].astype("string").eq(SOURCE_XGBOOST_MODEL_NAME)]
        if logistic_row.empty or xgboost_row.empty:
            continue
        counts.append(
            {
                "horizon_h": int(horizon_h),
                "n_feedforward_nn_hard": int(logistic_row.iloc[0]["n_left_hard"]),
                "n_logistic_hard": int(logistic_row.iloc[0]["n_right_hard"]),
                "n_xgboost_hard": int(xgboost_row.iloc[0]["n_right_hard"]),
            }
        )
    counts_df = pd.DataFrame(counts).sort_values("horizon_h").reset_index(drop=True)

    figure, axes = plt.subplots(2, 1, figsize=(8.5, 8.0), sharex=True)
    count_axis, overlap_axis = axes

    count_axis.plot(
        counts_df["horizon_h"],
        counts_df["n_logistic_hard"],
        marker="o",
        linewidth=2.0,
        label="Logistic hard cases",
        color="#1f77b4",
    )
    count_axis.plot(
        counts_df["horizon_h"],
        counts_df["n_xgboost_hard"],
        marker="o",
        linewidth=2.0,
        label="XGBoost hard cases",
        color="#ff7f0e",
    )
    count_axis.plot(
        counts_df["horizon_h"],
        counts_df["n_feedforward_nn_hard"],
        marker="o",
        linewidth=2.0,
        label="Feedforward NN hard cases",
        color="#2ca02c",
    )
    count_axis.set_ylabel("Hard-case count")
    count_axis.set_title("Chapter 1 feedforward-NN hard-case sensitivity across horizons")
    count_axis.grid(alpha=0.25, linewidth=0.6)
    count_axis.legend(loc="upper left")

    for comparator, color in (
        (SOURCE_LOGISTIC_MODEL_NAME, "#1f77b4"),
        (SOURCE_XGBOOST_MODEL_NAME, "#ff7f0e"),
    ):
        pair_df = ordered[ordered["right_model_name"].astype("string").eq(comparator)].copy()
        overlap_axis.plot(
            pair_df["horizon_h"],
            pair_df["jaccard_agreement"],
            marker="o",
            linewidth=2.0,
            label=f"Jaccard vs {_display_model_name(comparator)}",
            color=color,
        )
        overlap_axis.plot(
            pair_df["horizon_h"],
            pair_df["overlap_share_of_left_hard"],
            marker="s",
            linewidth=1.5,
            linestyle="--",
            label=f"NN overlap share vs {_display_model_name(comparator)}",
            color=color,
            alpha=0.75,
        )

    overlap_axis.set_xlabel("Horizon (h)")
    overlap_axis.set_ylabel("Agreement")
    overlap_axis.set_ylim(0.0, 1.0)
    overlap_axis.set_xticks(counts_df["horizon_h"].tolist())
    overlap_axis.grid(alpha=0.25, linewidth=0.6)
    overlap_axis.legend(loc="lower right")

    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _choose_resemblance(overlap_summary: pd.DataFrame) -> str:
    grouped = (
        overlap_summary.groupby("right_model_name", as_index=False)["jaccard_agreement"]
        .mean()
        .sort_values("jaccard_agreement", ascending=False)
        .reset_index(drop=True)
    )
    if grouped.shape[0] < 2:
        return "neither"
    top = grouped.iloc[0]
    runner_up = grouped.iloc[1]
    if float(top["jaccard_agreement"] - runner_up["jaccard_agreement"]) < 0.03:
        return "neither"
    return str(top["right_model_name"])


def _boundary_interpretation(boundary_summary: pd.DataFrame) -> str:
    shares = boundary_summary["boundary_near_share_of_disagreement"].dropna()
    if shares.empty:
        return "not assessed"
    mean_share = float(shares.mean())
    if mean_share >= 0.60:
        return "mostly boundary-driven"
    if mean_share >= 0.35:
        return "mixed between boundary-driven and substantive"
    return "more substantive than boundary-driven"


def _final_judgment(performance_summary: pd.DataFrame, overlap_summary: pd.DataFrame) -> str:
    evaluable_share = float(performance_summary["selected_split_evaluable"].astype(bool).mean())
    slopes = performance_summary["calibration_slope"].dropna()
    poor_calibration = bool(not slopes.empty and (slopes.sub(1.0).abs() > 0.5).mean() >= 0.5)
    mean_jaccard = float(overlap_summary["jaccard_agreement"].dropna().mean())
    if evaluable_share < 0.8 or poor_calibration:
        return "not meaningfully clarified by the NN baseline"
    if mean_jaccard >= 0.35:
        return "broadly unchanged"
    return "somewhat more definition-sensitive"


def build_feedforward_nn_hard_case_sensitivity_memo(
    *,
    performance_summary: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    boundary_summary: pd.DataFrame,
    three_way_overlap_24h: pd.DataFrame | None,
) -> str:
    resemblance = _choose_resemblance(overlap_summary)
    resemblance_text = {
        SOURCE_LOGISTIC_MODEL_NAME: "The NN most closely resembles logistic regression.",
        SOURCE_XGBOOST_MODEL_NAME: "The NN most closely resembles XGBoost.",
        "neither": "The NN does not clearly cluster with either logistic regression or XGBoost.",
    }[resemblance]

    nn_counts = overlap_summary[
        overlap_summary["left_model_name"].astype("string").eq(FEEDFORWARD_NN_MODEL_NAME)
    ].groupby("horizon_h")["n_left_hard"].first()
    logistic_counts = overlap_summary[
        overlap_summary["right_model_name"].astype("string").eq(SOURCE_LOGISTIC_MODEL_NAME)
    ].set_index("horizon_h")["n_right_hard"]
    xgboost_counts = overlap_summary[
        overlap_summary["right_model_name"].astype("string").eq(SOURCE_XGBOOST_MODEL_NAME)
    ].set_index("horizon_h")["n_right_hard"]

    if not nn_counts.empty and not logistic_counts.empty and nn_counts.mean() < logistic_counts.mean() * 0.90:
        count_text = "Adding the NN shrinks the hard-case pool relative to logistic regression."
    elif not nn_counts.empty and not xgboost_counts.empty and nn_counts.mean() > xgboost_counts.mean() * 1.10:
        count_text = "Adding the NN expands the hard-case pool relative to XGBoost."
    else:
        count_text = "Adding the NN keeps the hard-case pool within the span already defined by logistic regression and XGBoost."

    min_pairwise_jaccard = float(overlap_summary["jaccard_agreement"].dropna().min())
    if min_pairwise_jaccard >= 0.30:
        shared_core_text = "A shared hard-case core still persists across models."
    else:
        shared_core_text = "The shared hard-case core weakens once the NN is added."

    boundary_text = _boundary_interpretation(boundary_summary)
    judgment = _final_judgment(performance_summary, overlap_summary)

    primary_lines: list[str] = []
    primary_overlap = overlap_summary[overlap_summary["horizon_h"].astype(int).eq(PRIMARY_HORIZON)].copy()
    for row in primary_overlap.itertuples(index=False):
        primary_lines.append(
            f"- {int(row.horizon_h)}h {_display_model_name(row.left_model_name)} vs {_display_model_name(row.right_model_name)}: "
            f"Jaccard {float(row.jaccard_agreement):.3f}, overlap share of NN {float(row.overlap_share_of_left_hard):.3f}, "
            f"overlap share of comparator {float(row.overlap_share_of_right_hard):.3f}."
        )

    three_way_text = ""
    if three_way_overlap_24h is not None and not three_way_overlap_24h.empty:
        row = three_way_overlap_24h.iloc[0]
        three_way_text = (
            f" At 24h, {int(row['n_all_three_hard'])} fatal stays sit in the shared three-model core "
            f"out of {int(row['n_feedforward_nn_hard'])} NN hard cases."
        )

    performance_lines = []
    for row in performance_summary.itertuples(index=False):
        slope_text = "NA" if pd.isna(row.calibration_slope) else f"{float(row.calibration_slope):.3f}"
        performance_lines.append(
            f"- {int(row.horizon_h)}h used `{row.selected_split}`: AUROC {float(row.auroc):.3f}, "
            f"AUPRC {float(row.auprc):.3f}, calibration slope {slope_text}."
        )

    lines = [
        "# Feedforward NN Hard-Case Sensitivity Memo",
        "",
        resemblance_text,
        count_text,
        f"Disagreement looks {boundary_text}.",
        shared_core_text + three_way_text,
        f"Final judgment: the Chapter 1 hard-case concept looks **{judgment}**.",
        "",
        "NN performance summary:",
        *performance_lines,
        "",
        "Primary 24h overlap summary:",
        *primary_lines,
    ]
    return "\n".join(lines) + "\n"


def run_feedforward_nn_hard_case_sensitivity(
    *,
    baseline_input_root: Path = DEFAULT_BASELINE_ARTIFACT_ROOT,
    nn_output_dir: Path = DEFAULT_FEEDFORWARD_NN_BASELINE_OUTPUT_DIR,
    evaluation_output_dir: Path = DEFAULT_EVALUATION_OUTPUT_DIR,
    hard_case_output_dir: Path = DEFAULT_FEEDFORWARD_NN_HARD_CASE_OUTPUT_DIR,
    sensitivity_output_dir: Path = DEFAULT_FEEDFORWARD_NN_SENSITIVITY_OUTPUT_DIR,
    input_dataset_path: Path = DEFAULT_PRIMARY_MODEL_READY_DATASET_PATH,
    feature_set_definition_path: Path = DEFAULT_FEATURE_SET_DEFINITION_PATH,
    horizons: Sequence[int] | None = None,
    preprocessing_root: Path | None = None,
    standardized_input_dir: Path | None = None,
    standardized_input_format: str | None = None,
    run_config_path: Path | None = None,
    reuse_saved_nn_baseline: bool = True,
    reuse_saved_nn_evaluation: bool = True,
    boundary_margin: float = DEFAULT_BOUNDARY_MARGIN,
    primary_horizon: int = PRIMARY_HORIZON,
) -> FeedforwardNNHardCaseSensitivityRunResult:
    selected_horizons = tuple(int(horizon) for horizon in (horizons or DEFAULT_HORIZONS))
    baseline_input_root = Path(baseline_input_root)
    nn_output_dir = Path(nn_output_dir)
    evaluation_output_dir = Path(evaluation_output_dir)
    hard_case_output_dir = Path(hard_case_output_dir)
    sensitivity_output_dir = Path(sensitivity_output_dir)

    nn_predictions_ready = reuse_saved_nn_baseline and _baseline_predictions_complete(
        baseline_input_root,
        model_name=FEEDFORWARD_NN_MODEL_NAME,
        horizons=selected_horizons,
    )
    baseline_result = None
    if not nn_predictions_ready:
        baseline_result = run_asic_primary_feedforward_nn(
            input_dataset_path=input_dataset_path,
            feature_set_definition_path=feature_set_definition_path,
            output_dir=nn_output_dir,
            horizons=selected_horizons,
            preprocessing_root=preprocessing_root,
            standardized_input_dir=standardized_input_dir,
            standardized_input_format=standardized_input_format,
            run_config_path=run_config_path,
        )

    evaluation_files_ready = reuse_saved_nn_evaluation and _nn_evaluation_complete(
        evaluation_output_dir,
        horizons=selected_horizons,
    )
    evaluation_result = None
    if not evaluation_files_ready:
        evaluation_result = run_asic_baseline_evaluation(
            input_root=baseline_input_root,
            output_dir=evaluation_output_dir,
            models=[FEEDFORWARD_NN_MODEL_NAME],
            horizons=selected_horizons,
            primary_horizon=primary_horizon,
        )

    logistic_frames, logistic_sources = _load_prediction_frames(
        baseline_input_root,
        model_name=SOURCE_LOGISTIC_MODEL_NAME,
        horizons=selected_horizons,
    )
    xgboost_frames, xgboost_sources = _load_prediction_frames(
        baseline_input_root,
        model_name=SOURCE_XGBOOST_MODEL_NAME,
        horizons=selected_horizons,
    )
    nn_frames, nn_sources = _load_prediction_frames(
        baseline_input_root,
        model_name=FEEDFORWARD_NN_MODEL_NAME,
        horizons=selected_horizons,
    )

    logistic_stay_level, _ = _derive_hard_cases(
        logistic_frames,
        logistic_sources,
        model_name=SOURCE_LOGISTIC_MODEL_NAME,
        hard_case_rule=LOGISTIC_HARD_CASE_RULE,
    )
    xgboost_stay_level, _ = _derive_hard_cases(
        xgboost_frames,
        xgboost_sources,
        model_name=SOURCE_XGBOOST_MODEL_NAME,
        hard_case_rule=XGBOOST_HARD_CASE_RULE,
    )
    nn_stay_level, nn_horizon_summary = _derive_hard_cases(
        nn_frames,
        nn_sources,
        model_name=FEEDFORWARD_NN_MODEL_NAME,
        hard_case_rule=FEEDFORWARD_NN_HARD_CASE_RULE,
    )

    ensure_directory(hard_case_output_dir)
    nn_hard_case_stay_level_path = write_dataframe(
        nn_stay_level,
        hard_case_output_dir / "stay_level_hard_case_flags.csv",
        output_format="csv",
    )
    nn_hard_case_summary_path = write_dataframe(
        nn_horizon_summary,
        hard_case_output_dir / "horizon_hard_case_summary.csv",
        output_format="csv",
    )
    _write_json(
        {
            "timestamp_utc": _utc_timestamp(),
            "model_name": FEEDFORWARD_NN_MODEL_NAME,
            "hard_case_rule": FEEDFORWARD_NN_HARD_CASE_RULE,
            "input_root": str(baseline_input_root.resolve()),
            "output_dir": str(hard_case_output_dir.resolve()),
            "horizons_processed": list(selected_horizons),
            "prediction_paths_by_horizon": {str(k): v for k, v in nn_sources.items()},
        },
        hard_case_output_dir / "run_manifest.json",
    )

    nn_vs_logistic_stay_level, nn_vs_logistic_summary = build_pairwise_hard_case_overlap(
        nn_stay_level,
        logistic_stay_level,
        left_model_name=FEEDFORWARD_NN_MODEL_NAME,
        right_model_name=SOURCE_LOGISTIC_MODEL_NAME,
        left_prefix="nn",
        right_prefix="logistic",
    )
    nn_vs_xgboost_stay_level, nn_vs_xgboost_summary = build_pairwise_hard_case_overlap(
        nn_stay_level,
        xgboost_stay_level,
        left_model_name=FEEDFORWARD_NN_MODEL_NAME,
        right_model_name=SOURCE_XGBOOST_MODEL_NAME,
        left_prefix="nn",
        right_prefix="xgboost",
    )

    nn_vs_logistic_boundary = summarize_boundary_disagreement(
        nn_vs_logistic_stay_level,
        left_model_name=FEEDFORWARD_NN_MODEL_NAME,
        right_model_name=SOURCE_LOGISTIC_MODEL_NAME,
        left_prefix="nn",
        right_prefix="logistic",
        boundary_margin=boundary_margin,
    )
    nn_vs_xgboost_boundary = summarize_boundary_disagreement(
        nn_vs_xgboost_stay_level,
        left_model_name=FEEDFORWARD_NN_MODEL_NAME,
        right_model_name=SOURCE_XGBOOST_MODEL_NAME,
        left_prefix="nn",
        right_prefix="xgboost",
        boundary_margin=boundary_margin,
    )

    performance_summary = _load_nn_performance_summary(
        evaluation_output_dir,
        horizons=selected_horizons,
    )
    overlap_summary = pd.concat(
        [nn_vs_logistic_summary, nn_vs_xgboost_summary],
        ignore_index=True,
    ).sort_values(["pair_name", "horizon_h"], kind="stable").reset_index(drop=True)
    boundary_summary = pd.concat(
        [nn_vs_logistic_boundary, nn_vs_xgboost_boundary],
        ignore_index=True,
    ).sort_values(["pair_name", "horizon_h"], kind="stable").reset_index(drop=True)
    three_way_overlap_24h = build_three_way_overlap_24h_summary(
        logistic_stay_level,
        xgboost_stay_level,
        nn_stay_level,
        primary_horizon=primary_horizon,
    )

    ensure_directory(sensitivity_output_dir)
    performance_summary_path = write_dataframe(
        performance_summary,
        sensitivity_output_dir / "feedforward_nn_performance_summary.csv",
        output_format="csv",
    )
    overlap_summary_path = write_dataframe(
        overlap_summary,
        sensitivity_output_dir / "feedforward_nn_hard_case_overlap_summary.csv",
        output_format="csv",
    )
    boundary_summary_path = write_dataframe(
        boundary_summary,
        sensitivity_output_dir / "feedforward_nn_boundary_disagreement_summary.csv",
        output_format="csv",
    )
    three_way_overlap_24h_path: Path | None = None
    if not three_way_overlap_24h.empty:
        three_way_overlap_24h_path = write_dataframe(
            three_way_overlap_24h,
            sensitivity_output_dir / "feedforward_nn_three_way_overlap_24h.csv",
            output_format="csv",
        )

    figure_path = _plot_sensitivity_overview(
        overlap_summary,
        output_path=sensitivity_output_dir / "feedforward_nn_hard_case_comparison_by_horizon.png",
    )
    memo_text = build_feedforward_nn_hard_case_sensitivity_memo(
        performance_summary=performance_summary,
        overlap_summary=overlap_summary,
        boundary_summary=boundary_summary,
        three_way_overlap_24h=three_way_overlap_24h if not three_way_overlap_24h.empty else None,
    )
    memo_path = write_text(
        memo_text,
        sensitivity_output_dir / "feedforward_nn_hard_case_sensitivity_memo.md",
    )
    manifest_path = _write_json(
        {
            "timestamp_utc": _utc_timestamp(),
            "baseline_input_root": str(baseline_input_root.resolve()),
            "nn_output_dir": str(nn_output_dir.resolve()),
            "evaluation_output_dir": str(evaluation_output_dir.resolve()),
            "hard_case_output_dir": str(hard_case_output_dir.resolve()),
            "sensitivity_output_dir": str(sensitivity_output_dir.resolve()),
            "horizons_processed": list(selected_horizons),
            "reuse_saved_nn_baseline": reuse_saved_nn_baseline,
            "reuse_saved_nn_evaluation": reuse_saved_nn_evaluation,
            "nn_predictions_reused": nn_predictions_ready,
            "nn_evaluation_reused": evaluation_files_ready,
            "baseline_training_run_manifest": (
                str(baseline_result.manifest_path.resolve()) if baseline_result is not None else None
            ),
            "evaluation_run_manifest": (
                str(evaluation_result.manifest_path.resolve()) if evaluation_result is not None else None
            ),
            "nn_baseline_parameters": {
                **DEFAULT_FEEDFORWARD_NN_PARAMETERS,
                "random_state": DEFAULT_FEEDFORWARD_NN_RANDOM_STATE,
            },
            "artifacts": {
                "nn_hard_case_stay_level": str(nn_hard_case_stay_level_path.resolve()),
                "nn_hard_case_summary": str(nn_hard_case_summary_path.resolve()),
                "performance_summary": str(performance_summary_path.resolve()),
                "overlap_summary": str(overlap_summary_path.resolve()),
                "boundary_summary": str(boundary_summary_path.resolve()),
                "three_way_overlap_24h": (
                    str(three_way_overlap_24h_path.resolve()) if three_way_overlap_24h_path is not None else None
                ),
                "figure": str(figure_path.resolve()),
                "memo": str(memo_path.resolve()),
            },
        },
        sensitivity_output_dir / "run_manifest.json",
    )

    return FeedforwardNNHardCaseSensitivityRunResult(
        baseline_input_root=baseline_input_root,
        nn_output_dir=nn_output_dir,
        evaluation_output_dir=evaluation_output_dir,
        hard_case_output_dir=hard_case_output_dir,
        sensitivity_output_dir=sensitivity_output_dir,
        horizons_processed=selected_horizons,
        artifacts=FeedforwardNNHardCaseSensitivityArtifacts(
            nn_hard_case_stay_level_path=nn_hard_case_stay_level_path,
            nn_hard_case_summary_path=nn_hard_case_summary_path,
            performance_summary_path=performance_summary_path,
            overlap_summary_path=overlap_summary_path,
            boundary_summary_path=boundary_summary_path,
            three_way_overlap_24h_path=three_way_overlap_24h_path,
            figure_path=figure_path,
            memo_path=memo_path,
            manifest_path=manifest_path,
        ),
        performance_summary=performance_summary,
        overlap_summary=overlap_summary,
        boundary_summary=boundary_summary,
        three_way_overlap_24h=three_way_overlap_24h if not three_way_overlap_24h.empty else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Chapter 1 feedforward-NN baseline sensitivity for hard-case overlap "
            "against logistic regression and XGBoost."
        )
    )
    parser.add_argument(
        "--baseline-input-root",
        type=Path,
        default=DEFAULT_BASELINE_ARTIFACT_ROOT,
        help="Root baseline artifact directory containing logistic, xgboost, and feedforward_nn outputs.",
    )
    parser.add_argument(
        "--nn-output-dir",
        type=Path,
        default=DEFAULT_FEEDFORWARD_NN_BASELINE_OUTPUT_DIR,
        help="Output directory for saved feedforward-NN baseline prediction artifacts.",
    )
    parser.add_argument(
        "--evaluation-output-dir",
        type=Path,
        default=DEFAULT_EVALUATION_OUTPUT_DIR,
        help="Output directory for saved baseline evaluation artifacts.",
    )
    parser.add_argument(
        "--hard-case-output-dir",
        type=Path,
        default=DEFAULT_FEEDFORWARD_NN_HARD_CASE_OUTPUT_DIR,
        help="Output directory for saved feedforward-NN hard-case artifacts.",
    )
    parser.add_argument(
        "--sensitivity-output-dir",
        type=Path,
        default=DEFAULT_FEEDFORWARD_NN_SENSITIVITY_OUTPUT_DIR,
        help="Output directory for the feedforward-NN hard-case sensitivity package.",
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
        "--horizons",
        nargs="+",
        type=int,
        help="Optional subset of horizons to process.",
    )
    parser.add_argument(
        "--preprocessing-root",
        type=Path,
        help="Optional root directory containing Chapter 1 preprocessing artifacts.",
    )
    parser.add_argument(
        "--standardized-input-dir",
        type=Path,
        help="Optional standardized ASIC input directory used to rebuild all-valid scoring rows.",
    )
    parser.add_argument(
        "--standardized-input-format",
        choices=("csv", "parquet"),
        help="Format of the standardized ASIC input directory used for all-valid scoring.",
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        help="Optional Chapter 1 run config used to resolve the default standardized input directory.",
    )
    parser.add_argument(
        "--boundary-margin",
        type=float,
        default=DEFAULT_BOUNDARY_MARGIN,
        help="Absolute probability margin used for the threshold-near disagreement check.",
    )
    parser.add_argument(
        "--no-reuse-saved-nn-baseline",
        action="store_true",
        help="Force retraining of the feedforward-NN baseline even if saved predictions already exist.",
    )
    parser.add_argument(
        "--no-reuse-saved-nn-evaluation",
        action="store_true",
        help="Force regeneration of the saved feedforward-NN evaluation outputs.",
    )
    parser.add_argument(
        "--primary-horizon",
        type=int,
        default=PRIMARY_HORIZON,
        help="Primary horizon for the compact three-way overlap summary.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = run_feedforward_nn_hard_case_sensitivity(
        baseline_input_root=args.baseline_input_root,
        nn_output_dir=args.nn_output_dir,
        evaluation_output_dir=args.evaluation_output_dir,
        hard_case_output_dir=args.hard_case_output_dir,
        sensitivity_output_dir=args.sensitivity_output_dir,
        input_dataset_path=args.input_dataset,
        feature_set_definition_path=args.feature_set_definition,
        horizons=args.horizons,
        preprocessing_root=args.preprocessing_root,
        standardized_input_dir=args.standardized_input_dir,
        standardized_input_format=args.standardized_input_format,
        run_config_path=args.run_config,
        reuse_saved_nn_baseline=not args.no_reuse_saved_nn_baseline,
        reuse_saved_nn_evaluation=not args.no_reuse_saved_nn_evaluation,
        boundary_margin=args.boundary_margin,
        primary_horizon=args.primary_horizon,
    )

    print(f"Baseline input root: {result.baseline_input_root}")
    print(f"Horizons processed: {', '.join(str(horizon) for horizon in result.horizons_processed)}")
    print(f"Performance summary: {result.artifacts.performance_summary_path}")
    print(f"Overlap summary: {result.artifacts.overlap_summary_path}")
    print(f"Boundary summary: {result.artifacts.boundary_summary_path}")
    if result.artifacts.three_way_overlap_24h_path is not None:
        print(f"Three-way 24h overlap: {result.artifacts.three_way_overlap_24h_path}")
    print(f"Figure: {result.artifacts.figure_path}")
    print(f"Memo: {result.artifacts.memo_path}")
    print(f"Run manifest: {result.artifacts.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
