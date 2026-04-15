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

from chapter1_mortality_decomposition.asic_hard_case_comparison import (
    HARD_CASE_RULE,
    LOW_PREDICTED_FATAL_GROUP,
    OTHER_FATAL_GROUP,
    _continuous_standardized_difference,
    _proportion_standardized_difference,
)
from chapter1_mortality_decomposition.utils import (
    normalize_boolean_codes,
    read_dataframe,
    require_columns,
    write_dataframe,
    write_text,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "chapter1"
    / "evaluation"
    / "asic"
    / "hard_cases"
    / "primary_medians"
    / "logistic_regression"
    / "asic_observation_process_sensitivity"
)
DEFAULT_COMPARISON_DATASET_PATH = (
    REPO_ROOT
    / "cluster-results"
    / "chapter1_true_results"
    / "evaluation"
    / "asic"
    / "hard_cases"
    / "primary_medians"
    / "logistic_regression"
    / "asic_hard_case_comparison"
    / "stay_level_comparison_dataset.csv"
)
DEFAULT_HARD_CASE_PATH = (
    REPO_ROOT
    / "cluster-results"
    / "chapter1_true_results"
    / "evaluation"
    / "asic"
    / "hard_cases"
    / "primary_medians"
    / "logistic_regression"
    / "stay_level_hard_case_flags.csv"
)
DEFAULT_OBSERVATION_PROCESS_PATH = (
    REPO_ROOT
    / "cluster-results"
    / "chapter1_true_results"
    / "observation_process"
    / "chapter1_observation_process_block_features.csv"
)

DEFAULT_OUTPUT_DATASET_PATH = (
    DEFAULT_OUTPUT_ROOT / "stay_level_observation_process_dataset.csv"
)
DEFAULT_OUTPUT_COMPARISON_TABLE_PATH = (
    DEFAULT_OUTPUT_ROOT / "comparison_table.csv"
)
DEFAULT_OUTPUT_EFFECT_DETAILS_PATH = (
    DEFAULT_OUTPUT_ROOT / "effect_size_details.csv"
)
DEFAULT_OUTPUT_MANIFEST_PATH = (
    DEFAULT_OUTPUT_ROOT / "run_manifest.json"
)
DEFAULT_OUTPUT_FIGURE_PATH = DEFAULT_OUTPUT_ROOT / "effect_size_figure.png"
DEFAULT_OUTPUT_MEMO_PATH = DEFAULT_OUTPUT_ROOT / "memo.md"

TARGET_HORIZON_H = 24
OBSERVATION_INPUT_REQUIRED_COLUMNS = {
    "stay_id_global",
    "hospital_id",
    "block_index",
    "block_start_h",
    "block_end_h",
    "prediction_time_h",
    "obs_hr_grp_block",
    "obs_bp_grp_block",
    "obs_resp_grp_block",
    "obs_oxy_grp_block",
    "n_core_grps_obs_block",
    "tsl_hr_grp_h",
    "tsl_bp_grp_h",
    "tsl_resp_grp_h",
    "tsl_oxy_grp_h",
}
COMPARISON_INPUT_REQUIRED_COLUMNS = {
    "stay_id_global",
    "instance_id",
    "hard_case_flag",
    "hard_case_group",
    "prediction_time_h",
    "icu_end_time_proxy_hours",
    "hospital_id",
    "pf_ratio_last",
    "map_last",
    "creatinine_last",
    "peep_last",
}
HARD_CASE_INPUT_REQUIRED_COLUMNS = {
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
}

OBS_BLOCK_COLUMNS = [
    "obs_hr_grp_block",
    "obs_bp_grp_block",
    "obs_resp_grp_block",
    "obs_oxy_grp_block",
]
TSL_COLUMNS = [
    "tsl_hr_grp_h",
    "tsl_bp_grp_h",
    "tsl_resp_grp_h",
    "tsl_oxy_grp_h",
]
FROZEN_PROXY_COLUMNS = [
    "pf_ratio_last",
    "map_last",
    "creatinine_last",
    "peep_last",
]

VARIABLE_SPECS = (
    {
        "name": "n_core_groups_fresh_block",
        "label": "Fresh core groups in anchor block",
        "kind": "continuous",
    },
    {
        "name": "core_block_complete_all4",
        "label": "All 4 core groups observed in anchor block",
        "kind": "binary",
    },
    {
        "name": "n_core_groups_historical_only",
        "label": "Core groups historical-only at anchor",
        "kind": "continuous",
    },
    {
        "name": "n_core_groups_never_observed",
        "label": "Core groups never observed by anchor",
        "kind": "continuous",
    },
    {
        "name": "n_frozen_proxy_missing",
        "label": "Missing frozen physiologic proxies",
        "kind": "continuous",
    },
    {
        "name": "time_since_last_any_core_h",
        "label": "Time since any core group observed (h)",
        "kind": "continuous",
    },
    {
        "name": "max_time_since_last_core_h",
        "label": "Longest core-group recency gap (h)",
        "kind": "continuous",
    },
    {
        "name": "any_stale_core_ge_8h_flag",
        "label": "Any core group stale >=8h",
        "kind": "binary",
    },
)

INTEGER_SUMMARY_VARIABLES = {
    "n_core_groups_fresh_block",
    "n_core_groups_historical_only",
    "n_core_groups_never_observed",
    "n_frozen_proxy_missing",
}


@dataclass(frozen=True)
class ASICObservationProcessSensitivityArtifacts:
    dataset_path: Path
    comparison_table_path: Path
    effect_details_path: Path
    figure_path: Path
    memo_path: Path
    manifest_path: Path
    promoted_memo_path: Path | None = None


@dataclass(frozen=True)
class ASICObservationProcessSensitivityRunResult:
    dataset: pd.DataFrame
    comparison_table: pd.DataFrame
    effect_details: pd.DataFrame
    memo_markdown: str
    artifacts: ASICObservationProcessSensitivityArtifacts


def _require_matplotlib() -> None:
    if plt is None:
        raise ImportError(
            "matplotlib is required for the ASIC observation-process sensitivity output."
        ) from MATPLOTLIB_IMPORT_ERROR


def _resolve_existing_path(path: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    alternate_suffix = ".parquet" if candidate.suffix == ".csv" else ".csv"
    alternate = candidate.with_suffix(alternate_suffix)
    if alternate.exists():
        return alternate.resolve()
    raise FileNotFoundError(f"Required artifact is missing: {candidate}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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


def _write_promoted_memo(memo_markdown: str, promoted_memo_path: Path | None) -> Path | None:
    if promoted_memo_path is None:
        return None
    return write_text(memo_markdown, promoted_memo_path)


def _normalize_string_keys(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    normalized = frame.copy()
    for column in columns:
        normalized[column] = normalized[column].astype("string")
    return normalized


def _normalize_numeric_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    normalized = frame.copy()
    for column in columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


def load_authoritative_observation_process_anchor(
    *,
    comparison_dataset_path: Path = DEFAULT_COMPARISON_DATASET_PATH,
    hard_case_path: Path = DEFAULT_HARD_CASE_PATH,
    observation_process_path: Path = DEFAULT_OBSERVATION_PROCESS_PATH,
) -> tuple[pd.DataFrame, dict[str, object]]:
    resolved_comparison_path = _resolve_existing_path(comparison_dataset_path)
    resolved_hard_case_path = _resolve_existing_path(hard_case_path)
    resolved_observation_process_path = _resolve_existing_path(observation_process_path)

    comparison_dataset = read_dataframe(resolved_comparison_path)
    require_columns(
        comparison_dataset,
        COMPARISON_INPUT_REQUIRED_COLUMNS,
        "authoritative saved hard-case comparison dataset",
    )
    comparison_dataset = _normalize_string_keys(
        comparison_dataset,
        ["stay_id_global", "instance_id", "hard_case_group", "hospital_id"],
    )
    comparison_dataset = _normalize_numeric_columns(
        comparison_dataset,
        ["prediction_time_h", "icu_end_time_proxy_hours", *FROZEN_PROXY_COLUMNS],
    )
    comparison_dataset["hard_case_flag"] = normalize_boolean_codes(
        comparison_dataset["hard_case_flag"]
    )
    if comparison_dataset["hard_case_flag"].isna().any():
        raise ValueError("Saved hard-case comparison dataset contains missing hard_case_flag values.")

    expected_group = pd.Series(
        np.where(
            comparison_dataset["hard_case_flag"].astype(bool),
            LOW_PREDICTED_FATAL_GROUP,
            OTHER_FATAL_GROUP,
        ),
        index=comparison_dataset.index,
        dtype="string",
    )
    if not comparison_dataset["hard_case_group"].eq(expected_group).all():
        raise ValueError(
            "Saved hard-case comparison dataset contains hard_case_group values that are "
            "inconsistent with hard_case_flag."
        )

    hard_case_flags = read_dataframe(resolved_hard_case_path)
    require_columns(
        hard_case_flags,
        HARD_CASE_INPUT_REQUIRED_COLUMNS,
        "saved stay-level hard-case flags",
    )
    hard_case_flags = _normalize_string_keys(
        hard_case_flags,
        ["stay_id_global", "instance_id", "hospital_id", "hard_case_rule"],
    )
    hard_case_flags = _normalize_numeric_columns(
        hard_case_flags,
        [
            "horizon_h",
            "label_value",
            "block_index",
            "prediction_time_h",
            "predicted_probability",
            "nonfatal_q75_threshold",
        ],
    )
    hard_case_flags["hard_case_flag"] = normalize_boolean_codes(hard_case_flags["hard_case_flag"])
    hard_case_flags = hard_case_flags[
        hard_case_flags["horizon_h"].eq(TARGET_HORIZON_H)
        & hard_case_flags["label_value"].eq(1)
    ].copy()
    if hard_case_flags.empty:
        raise ValueError("The authoritative 24h fatal stay-level hard-case slice is empty.")
    if not hard_case_flags["hard_case_rule"].eq(HARD_CASE_RULE).all():
        raise ValueError(
            "The authoritative hard-case flags do not match the frozen logistic hard-case rule."
        )

    anchor = comparison_dataset.merge(
        hard_case_flags[
            [
                "stay_id_global",
                "instance_id",
                "hospital_id",
                "block_index",
                "prediction_time_h",
                "predicted_probability",
                "nonfatal_q75_threshold",
                "hard_case_rule",
            ]
        ],
        on=["stay_id_global", "instance_id", "hospital_id", "prediction_time_h"],
        how="inner",
        validate="one_to_one",
    )
    if int(anchor.shape[0]) != int(comparison_dataset.shape[0]):
        raise ValueError(
            "Authoritative hard-case comparison dataset could not be matched one-to-one to the "
            "24h fatal hard-case flags."
        )

    observation_process = read_dataframe(resolved_observation_process_path)
    require_columns(
        observation_process,
        OBSERVATION_INPUT_REQUIRED_COLUMNS,
        "authoritative observation-process block feature table",
    )
    observation_process = _normalize_string_keys(
        observation_process,
        ["stay_id_global", "hospital_id"],
    )
    observation_process = _normalize_numeric_columns(
        observation_process,
        [
            "block_index",
            "block_start_h",
            "block_end_h",
            "prediction_time_h",
            "n_core_grps_obs_block",
            *OBS_BLOCK_COLUMNS,
            *TSL_COLUMNS,
        ],
    )

    merged = anchor.merge(
        observation_process,
        on=["stay_id_global", "hospital_id", "block_index", "prediction_time_h"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError(
            "Some authoritative 24h fatal hard-case anchor rows could not be linked to the saved "
            "observation-process block feature table."
        )
    merged = merged.drop(columns="_merge")

    metadata = {
        "target_horizon_h": TARGET_HORIZON_H,
        "hard_case_rule": HARD_CASE_RULE,
        "group_counts": {
            LOW_PREDICTED_FATAL_GROUP: int(merged["hard_case_flag"].astype(bool).sum()),
            OTHER_FATAL_GROUP: int((~merged["hard_case_flag"].astype(bool)).sum()),
            "total_fatal_stays": int(merged.shape[0]),
        },
        "source_paths": {
            "comparison_dataset_path": resolved_comparison_path,
            "hard_case_path": resolved_hard_case_path,
            "observation_process_path": resolved_observation_process_path,
        },
        "bundle_gaps_encountered": [
            "The local cluster-results mirror excluded the row-level Issue 3.2 comparison dataset by default, so that producer file had to be synced separately from the true cluster artifact directory.",
            "The local cluster-results mirror excluded the row-level observation-process block-feature table by default, so that producer file had to be synced separately from the true cluster artifact directory.",
        ],
    }
    return merged, metadata


def derive_observation_process_sensitivity_dataset(anchor_dataset: pd.DataFrame) -> pd.DataFrame:
    derived = anchor_dataset.copy()

    for column in OBS_BLOCK_COLUMNS:
        derived[column] = pd.to_numeric(derived[column], errors="coerce").fillna(0).astype("int64")
    for column in TSL_COLUMNS:
        derived[column] = pd.to_numeric(derived[column], errors="coerce")

    derived["n_core_groups_fresh_block"] = pd.to_numeric(
        derived["n_core_grps_obs_block"],
        errors="coerce",
    ).astype("int64")
    derived["n_core_groups_historical_only"] = sum(
        ((derived[tsl].notna()) & derived[obs].eq(0)).astype("int64")
        for tsl, obs in zip(TSL_COLUMNS, OBS_BLOCK_COLUMNS)
    )
    derived["n_core_groups_never_observed"] = sum(
        derived[column].isna().astype("int64") for column in TSL_COLUMNS
    )
    derived["n_core_groups_stale_ge_8h"] = sum(
        derived[column].ge(8).fillna(False).astype("int64") for column in TSL_COLUMNS
    )
    derived["time_since_last_any_core_h"] = derived[TSL_COLUMNS].min(axis=1, skipna=True)
    derived["median_time_since_last_core_h"] = derived[TSL_COLUMNS].median(axis=1, skipna=True)
    derived["max_time_since_last_core_h"] = derived[TSL_COLUMNS].max(axis=1, skipna=True)
    derived["core_block_complete_all4"] = derived["n_core_groups_fresh_block"].eq(4)
    derived["core_block_incomplete_any"] = derived["n_core_groups_fresh_block"].lt(4)
    derived["any_stale_core_ge_8h_flag"] = derived["n_core_groups_stale_ge_8h"].gt(0)
    derived["any_never_observed_core_flag"] = derived["n_core_groups_never_observed"].gt(0)

    for column in FROZEN_PROXY_COLUMNS:
        derived[f"{column}_missing"] = derived[column].isna()
    derived["n_frozen_proxy_missing"] = sum(
        derived[f"{column}_missing"].astype("int64") for column in FROZEN_PROXY_COLUMNS
    )

    ordered_columns = [
        "stay_id_global",
        "instance_id",
        "hospital_id",
        "block_index",
        "block_start_h",
        "block_end_h",
        "prediction_time_h",
        "icu_end_time_proxy_hours",
        "hard_case_flag",
        "hard_case_group",
        "predicted_probability",
        "nonfatal_q75_threshold",
        "hard_case_rule",
        *OBS_BLOCK_COLUMNS,
        "n_core_grps_obs_block",
        *TSL_COLUMNS,
        "n_core_groups_fresh_block",
        "n_core_groups_historical_only",
        "n_core_groups_never_observed",
        "n_core_groups_stale_ge_8h",
        "time_since_last_any_core_h",
        "median_time_since_last_core_h",
        "max_time_since_last_core_h",
        "core_block_complete_all4",
        "core_block_incomplete_any",
        "any_stale_core_ge_8h_flag",
        "any_never_observed_core_flag",
        *FROZEN_PROXY_COLUMNS,
        *[f"{column}_missing" for column in FROZEN_PROXY_COLUMNS],
        "n_frozen_proxy_missing",
    ]
    return derived.loc[:, ordered_columns].sort_values(
        ["hard_case_flag", "hospital_id", "stay_id_global"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def _format_count_pct(count: int, total: int) -> str:
    pct = (100.0 * count / total) if total else 0.0
    return f"{count} ({pct:.1f}%)"


def _format_numeric_value(value: float, *, variable_name: str) -> str:
    if pd.isna(value):
        return "NA"
    if variable_name in INTEGER_SUMMARY_VARIABLES:
        return f"{float(value):.0f}"
    return f"{float(value):.2f}"


def _format_continuous_summary(series: pd.Series, *, variable_name: str) -> str:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return "NA (n=0)"

    q1 = float(numeric.quantile(0.25))
    median = float(numeric.quantile(0.50))
    q3 = float(numeric.quantile(0.75))
    return (
        f"{_format_numeric_value(median, variable_name=variable_name)} "
        f"[{_format_numeric_value(q1, variable_name=variable_name)}, "
        f"{_format_numeric_value(q3, variable_name=variable_name)}] "
        f"(n={int(numeric.shape[0])})"
    )


def build_observation_process_comparison_outputs(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hard_group = dataset[dataset["hard_case_flag"].astype(bool)].copy()
    other_group = dataset[~dataset["hard_case_flag"].astype(bool)].copy()
    n_hard = int(hard_group.shape[0])
    n_other = int(other_group.shape[0])
    if n_hard == 0 or n_other == 0:
        raise ValueError("Both fatal comparison groups must be non-empty.")

    comparison_rows: list[dict[str, object]] = []
    effect_rows: list[dict[str, object]] = []
    for spec in VARIABLE_SPECS:
        variable_name = spec["name"]
        variable_label = spec["label"]

        if spec["kind"] == "binary":
            hard_fraction = float(hard_group[variable_name].astype(bool).mean())
            other_fraction = float(other_group[variable_name].astype(bool).mean())
            standardized_difference = _proportion_standardized_difference(
                hard_fraction,
                other_fraction,
            )
            comparison_rows.append(
                {
                    "variable": variable_name,
                    "variable_label": variable_label,
                    "low_predicted_fatal_stays": _format_count_pct(
                        int(hard_group[variable_name].astype(bool).sum()),
                        n_hard,
                    ),
                    "other_fatal_stays": _format_count_pct(
                        int(other_group[variable_name].astype(bool).sum()),
                        n_other,
                    ),
                    "effect_size_type": "standardized difference in proportions",
                    "effect_size_basis": "flag proportion",
                    "standardized_difference": standardized_difference,
                    "absolute_standardized_difference": (
                        abs(standardized_difference) if np.isfinite(standardized_difference) else np.nan
                    ),
                    "low_predicted_fatal_n": n_hard,
                    "other_fatal_n": n_other,
                }
            )
            effect_rows.append(
                {
                    "variable": variable_name,
                    "variable_label": variable_label,
                    "effect_size_type": "standardized difference in proportions",
                    "effect_size_basis": "flag proportion",
                    "standardized_difference": standardized_difference,
                    "absolute_standardized_difference": (
                        abs(standardized_difference) if np.isfinite(standardized_difference) else np.nan
                    ),
                    "low_predicted_value": hard_fraction,
                    "other_value": other_fraction,
                }
            )
            continue

        standardized_difference, details = _continuous_standardized_difference(
            hard_group[variable_name],
            other_group[variable_name],
        )
        comparison_rows.append(
            {
                "variable": variable_name,
                "variable_label": variable_label,
                "low_predicted_fatal_stays": _format_continuous_summary(
                    hard_group[variable_name],
                    variable_name=variable_name,
                ),
                "other_fatal_stays": _format_continuous_summary(
                    other_group[variable_name],
                    variable_name=variable_name,
                ),
                "effect_size_type": "continuous pooled-SD standardized mean difference",
                "effect_size_basis": "available values",
                "standardized_difference": standardized_difference,
                "absolute_standardized_difference": (
                    abs(standardized_difference) if np.isfinite(standardized_difference) else np.nan
                ),
                "low_predicted_fatal_n": n_hard,
                "other_fatal_n": n_other,
            }
        )
        effect_rows.append(
            {
                "variable": variable_name,
                "variable_label": variable_label,
                "effect_size_type": "continuous pooled-SD standardized mean difference",
                "effect_size_basis": "available values",
                "standardized_difference": standardized_difference,
                "absolute_standardized_difference": (
                    abs(standardized_difference) if np.isfinite(standardized_difference) else np.nan
                ),
                "low_predicted_value": details.get("hard_mean"),
                "other_value": details.get("other_mean"),
            }
        )

    comparison_table = pd.DataFrame(comparison_rows)
    effect_details = pd.DataFrame(effect_rows).sort_values(
        "absolute_standardized_difference",
        ascending=False,
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)

    for frame in (comparison_table, effect_details):
        frame["standardized_difference"] = frame["standardized_difference"].round(3)
        frame["absolute_standardized_difference"] = frame["absolute_standardized_difference"].round(3)

    return comparison_table, effect_details


def _plot_effect_sizes(effect_details: pd.DataFrame, *, output_path: Path) -> Path:
    _require_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_frame = effect_details[np.isfinite(effect_details["absolute_standardized_difference"])].copy()
    if plot_frame.empty:
        raise ValueError("No finite effect sizes are available for plotting.")

    plot_frame = plot_frame.sort_values(
        "absolute_standardized_difference",
        ascending=True,
        kind="stable",
    ).reset_index(drop=True)
    figure_height = max(3.6, 0.5 * plot_frame.shape[0] + 1.0)
    figure, axis = plt.subplots(figsize=(8.2, figure_height))
    axis.barh(
        np.arange(plot_frame.shape[0]),
        plot_frame["absolute_standardized_difference"],
        color="#5d7fa3",
        edgecolor="#274766",
        linewidth=0.7,
    )
    axis.set_yticks(np.arange(plot_frame.shape[0]))
    axis.set_yticklabels(plot_frame["variable_label"].tolist(), fontsize=9)
    axis.set_xlabel("Absolute standardized difference")
    axis.set_title("ASIC 24h hard-case observation-process sensitivity", fontsize=12)
    axis.grid(axis="x", color="#d8d8d8", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.text(
        0.99,
        -0.12,
        "Higher values indicate larger separation between low-predicted and other fatal stays.",
        fontsize=8,
        color="#555555",
        ha="right",
        va="top",
        transform=axis.transAxes,
    )
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _comparison_row(table: pd.DataFrame, variable: str) -> pd.Series:
    row = table[table["variable"].eq(variable)]
    if row.empty:
        raise KeyError(f"Variable {variable!r} is missing from the observation-process comparison table.")
    return row.iloc[0]


def build_observation_process_sensitivity_memo(
    dataset: pd.DataFrame,
    comparison_table: pd.DataFrame,
    *,
    metadata: dict[str, object],
) -> str:
    n_hard = int(dataset["hard_case_flag"].astype(bool).sum())
    n_total = int(dataset.shape[0])
    n_other = n_total - n_hard

    completeness = _comparison_row(comparison_table, "core_block_complete_all4")
    stale = _comparison_row(comparison_table, "any_stale_core_ge_8h_flag")
    proxy_missing = _comparison_row(comparison_table, "n_frozen_proxy_missing")
    any_recent = _comparison_row(comparison_table, "time_since_last_any_core_h")
    longest_gap = _comparison_row(comparison_table, "max_time_since_last_core_h")

    memo_lines = [
        "# ASIC Observation-Process Sensitivity Memo",
        "",
        "## Scope",
        "- Issue 4.1 only.",
        "- Frozen ASIC 24h logistic hard-case definition only.",
        "- Fatal stays only: low-predicted fatal stays versus other fatal stays.",
        "- This memo treats observation process as an interpretive threat check, not a redesign of the Chapter 1 claim.",
        "",
        "## Cohort",
        f"- Total fatal stays in the authoritative 24h comparison anchor: `{n_total}`.",
        f"- Low-predicted fatal stays: `{n_hard}`. Other fatal stays: `{n_other}`.",
        f"- Hard-case rule: `{HARD_CASE_RULE}`.",
        "",
        "## Main Descriptive Pattern",
        (
            f"- Anchor-block completeness was lower in low-predicted fatal stays "
            f"({completeness['low_predicted_fatal_stays']} vs {completeness['other_fatal_stays']}; "
            f"absolute standardized difference `{completeness['absolute_standardized_difference']}`)."
        ),
        (
            f"- Stale monitoring was more common in low-predicted fatal stays "
            f"({stale['low_predicted_fatal_stays']} vs {stale['other_fatal_stays']}; "
            f"absolute standardized difference `{stale['absolute_standardized_difference']}`)."
        ),
        (
            f"- Missingness in the four frozen physiologic comparison proxies was also higher "
            f"({proxy_missing['low_predicted_fatal_stays']} vs {proxy_missing['other_fatal_stays']}; "
            f"absolute standardized difference `{proxy_missing['absolute_standardized_difference']}`)."
        ),
        (
            f"- Immediate recency of any core-group observation stayed close between groups "
            f"({any_recent['low_predicted_fatal_stays']} vs {any_recent['other_fatal_stays']}; "
            f"absolute standardized difference `{any_recent['absolute_standardized_difference']}`), "
            f"while the longest core-group recency gap showed only modest separation "
            f"({longest_gap['low_predicted_fatal_stays']} vs {longest_gap['other_fatal_stays']}; "
            f"absolute standardized difference `{longest_gap['absolute_standardized_difference']}`)."
        ),
        "",
        "## Interpretation",
        (
            "- Are hard cases enriched for sparse or irregular monitoring? Yes, but only modestly. "
            "The low-predicted fatal group is less likely to have all four core groups freshly observed "
            "in the anchor block and is more likely to show stale or historical-only monitoring patterns."
        ),
        (
            "- Do observation-process artifacts appear to explain a meaningful share of the hard-case pattern? "
            "They plausibly explain some share of the pattern, especially through incomplete anchor-block coverage "
            "and higher proxy missingness, but the effect sizes remain in the modest range rather than indicating "
            "an overwhelming documentation-only explanation."
        ),
        (
            "- Does this strengthen the bounded descriptive-core interpretation, narrow it, or materially weaken it? "
            "This narrows rather than materially weakens the descriptive core. The Chapter 1 read should stay "
            "conditional on the observed feature set, the documentation process, and the 8-hour temporal aggregation, "
            "with explicit acknowledgement that some low-predicted fatal stays appear more weakly captured at the observation layer."
        ),
        "",
        "## Bounded Chapter 1 Language",
        "- These results do not support biological subtype claims.",
        "- These results do not support irreducible-stochasticity claims.",
        "- These results do not support causal claims about monitoring intensity or treatment decisions.",
        "- All interpretation remains conditional on the observed feature set, documentation process, saved hard-case anchor, and temporal aggregation used in the current ASIC pipeline.",
        "",
        "## Data Availability Notes",
        f"- Authoritative comparison dataset source: `{_display_path(Path(metadata['source_paths']['comparison_dataset_path']))}`.",
        f"- Authoritative hard-case flags source: `{_display_path(Path(metadata['source_paths']['hard_case_path']))}`.",
        f"- Authoritative observation-process block-feature source: `{_display_path(Path(metadata['source_paths']['observation_process_path']))}`.",
        "- The default local-review mirror omitted both row-level producer files above, so they had to be synced separately from the cluster-side true artifact directories for this issue.",
    ]
    return "\n".join(memo_lines) + "\n"


def run_asic_observation_process_sensitivity(
    *,
    comparison_dataset_path: Path = DEFAULT_COMPARISON_DATASET_PATH,
    hard_case_path: Path = DEFAULT_HARD_CASE_PATH,
    observation_process_path: Path = DEFAULT_OBSERVATION_PROCESS_PATH,
    output_dataset_path: Path = DEFAULT_OUTPUT_DATASET_PATH,
    output_comparison_table_path: Path = DEFAULT_OUTPUT_COMPARISON_TABLE_PATH,
    output_effect_details_path: Path = DEFAULT_OUTPUT_EFFECT_DETAILS_PATH,
    output_figure_path: Path = DEFAULT_OUTPUT_FIGURE_PATH,
    output_memo_path: Path = DEFAULT_OUTPUT_MEMO_PATH,
    output_manifest_path: Path = DEFAULT_OUTPUT_MANIFEST_PATH,
    promoted_memo_path: Path | None = None,
) -> ASICObservationProcessSensitivityRunResult:
    anchor_dataset, metadata = load_authoritative_observation_process_anchor(
        comparison_dataset_path=comparison_dataset_path,
        hard_case_path=hard_case_path,
        observation_process_path=observation_process_path,
    )
    dataset = derive_observation_process_sensitivity_dataset(anchor_dataset)
    comparison_table, effect_details = build_observation_process_comparison_outputs(dataset)
    memo_markdown = build_observation_process_sensitivity_memo(
        dataset,
        comparison_table,
        metadata=metadata,
    )

    dataset_path = write_dataframe(dataset, output_dataset_path, output_format="csv")
    comparison_table_path = write_dataframe(
        comparison_table,
        output_comparison_table_path,
        output_format="csv",
    )
    effect_details_path = write_dataframe(
        effect_details,
        output_effect_details_path,
        output_format="csv",
    )
    figure_path = _plot_effect_sizes(effect_details, output_path=output_figure_path)
    memo_path = write_text(memo_markdown, output_memo_path)
    promoted_memo_result_path = _write_promoted_memo(memo_markdown, promoted_memo_path)
    manifest_path = _write_json(
        {
            "timestamp_utc": _utc_timestamp(),
            "target_horizon_h": TARGET_HORIZON_H,
            "hard_case_rule": HARD_CASE_RULE,
            "group_counts": metadata["group_counts"],
            "source_paths": {
                key: str(Path(value).resolve()) for key, value in metadata["source_paths"].items()
            },
            "bundle_gaps_encountered": metadata["bundle_gaps_encountered"],
            "derived_variable_set": [spec["name"] for spec in VARIABLE_SPECS],
            "output_paths": {
                "dataset": str(Path(dataset_path).resolve()),
                "comparison_table": str(Path(comparison_table_path).resolve()),
                "effect_details": str(Path(effect_details_path).resolve()),
                "figure": str(Path(figure_path).resolve()),
                "memo": str(Path(memo_path).resolve()),
                "promoted_memo": (
                    str(Path(promoted_memo_result_path).resolve())
                    if promoted_memo_result_path is not None
                    else None
                ),
            },
        },
        output_manifest_path,
    )

    return ASICObservationProcessSensitivityRunResult(
        dataset=dataset,
        comparison_table=comparison_table,
        effect_details=effect_details,
        memo_markdown=memo_markdown,
        artifacts=ASICObservationProcessSensitivityArtifacts(
            dataset_path=dataset_path,
            comparison_table_path=comparison_table_path,
            effect_details_path=effect_details_path,
            figure_path=figure_path,
            memo_path=memo_path,
            manifest_path=manifest_path,
            promoted_memo_path=promoted_memo_result_path,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen ASIC 24h hard-case observation-process sensitivity comparison "
            "using the authoritative saved cluster-side artifacts."
        )
    )
    parser.add_argument(
        "--comparison-dataset-path",
        type=Path,
        default=DEFAULT_COMPARISON_DATASET_PATH,
        help="Authoritative saved 24h hard-case comparison dataset.",
    )
    parser.add_argument(
        "--hard-case-path",
        type=Path,
        default=DEFAULT_HARD_CASE_PATH,
        help="Authoritative stay-level hard-case flag artifact.",
    )
    parser.add_argument(
        "--observation-process-path",
        type=Path,
        default=DEFAULT_OBSERVATION_PROCESS_PATH,
        help="Authoritative observation-process block-feature artifact.",
    )
    parser.add_argument(
        "--output-dataset-path",
        type=Path,
        default=DEFAULT_OUTPUT_DATASET_PATH,
        help="Path for the derived stay-level observation-process dataset.",
    )
    parser.add_argument(
        "--output-comparison-table-path",
        type=Path,
        default=DEFAULT_OUTPUT_COMPARISON_TABLE_PATH,
        help="Path for the observation-process comparison table.",
    )
    parser.add_argument(
        "--output-effect-details-path",
        type=Path,
        default=DEFAULT_OUTPUT_EFFECT_DETAILS_PATH,
        help="Path for the effect-size detail table.",
    )
    parser.add_argument(
        "--output-figure-path",
        type=Path,
        default=DEFAULT_OUTPUT_FIGURE_PATH,
        help="Path for the compact effect-size figure.",
    )
    parser.add_argument(
        "--output-memo-path",
        type=Path,
        default=DEFAULT_OUTPUT_MEMO_PATH,
        help="Path for the short interpretation memo.",
    )
    parser.add_argument(
        "--output-manifest-path",
        type=Path,
        default=DEFAULT_OUTPUT_MANIFEST_PATH,
        help="Path for the run manifest.",
    )
    parser.add_argument(
        "--promoted-memo-path",
        type=Path,
        default=None,
        help=(
            "Optional repo-level memo copy. Leave unset to keep the canonical memo only inside "
            "the dedicated artifact package."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_asic_observation_process_sensitivity(
        comparison_dataset_path=args.comparison_dataset_path,
        hard_case_path=args.hard_case_path,
        observation_process_path=args.observation_process_path,
        output_dataset_path=args.output_dataset_path,
        output_comparison_table_path=args.output_comparison_table_path,
        output_effect_details_path=args.output_effect_details_path,
        output_figure_path=args.output_figure_path,
        output_memo_path=args.output_memo_path,
        output_manifest_path=args.output_manifest_path,
        promoted_memo_path=args.promoted_memo_path,
    )
    print(f"Derived dataset: {result.artifacts.dataset_path}")
    print(f"Comparison table: {result.artifacts.comparison_table_path}")
    print(f"Effect details: {result.artifacts.effect_details_path}")
    print(f"Figure: {result.artifacts.figure_path}")
    print(f"Memo: {result.artifacts.memo_path}")
    if result.artifacts.promoted_memo_path is not None:
        print(f"Promoted memo: {result.artifacts.promoted_memo_path}")
    print(f"Manifest: {result.artifacts.manifest_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
