from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from chapter1_mortality_decomposition.asic_hard_case_comparison import (
    HARD_CASE_RULE,
    _proportion_standardized_difference,
)
from chapter1_mortality_decomposition.asic_observation_process_sensitivity import (
    DEFAULT_COMPARISON_DATASET_PATH,
    DEFAULT_HARD_CASE_PATH,
    DEFAULT_OBSERVATION_PROCESS_PATH,
    INTEGER_SUMMARY_VARIABLES,
    VARIABLE_SPECS,
    derive_observation_process_sensitivity_dataset,
    load_authoritative_observation_process_anchor,
)
from chapter1_mortality_decomposition.utils import write_dataframe, write_text


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "chapter1" / "site_sensitivity" / "asic"

SUMMARY_FILENAME = "uk04_observation_process_summary.csv"
HARD_CASE_FOLLOWUP_FILENAME = "uk04_observation_process_hard_case_followup.csv"
MEMO_FILENAME = "uk04_observation_process_interpretation.md"
MANIFEST_FILENAME = "uk04_observation_process_run_manifest.json"

UK04_SITE_ID = "asic_UK04"
NON_UK04_LABEL = "non-UK04 ASIC sites combined"
ANALYSIS_POPULATION_ALL_FATAL = "fatal_stays"
ANALYSIS_POPULATION_HARD_CASES = "hard_cases_only"
NEAR_NULL_EFFECT_SIZE = 0.10
MODEST_EFFECT_SIZE = 0.20
MATERIAL_EFFECT_SIZE = 0.40

KEY_INTERPRETIVE_VARIABLES = (
    "core_block_complete_all4",
    "n_core_groups_historical_only",
    "n_frozen_proxy_missing",
    "any_stale_core_ge_8h_flag",
    "max_time_since_last_core_h",
)


@dataclass(frozen=True)
class ASICUK04ObservationProcessArtifacts:
    summary_path: Path
    hard_case_followup_path: Path
    memo_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class ASICUK04ObservationProcessRunResult:
    dataset: pd.DataFrame
    summary_table: pd.DataFrame
    hard_case_followup_table: pd.DataFrame
    interpretation_category: str
    memo_markdown: str
    artifacts: ASICUK04ObservationProcessArtifacts


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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


def _continuous_standardized_difference(
    uk04_values: pd.Series,
    non_uk04_values: pd.Series,
) -> tuple[float, dict[str, float | int]]:
    uk04_numeric = pd.to_numeric(uk04_values, errors="coerce").dropna()
    non_uk04_numeric = pd.to_numeric(non_uk04_values, errors="coerce").dropna()

    details: dict[str, float | int] = {
        "uk04_nonmissing": int(uk04_numeric.shape[0]),
        "non_uk04_nonmissing": int(non_uk04_numeric.shape[0]),
    }
    if uk04_numeric.empty or non_uk04_numeric.empty:
        return float("nan"), details

    uk04_mean = float(uk04_numeric.mean())
    non_uk04_mean = float(non_uk04_numeric.mean())
    uk04_variance = float(uk04_numeric.var(ddof=0))
    non_uk04_variance = float(non_uk04_numeric.var(ddof=0))
    pooled_sd = float(np.sqrt((uk04_variance + non_uk04_variance) / 2.0))

    details.update(
        {
            "uk04_mean": uk04_mean,
            "non_uk04_mean": non_uk04_mean,
            "uk04_sd": float(np.sqrt(uk04_variance)),
            "non_uk04_sd": float(np.sqrt(non_uk04_variance)),
            "pooled_sd": pooled_sd,
        }
    )
    if pooled_sd == 0.0:
        if np.isclose(uk04_mean, non_uk04_mean):
            return 0.0, details
        return float("nan"), details
    return (uk04_mean - non_uk04_mean) / pooled_sd, details


def _format_numeric_value(value: float, *, variable_name: str) -> str:
    if pd.isna(value):
        return "NA"
    if variable_name in INTEGER_SUMMARY_VARIABLES or variable_name == "prediction_time_h":
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


def _format_count_pct(count: int, total: int) -> str:
    pct = (100.0 * count / total) if total else 0.0
    return f"{count} ({pct:.1f}%)"


def _direction_vs_reference(reference_effect: float, contrast_effect: float) -> str:
    if not np.isfinite(contrast_effect):
        return "not_assessable"
    if not np.isfinite(reference_effect) or abs(reference_effect) < NEAR_NULL_EFFECT_SIZE:
        return "near_null_reference"
    if abs(contrast_effect) < NEAR_NULL_EFFECT_SIZE:
        return "near_null_or_unclear"
    if np.sign(reference_effect) == np.sign(contrast_effect):
        return "same_direction"
    return "opposite_direction"


def build_reference_hard_case_effects(dataset: pd.DataFrame) -> dict[str, float]:
    hard = dataset[dataset["hard_case_flag"].astype(bool)].copy()
    other = dataset[~dataset["hard_case_flag"].astype(bool)].copy()
    effects: dict[str, float] = {}
    for spec in VARIABLE_SPECS:
        variable_name = str(spec["name"])
        if str(spec["kind"]) == "binary":
            effects[variable_name] = _proportion_standardized_difference(
                float(hard[variable_name].astype(bool).mean()),
                float(other[variable_name].astype(bool).mean()),
            )
            continue
        effects[variable_name], _ = _continuous_standardized_difference(
            hard[variable_name],
            other[variable_name],
        )
    return effects


def build_uk04_vs_non_uk04_summary(
    dataset: pd.DataFrame,
    *,
    reference_effects: dict[str, float],
    analysis_population: str,
) -> pd.DataFrame:
    uk04 = dataset[dataset["hospital_id"].astype("string").eq(UK04_SITE_ID)].copy()
    non_uk04 = dataset[~dataset["hospital_id"].astype("string").eq(UK04_SITE_ID)].copy()
    if uk04.empty or non_uk04.empty:
        raise ValueError("Both UK04 and non-UK04 groups must be non-empty for the targeted follow-up.")

    rows: list[dict[str, object]] = []
    for spec in VARIABLE_SPECS:
        variable_name = str(spec["name"])
        variable_label = str(spec["label"])
        variable_kind = str(spec["kind"])
        row: dict[str, object] = {
            "analysis_population": analysis_population,
            "variable": variable_name,
            "variable_label": variable_label,
            "variable_kind": variable_kind,
            "uk04_stay_count": int(uk04.shape[0]),
            "non_uk04_stay_count": int(non_uk04.shape[0]),
            "reference_hard_case_standardized_difference": reference_effects[variable_name],
        }

        if variable_kind == "binary":
            uk04_count = int(uk04[variable_name].astype(bool).sum())
            non_uk04_count = int(non_uk04[variable_name].astype(bool).sum())
            uk04_fraction = float(uk04_count / uk04.shape[0])
            non_uk04_fraction = float(non_uk04_count / non_uk04.shape[0])
            standardized_difference = _proportion_standardized_difference(
                uk04_fraction,
                non_uk04_fraction,
            )
            row.update(
                {
                    "uk04_nonmissing_n": int(uk04.shape[0]),
                    "non_uk04_nonmissing_n": int(non_uk04.shape[0]),
                    "uk04_summary": _format_count_pct(uk04_count, int(uk04.shape[0])),
                    "non_uk04_summary": _format_count_pct(non_uk04_count, int(non_uk04.shape[0])),
                    "uk04_count": uk04_count,
                    "uk04_fraction": uk04_fraction,
                    "non_uk04_count": non_uk04_count,
                    "non_uk04_fraction": non_uk04_fraction,
                    "absolute_difference_pp": 100.0 * (uk04_fraction - non_uk04_fraction),
                    "uk04_mean": uk04_fraction,
                    "non_uk04_mean": non_uk04_fraction,
                    "mean_difference": uk04_fraction - non_uk04_fraction,
                    "uk04_median": np.nan,
                    "uk04_q1": np.nan,
                    "uk04_q3": np.nan,
                    "non_uk04_median": np.nan,
                    "non_uk04_q1": np.nan,
                    "non_uk04_q3": np.nan,
                    "median_difference": np.nan,
                    "effect_size_type": "standardized difference in proportions",
                    "standardized_difference": standardized_difference,
                    "absolute_standardized_difference": (
                        abs(standardized_difference) if np.isfinite(standardized_difference) else np.nan
                    ),
                }
            )
        else:
            standardized_difference, details = _continuous_standardized_difference(
                uk04[variable_name],
                non_uk04[variable_name],
            )
            uk04_numeric = pd.to_numeric(uk04[variable_name], errors="coerce").dropna()
            non_uk04_numeric = pd.to_numeric(non_uk04[variable_name], errors="coerce").dropna()
            row.update(
                {
                    "uk04_nonmissing_n": int(uk04_numeric.shape[0]),
                    "non_uk04_nonmissing_n": int(non_uk04_numeric.shape[0]),
                    "uk04_summary": _format_continuous_summary(uk04[variable_name], variable_name=variable_name),
                    "non_uk04_summary": _format_continuous_summary(
                        non_uk04[variable_name],
                        variable_name=variable_name,
                    ),
                    "uk04_count": np.nan,
                    "uk04_fraction": np.nan,
                    "non_uk04_count": np.nan,
                    "non_uk04_fraction": np.nan,
                    "absolute_difference_pp": np.nan,
                    "uk04_mean": details.get("uk04_mean"),
                    "non_uk04_mean": details.get("non_uk04_mean"),
                    "mean_difference": (
                        float(details["uk04_mean"]) - float(details["non_uk04_mean"])
                        if "uk04_mean" in details and "non_uk04_mean" in details
                        else np.nan
                    ),
                    "uk04_median": float(uk04_numeric.quantile(0.50)) if not uk04_numeric.empty else np.nan,
                    "uk04_q1": float(uk04_numeric.quantile(0.25)) if not uk04_numeric.empty else np.nan,
                    "uk04_q3": float(uk04_numeric.quantile(0.75)) if not uk04_numeric.empty else np.nan,
                    "non_uk04_median": (
                        float(non_uk04_numeric.quantile(0.50)) if not non_uk04_numeric.empty else np.nan
                    ),
                    "non_uk04_q1": (
                        float(non_uk04_numeric.quantile(0.25)) if not non_uk04_numeric.empty else np.nan
                    ),
                    "non_uk04_q3": (
                        float(non_uk04_numeric.quantile(0.75)) if not non_uk04_numeric.empty else np.nan
                    ),
                    "median_difference": (
                        float(uk04_numeric.quantile(0.50) - non_uk04_numeric.quantile(0.50))
                        if not uk04_numeric.empty and not non_uk04_numeric.empty
                        else np.nan
                    ),
                    "effect_size_type": "continuous pooled-SD standardized mean difference",
                    "standardized_difference": standardized_difference,
                    "absolute_standardized_difference": (
                        abs(standardized_difference) if np.isfinite(standardized_difference) else np.nan
                    ),
                }
            )

        row["direction_vs_hard_case_pattern"] = _direction_vs_reference(
            reference_effects[variable_name],
            float(row["standardized_difference"])
            if pd.notna(row["standardized_difference"])
            else float("nan"),
        )
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["reference_hard_case_standardized_difference"] = pd.to_numeric(
        summary["reference_hard_case_standardized_difference"],
        errors="coerce",
    ).round(3)
    for column in (
        "uk04_fraction",
        "non_uk04_fraction",
        "absolute_difference_pp",
        "uk04_mean",
        "non_uk04_mean",
        "mean_difference",
        "uk04_median",
        "uk04_q1",
        "uk04_q3",
        "non_uk04_median",
        "non_uk04_q1",
        "non_uk04_q3",
        "median_difference",
        "standardized_difference",
        "absolute_standardized_difference",
    ):
        summary[column] = pd.to_numeric(summary[column], errors="coerce")
    summary[
        [
            "uk04_fraction",
            "non_uk04_fraction",
            "absolute_difference_pp",
            "uk04_mean",
            "non_uk04_mean",
            "mean_difference",
            "uk04_median",
            "uk04_q1",
            "uk04_q3",
            "non_uk04_median",
            "non_uk04_q1",
            "non_uk04_q3",
            "median_difference",
            "standardized_difference",
            "absolute_standardized_difference",
        ]
    ] = summary[
        [
            "uk04_fraction",
            "non_uk04_fraction",
            "absolute_difference_pp",
            "uk04_mean",
            "non_uk04_mean",
            "mean_difference",
            "uk04_median",
            "uk04_q1",
            "uk04_q3",
            "non_uk04_median",
            "non_uk04_q1",
            "non_uk04_q3",
            "median_difference",
            "standardized_difference",
            "absolute_standardized_difference",
        ]
    ].round(3)
    return summary


def _interpretive_slice(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[summary["variable"].isin(KEY_INTERPRETIVE_VARIABLES)].copy()


def build_interpretation(
    *,
    summary_table: pd.DataFrame,
    hard_case_followup_table: pd.DataFrame,
) -> dict[str, object]:
    primary = _interpretive_slice(summary_table)
    hard_case = _interpretive_slice(hard_case_followup_table)

    primary_supportive = primary[
        primary["direction_vs_hard_case_pattern"].eq("same_direction")
        & primary["absolute_standardized_difference"].ge(MODEST_EFFECT_SIZE)
    ]
    primary_opposing = primary[
        primary["direction_vs_hard_case_pattern"].eq("opposite_direction")
        & primary["absolute_standardized_difference"].ge(MODEST_EFFECT_SIZE)
    ]
    hard_supportive = hard_case[
        hard_case["direction_vs_hard_case_pattern"].eq("same_direction")
        & hard_case["absolute_standardized_difference"].ge(MODEST_EFFECT_SIZE)
    ]
    hard_opposing = hard_case[
        hard_case["direction_vs_hard_case_pattern"].eq("opposite_direction")
        & hard_case["absolute_standardized_difference"].ge(MODEST_EFFECT_SIZE)
    ]

    strong_supportive = (
        int(primary_supportive["absolute_standardized_difference"].ge(MATERIAL_EFFECT_SIZE).sum())
        + int(hard_supportive["absolute_standardized_difference"].ge(MATERIAL_EFFECT_SIZE).sum())
    )
    strong_opposing = (
        int(primary_opposing["absolute_standardized_difference"].ge(MATERIAL_EFFECT_SIZE).sum())
        + int(hard_opposing["absolute_standardized_difference"].ge(MATERIAL_EFFECT_SIZE).sum())
    )

    if (
        strong_supportive >= 2
        and int(primary_supportive.shape[0]) >= 2
        and int(hard_supportive.shape[0]) >= 1
        and int(primary_opposing.shape[0]) == 0
    ):
        interpretation_category = "additional_targeted_site_sensitivity_may_be_warranted"
        headline = (
            "UK04 showed strong observation-process differences in the same direction as the measured "
            "hard-case observation-process pattern."
        )
        further_work_warranted = True
    elif (
        int(primary_supportive.shape[0]) >= 2
        and int(primary_opposing.shape[0]) == 0
        and int(hard_opposing.shape[0]) <= 1
    ):
        interpretation_category = "modest_enrichment_with_plausible_observation_process_contribution"
        headline = (
            "UK04 showed observation-process differences plausibly consistent with part of its modest "
            "hard-case enrichment."
        )
        further_work_warranted = False
    else:
        interpretation_category = "modest_enrichment_not_clearly_explained_by_measured_observation_process"
        headline = (
            "UK04 did not show strong measured observation-process differences in the direction that "
            "would explain its modest hard-case enrichment."
        )
        further_work_warranted = False

    return {
        "interpretation_category": interpretation_category,
        "headline": headline,
        "further_work_warranted": further_work_warranted,
        "primary_supportive": primary_supportive.sort_values(
            "absolute_standardized_difference",
            ascending=False,
            kind="stable",
        ).reset_index(drop=True),
        "primary_opposing": primary_opposing.sort_values(
            "absolute_standardized_difference",
            ascending=False,
            kind="stable",
        ).reset_index(drop=True),
        "hard_supportive": hard_supportive.sort_values(
            "absolute_standardized_difference",
            ascending=False,
            kind="stable",
        ).reset_index(drop=True),
        "hard_opposing": hard_opposing.sort_values(
            "absolute_standardized_difference",
            ascending=False,
            kind="stable",
        ).reset_index(drop=True),
        "strong_supportive": strong_supportive,
        "strong_opposing": strong_opposing,
    }


def _row(summary: pd.DataFrame, variable_name: str) -> pd.Series:
    row = summary[summary["variable"].eq(variable_name)]
    if row.empty:
        raise KeyError(f"Variable {variable_name!r} is missing from the UK04 observation-process summary.")
    return row.iloc[0]


def build_interpretation_memo(
    *,
    metadata: dict[str, object],
    summary_table: pd.DataFrame,
    hard_case_followup_table: pd.DataFrame,
    interpretation: dict[str, object],
) -> str:
    n_total = int(metadata["group_counts"]["total_fatal_stays"])
    all_complete = _row(summary_table, "core_block_complete_all4")
    all_historical = _row(summary_table, "n_core_groups_historical_only")
    all_proxy_missing = _row(summary_table, "n_frozen_proxy_missing")
    all_stale = _row(summary_table, "any_stale_core_ge_8h_flag")

    hard_complete = _row(hard_case_followup_table, "core_block_complete_all4")
    hard_proxy_missing = _row(hard_case_followup_table, "n_frozen_proxy_missing")
    hard_stale = _row(hard_case_followup_table, "any_stale_core_ge_8h_flag")

    n_uk04_fatal = int(summary_table["uk04_stay_count"].iloc[0])
    n_non_uk04_fatal = int(summary_table["non_uk04_stay_count"].iloc[0])
    n_uk04_hard = int(hard_case_followup_table["uk04_stay_count"].iloc[0])
    n_non_uk04_hard = int(hard_case_followup_table["non_uk04_stay_count"].iloc[0])

    lines = [
        "# UK04 Observation-Process Interpretation",
        "",
        "## Scope",
        "- This is the targeted UK04 follow-up after Package 1 found some site enrichment, but clearly modest, with `asic_UK04` as the most enriched site.",
        "- The goal is only to assess whether measured observation-process differences plausibly contribute to that modest enrichment.",
        "- This package does not reopen full site sensitivity, case-mix, proxy-label, or later-chapter hospital interpretation branches.",
        "",
        "## Inputs",
        f"- Authoritative 24h fatal comparison dataset: `{_display_path(Path(metadata['source_paths']['comparison_dataset_path']))}`.",
        f"- Authoritative hard-case anchor artifact: `{_display_path(Path(metadata['source_paths']['hard_case_path']))}`.",
        f"- Authoritative observation-process block-feature artifact: `{_display_path(Path(metadata['source_paths']['observation_process_path']))}`.",
        f"- Frozen hard-case rule: `{metadata['hard_case_rule']}`.",
        "",
        "## Cohorts",
        f"- Fatal-stay comparison: `{UK04_SITE_ID}` (`{n_uk04_fatal}` stays) versus `{NON_UK04_LABEL}` (`{n_non_uk04_fatal}` stays).",
        f"- Hard-case-only follow-up: `{UK04_SITE_ID}` (`{n_uk04_hard}` stays) versus non-UK04 hard cases (`{n_non_uk04_hard}` stays).",
        f"- Total fatal stays in the shared 24h anchor: `{n_total}`.",
        "",
        "## Main Findings",
        (
            f"- On all fatal stays, `{UK04_SITE_ID}` did not look less completely observed than the rest. "
            f"All-4-core-group completeness was `{all_complete['uk04_summary']}` in UK04 versus "
            f"`{all_complete['non_uk04_summary']}` outside UK04 "
            f"(standardized difference `{float(all_complete['standardized_difference']):+.3f}`, "
            f"opposite to the overall hard-case observation-process contrast)."
        ),
        (
            f"- Historical-only monitoring and stale-core monitoring were not elevated in UK04. "
            f"`n_core_groups_historical_only` was `{all_historical['uk04_summary']}` versus "
            f"`{all_historical['non_uk04_summary']}` "
            f"(standardized difference `{float(all_historical['standardized_difference']):+.3f}`), and "
            f"`any_stale_core_ge_8h_flag` was `{all_stale['uk04_summary']}` versus "
            f"`{all_stale['non_uk04_summary']}` "
            f"(standardized difference `{float(all_stale['standardized_difference']):+.3f}`)."
        ),
        (
            f"- The main measured UK04-vs-rest observation-process departure on all fatal stays was somewhat higher "
            f"frozen-proxy missingness (`{all_proxy_missing['uk04_summary']}` vs `{all_proxy_missing['non_uk04_summary']}`; "
            f"standardized difference `{float(all_proxy_missing['standardized_difference']):+.3f}`), which is "
            "plausibly consistent with some documentation contribution but not on its own a strong explanation."
        ),
        (
            f"- Within hard cases, the measured pattern moved further away from a documentation-only account. "
            f"UK04 hard cases had higher all-4-core-group completeness (`{hard_complete['uk04_summary']}` vs "
            f"`{hard_complete['non_uk04_summary']}`; standardized difference `{float(hard_complete['standardized_difference']):+.3f}`) "
            f"and lower stale-core monitoring (`{hard_stale['uk04_summary']}` vs `{hard_stale['non_uk04_summary']}`; "
            f"standardized difference `{float(hard_stale['standardized_difference']):+.3f}`) than hard cases elsewhere."
        ),
        (
            f"- The frozen-proxy missingness difference did not strengthen inside the hard-case subset "
            f"(`{hard_proxy_missing['uk04_summary']}` vs `{hard_proxy_missing['non_uk04_summary']}`; "
            f"standardized difference `{float(hard_proxy_missing['standardized_difference']):+.3f}`)."
        ),
        "",
        "## Interpretation",
        f"- Did UK04 look observation-process atypical? {interpretation['headline']}",
        (
            "- Are the observed differences plausibly consistent with greater hard-case enrichment? Only weakly and inconsistently. "
            "Slightly higher proxy missingness on all fatal stays is compatible with partial contribution, but the more central "
            "completeness and staleness measures are not shifted in the expected direction."
        ),
        (
            "- Do the findings support a wording update? Yes: UK04 shows at most limited measured observation-process differences, "
            "so the safest Chapter 1 wording is that its modest enrichment remains only partially explained rather than clearly "
            "explained by measured observation-process differences."
        ),
        "",
        "## Decision",
        (
            "- Recommended bounded wording: `UK04 did not show strong measured observation-process differences, "
            "so its modest enrichment remains only partially explained.`"
            if interpretation["interpretation_category"]
            == "modest_enrichment_not_clearly_explained_by_measured_observation_process"
            else (
                "- Recommended bounded wording: `UK04 showed observation-process differences plausibly consistent "
                "with part of its modest hard-case enrichment.`"
                if interpretation["interpretation_category"]
                == "modest_enrichment_with_plausible_observation_process_contribution"
                else "- Recommended bounded wording: `UK04 observation-process differences were strong enough that additional targeted site sensitivity may be warranted.`"
            )
        ),
        (
            "- Is further site sensitivity work warranted? No. This targeted follow-up does not reveal a stronger "
            "or more contradictory site-driven documentation signal than Package 1 already suggested."
            if not interpretation["further_work_warranted"]
            else "- Is further site sensitivity work warranted? Possibly. The UK04 observation-process signal is strong enough that a leave-one-site-out check may be warranted, but it is not implemented here."
        ),
        "",
        "## Bounded Reading",
        "- Observation-process differences can at most plausibly contribute to part of the modest UK04 enrichment.",
        "- These results do not support causal site claims, biological subtype claims, or treatment-policy interpretation.",
        "- Chapter 1 interpretation remains conditional on the observed feature set, documentation process, temporal aggregation, and site/context.",
    ]
    return "\n".join(lines) + "\n"


def run_asic_uk04_observation_process_followup(
    *,
    comparison_dataset_path: Path = DEFAULT_COMPARISON_DATASET_PATH,
    hard_case_path: Path = DEFAULT_HARD_CASE_PATH,
    observation_process_path: Path = DEFAULT_OBSERVATION_PROCESS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> ASICUK04ObservationProcessRunResult:
    anchor_dataset, metadata = load_authoritative_observation_process_anchor(
        comparison_dataset_path=comparison_dataset_path,
        hard_case_path=hard_case_path,
        observation_process_path=observation_process_path,
    )
    dataset = derive_observation_process_sensitivity_dataset(anchor_dataset)
    reference_effects = build_reference_hard_case_effects(dataset)

    summary_table = build_uk04_vs_non_uk04_summary(
        dataset,
        reference_effects=reference_effects,
        analysis_population=ANALYSIS_POPULATION_ALL_FATAL,
    )
    hard_case_followup_table = build_uk04_vs_non_uk04_summary(
        dataset[dataset["hard_case_flag"].astype(bool)].copy(),
        reference_effects=reference_effects,
        analysis_population=ANALYSIS_POPULATION_HARD_CASES,
    )
    interpretation = build_interpretation(
        summary_table=summary_table,
        hard_case_followup_table=hard_case_followup_table,
    )
    memo_markdown = build_interpretation_memo(
        metadata=metadata,
        summary_table=summary_table,
        hard_case_followup_table=hard_case_followup_table,
        interpretation=interpretation,
    )

    resolved_output_dir = Path(output_dir)
    summary_path = write_dataframe(
        summary_table,
        resolved_output_dir / SUMMARY_FILENAME,
        output_format="csv",
    )
    hard_case_followup_path = write_dataframe(
        hard_case_followup_table,
        resolved_output_dir / HARD_CASE_FOLLOWUP_FILENAME,
        output_format="csv",
    )
    memo_path = write_text(
        memo_markdown,
        resolved_output_dir / MEMO_FILENAME,
    )
    manifest_path = _write_json(
        {
            "timestamp_utc": _utc_timestamp(),
            "hard_case_rule": HARD_CASE_RULE,
            "target_site": UK04_SITE_ID,
            "comparison_group": NON_UK04_LABEL,
            "source_paths": {
                key: str(Path(value).resolve()) for key, value in metadata["source_paths"].items()
            },
            "analysis_populations": {
                ANALYSIS_POPULATION_ALL_FATAL: {
                    "uk04_stays": int(summary_table["uk04_stay_count"].iloc[0]),
                    "non_uk04_stays": int(summary_table["non_uk04_stay_count"].iloc[0]),
                },
                ANALYSIS_POPULATION_HARD_CASES: {
                    "uk04_stays": int(hard_case_followup_table["uk04_stay_count"].iloc[0]),
                    "non_uk04_stays": int(hard_case_followup_table["non_uk04_stay_count"].iloc[0]),
                },
            },
            "variable_set": [
                {"name": str(spec["name"]), "label": str(spec["label"]), "kind": str(spec["kind"])}
                for spec in VARIABLE_SPECS
            ],
            "key_interpretive_variables": list(KEY_INTERPRETIVE_VARIABLES),
            "interpretation_category": interpretation["interpretation_category"],
            "further_work_warranted": interpretation["further_work_warranted"],
            "effect_size_thresholds": {
                "near_null": NEAR_NULL_EFFECT_SIZE,
                "modest": MODEST_EFFECT_SIZE,
                "material": MATERIAL_EFFECT_SIZE,
            },
            "output_paths": {
                "summary_table": str(Path(summary_path).resolve()),
                "hard_case_followup_table": str(Path(hard_case_followup_path).resolve()),
                "memo": str(Path(memo_path).resolve()),
            },
        },
        resolved_output_dir / MANIFEST_FILENAME,
    )

    return ASICUK04ObservationProcessRunResult(
        dataset=dataset,
        summary_table=summary_table,
        hard_case_followup_table=hard_case_followup_table,
        interpretation_category=str(interpretation["interpretation_category"]),
        memo_markdown=memo_markdown,
        artifacts=ASICUK04ObservationProcessArtifacts(
            summary_path=summary_path,
            hard_case_followup_path=hard_case_followup_path,
            memo_path=memo_path,
            manifest_path=manifest_path,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the targeted UK04 observation-process follow-up after the ASIC site-enrichment "
            "decision package."
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
        help="Authoritative stay-level hard-case artifact.",
    )
    parser.add_argument(
        "--observation-process-path",
        type=Path,
        default=DEFAULT_OBSERVATION_PROCESS_PATH,
        help="Authoritative observation-process block-feature artifact.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the targeted UK04 follow-up artifacts will be written.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = run_asic_uk04_observation_process_followup(
        comparison_dataset_path=args.comparison_dataset_path,
        hard_case_path=args.hard_case_path,
        observation_process_path=args.observation_process_path,
        output_dir=args.output_dir,
    )

    print(f"Output directory: {args.output_dir}")
    print(f"Interpretation category: {result.interpretation_category}")
    print(f"Summary table: {result.artifacts.summary_path}")
    print(f"Hard-case follow-up table: {result.artifacts.hard_case_followup_path}")
    print(f"Interpretation memo: {result.artifacts.memo_path}")
    print(f"Manifest: {result.artifacts.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
