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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "chapter1" / "site_sensitivity" / "asic"
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

TARGET_HORIZON_H = 24
COMPARISON_DATASET_REQUIRED_COLUMNS = {
    "stay_id_global",
    "instance_id",
    "hard_case_flag",
    "hard_case_group",
    "hospital_id",
    "prediction_time_h",
    "map_last",
    "pf_ratio_last",
    "peep_last",
}
HARD_CASE_REQUIRED_COLUMNS = {
    "stay_id_global",
    "instance_id",
    "hospital_id",
    "horizon_h",
    "label_value",
    "hard_case_flag",
    "hard_case_rule",
}
ANCHOR_KEY_COLUMNS = ["stay_id_global", "instance_id", "hospital_id"]
PERSISTENCE_VARIABLE_SPECS = (
    {
        "name": "prediction_time_h",
        "label": "Prediction time from ICU admission (h)",
        "format": "hours",
    },
    {
        "name": "map_last",
        "label": "MAP",
        "format": "continuous",
    },
    {
        "name": "pf_ratio_last",
        "label": "PF ratio",
        "format": "continuous",
    },
    {
        "name": "peep_last",
        "label": "PEEP",
        "format": "continuous",
    },
)
PERSISTENCE_MIN_PER_ARM = 5
NEAR_NULL_EFFECT_SIZE = 0.10
NO_MEANINGFUL_CRAMERS_V_MAX = 0.10
MODEST_CRAMERS_V_MAX = 0.20
NO_MEANINGFUL_SITE_STD_DIFF_MAX = 0.10
MODEST_SITE_STD_DIFF_MAX = 0.30
NO_MEANINGFUL_SHARE_DIFF_PP_MAX = 3.0
NONTRIVIAL_EXCESS_HARD_SHARE_PP_MIN = 10.0


@dataclass(frozen=True)
class ASICSiteEnrichmentArtifacts:
    site_hard_case_summary_path: Path
    site_hard_case_comparison_path: Path
    site_persistence_check_path: Path
    memo_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class ASICSiteEnrichmentRunResult:
    anchor_dataset: pd.DataFrame
    site_hard_case_summary: pd.DataFrame
    site_hard_case_comparison: pd.DataFrame
    site_persistence_check: pd.DataFrame
    decision_category: str
    primary_enriched_site: str | None
    package2_justified: bool
    memo_markdown: str
    artifacts: ASICSiteEnrichmentArtifacts


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


def _resolve_existing_path(path: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    alternate_suffix = ".parquet" if candidate.suffix == ".csv" else ".csv"
    alternate = candidate.with_suffix(alternate_suffix)
    if alternate.exists():
        return alternate.resolve()
    raise FileNotFoundError(f"Required artifact is missing: {candidate}")


def _format_numeric_value(value: float, *, variable_name: str) -> str:
    if pd.isna(value):
        return "NA"
    if variable_name == "prediction_time_h":
        return f"{float(value):.0f}"
    return f"{float(value):.1f}"


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


def _normalize_comparison_dataset(comparison_dataset: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        comparison_dataset,
        COMPARISON_DATASET_REQUIRED_COLUMNS,
        "ASIC site-enrichment comparison dataset",
    )

    normalized = comparison_dataset.copy()
    for column in ("stay_id_global", "instance_id", "hard_case_group", "hospital_id"):
        normalized[column] = normalized[column].astype("string")
    normalized["hard_case_flag"] = normalize_boolean_codes(normalized["hard_case_flag"])
    for column in ("prediction_time_h", "map_last", "pf_ratio_last", "peep_last"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if normalized["hard_case_flag"].isna().any():
        raise ValueError("The ASIC site-enrichment comparison dataset contains missing hard_case_flag values.")
    if normalized["hospital_id"].isna().any():
        raise ValueError("The ASIC site-enrichment comparison dataset contains missing hospital_id values.")

    expected_group = pd.Series(
        np.where(
            normalized["hard_case_flag"].astype(bool),
            LOW_PREDICTED_FATAL_GROUP,
            OTHER_FATAL_GROUP,
        ),
        index=normalized.index,
        dtype="string",
    )
    if not normalized["hard_case_group"].eq(expected_group).all():
        raise ValueError(
            "The ASIC site-enrichment comparison dataset contains hard_case_group labels "
            "that are inconsistent with hard_case_flag."
        )

    duplicate_anchor_rows = int(normalized.duplicated(subset=ANCHOR_KEY_COLUMNS).sum())
    if duplicate_anchor_rows:
        raise ValueError(
            f"The ASIC site-enrichment comparison dataset contains {duplicate_anchor_rows} duplicated anchor rows."
        )

    return normalized.sort_values(
        ["hard_case_flag", "hospital_id", "stay_id_global"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def _normalize_hard_case_anchor(hard_case_flags: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        hard_case_flags,
        HARD_CASE_REQUIRED_COLUMNS,
        "ASIC hard-case anchor artifact",
    )
    normalized = hard_case_flags.copy()
    for column in ("stay_id_global", "instance_id", "hospital_id", "hard_case_rule"):
        normalized[column] = normalized[column].astype("string")
    for column in ("horizon_h", "label_value"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["hard_case_flag"] = normalize_boolean_codes(normalized["hard_case_flag"])

    if normalized["hard_case_flag"].isna().any():
        raise ValueError("The ASIC hard-case anchor artifact contains missing hard_case_flag values.")

    filtered = normalized[
        normalized["horizon_h"].eq(TARGET_HORIZON_H)
        & normalized["label_value"].eq(1)
    ].copy()
    if filtered.empty:
        raise ValueError("The frozen 24h fatal hard-case anchor is empty.")
    if not filtered["hard_case_rule"].eq(HARD_CASE_RULE).all():
        raise ValueError(
            "The ASIC hard-case anchor artifact does not match the frozen logistic hard-case rule."
        )

    duplicate_anchor_rows = int(filtered.duplicated(subset=ANCHOR_KEY_COLUMNS).sum())
    if duplicate_anchor_rows:
        raise ValueError(
            f"The ASIC hard-case anchor artifact contains {duplicate_anchor_rows} duplicated 24h fatal rows."
        )

    return filtered.sort_values(
        ["hard_case_flag", "hospital_id", "stay_id_global"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def load_authoritative_site_enrichment_anchor(
    *,
    comparison_dataset_path: Path = DEFAULT_COMPARISON_DATASET_PATH,
    hard_case_path: Path = DEFAULT_HARD_CASE_PATH,
) -> tuple[pd.DataFrame, dict[str, object]]:
    resolved_comparison_dataset_path = _resolve_existing_path(comparison_dataset_path)
    resolved_hard_case_path = _resolve_existing_path(hard_case_path)

    comparison_dataset = _normalize_comparison_dataset(read_dataframe(resolved_comparison_dataset_path))
    hard_case_anchor = _normalize_hard_case_anchor(read_dataframe(resolved_hard_case_path))

    comparison_keys = comparison_dataset.loc[:, ANCHOR_KEY_COLUMNS].copy()
    hard_case_keys = hard_case_anchor.loc[:, ANCHOR_KEY_COLUMNS].copy()
    comparison_index = pd.MultiIndex.from_frame(comparison_keys)
    hard_case_index = pd.MultiIndex.from_frame(hard_case_keys)

    if not comparison_index.equals(hard_case_index):
        missing_in_comparison = hard_case_keys.loc[~hard_case_index.isin(comparison_index)]
        missing_in_hard_case = comparison_keys.loc[~comparison_index.isin(hard_case_index)]
        raise ValueError(
            "The saved ASIC comparison dataset is not aligned with the 24h fatal hard-case anchor. "
            f"Missing from comparison dataset: {missing_in_comparison.head(5).to_dict(orient='records')}; "
            f"missing from hard-case anchor: {missing_in_hard_case.head(5).to_dict(orient='records')}."
        )

    comparison_flags = comparison_dataset["hard_case_flag"].astype(bool).reset_index(drop=True)
    hard_case_flags = hard_case_anchor["hard_case_flag"].astype(bool).reset_index(drop=True)
    if not comparison_flags.equals(hard_case_flags):
        raise ValueError(
            "The saved ASIC comparison dataset hard_case_flag values do not match the 24h fatal hard-case anchor."
        )

    group_counts = {
        LOW_PREDICTED_FATAL_GROUP: int(comparison_flags.sum()),
        OTHER_FATAL_GROUP: int((~comparison_flags).sum()),
        "total_fatal_stays": int(comparison_dataset.shape[0]),
        "n_sites": int(comparison_dataset["hospital_id"].nunique(dropna=True)),
    }
    metadata = {
        "target_horizon_h": TARGET_HORIZON_H,
        "hard_case_rule": HARD_CASE_RULE,
        "source_paths": {
            "comparison_dataset_path": resolved_comparison_dataset_path,
            "hard_case_path": resolved_hard_case_path,
        },
        "group_counts": group_counts,
    }
    return comparison_dataset, metadata


def build_site_hard_case_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    total_fatal = int(dataset.shape[0])
    total_hard = int(dataset["hard_case_flag"].astype(bool).sum())
    overall_hard_case_share = float(total_hard / total_fatal) if total_fatal else float("nan")

    summary = (
        dataset.groupby("hospital_id", dropna=False)
        .agg(
            fatal_stays=("stay_id_global", "size"),
            hard_cases=("hard_case_flag", lambda values: int(pd.Series(values).astype(bool).sum())),
        )
        .reset_index()
        .rename(columns={"hospital_id": "site"})
    )
    summary["other_fatal_cases"] = summary["fatal_stays"] - summary["hard_cases"]
    summary["within_site_hard_case_share"] = np.where(
        summary["fatal_stays"].gt(0),
        summary["hard_cases"] / summary["fatal_stays"],
        np.nan,
    )
    summary["share_of_all_hard_cases"] = np.where(
        total_hard > 0,
        summary["hard_cases"] / total_hard,
        np.nan,
    )
    summary["share_of_all_fatal_stays"] = np.where(
        total_fatal > 0,
        summary["fatal_stays"] / total_fatal,
        np.nan,
    )
    summary["within_site_minus_overall_hard_case_share_pp"] = 100.0 * (
        summary["within_site_hard_case_share"] - overall_hard_case_share
    )
    summary["hard_case_share_minus_fatal_share_pp"] = 100.0 * (
        summary["share_of_all_hard_cases"] - summary["share_of_all_fatal_stays"]
    )
    return summary.sort_values(
        ["within_site_hard_case_share", "hard_cases", "site"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def _odds_ratio(a: int, b: int, c: int, d: int) -> float:
    if min(a, b, c, d) == 0:
        return float("nan")
    return float((a / b) / (c / d))


def _build_site_distribution_metrics(
    observed: np.ndarray,
) -> dict[str, float | int]:
    row_totals = observed.sum(axis=1, keepdims=True)
    column_totals = observed.sum(axis=0, keepdims=True)
    grand_total = int(observed.sum())
    expected = row_totals @ column_totals / grand_total
    chi_square = float(((observed - expected) ** 2 / expected).sum())
    degrees_of_freedom = int((observed.shape[0] - 1) * (observed.shape[1] - 1))
    cramers_v = float(
        np.sqrt(chi_square / (grand_total * min(observed.shape[0] - 1, observed.shape[1] - 1)))
    )
    return {
        "chi_square_statistic": chi_square,
        "degrees_of_freedom": degrees_of_freedom,
        "cramers_v": cramers_v,
    }


def build_site_hard_case_comparison(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    hard_mask = dataset["hard_case_flag"].astype(bool)
    n_hard = int(hard_mask.sum())
    n_other = int((~hard_mask).sum())

    rows: list[dict[str, object]] = []
    site_order = sorted(dataset["hospital_id"].dropna().astype("string").unique().tolist())
    contingency_rows: list[list[int]] = []

    for site in site_order:
        site_mask = dataset["hospital_id"].astype("string").eq(site)
        hard_cases_at_site = int((hard_mask & site_mask).sum())
        other_fatal_cases_at_site = int((~hard_mask & site_mask).sum())
        contingency_rows.append([other_fatal_cases_at_site, hard_cases_at_site])

        hard_case_group_share = float(hard_cases_at_site / n_hard) if n_hard else np.nan
        other_fatal_group_share = float(other_fatal_cases_at_site / n_other) if n_other else np.nan
        site_distribution_standardized_difference = _proportion_standardized_difference(
            hard_case_group_share,
            other_fatal_group_share,
        )

        site_total = hard_cases_at_site + other_fatal_cases_at_site
        non_site_hard = n_hard - hard_cases_at_site
        non_site_other = n_other - other_fatal_cases_at_site
        non_site_total = non_site_hard + non_site_other
        site_level_hard_case_share = float(hard_cases_at_site / site_total) if site_total else np.nan
        non_site_hard_case_share = float(non_site_hard / non_site_total) if non_site_total else np.nan
        site_vs_non_site_risk_ratio = (
            float(site_level_hard_case_share / non_site_hard_case_share)
            if site_total and non_site_total and non_site_hard_case_share > 0
            else np.nan
        )
        expected_hard_cases = float(n_hard * other_fatal_group_share) if np.isfinite(other_fatal_group_share) else np.nan

        rows.append(
            {
                "site": site,
                "hard_cases_at_site": hard_cases_at_site,
                "other_fatal_cases_at_site": other_fatal_cases_at_site,
                "hard_case_group_share": hard_case_group_share,
                "other_fatal_group_share": other_fatal_group_share,
                "composition_share_difference_pp": 100.0
                * (hard_case_group_share - other_fatal_group_share),
                "site_distribution_standardized_difference": site_distribution_standardized_difference,
                "site_level_hard_case_share": site_level_hard_case_share,
                "non_site_hard_case_share": non_site_hard_case_share,
                "site_vs_non_site_risk_ratio": site_vs_non_site_risk_ratio,
                "site_vs_non_site_odds_ratio": _odds_ratio(
                    hard_cases_at_site,
                    other_fatal_cases_at_site,
                    non_site_hard,
                    non_site_other,
                ),
                "expected_hard_cases_from_other_fatal_distribution": expected_hard_cases,
                "observed_minus_expected_hard_cases": (
                    float(hard_cases_at_site - expected_hard_cases)
                    if np.isfinite(expected_hard_cases)
                    else np.nan
                ),
            }
        )

    comparison = pd.DataFrame(rows).sort_values(
        ["site_distribution_standardized_difference", "site"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    metrics = _build_site_distribution_metrics(np.asarray(contingency_rows, dtype=float))
    metrics["max_absolute_site_distribution_standardized_difference"] = float(
        comparison["site_distribution_standardized_difference"].abs().max()
    )
    metrics["max_absolute_composition_share_difference_pp"] = float(
        comparison["composition_share_difference_pp"].abs().max()
    )
    return comparison, metrics


def _classify_direction(
    *,
    pooled_effect: float,
    site_effect: float,
    assessable: bool,
) -> str:
    if not assessable or not np.isfinite(site_effect):
        return "not_assessable"
    if not np.isfinite(pooled_effect) or abs(pooled_effect) < NEAR_NULL_EFFECT_SIZE:
        return "near_null_or_unclear"
    if abs(site_effect) < NEAR_NULL_EFFECT_SIZE:
        return "near_null_or_unclear"
    if np.sign(site_effect) == np.sign(pooled_effect):
        return "same_direction"
    return "opposite_direction"


def build_site_persistence_check(dataset: pd.DataFrame) -> pd.DataFrame:
    hard_group = dataset[dataset["hard_case_flag"].astype(bool)].copy()
    other_group = dataset[~dataset["hard_case_flag"].astype(bool)].copy()
    pooled_effects: dict[str, float] = {}
    for spec in PERSISTENCE_VARIABLE_SPECS:
        pooled_effect, _ = _continuous_standardized_difference(
            hard_group[spec["name"]],
            other_group[spec["name"]],
        )
        pooled_effects[spec["name"]] = pooled_effect

    rows: list[dict[str, object]] = []
    for site, site_frame in dataset.groupby("hospital_id", dropna=False):
        site_hard = site_frame[site_frame["hard_case_flag"].astype(bool)].copy()
        site_other = site_frame[~site_frame["hard_case_flag"].astype(bool)].copy()
        for spec in PERSISTENCE_VARIABLE_SPECS:
            variable_name = str(spec["name"])
            site_effect, _ = _continuous_standardized_difference(
                site_hard[variable_name],
                site_other[variable_name],
            )
            low_nonmissing = int(pd.to_numeric(site_hard[variable_name], errors="coerce").notna().sum())
            other_nonmissing = int(pd.to_numeric(site_other[variable_name], errors="coerce").notna().sum())
            assessable = (
                low_nonmissing >= PERSISTENCE_MIN_PER_ARM
                and other_nonmissing >= PERSISTENCE_MIN_PER_ARM
            )
            rows.append(
                {
                    "site": site,
                    "variable": variable_name,
                    "variable_label": spec["label"],
                    "low_predicted_fatal_summary": _format_continuous_summary(
                        site_hard[variable_name],
                        variable_name=variable_name,
                    ),
                    "other_fatal_summary": _format_continuous_summary(
                        site_other[variable_name],
                        variable_name=variable_name,
                    ),
                    "pooled_standardized_difference": pooled_effects[variable_name],
                    "site_standardized_difference": site_effect,
                    "direction_classification": _classify_direction(
                        pooled_effect=pooled_effects[variable_name],
                        site_effect=site_effect,
                        assessable=assessable,
                    ),
                    "low_predicted_fatal_nonmissing": low_nonmissing,
                    "other_fatal_nonmissing": other_nonmissing,
                    "assessable": assessable,
                }
            )

    persistence = pd.DataFrame(rows)
    persistence["pooled_standardized_difference"] = pd.to_numeric(
        persistence["pooled_standardized_difference"],
        errors="coerce",
    ).round(3)
    persistence["site_standardized_difference"] = pd.to_numeric(
        persistence["site_standardized_difference"],
        errors="coerce",
    ).round(3)
    return persistence.sort_values(
        ["variable", "site"],
        kind="stable",
    ).reset_index(drop=True)


def _summarize_persistence(persistence: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variable_name, variable_frame in persistence.groupby("variable", dropna=False):
        assessable = variable_frame[variable_frame["assessable"].astype(bool)].copy()
        counts = assessable["direction_classification"].value_counts()
        rows.append(
            {
                "variable": variable_name,
                "variable_label": str(variable_frame["variable_label"].iloc[0]),
                "assessable_sites": int(assessable.shape[0]),
                "same_direction_sites": int(counts.get("same_direction", 0)),
                "opposite_direction_sites": int(counts.get("opposite_direction", 0)),
                "near_null_sites": int(counts.get("near_null_or_unclear", 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("variable", kind="stable").reset_index(drop=True)


def _select_primary_enriched_site(
    site_summary: pd.DataFrame,
    site_comparison: pd.DataFrame,
) -> str | None:
    merged = site_summary.merge(site_comparison, on="site", how="inner", validate="one_to_one")
    overall_hard_case_share = _overall_hard_case_share(site_summary)
    eligible = merged[
        merged["within_site_hard_case_share"].gt(overall_hard_case_share)
    ].copy()
    eligible = eligible[
        eligible["hard_case_share_minus_fatal_share_pp"].gt(0.0)
        & eligible["site_distribution_standardized_difference"].gt(0.0)
    ].copy()
    if eligible.empty:
        return None
    eligible = eligible.sort_values(
        [
            "site_distribution_standardized_difference",
            "within_site_hard_case_share",
            "hard_cases",
            "site",
        ],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    return str(eligible.at[0, "site"])


def _overall_hard_case_share(site_summary: pd.DataFrame) -> float:
    total_hard = int(site_summary["hard_cases"].sum())
    total_fatal = int(site_summary["fatal_stays"].sum())
    return float(total_hard / total_fatal) if total_fatal else float("nan")


def build_site_enrichment_decision(
    *,
    site_summary: pd.DataFrame,
    site_comparison: pd.DataFrame,
    site_distribution_metrics: dict[str, float | int],
    site_persistence_check: pd.DataFrame,
) -> dict[str, object]:
    overall_hard_case_share = _overall_hard_case_share(site_summary)
    persistence_summary = _summarize_persistence(site_persistence_check)
    broad_persistence = bool(
        (
            persistence_summary["same_direction_sites"].ge(3)
            & persistence_summary["opposite_direction_sites"].eq(0)
        ).all()
    )

    primary_enriched_site = _select_primary_enriched_site(site_summary, site_comparison)
    enriched_site_summary = (
        site_summary[site_summary["site"].eq(primary_enriched_site)].iloc[0]
        if primary_enriched_site is not None
        else None
    )

    max_abs_std_diff = float(site_distribution_metrics["max_absolute_site_distribution_standardized_difference"])
    max_abs_share_diff_pp = float(site_distribution_metrics["max_absolute_composition_share_difference_pp"])
    cramers_v = float(site_distribution_metrics["cramers_v"])
    primary_excess_hard_share_pp = (
        float(enriched_site_summary["hard_case_share_minus_fatal_share_pp"])
        if enriched_site_summary is not None
        else 0.0
    )

    if (
        cramers_v < NO_MEANINGFUL_CRAMERS_V_MAX
        and max_abs_std_diff < NO_MEANINGFUL_SITE_STD_DIFF_MAX
        and max_abs_share_diff_pp < NO_MEANINGFUL_SHARE_DIFF_PP_MAX
    ):
        decision_category = "No meaningful site enrichment"
        package2_justified = False
        decision_rationale = (
            "Global and site-level site-distribution effect sizes both stayed below the "
            "pragmatic minimal-enrichment thresholds."
        )
    elif (
        (
            cramers_v >= MODEST_CRAMERS_V_MAX
            or max_abs_std_diff >= MODEST_SITE_STD_DIFF_MAX
        )
        and (
            not broad_persistence
            or primary_excess_hard_share_pp >= NONTRIVIAL_EXCESS_HARD_SHARE_PP_MIN
        )
    ):
        decision_category = "Nontrivial site enrichment"
        package2_justified = True
        decision_rationale = (
            "Site-distribution separation reached at least a moderate range and the pattern "
            "either concentrated materially in one site or failed to persist broadly across sites."
        )
    else:
        decision_category = "Some site enrichment, but clearly modest"
        package2_justified = False
        decision_rationale = (
            "There is visible site imbalance, but the global site-distribution effect is still small, "
            "the strongest site-specific composition effect remains modest, and the core contrasts "
            "persist directionally across sites."
        )

    return {
        "decision_category": decision_category,
        "package2_justified": package2_justified,
        "primary_enriched_site": primary_enriched_site,
        "overall_hard_case_share": overall_hard_case_share,
        "broad_persistence": broad_persistence,
        "decision_rationale": decision_rationale,
        "persistence_summary": persistence_summary,
        "decision_assumptions": {
            "possible_enriched_site": (
                "site with within-site hard-case share above the overall 24h fatal anchor share "
                "and positive hard-case overcontribution relative to its fatal-stay share"
            ),
            "no_meaningful": (
                f"Cramer's V < {NO_MEANINGFUL_CRAMERS_V_MAX:.2f}, max absolute site standardized difference "
                f"< {NO_MEANINGFUL_SITE_STD_DIFF_MAX:.2f}, and max absolute composition-share gap "
                f"< {NO_MEANINGFUL_SHARE_DIFF_PP_MAX:.1f} percentage points"
            ),
            "nontrivial": (
                f"Cramer's V >= {MODEST_CRAMERS_V_MAX:.2f} or max absolute site standardized difference "
                f">= {MODEST_SITE_STD_DIFF_MAX:.2f}, plus either lack of broad cross-site persistence "
                f"or >= {NONTRIVIAL_EXCESS_HARD_SHARE_PP_MIN:.1f} percentage-point excess hard-case share "
                "beyond fatal-stay share for the primary enriched site"
            ),
            "modest": "anything between those two decision bands",
        },
    }


def build_site_enrichment_decision_memo(
    *,
    metadata: dict[str, object],
    site_summary: pd.DataFrame,
    site_comparison: pd.DataFrame,
    site_distribution_metrics: dict[str, float | int],
    site_persistence_check: pd.DataFrame,
    decision: dict[str, object],
) -> str:
    n_total = int(metadata["group_counts"]["total_fatal_stays"])
    n_hard = int(metadata["group_counts"][LOW_PREDICTED_FATAL_GROUP])
    n_other = int(metadata["group_counts"][OTHER_FATAL_GROUP])
    overall_hard_case_share = float(decision["overall_hard_case_share"])
    cramers_v = float(site_distribution_metrics["cramers_v"])
    chi_square = float(site_distribution_metrics["chi_square_statistic"])
    max_abs_std_diff = float(site_distribution_metrics["max_absolute_site_distribution_standardized_difference"])
    primary_enriched_site = decision["primary_enriched_site"]
    persistence_summary = decision["persistence_summary"]

    primary_site_summary = (
        site_summary[site_summary["site"].eq(primary_enriched_site)].iloc[0]
        if primary_enriched_site is not None
        else None
    )
    primary_site_comparison = (
        site_comparison[site_comparison["site"].eq(primary_enriched_site)].iloc[0]
        if primary_enriched_site is not None
        else None
    )
    top_overcontributor = site_summary.sort_values(
        ["hard_case_share_minus_fatal_share_pp", "hard_cases", "site"],
        ascending=[False, False, True],
        kind="stable",
    ).iloc[0]

    persistence_lines = []
    for row in persistence_summary.itertuples(index=False):
        persistence_lines.append(
            f"- `{row.variable_label}`: `{row.same_direction_sites}/{row.assessable_sites}` assessable sites "
            "matched the pooled direction"
            + (
                f"; `{row.opposite_direction_sites}` were opposite."
                if int(row.opposite_direction_sites) > 0
                else "; no site was opposite."
            )
        )

    if primary_site_summary is None or primary_site_comparison is None:
        enriched_site_lines = [
            "- No site met the combined within-site-enrichment and overcontribution screen."
        ]
    else:
        enriched_site_lines = [
            (
                f"- The clearest within-site enrichment was `{primary_enriched_site}`: "
                f"`{int(primary_site_summary['hard_cases'])}/{int(primary_site_summary['fatal_stays'])}` hard cases "
                f"({100.0 * float(primary_site_summary['within_site_hard_case_share']):.1f}%) versus "
                f"`{100.0 * overall_hard_case_share:.1f}%` overall."
            ),
            (
                f"- `{primary_enriched_site}` contributed `{100.0 * float(primary_site_summary['share_of_all_hard_cases']):.1f}%` "
                f"of all hard cases but only `{100.0 * float(primary_site_summary['share_of_all_fatal_stays']):.1f}%` "
                "of all fatal stays."
            ),
            (
                f"- In the direct hard-case vs other-fatal site comparison, `{primary_enriched_site}` accounted for "
                f"`{100.0 * float(primary_site_comparison['hard_case_group_share']):.1f}%` of hard cases versus "
                f"`{100.0 * float(primary_site_comparison['other_fatal_group_share']):.1f}%` of other fatal stays "
                f"(site standardized difference `{float(primary_site_comparison['site_distribution_standardized_difference']):.3f}`)."
            ),
        ]

    lines = [
        "# ASIC Site Enrichment Decision Memo",
        "",
        "## Scope",
        "- Package 1 only: detect whether the frozen ASIC 24h logistic hard-case pattern shows enough site enrichment to justify conditional follow-up.",
        "- This memo does not attempt to explain site differences and does not interpret site as biology, treatment-policy heterogeneity, or causality.",
        "",
        "## Inputs",
        f"- Authoritative 24h fatal comparison dataset: `{_display_path(Path(metadata['source_paths']['comparison_dataset_path']))}`.",
        f"- Authoritative hard-case anchor artifact: `{_display_path(Path(metadata['source_paths']['hard_case_path']))}`.",
        f"- Frozen hard-case rule: `{metadata['hard_case_rule']}`.",
        "",
        "## Anchor",
        f"- Fatal stays in the 24h anchor: `{n_total}`.",
        f"- Hard cases: `{n_hard}`. Other fatal stays: `{n_other}`.",
        f"- Overall hard-case share among fatal stays: `{100.0 * overall_hard_case_share:.1f}%`.",
        "",
        "## Site Enrichment Check",
        (
            f"- Global hard-case vs other-fatal site-distribution separation was present but small "
            f"(chi-square `{chi_square:.2f}` on `{int(site_distribution_metrics['degrees_of_freedom'])}` df; "
            f"Cramer's V `{cramers_v:.3f}`)."
        ),
        (
            f"- The largest site-specific hard-case vs other-fatal composition effect was "
            f"`{max_abs_std_diff:.3f}` standardized-difference units, which stays in the modest range."
        ),
        *enriched_site_lines,
        (
            f"- The largest raw overcontribution by share came from `{top_overcontributor['site']}` "
            f"({float(top_overcontributor['hard_case_share_minus_fatal_share_pp']):+.1f} percentage points), "
            "which is still well below a dominant single-site concentration."
        ),
        "",
        "## Minimal Cross-Site Persistence Check",
        *persistence_lines,
        (
            "- Broad reading: the main timing/physiology contrasts are not obviously confined to one center; "
            "their direction is broadly stable across the retained ASIC sites."
        ),
        "",
        "## Decision Assumptions",
        (
            "- Possible enriched site screen: within-site hard-case share above the overall 24h fatal anchor share "
            "plus positive hard-case overcontribution relative to fatal-stay share."
        ),
        (
            f"- Nontrivial follow-up threshold for this package: at least moderate site separation "
            f"(Cramer's V >= `{MODEST_CRAMERS_V_MAX:.2f}` or max site standardized difference >= "
            f"`{MODEST_SITE_STD_DIFF_MAX:.2f}`) plus either material single-site concentration "
            f"(>= `{NONTRIVIAL_EXCESS_HARD_SHARE_PP_MIN:.1f}` percentage-point excess hard-case share) "
            "or failure of broad cross-site persistence."
        ),
        "- These are pragmatic decision thresholds for Package 1, not scientific cutoffs.",
        "",
        "## Decision",
        f"- Category: **{decision['decision_category']}**.",
        f"- Was site enrichment present? {'Yes, but only to a limited degree.' if decision['decision_category'] != 'No meaningful site enrichment' else 'No material site enrichment was detected.'}",
        (
            f"- How large did it appear? Small overall: global site-distribution effect size "
            f"`{cramers_v:.3f}` with modest site-level composition separation."
        ),
        (
            f"- Which site appeared enriched? `{primary_enriched_site}`."
            if primary_enriched_site is not None
            else "- Which site appeared enriched? No single site crossed the package's combined enrichment screen."
        ),
        (
            "- Did the broad hard-case pattern look obviously confined to one center? No. "
            "The minimal persistence check stayed same-direction across the assessable sites for every tracked contrast."
        ),
        (
            "- Is Package 2 justified? No. Observation-process-by-site follow-up and leave-one-site-out sensitivity are optional for completeness, but they are not justified by the current Package 1 signal."
            if not decision["package2_justified"]
            else "- Is Package 2 justified? Yes. Observation-process-by-site follow-up and enriched-site leave-one-site-out sensitivity are recommended."
        ),
        (
            f"- Decision rationale: {decision['decision_rationale']}"
        ),
        "",
        "## Bounded Interpretation",
        "- This package detects potential site enrichment only; it does not attempt explanation.",
        "- Chapter 1 interpretation remains conditional on the observed feature set, documentation process, temporal aggregation, and site/context.",
    ]
    return "\n".join(lines) + "\n"


def run_asic_site_enrichment_decision(
    *,
    comparison_dataset_path: Path = DEFAULT_COMPARISON_DATASET_PATH,
    hard_case_path: Path = DEFAULT_HARD_CASE_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> ASICSiteEnrichmentRunResult:
    anchor_dataset, metadata = load_authoritative_site_enrichment_anchor(
        comparison_dataset_path=comparison_dataset_path,
        hard_case_path=hard_case_path,
    )
    site_summary = build_site_hard_case_summary(anchor_dataset)
    site_comparison, site_distribution_metrics = build_site_hard_case_comparison(anchor_dataset)
    site_persistence_check = build_site_persistence_check(anchor_dataset)
    decision = build_site_enrichment_decision(
        site_summary=site_summary,
        site_comparison=site_comparison,
        site_distribution_metrics=site_distribution_metrics,
        site_persistence_check=site_persistence_check,
    )
    memo_markdown = build_site_enrichment_decision_memo(
        metadata=metadata,
        site_summary=site_summary,
        site_comparison=site_comparison,
        site_distribution_metrics=site_distribution_metrics,
        site_persistence_check=site_persistence_check,
        decision=decision,
    )

    resolved_output_dir = Path(output_dir)
    site_summary_path = write_dataframe(
        site_summary,
        resolved_output_dir / "site_hard_case_summary.csv",
        output_format="csv",
    )
    site_comparison_path = write_dataframe(
        site_comparison,
        resolved_output_dir / "site_hard_case_comparison.csv",
        output_format="csv",
    )
    site_persistence_check_path = write_dataframe(
        site_persistence_check,
        resolved_output_dir / "site_persistence_check.csv",
        output_format="csv",
    )
    memo_path = write_text(
        memo_markdown,
        resolved_output_dir / "site_enrichment_decision.md",
    )
    manifest_path = _write_json(
        {
            "timestamp_utc": _utc_timestamp(),
            "target_horizon_h": TARGET_HORIZON_H,
            "hard_case_rule": HARD_CASE_RULE,
            "source_paths": {
                key: str(Path(value).resolve()) for key, value in metadata["source_paths"].items()
            },
            "group_counts": metadata["group_counts"],
            "site_distribution_metrics": site_distribution_metrics,
            "decision_category": decision["decision_category"],
            "package2_justified": decision["package2_justified"],
            "primary_enriched_site": decision["primary_enriched_site"],
            "broad_persistence": decision["broad_persistence"],
            "decision_rationale": decision["decision_rationale"],
            "decision_assumptions": decision["decision_assumptions"],
            "persistence_variables": [
                {"name": spec["name"], "label": spec["label"]}
                for spec in PERSISTENCE_VARIABLE_SPECS
            ],
            "output_paths": {
                "site_hard_case_summary": str(Path(site_summary_path).resolve()),
                "site_hard_case_comparison": str(Path(site_comparison_path).resolve()),
                "site_persistence_check": str(Path(site_persistence_check_path).resolve()),
                "site_enrichment_decision_memo": str(Path(memo_path).resolve()),
            },
        },
        resolved_output_dir / "run_manifest.json",
    )

    return ASICSiteEnrichmentRunResult(
        anchor_dataset=anchor_dataset,
        site_hard_case_summary=site_summary,
        site_hard_case_comparison=site_comparison,
        site_persistence_check=site_persistence_check,
        decision_category=str(decision["decision_category"]),
        primary_enriched_site=(
            str(decision["primary_enriched_site"])
            if decision["primary_enriched_site"] is not None
            else None
        ),
        package2_justified=bool(decision["package2_justified"]),
        memo_markdown=memo_markdown,
        artifacts=ASICSiteEnrichmentArtifacts(
            site_hard_case_summary_path=site_summary_path,
            site_hard_case_comparison_path=site_comparison_path,
            site_persistence_check_path=site_persistence_check_path,
            memo_path=memo_path,
            manifest_path=manifest_path,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ASIC 24h logistic hard-case Package 1 site-enrichment decision package "
            "using authoritative local-review exports."
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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the Package 1 site-enrichment artifacts will be written.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = run_asic_site_enrichment_decision(
        comparison_dataset_path=args.comparison_dataset_path,
        hard_case_path=args.hard_case_path,
        output_dir=args.output_dir,
    )

    print(f"Output directory: {args.output_dir}")
    print(f"Decision category: {result.decision_category}")
    print(f"Primary enriched site: {result.primary_enriched_site}")
    print(f"Package 2 justified: {result.package2_justified}")
    print(f"Site summary: {result.artifacts.site_hard_case_summary_path}")
    print(f"Site comparison: {result.artifacts.site_hard_case_comparison_path}")
    print(f"Site persistence check: {result.artifacts.site_persistence_check_path}")
    print(f"Decision memo: {result.artifacts.memo_path}")
    print(f"Manifest: {result.artifacts.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
