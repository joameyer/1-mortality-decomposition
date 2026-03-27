from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from chapter1_mortality_decomposition.config import (
    Chapter1Config,
    chapter1_group_definitions,
    default_chapter1_config,
)
from chapter1_mortality_decomposition.utils import require_columns


@dataclass(frozen=True)
class Chapter1ValidInstanceResult:
    candidate_instances: pd.DataFrame
    valid_instances: pd.DataFrame
    counts_by_horizon: pd.DataFrame
    exclusion_summary: pd.DataFrame


def _group_observed_in_block(
    blocked_dynamic_features: pd.DataFrame,
    candidate_variables: tuple[str, ...],
) -> pd.Series:
    obs_count_columns = [
        f"{variable}_obs_count"
        for variable in candidate_variables
        if f"{variable}_obs_count" in blocked_dynamic_features.columns
    ]
    if obs_count_columns:
        return (
            blocked_dynamic_features[obs_count_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .gt(0)
            .any(axis=1)
        )

    fallback_columns = [
        column
        for variable in candidate_variables
        for column in blocked_dynamic_features.columns
        if column.startswith(f"{variable}_") and not column.endswith("_obs_count")
    ]
    if fallback_columns:
        return blocked_dynamic_features[fallback_columns].notna().any(axis=1)

    return pd.Series(False, index=blocked_dynamic_features.index, dtype="boolean")


def build_chapter1_valid_instances(
    retained_cohort: pd.DataFrame,
    block_index: pd.DataFrame,
    blocked_dynamic_features: pd.DataFrame,
    config: Chapter1Config | None = None,
) -> Chapter1ValidInstanceResult:
    config = config or default_chapter1_config()

    require_columns(
        retained_cohort,
        {
            "stay_id_global",
            "hospital_id",
            "icu_end_time_proxy_hours",
            "icu_mortality",
            "mech_vent_ge_24h_qc",
        },
        "retained_cohort",
    )
    require_columns(
        block_index,
        {
            "stay_id_global",
            "hospital_id",
            "block_index",
            "block_start_h",
            "block_end_h",
            "prediction_time_h",
        },
        "block_index",
    )
    require_columns(
        blocked_dynamic_features,
        {
            "stay_id_global",
            "hospital_id",
            "block_index",
            "block_start_h",
            "block_end_h",
            "prediction_time_h",
        },
        "blocked_dynamic_features",
    )

    retained_stays = retained_cohort[
        [
            "stay_id_global",
            "hospital_id",
            "icu_end_time_proxy_hours",
            "icu_mortality",
            "mech_vent_ge_24h_qc",
        ]
    ].copy()
    retained_stays["stay_id_global"] = retained_stays["stay_id_global"].astype("string")
    retained_stays["hospital_id"] = retained_stays["hospital_id"].astype("string")
    retained_stays["icu_end_time_proxy_hours"] = pd.to_numeric(
        retained_stays["icu_end_time_proxy_hours"],
        errors="coerce",
    )
    retained_stays["icu_mortality"] = pd.to_numeric(retained_stays["icu_mortality"], errors="coerce")

    retained_blocks = (
        block_index.merge(
            retained_stays,
            on=["stay_id_global", "hospital_id"],
            how="inner",
        )
        .merge(
            blocked_dynamic_features,
            on=[
                "stay_id_global",
                "hospital_id",
                "block_index",
                "block_start_h",
                "block_end_h",
                "prediction_time_h",
            ],
            how="left",
        )
        .copy()
    )

    retained_blocks["block_index"] = pd.to_numeric(retained_blocks["block_index"], errors="coerce")
    retained_blocks["block_start_h"] = pd.to_numeric(
        retained_blocks["block_start_h"], errors="coerce"
    )
    retained_blocks["block_end_h"] = pd.to_numeric(retained_blocks["block_end_h"], errors="coerce")
    retained_blocks["prediction_time_h"] = pd.to_numeric(
        retained_blocks["prediction_time_h"], errors="coerce"
    )
    retained_blocks["block_exists_structurally"] = True
    retained_blocks["has_icu_end_time_proxy"] = retained_blocks["icu_end_time_proxy_hours"].notna()
    retained_blocks["block_end_not_after_icu_end_proxy"] = (
        retained_blocks["has_icu_end_time_proxy"]
        & retained_blocks["block_end_h"].le(retained_blocks["icu_end_time_proxy_hours"])
    )

    group_definitions = chapter1_group_definitions()
    group_column_lookup = {
        "cardiac_rate": "cardiac_rate_group_observed_in_block",
        "blood_pressure": "blood_pressure_group_observed_in_block",
        "respiratory": "respiratory_group_observed_in_block",
        "oxygenation": "oxygenation_group_observed_in_block",
    }
    for group_name, candidate_variables in group_definitions.items():
        retained_blocks[group_column_lookup[group_name]] = _group_observed_in_block(
            retained_blocks,
            candidate_variables,
        ).astype("boolean")

    group_columns = list(group_column_lookup.values())
    retained_blocks["covered_core_vital_group_count"] = (
        retained_blocks[group_columns].fillna(False).astype("int64").sum(axis=1).astype("Int64")
    )
    retained_blocks["core_vital_coverage_sufficient"] = retained_blocks[
        "covered_core_vital_group_count"
    ].ge(config.min_required_core_groups)
    retained_blocks["valid_instance"] = (
        retained_blocks["block_exists_structurally"]
        & retained_blocks["block_end_not_after_icu_end_proxy"]
        & retained_blocks["core_vital_coverage_sufficient"]
    )

    rows = []
    for block in retained_blocks.itertuples(index=False):
        for horizon_h in config.horizons_hours:
            future_window_end_h = int(block.prediction_time_h) + int(horizon_h)
            invalid_reasons: list[str] = []
            if not block.has_icu_end_time_proxy:
                invalid_reasons.append("missing_icu_end_time_proxy_hours")
            elif not block.block_end_not_after_icu_end_proxy:
                invalid_reasons.append("block_end_after_icu_end_proxy")
            if not block.core_vital_coverage_sufficient:
                invalid_reasons.append("insufficient_core_vital_group_coverage_in_block")

            rows.append(
                {
                    "instance_id": (
                        f"{block.stay_id_global}__b{int(block.block_index)}__h{int(horizon_h)}"
                    ),
                    "stay_id_global": block.stay_id_global,
                    "hospital_id": block.hospital_id,
                    "block_index": int(block.block_index),
                    "block_start_h": int(block.block_start_h),
                    "block_end_h": int(block.block_end_h),
                    "prediction_time_h": int(block.prediction_time_h),
                    "horizon_h": int(horizon_h),
                    "future_window_end_h": int(future_window_end_h),
                    "icu_end_time_proxy_hours": block.icu_end_time_proxy_hours,
                    "mech_vent_ge_24h_qc": block.mech_vent_ge_24h_qc,
                    "block_exists_structurally": bool(block.block_exists_structurally),
                    "has_icu_end_time_proxy": bool(block.has_icu_end_time_proxy),
                    "block_end_not_after_icu_end_proxy": bool(
                        block.block_end_not_after_icu_end_proxy
                    ),
                    "cardiac_rate_group_observed_in_block": bool(
                        block.cardiac_rate_group_observed_in_block
                    ),
                    "blood_pressure_group_observed_in_block": bool(
                        block.blood_pressure_group_observed_in_block
                    ),
                    "respiratory_group_observed_in_block": bool(
                        block.respiratory_group_observed_in_block
                    ),
                    "oxygenation_group_observed_in_block": bool(
                        block.oxygenation_group_observed_in_block
                    ),
                    "covered_core_vital_group_count": int(block.covered_core_vital_group_count),
                    "core_vital_coverage_sufficient": bool(block.core_vital_coverage_sufficient),
                    "valid_instance": bool(block.valid_instance),
                    "exclusion_reasons": invalid_reasons,
                    "exclusion_reason": "; ".join(invalid_reasons) if invalid_reasons else pd.NA,
                }
            )

    candidate_columns = [
        "instance_id",
        "stay_id_global",
        "hospital_id",
        "block_index",
        "block_start_h",
        "block_end_h",
        "prediction_time_h",
        "horizon_h",
        "future_window_end_h",
        "icu_end_time_proxy_hours",
        "mech_vent_ge_24h_qc",
        "block_exists_structurally",
        "has_icu_end_time_proxy",
        "block_end_not_after_icu_end_proxy",
        "cardiac_rate_group_observed_in_block",
        "blood_pressure_group_observed_in_block",
        "respiratory_group_observed_in_block",
        "oxygenation_group_observed_in_block",
        "covered_core_vital_group_count",
        "core_vital_coverage_sufficient",
        "valid_instance",
        "exclusion_reasons",
        "exclusion_reason",
    ]

    if rows:
        candidate_instances = pd.DataFrame(rows)[candidate_columns].sort_values(
            ["hospital_id", "stay_id_global", "block_index", "horizon_h"]
        ).reset_index(drop=True)
    else:
        candidate_instances = pd.DataFrame(columns=candidate_columns)

    valid_instances = candidate_instances[candidate_instances["valid_instance"]].reset_index(
        drop=True
    )

    if candidate_instances.empty:
        counts_by_horizon = pd.DataFrame(
            columns=["horizon_h", "candidate_instances", "valid_instances", "excluded_instances"]
        )
        exclusion_summary = pd.DataFrame(
            columns=["horizon_h", "exclusion_reason", "instance_count"]
        )
    else:
        counts_by_horizon = (
            candidate_instances.groupby("horizon_h", dropna=False)["valid_instance"]
            .agg(candidate_instances="size", valid_instances="sum")
            .reset_index()
        )
        counts_by_horizon["excluded_instances"] = (
            counts_by_horizon["candidate_instances"] - counts_by_horizon["valid_instances"]
        )

        exclusion_summary = (
            candidate_instances.loc[~candidate_instances["valid_instance"]]
            .groupby(["horizon_h", "exclusion_reason"], dropna=False)
            .size()
            .rename("instance_count")
            .reset_index()
            .sort_values(["horizon_h", "instance_count"], ascending=[True, False])
            .reset_index(drop=True)
        )

    return Chapter1ValidInstanceResult(
        candidate_instances=candidate_instances,
        valid_instances=valid_instances,
        counts_by_horizon=counts_by_horizon,
        exclusion_summary=exclusion_summary,
    )
