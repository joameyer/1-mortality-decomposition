from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from chapter1_mortality_decomposition.config import (
    Chapter1Config,
    default_chapter1_config,
)
from chapter1_mortality_decomposition.utils import require_columns


@dataclass(frozen=True)
class Chapter1ValidInstanceResult:
    candidate_instances: pd.DataFrame
    valid_instances: pd.DataFrame
    counts_by_horizon: pd.DataFrame
    exclusion_summary: pd.DataFrame


def _feature_obs_count_columns(feature_set_definition: pd.DataFrame) -> list[str]:
    return (
        feature_set_definition.loc[
            feature_set_definition["statistic"].eq("obs_count")
            & feature_set_definition["selected_for_model"],
            "feature_name",
        ]
        .astype("string")
        .tolist()
    )


def build_chapter1_valid_instances(
    retained_cohort: pd.DataFrame,
    block_index: pd.DataFrame,
    blocked_dynamic_features: pd.DataFrame,
    feature_set_definition: pd.DataFrame,
    config: Chapter1Config | None = None,
) -> Chapter1ValidInstanceResult:
    config = config or default_chapter1_config()

    require_columns(
        retained_cohort,
        {"stay_id_global", "hospital_id", "icu_end_time_proxy_hours", "icu_mortality"},
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
    require_columns(
        feature_set_definition,
        {"feature_name", "statistic", "selected_for_model"},
        "feature_set_definition",
    )

    obs_count_columns = [
        column
        for column in _feature_obs_count_columns(feature_set_definition)
        if column in blocked_dynamic_features.columns
    ]

    retained_stays = retained_cohort[
        ["stay_id_global", "hospital_id", "icu_end_time_proxy_hours", "icu_mortality"]
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

    if obs_count_columns:
        retained_blocks["chapter1_feature_obs_count_in_block"] = (
            retained_blocks[obs_count_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .sum(axis=1)
            .astype("Int64")
        )
    else:
        retained_blocks["chapter1_feature_obs_count_in_block"] = pd.Series(
            0,
            index=retained_blocks.index,
            dtype="Int64",
        )

    retained_blocks["has_chapter1_feature_data_in_block"] = retained_blocks[
        "chapter1_feature_obs_count_in_block"
    ].gt(0)
    retained_blocks["has_icu_end_time_proxy"] = retained_blocks["icu_end_time_proxy_hours"].notna()
    retained_blocks["prediction_before_icu_end"] = (
        retained_blocks["has_icu_end_time_proxy"]
        & (retained_blocks["prediction_time_h"] < retained_blocks["icu_end_time_proxy_hours"])
    )
    retained_blocks["has_usable_icu_mortality"] = retained_blocks["icu_mortality"].notna()
    retained_blocks["valid_for_label_generation"] = (
        retained_blocks["has_usable_icu_mortality"] & retained_blocks["prediction_before_icu_end"]
    )

    rows = []
    for block in retained_blocks.itertuples(index=False):
        for horizon_h in config.horizons_hours:
            future_window_end_h = int(block.prediction_time_h) + int(horizon_h)
            valid_instance = bool(
                block.has_chapter1_feature_data_in_block and block.valid_for_label_generation
            )
            exclusion_reason = pd.NA
            if not block.has_chapter1_feature_data_in_block:
                exclusion_reason = "no_chapter1_feature_data_in_block"
            elif not block.has_icu_end_time_proxy:
                exclusion_reason = "missing_icu_end_time_proxy_hours"
            elif not block.prediction_before_icu_end:
                exclusion_reason = "prediction_time_not_before_icu_end"
            elif not block.has_usable_icu_mortality:
                exclusion_reason = "missing_icu_mortality"

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
                    "chapter1_feature_obs_count_in_block": int(
                        block.chapter1_feature_obs_count_in_block
                    ),
                    "has_chapter1_feature_data_in_block": bool(
                        block.has_chapter1_feature_data_in_block
                    ),
                    "has_icu_end_time_proxy": bool(block.has_icu_end_time_proxy),
                    "prediction_before_icu_end": bool(block.prediction_before_icu_end),
                    "valid_for_label_generation": bool(block.valid_for_label_generation),
                    "valid_instance": valid_instance,
                    "exclusion_reason": exclusion_reason,
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
        "chapter1_feature_obs_count_in_block",
        "has_chapter1_feature_data_in_block",
        "has_icu_end_time_proxy",
        "prediction_before_icu_end",
        "valid_for_label_generation",
        "valid_instance",
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
