from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from chapter1_mortality_decomposition.carry_forward import (
    Chapter1CarryForwardResult,
    INSTANCE_KEY_COLUMNS,
    build_chapter1_locf_feature_frame,
)
from chapter1_mortality_decomposition.utils import require_columns


@dataclass(frozen=True)
class Chapter1ModelReadyResult:
    table: pd.DataFrame
    readiness_summary: pd.DataFrame
    feature_availability_by_horizon: pd.DataFrame
    split_summary: pd.DataFrame
    split_verification_summary: pd.DataFrame
    locf_feature_summary: pd.DataFrame
    ventilator_locf_summary: pd.DataFrame
    missingness_by_hospital_and_family: pd.DataFrame
    carry_forward_verification_summary: pd.DataFrame


def _build_model_ready_split_summary(
    model_ready: pd.DataFrame,
    *,
    feature_set_name: str,
) -> pd.DataFrame:
    base_columns = [
        "feature_set_name",
        "summary_level",
        "split",
        "hospital_id",
        "horizon_h",
        "stay_count",
        "instance_count",
        "positive_labels",
        "negative_labels",
        "label_prevalence",
    ]
    if model_ready.empty:
        return pd.DataFrame(columns=base_columns)

    rows: list[dict[str, object]] = []

    def append_summary(summary_level: str, hospital_id: object, group_df: pd.DataFrame) -> None:
        for split, split_df in group_df.groupby("split", dropna=False):
            rows.append(
                {
                    "feature_set_name": feature_set_name,
                    "summary_level": summary_level,
                    "split": split,
                    "hospital_id": hospital_id,
                    "horizon_h": pd.NA,
                    "stay_count": int(split_df["stay_id_global"].nunique(dropna=True)),
                    "instance_count": int(split_df.shape[0]),
                    "positive_labels": int(split_df["label_value"].eq(1).sum()),
                    "negative_labels": int(split_df["label_value"].eq(0).sum()),
                    "label_prevalence": (
                        float(split_df["label_value"].eq(1).mean())
                        if not split_df.empty
                        else pd.NA
                    ),
                }
            )
            for horizon_h, horizon_df in split_df.groupby("horizon_h", dropna=False):
                rows.append(
                    {
                        "feature_set_name": feature_set_name,
                        "summary_level": f"{summary_level}_horizon",
                        "split": split,
                        "hospital_id": hospital_id,
                        "horizon_h": int(horizon_h),
                        "stay_count": int(horizon_df["stay_id_global"].nunique(dropna=True)),
                        "instance_count": int(horizon_df.shape[0]),
                        "positive_labels": int(horizon_df["label_value"].eq(1).sum()),
                        "negative_labels": int(horizon_df["label_value"].eq(0).sum()),
                        "label_prevalence": (
                            float(horizon_df["label_value"].eq(1).mean())
                            if not horizon_df.empty
                            else pd.NA
                        ),
                    }
                )

    append_summary("overall", pd.NA, model_ready)
    hospitals = sorted(model_ready["hospital_id"].dropna().astype("string").unique().tolist())
    for hospital_id in hospitals:
        append_summary(
            "hospital",
            hospital_id,
            model_ready[model_ready["hospital_id"].astype("string") == hospital_id],
        )

    return pd.DataFrame(rows)


def _build_model_ready_split_verification_summary(
    model_ready: pd.DataFrame,
    *,
    feature_set_name: str,
) -> pd.DataFrame:
    if model_ready.empty:
        return pd.DataFrame(
            [
                {
                    "feature_set_name": feature_set_name,
                    "check_id": "no_unassigned_instances_in_model_ready",
                    "passed": True,
                    "detail": "No model-ready rows are present, so there are no unassigned instances.",
                },
                {
                    "feature_set_name": feature_set_name,
                    "check_id": "all_instances_from_stay_share_same_split",
                    "passed": True,
                    "detail": "No model-ready rows are present, so there is no stay-level split leakage.",
                },
            ]
        )

    stay_split_counts = model_ready.groupby("stay_id_global", dropna=False)["split"].nunique(dropna=True)
    return pd.DataFrame(
        [
            {
                "feature_set_name": feature_set_name,
                "check_id": "no_unassigned_instances_in_model_ready",
                "passed": bool(model_ready["split"].notna().all()),
                "detail": "Every model-ready row inherits a split from the stay-level split table.",
            },
            {
                "feature_set_name": feature_set_name,
                "check_id": "all_instances_from_stay_share_same_split",
                "passed": bool(stay_split_counts.le(1).all()),
                "detail": "All model-ready rows from the same stay_id_global share one split.",
            },
        ]
    )


def build_chapter1_model_ready_dataset(
    usable_labels: pd.DataFrame,
    blocked_dynamic_features: pd.DataFrame,
    feature_set_definition: pd.DataFrame,
    feature_set_name: str,
    mech_vent_episode_level: pd.DataFrame,
    stay_split_assignments: pd.DataFrame | None = None,
    config=None,
) -> Chapter1ModelReadyResult:
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
        {
            "feature_name",
            "base_variable",
            "selected_for_model",
            "available_in_blocked_schema",
            "base_feature_available_in_blocked_schema",
            "base_feature_present_in_retained_data",
        },
        "feature_set_definition",
    )

    selected_feature_columns = (
        feature_set_definition.loc[feature_set_definition["selected_for_model"], "feature_name"]
        .astype("string")
        .tolist()
    )

    carry_forward = build_chapter1_locf_feature_frame(
        instance_index=usable_labels[INSTANCE_KEY_COLUMNS],
        blocked_dynamic_features=blocked_dynamic_features,
        feature_set_definition=feature_set_definition,
        mech_vent_episode_level=mech_vent_episode_level,
        config=config,
        feature_set_name=feature_set_name,
    )
    feature_indicator_columns = [
        column
        for column in carry_forward.feature_frame.columns
        if column.endswith("_observed_in_block")
        or column.endswith("_filled_by_locf")
        or column.endswith("_missing_after_locf")
        or column.endswith("_ventilation_window_active")
    ]

    model_ready = usable_labels.merge(
        carry_forward.feature_frame,
        on=INSTANCE_KEY_COLUMNS,
        how="left",
    )
    model_ready["feature_set_name"] = feature_set_name

    if stay_split_assignments is not None:
        require_columns(
            stay_split_assignments,
            {"stay_id_global", "hospital_id", "split"},
            "stay_split_assignments",
        )
        split_frame = stay_split_assignments[["stay_id_global", "hospital_id", "split"]].copy()
        split_frame["stay_id_global"] = split_frame["stay_id_global"].astype("string")
        split_frame["hospital_id"] = split_frame["hospital_id"].astype("string")
        model_ready = model_ready.merge(
            split_frame,
            on=["stay_id_global", "hospital_id"],
            how="left",
        )
    else:
        model_ready["split"] = pd.Series(pd.NA, index=model_ready.index, dtype="string")

    configured_base_feature_count = int(feature_set_definition["base_variable"].nunique(dropna=True))
    missing_base_features = (
        feature_set_definition.loc[
            ~feature_set_definition["base_feature_available_in_blocked_schema"],
            "base_variable",
        ]
        .astype("string")
        .drop_duplicates()
        .tolist()
    )
    absent_from_retained_data = (
        feature_set_definition.loc[
            ~feature_set_definition["base_feature_present_in_retained_data"],
            "base_variable",
        ]
        .astype("string")
        .drop_duplicates()
        .tolist()
    )
    readiness_rows: list[dict[str, object]] = [
        {
            "feature_set_name": feature_set_name,
            "metric": "model_ready_rows_total",
            "value": int(model_ready.shape[0]),
            "note": (
                "Rows after valid-instance selection, proxy label availability filtering, "
                "bounded LOCF preprocessing, and split annotation."
            ),
        },
        {
            "feature_set_name": feature_set_name,
            "metric": "configured_base_features_total",
            "value": configured_base_feature_count,
            "note": "Configured base features for this Chapter 1 feature set.",
        },
        {
            "feature_set_name": feature_set_name,
            "metric": "selected_feature_columns_total",
            "value": int(len(selected_feature_columns)),
            "note": (
                "Blocked dynamic feature columns selected by the Chapter 1 feature config. "
                "obs_count columns remain raw; value-summary columns may be LOCF-filled."
            ),
        },
        {
            "feature_set_name": feature_set_name,
            "metric": "locf_missingness_indicator_columns_total",
            "value": int(len(feature_indicator_columns)),
            "note": (
                "Per-feature indicators appended to distinguish originally observed values, "
                "bounded-LOCF fills, and values still missing after LOCF."
            ),
        },
        {
            "feature_set_name": feature_set_name,
            "metric": "distinct_horizons_total",
            "value": int(model_ready["horizon_h"].nunique(dropna=True)) if not model_ready.empty else 0,
            "note": "Configured prediction horizons represented in the model-ready table.",
        },
        {
            "feature_set_name": feature_set_name,
            "metric": "distinct_splits_total",
            "value": int(model_ready["split"].nunique(dropna=True)) if not model_ready.empty else 0,
            "note": "Stay-level train/validation/test splits represented in the model-ready table.",
        },
        {
            "feature_set_name": feature_set_name,
            "metric": "bounded_locf_applied_in_export",
            "value": True,
            "note": (
                "Configured feature families use bounded preprocessing-time LOCF. "
                "Values remain missing when no valid carry-forward source exists."
            ),
        },
        {
            "feature_set_name": feature_set_name,
            "metric": "ventilator_locf_restricted_to_supported_windows",
            "value": True,
            "note": (
                "Ventilator-variable LOCF is restricted to blocks overlapping upstream "
                "ventilation-supported episodes."
            ),
        },
        {
            "feature_set_name": feature_set_name,
            "metric": "global_median_imputation_applied_in_export",
            "value": False,
            "note": (
                "No global median or other final imputation is applied in the preprocessing export."
            ),
        },
        {
            "feature_set_name": feature_set_name,
            "metric": "downstream_imputation_policy",
            "value": "deferred_to_model_training_stage",
            "note": (
                "Any final imputation must be fit later on the training split only, "
                "not during preprocessing export generation."
            ),
        },
        {
            "feature_set_name": feature_set_name,
            "metric": "configured_base_features_missing_from_blocked_schema",
            "value": ", ".join(missing_base_features),
            "note": "Configured base features absent from the current blocked schema.",
        },
        {
            "feature_set_name": feature_set_name,
            "metric": "configured_base_features_absent_from_retained_data",
            "value": ", ".join(absent_from_retained_data),
            "note": "Configured base features absent from current retained-hospital ASIC data.",
        },
    ]
    if "label_definition_id" in model_ready.columns:
        label_definitions = sorted(
            model_ready["label_definition_id"].dropna().astype("string").unique().tolist()
        )
        readiness_rows.append(
            {
                "feature_set_name": feature_set_name,
                "metric": "label_definition_in_model_ready_dataset",
                "value": ", ".join(label_definitions),
                "note": (
                    "Current model-ready rows use the explicit ASIC proxy within-horizon "
                    "label definition."
                ),
            }
        )

    readiness_summary = pd.DataFrame(readiness_rows)

    feature_availability_rows = []
    for horizon_h, horizon_df in model_ready.groupby("horizon_h", dropna=False):
        horizon_size = int(horizon_df.shape[0])
        for feature_name in selected_feature_columns:
            non_missing_count = int(horizon_df[feature_name].notna().sum())
            feature_availability_rows.append(
                {
                    "feature_set_name": feature_set_name,
                    "horizon_h": int(horizon_h),
                    "feature_name": feature_name,
                    "non_missing_count": non_missing_count,
                    "rows_in_horizon": horizon_size,
                    "non_missing_proportion": (
                        float(non_missing_count / horizon_size) if horizon_size else pd.NA
                    ),
                }
            )

    if feature_availability_rows:
        feature_availability_by_horizon = pd.DataFrame(feature_availability_rows).sort_values(
            ["horizon_h", "feature_name"]
        ).reset_index(drop=True)
    else:
        feature_availability_by_horizon = pd.DataFrame(
            columns=[
                "horizon_h",
                "feature_set_name",
                "feature_name",
                "non_missing_count",
                "rows_in_horizon",
                "non_missing_proportion",
            ]
        )

    split_summary = _build_model_ready_split_summary(
        model_ready,
        feature_set_name=feature_set_name,
    )
    split_verification_summary = _build_model_ready_split_verification_summary(
        model_ready,
        feature_set_name=feature_set_name,
    )

    return Chapter1ModelReadyResult(
        table=model_ready,
        readiness_summary=readiness_summary,
        feature_availability_by_horizon=feature_availability_by_horizon,
        split_summary=split_summary,
        split_verification_summary=split_verification_summary,
        locf_feature_summary=carry_forward.feature_summary,
        ventilator_locf_summary=carry_forward.ventilator_summary,
        missingness_by_hospital_and_family=carry_forward.missingness_by_hospital_and_family,
        carry_forward_verification_summary=carry_forward.verification_summary,
    )
