from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from chapter1_mortality_decomposition.utils import require_columns


KNOWN_UNLABELED_REASONS = (
    "missing_required_fields",
    "prediction_time_not_strictly_before_proxy_end",
    "survivor_without_full_horizon_observation",
    "non_survivor_proxy_end_not_within_horizon",
    "unsupported_icu_mortality_code",
)


@dataclass(frozen=True)
class Chapter1LabelResult:
    labels: pd.DataFrame
    usable_labels: pd.DataFrame
    summary_by_horizon: pd.DataFrame
    unlabeled_reason_summary: pd.DataFrame
    notes: pd.DataFrame


def _build_unlabeled_reason_summary(labels: pd.DataFrame) -> pd.DataFrame:
    base_columns = ["horizon_h", "unlabeled_reason", "instance_count"]
    if labels.empty:
        return pd.DataFrame(columns=base_columns)

    observed_reasons = (
        labels["unlabeled_reason"]
        .dropna()
        .astype("string")
        .drop_duplicates()
        .tolist()
    )
    reason_order = list(KNOWN_UNLABELED_REASONS)
    for reason in observed_reasons:
        if reason not in reason_order:
            reason_order.append(reason)

    horizons = sorted(labels["horizon_h"].dropna().astype(int).unique().tolist())
    base_index = pd.MultiIndex.from_product(
        [horizons, reason_order],
        names=["horizon_h", "unlabeled_reason"],
    )
    observed = (
        labels.loc[labels["proxy_horizon_labelable"].eq(False)]
        .groupby(["horizon_h", "unlabeled_reason"], dropna=False)
        .size()
        .rename("instance_count")
        .reindex(base_index, fill_value=0)
        .reset_index()
    )
    return observed[base_columns]


def build_chapter1_proxy_horizon_labels(
    valid_instances: pd.DataFrame,
    retained_cohort: pd.DataFrame,
) -> Chapter1LabelResult:
    require_columns(
        valid_instances,
        {
            "stay_id_global",
            "hospital_id",
            "future_window_end_h",
            "horizon_h",
            "icu_end_time_proxy_hours",
        },
        "valid_instances",
    )
    require_columns(
        retained_cohort,
        {"stay_id_global", "hospital_id", "icu_mortality"},
        "retained_cohort",
    )

    label_lookup = retained_cohort[["stay_id_global", "hospital_id", "icu_mortality"]].copy()
    label_lookup["stay_id_global"] = label_lookup["stay_id_global"].astype("string")
    label_lookup["hospital_id"] = label_lookup["hospital_id"].astype("string")
    label_lookup["icu_mortality"] = pd.to_numeric(label_lookup["icu_mortality"], errors="coerce")

    labels = valid_instances.merge(
        label_lookup,
        on=["stay_id_global", "hospital_id"],
        how="left",
    )
    labels["label_name"] = "proxy_within_horizon_icu_mortality"
    labels["label_definition_id"] = "proxy_within_horizon_icu_mortality_v1"
    labels["label_definition_status"] = "approved_proxy"
    labels["event_time_proxy_h"] = pd.to_numeric(
        labels["icu_end_time_proxy_hours"],
        errors="coerce",
    )
    labels["prediction_time_h"] = pd.to_numeric(labels["prediction_time_h"], errors="coerce")
    labels["future_window_end_h"] = pd.to_numeric(labels["future_window_end_h"], errors="coerce")
    labels["proxy_horizon_labelable"] = False
    labels["label_value"] = pd.Series(pd.NA, index=labels.index, dtype="Int64")
    labels["unlabeled_reason"] = pd.Series(pd.NA, index=labels.index, dtype="string")

    positive_mask = (
        labels["icu_mortality"].eq(1)
        & labels["event_time_proxy_h"].gt(labels["prediction_time_h"])
        & labels["event_time_proxy_h"].le(labels["future_window_end_h"])
    )
    negative_mask = (
        labels["icu_mortality"].eq(0)
        & labels["event_time_proxy_h"].ge(labels["future_window_end_h"])
    )
    missing_required_fields_mask = (
        labels["icu_mortality"].isna()
        | labels["event_time_proxy_h"].isna()
        | labels["prediction_time_h"].isna()
        | labels["future_window_end_h"].isna()
    )
    unsupported_icu_mortality_mask = (
        labels["icu_mortality"].notna()
        & ~labels["icu_mortality"].isin([0, 1])
    )
    prediction_not_before_proxy_end_mask = (
        ~missing_required_fields_mask
        & labels["event_time_proxy_h"].le(labels["prediction_time_h"])
    )
    survivor_without_full_horizon_mask = (
        ~missing_required_fields_mask
        & ~unsupported_icu_mortality_mask
        & ~prediction_not_before_proxy_end_mask
        & labels["icu_mortality"].eq(0)
        & ~negative_mask
    )
    non_survivor_not_within_horizon_mask = (
        ~missing_required_fields_mask
        & ~unsupported_icu_mortality_mask
        & ~prediction_not_before_proxy_end_mask
        & labels["icu_mortality"].eq(1)
        & ~positive_mask
    )

    labelable_mask = positive_mask | negative_mask
    labels.loc[labelable_mask, "proxy_horizon_labelable"] = True
    labels.loc[positive_mask, "label_value"] = 1
    labels.loc[negative_mask, "label_value"] = 0
    labels["label_available"] = labels["proxy_horizon_labelable"]

    labels.loc[missing_required_fields_mask, "unlabeled_reason"] = "missing_required_fields"
    labels.loc[
        unsupported_icu_mortality_mask & labels["unlabeled_reason"].isna(),
        "unlabeled_reason",
    ] = "unsupported_icu_mortality_code"
    labels.loc[
        prediction_not_before_proxy_end_mask & labels["unlabeled_reason"].isna(),
        "unlabeled_reason",
    ] = "prediction_time_not_strictly_before_proxy_end"
    labels.loc[
        survivor_without_full_horizon_mask & labels["unlabeled_reason"].isna(),
        "unlabeled_reason",
    ] = "survivor_without_full_horizon_observation"
    labels.loc[
        non_survivor_not_within_horizon_mask & labels["unlabeled_reason"].isna(),
        "unlabeled_reason",
    ] = "non_survivor_proxy_end_not_within_horizon"

    labels["label_semantics"] = (
        "Proxy within-horizon in-ICU mortality label using icu_mortality and "
        "icu_end_time_proxy_hours because true ICU discharge and death timestamps "
        "are unavailable in standardized ASIC artifacts."
    )
    labels = labels.drop(columns=["icu_mortality"])

    label_columns = list(valid_instances.columns) + [
        "label_name",
        "label_definition_id",
        "label_definition_status",
        "event_time_proxy_h",
        "proxy_horizon_labelable",
        "label_value",
        "label_available",
        "unlabeled_reason",
        "label_semantics",
    ]
    labels = labels[label_columns]
    usable_labels = labels[labels["proxy_horizon_labelable"]].reset_index(drop=True)

    if labels.empty:
        summary_by_horizon = pd.DataFrame(
            columns=[
                "horizon_h",
                "total_valid_prediction_instances",
                "labelable_instances",
                "positive_labels",
                "negative_labels",
                "unlabeled_instances",
            ]
        )
    else:
        summary_by_horizon = (
            labels.groupby("horizon_h", dropna=False)
            .agg(
                total_valid_prediction_instances=("instance_id", "size"),
                labelable_instances=("proxy_horizon_labelable", "sum"),
                positive_labels=(
                    "label_value",
                    lambda series: int(series.fillna(-1).eq(1).sum()),
                ),
                negative_labels=(
                    "label_value",
                    lambda series: int(series.fillna(-1).eq(0).sum()),
                ),
            )
            .reset_index()
        )
        summary_by_horizon["unlabeled_instances"] = (
            summary_by_horizon["total_valid_prediction_instances"]
            - summary_by_horizon["labelable_instances"]
        )

    unlabeled_reason_summary = _build_unlabeled_reason_summary(labels)

    notes = pd.DataFrame(
        [
            {
                "note_id": "proxy_horizon_label_definition",
                "category": "label_semantics",
                "note": (
                    "ASIC Chapter 1 uses proxy within-horizon in-ICU mortality labels. "
                    "Positive labels require icu_mortality=1 with icu_end_time_proxy_hours in "
                    "(t, t+H]; negative labels require icu_mortality=0 with "
                    "icu_end_time_proxy_hours >= t+H; all other cases remain unlabeled."
                ),
            },
            {
                "note_id": "proxy_horizon_label_limitation",
                "category": "label_limitation",
                "note": (
                    "True ICU discharge timestamps and true death timestamps are unavailable in "
                    "standardized ASIC artifacts, so these are explicit proxy horizon labels "
                    "rather than true event-timed labels."
                ),
            }
        ]
    )

    return Chapter1LabelResult(
        labels=labels,
        usable_labels=usable_labels,
        summary_by_horizon=summary_by_horizon,
        unlabeled_reason_summary=unlabeled_reason_summary,
        notes=notes,
    )
