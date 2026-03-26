from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from chapter1_mortality_decomposition.utils import require_columns


@dataclass(frozen=True)
class Chapter1LabelResult:
    labels: pd.DataFrame
    usable_labels: pd.DataFrame
    summary_by_horizon: pd.DataFrame
    notes: pd.DataFrame


def build_chapter1_horizon_labels(
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
    labels["label_name"] = "icu_mortality_within_horizon_proxy"
    labels["event_time_proxy_h"] = pd.to_numeric(labels["icu_end_time_proxy_hours"], errors="coerce")
    labels["label_available"] = (
        labels["icu_mortality"].notna() & labels["event_time_proxy_h"].notna()
    )
    labels["label_value"] = pd.Series(pd.NA, index=labels.index, dtype="Int64")
    available_mask = labels["label_available"]
    labels.loc[available_mask, "label_value"] = (
        (
            labels.loc[available_mask, "icu_mortality"].eq(1)
            & (
                labels.loc[available_mask, "event_time_proxy_h"]
                <= labels.loc[available_mask, "future_window_end_h"]
            )
        )
        .astype("Int64")
        .to_numpy()
    )
    labels["label_semantics"] = (
        "Proxy-based within-horizon ICU mortality label using icu_end_time_proxy_hours "
        "as the only standardized event-time surrogate. Exact death timestamps are not "
        "available in the standardized artifacts."
    )
    labels = labels.drop(columns=["icu_mortality"])

    label_columns = list(valid_instances.columns) + [
        "label_name",
        "event_time_proxy_h",
        "label_value",
        "label_available",
        "label_semantics",
    ]
    labels = labels[label_columns]
    usable_labels = labels[labels["label_available"]].reset_index(drop=True)

    if labels.empty:
        summary_by_horizon = pd.DataFrame(
            columns=[
                "horizon_h",
                "valid_instances",
                "labeled_instances",
                "positive_labels",
                "negative_labels",
                "missing_label_instances",
            ]
        )
    else:
        summary_by_horizon = (
            labels.groupby("horizon_h", dropna=False)
            .agg(
                valid_instances=("instance_id", "size"),
                labeled_instances=("label_available", "sum"),
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
        summary_by_horizon["missing_label_instances"] = (
            summary_by_horizon["valid_instances"] - summary_by_horizon["labeled_instances"]
        )

    notes = pd.DataFrame(
        [
            {
                "note_id": "proxy_horizon_label",
                "category": "label_semantics",
                "note": (
                    "Chapter 1 emits proxy-based horizon labels derived from ICU mortality "
                    "and icu_end_time_proxy_hours. Exact death timestamps are not available, "
                    "so the ICU end-time proxy is used as the event-time surrogate."
                ),
            }
        ]
    )

    return Chapter1LabelResult(
        labels=labels,
        usable_labels=usable_labels,
        summary_by_horizon=summary_by_horizon,
        notes=notes,
    )
