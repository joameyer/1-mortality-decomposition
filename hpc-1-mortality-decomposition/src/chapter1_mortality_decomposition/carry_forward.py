from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from chapter1_mortality_decomposition.config import Chapter1Config, default_chapter1_config
from chapter1_mortality_decomposition.utils import require_columns


FAST_BEDSIDE_PHYSIOLOGY = {
    "heart_rate",
    "sbp",
    "map",
    "dbp",
    "resp_rate",
    "spo2",
    "sao2",
    "core_temp",
}
BLOOD_GAS_ACID_BASE = {
    "pao2",
    "paco2",
    "ph_art",
    "bicarbonate_art",
    "base_excess_art",
}
LACTATE = {"lactate_art"}
VENTILATOR_VARIABLES = {"fio2", "peep", "vt", "vt_per_kg_ibw", "etco2"}
STANDARD_LABS = {
    "hemoglobin",
    "hematocrit",
    "wbc",
    "platelets",
    "inr",
    "ptt",
    "crp",
}
SLOWER_LABS = {"albumin", "bilirubin_total", "urea", "creatinine"}

FEATURE_FAMILY_BY_BASE_VARIABLE = {
    **{feature_name: "fast_bedside_physiology" for feature_name in FAST_BEDSIDE_PHYSIOLOGY},
    **{feature_name: "blood_gas_acid_base" for feature_name in BLOOD_GAS_ACID_BASE},
    **{feature_name: "lactate" for feature_name in LACTATE},
    **{feature_name: "ventilator_variables" for feature_name in VENTILATOR_VARIABLES},
    **{feature_name: "standard_labs" for feature_name in STANDARD_LABS},
    **{feature_name: "slower_labs" for feature_name in SLOWER_LABS},
}
LOCF_WINDOW_HOURS_BY_BASE_VARIABLE = {
    **{feature_name: 4 for feature_name in FAST_BEDSIDE_PHYSIOLOGY},
    **{feature_name: 12 for feature_name in BLOOD_GAS_ACID_BASE},
    **{feature_name: 6 for feature_name in LACTATE},
    **{feature_name: 24 for feature_name in VENTILATOR_VARIABLES},
    **{feature_name: 24 for feature_name in STANDARD_LABS},
    **{feature_name: 48 for feature_name in SLOWER_LABS},
}

INSTANCE_KEY_COLUMNS = [
    "stay_id_global",
    "hospital_id",
    "block_index",
    "block_start_h",
    "block_end_h",
    "prediction_time_h",
]


@dataclass(frozen=True)
class Chapter1CarryForwardResult:
    feature_frame: pd.DataFrame
    feature_summary: pd.DataFrame
    ventilator_summary: pd.DataFrame
    missingness_by_hospital_and_family: pd.DataFrame
    verification_summary: pd.DataFrame


def _value_columns_for_base_variable(
    base_variable: str,
    *,
    feature_statistics: tuple[str, ...],
    available_columns: set[str],
) -> list[str]:
    return [
        f"{base_variable}_{statistic}"
        for statistic in feature_statistics
        if statistic != "obs_count" and f"{base_variable}_{statistic}" in available_columns
    ]


def _obs_count_column_for_base_variable(
    base_variable: str,
    *,
    available_columns: set[str],
) -> str | None:
    candidate = f"{base_variable}_obs_count"
    return candidate if candidate in available_columns else None


def _selected_base_variables(feature_set_definition: pd.DataFrame) -> list[str]:
    return (
        feature_set_definition.loc[feature_set_definition["selected_for_model"], "base_variable"]
        .astype("string")
        .dropna()
        .drop_duplicates()
        .tolist()
    )


def _build_ventilation_window_flags(
    block_history: pd.DataFrame,
    mech_vent_episode_level: pd.DataFrame,
) -> pd.Series:
    if block_history.empty:
        return pd.Series(pd.array([], dtype="boolean"), index=block_history.index)

    require_columns(
        mech_vent_episode_level,
        {
            "stay_id_global",
            "hospital_id",
            "episode_start_time",
            "episode_end_time",
        },
        "mech_vent_episode_level",
    )

    episodes = mech_vent_episode_level[
        ["stay_id_global", "hospital_id", "episode_start_time", "episode_end_time"]
    ].copy()
    episodes["stay_id_global"] = episodes["stay_id_global"].astype("string")
    episodes["hospital_id"] = episodes["hospital_id"].astype("string")
    episodes["episode_start_h"] = (
        pd.to_timedelta(episodes["episode_start_time"], errors="coerce").dt.total_seconds()
        / 3600.0
    )
    episodes["episode_end_h"] = (
        pd.to_timedelta(episodes["episode_end_time"], errors="coerce").dt.total_seconds() / 3600.0
    )
    episodes = episodes.dropna(subset=["episode_start_h", "episode_end_h"]).reset_index(drop=True)

    ventilation_window_active = pd.Series(False, index=block_history.index, dtype="boolean")
    history = block_history.copy()
    history["stay_id_global"] = history["stay_id_global"].astype("string")
    history["hospital_id"] = history["hospital_id"].astype("string")
    history["block_start_h"] = pd.to_numeric(history["block_start_h"], errors="coerce")
    history["block_end_h"] = pd.to_numeric(history["block_end_h"], errors="coerce")

    for (stay_id_global, hospital_id), group_index in history.groupby(
        ["stay_id_global", "hospital_id"],
        dropna=False,
    ).groups.items():
        group = history.loc[group_index]
        stay_episodes = episodes[
            episodes["stay_id_global"].eq(stay_id_global)
            & episodes["hospital_id"].eq(hospital_id)
        ]
        if stay_episodes.empty:
            continue

        active = pd.Series(False, index=group.index, dtype="boolean")
        for episode in stay_episodes.itertuples(index=False):
            active = active | (
                group["block_start_h"].lt(float(episode.episode_end_h))
                & group["block_end_h"].gt(float(episode.episode_start_h))
            )
        ventilation_window_active.loc[group.index] = active

    return ventilation_window_active


def _apply_bounded_locf_for_base_variable(
    block_history: pd.DataFrame,
    *,
    base_variable: str,
    value_columns: list[str],
    obs_count_column: str | None,
    locf_window_hours: int | None,
    ventilation_window_active: pd.Series,
) -> pd.DataFrame:
    observed_in_block = pd.Series(False, index=block_history.index, dtype="boolean")
    if obs_count_column is not None:
        observed_in_block = observed_in_block | (
            pd.to_numeric(block_history[obs_count_column], errors="coerce").fillna(0).gt(0)
        )
    if value_columns:
        observed_in_block = observed_in_block | block_history[value_columns].notna().any(axis=1)

    filled_by_locf = pd.Series(False, index=block_history.index, dtype="boolean")
    processed = block_history.copy()
    prediction_time_hours = pd.to_numeric(processed["prediction_time_h"], errors="coerce")
    is_ventilator_variable = base_variable in VENTILATOR_VARIABLES

    if locf_window_hours is not None and value_columns:
        for _, group in processed.groupby("stay_id_global", dropna=False, sort=False):
            ordered_index = group.sort_values(
                ["prediction_time_h", "block_index"],
                kind="stable",
            ).index.tolist()
            last_observed_values: pd.Series | None = None
            last_observed_time_h: float | None = None
            last_observed_within_ventilation_window = False

            for row_index in ordered_index:
                if bool(observed_in_block.loc[row_index]):
                    last_observed_values = processed.loc[row_index, value_columns].copy()
                    last_observed_time_h = float(prediction_time_hours.loc[row_index])
                    last_observed_within_ventilation_window = bool(
                        ventilation_window_active.loc[row_index]
                    )
                    continue

                if last_observed_values is None or last_observed_time_h is None:
                    continue

                current_prediction_time_h = float(prediction_time_hours.loc[row_index])
                hours_since_last_observed = current_prediction_time_h - last_observed_time_h
                if hours_since_last_observed <= 0 or hours_since_last_observed > locf_window_hours:
                    continue

                if is_ventilator_variable and (
                    not bool(ventilation_window_active.loc[row_index])
                    or not last_observed_within_ventilation_window
                ):
                    continue

                processed.loc[row_index, value_columns] = last_observed_values.to_numpy()
                filled_by_locf.loc[row_index] = True

    missing_after_locf = (
        processed[value_columns].isna().all(axis=1)
        if value_columns
        else pd.Series(True, index=processed.index, dtype="boolean")
    )
    processed[f"{base_variable}_observed_in_block"] = observed_in_block.astype("boolean")
    processed[f"{base_variable}_filled_by_locf"] = filled_by_locf.astype("boolean")
    processed[f"{base_variable}_missing_after_locf"] = missing_after_locf.astype("boolean")
    if is_ventilator_variable:
        processed[f"{base_variable}_ventilation_window_active"] = (
            ventilation_window_active.astype("boolean")
        )
    return processed


def _build_feature_summary(
    feature_frame: pd.DataFrame,
    *,
    feature_set_name: str,
    selected_base_variables: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total_prediction_instances = int(feature_frame.shape[0])

    for base_variable in selected_base_variables:
        observed_column = f"{base_variable}_observed_in_block"
        filled_column = f"{base_variable}_filled_by_locf"
        missing_column = f"{base_variable}_missing_after_locf"
        if observed_column not in feature_frame.columns:
            continue

        rows.append(
            {
                "feature_set_name": feature_set_name,
                "qc_unit": "unique_prediction_instances_before_horizon_duplication",
                "base_variable": base_variable,
                "feature_family": FEATURE_FAMILY_BY_BASE_VARIABLE.get(
                    base_variable,
                    "no_bounded_locf_configured",
                ),
                "locf_window_hours": LOCF_WINDOW_HOURS_BY_BASE_VARIABLE.get(base_variable, pd.NA),
                "ventilation_window_restricted": base_variable in VENTILATOR_VARIABLES,
                "prediction_instances_total": total_prediction_instances,
                "originally_observed_instances": int(feature_frame[observed_column].eq(True).sum()),
                "locf_filled_instances": int(feature_frame[filled_column].eq(True).sum()),
                "remaining_missing_instances": int(feature_frame[missing_column].eq(True).sum()),
            }
        )

    return pd.DataFrame(rows)


def _build_ventilator_summary(
    feature_frame: pd.DataFrame,
    *,
    feature_set_name: str,
    selected_base_variables: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for base_variable in selected_base_variables:
        if base_variable not in VENTILATOR_VARIABLES:
            continue

        filled_column = f"{base_variable}_filled_by_locf"
        window_column = f"{base_variable}_ventilation_window_active"
        if filled_column not in feature_frame.columns or window_column not in feature_frame.columns:
            continue

        filled_inside_window = feature_frame[filled_column].eq(True) & feature_frame[window_column].eq(
            True
        )
        filled_outside_window = feature_frame[filled_column].eq(True) & feature_frame[
            window_column
        ].ne(True)
        rows.append(
            {
                "feature_set_name": feature_set_name,
                "qc_unit": "unique_prediction_instances_before_horizon_duplication",
                "base_variable": base_variable,
                "locf_fills_inside_ventilation_window": int(filled_inside_window.sum()),
                "locf_fills_outside_ventilation_window": int(filled_outside_window.sum()),
                "no_locf_fills_outside_ventilation_window": int(filled_outside_window.sum()) == 0,
            }
        )
    return pd.DataFrame(rows)


def _build_missingness_by_hospital_and_family(
    feature_frame: pd.DataFrame,
    *,
    feature_set_name: str,
    selected_base_variables: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if feature_frame.empty:
        return pd.DataFrame(
            columns=[
                "feature_set_name",
                "qc_unit",
                "hospital_id",
                "feature_family",
                "base_feature_count",
                "prediction_instances_total",
                "feature_instance_cells_total",
                "missing_before_locf_count",
                "missing_after_locf_count",
                "missing_before_locf_proportion",
                "missing_after_locf_proportion",
            ]
        )

    family_lookup: dict[str, list[str]] = {}
    for base_variable in selected_base_variables:
        family_lookup.setdefault(
            FEATURE_FAMILY_BY_BASE_VARIABLE.get(base_variable, "no_bounded_locf_configured"),
            [],
        ).append(base_variable)

    for hospital_id, hospital_df in feature_frame.groupby("hospital_id", dropna=False):
        prediction_instances_total = int(hospital_df.shape[0])
        for feature_family, base_variables in family_lookup.items():
            missing_before_count = 0
            missing_after_count = 0
            for base_variable in base_variables:
                observed_column = f"{base_variable}_observed_in_block"
                missing_after_column = f"{base_variable}_missing_after_locf"
                if observed_column not in hospital_df.columns:
                    continue
                missing_before_count += int(hospital_df[observed_column].ne(True).sum())
                missing_after_count += int(hospital_df[missing_after_column].eq(True).sum())

            feature_instance_cells_total = int(prediction_instances_total * len(base_variables))
            rows.append(
                {
                    "feature_set_name": feature_set_name,
                    "qc_unit": "unique_prediction_instances_before_horizon_duplication",
                    "hospital_id": hospital_id,
                    "feature_family": feature_family,
                    "base_feature_count": int(len(base_variables)),
                    "prediction_instances_total": prediction_instances_total,
                    "feature_instance_cells_total": feature_instance_cells_total,
                    "missing_before_locf_count": missing_before_count,
                    "missing_after_locf_count": missing_after_count,
                    "missing_before_locf_proportion": (
                        float(missing_before_count / feature_instance_cells_total)
                        if feature_instance_cells_total
                        else pd.NA
                    ),
                    "missing_after_locf_proportion": (
                        float(missing_after_count / feature_instance_cells_total)
                        if feature_instance_cells_total
                        else pd.NA
                    ),
                }
            )

    return pd.DataFrame(rows)


def _build_carry_forward_verification_summary(
    *,
    feature_set_name: str,
    ventilator_summary: pd.DataFrame,
) -> pd.DataFrame:
    no_outside_window_fills = (
        bool(ventilator_summary["no_locf_fills_outside_ventilation_window"].all())
        if not ventilator_summary.empty
        else True
    )
    return pd.DataFrame(
        [
            {
                "feature_set_name": feature_set_name,
                "check_id": "no_global_median_imputation_applied_in_export",
                "passed": True,
                "detail": (
                    "The Chapter 1 preprocessing export applies bounded LOCF only. "
                    "No global median or other final imputation is applied."
                ),
            },
            {
                "feature_set_name": feature_set_name,
                "check_id": "no_ventilator_locf_outside_supported_windows",
                "passed": no_outside_window_fills,
                "detail": (
                    "Ventilator-variable LOCF fills are restricted to blocks overlapping "
                    "upstream ventilation-supported episodes."
                ),
            },
        ]
    )


def build_chapter1_locf_feature_frame(
    instance_index: pd.DataFrame,
    blocked_dynamic_features: pd.DataFrame,
    feature_set_definition: pd.DataFrame,
    mech_vent_episode_level: pd.DataFrame,
    config: Chapter1Config | None = None,
    *,
    feature_set_name: str,
) -> Chapter1CarryForwardResult:
    config = config or default_chapter1_config()

    require_columns(instance_index, set(INSTANCE_KEY_COLUMNS), "instance_index")
    require_columns(
        blocked_dynamic_features,
        set(INSTANCE_KEY_COLUMNS),
        "blocked_dynamic_features",
    )
    require_columns(
        feature_set_definition,
        {"base_variable", "feature_name", "selected_for_model"},
        "feature_set_definition",
    )

    instance_frame = instance_index[INSTANCE_KEY_COLUMNS].drop_duplicates().copy()
    instance_frame["stay_id_global"] = instance_frame["stay_id_global"].astype("string")
    instance_frame["hospital_id"] = instance_frame["hospital_id"].astype("string")

    selected_base_variables = _selected_base_variables(feature_set_definition)
    available_columns = set(blocked_dynamic_features.columns)
    selected_feature_columns = (
        feature_set_definition.loc[feature_set_definition["selected_for_model"], "feature_name"]
        .astype("string")
        .tolist()
    )

    history = blocked_dynamic_features[
        [*INSTANCE_KEY_COLUMNS, *[column for column in selected_feature_columns if column in available_columns]]
    ].copy()
    history["stay_id_global"] = history["stay_id_global"].astype("string")
    history["hospital_id"] = history["hospital_id"].astype("string")

    stay_index = instance_frame[["stay_id_global", "hospital_id"]].drop_duplicates()
    history = history.merge(stay_index, on=["stay_id_global", "hospital_id"], how="inner")
    history = history.sort_values(
        ["stay_id_global", "prediction_time_h", "block_index"],
        kind="stable",
    ).reset_index(drop=True)

    ventilation_window_active = _build_ventilation_window_flags(history, mech_vent_episode_level)
    if selected_base_variables:
        for base_variable in selected_base_variables:
            value_columns = _value_columns_for_base_variable(
                base_variable,
                feature_statistics=config.feature_statistics,
                available_columns=available_columns,
            )
            obs_count_column = _obs_count_column_for_base_variable(
                base_variable,
                available_columns=available_columns,
            )
            history = _apply_bounded_locf_for_base_variable(
                history,
                base_variable=base_variable,
                value_columns=value_columns,
                obs_count_column=obs_count_column,
                locf_window_hours=LOCF_WINDOW_HOURS_BY_BASE_VARIABLE.get(base_variable),
                ventilation_window_active=ventilation_window_active,
            )

    feature_indicator_columns = []
    for base_variable in selected_base_variables:
        for suffix in (
            "observed_in_block",
            "filled_by_locf",
            "missing_after_locf",
            "ventilation_window_active",
        ):
            column = f"{base_variable}_{suffix}"
            if column in history.columns:
                feature_indicator_columns.append(column)

    feature_frame = history.merge(instance_frame, on=INSTANCE_KEY_COLUMNS, how="inner")
    feature_frame = feature_frame[
        [
            *INSTANCE_KEY_COLUMNS,
            *[column for column in selected_feature_columns if column in feature_frame.columns],
            *feature_indicator_columns,
        ]
    ].copy()

    feature_summary = _build_feature_summary(
        feature_frame,
        feature_set_name=feature_set_name,
        selected_base_variables=selected_base_variables,
    )
    ventilator_summary = _build_ventilator_summary(
        feature_frame,
        feature_set_name=feature_set_name,
        selected_base_variables=selected_base_variables,
    )
    missingness_by_hospital_and_family = _build_missingness_by_hospital_and_family(
        feature_frame,
        feature_set_name=feature_set_name,
        selected_base_variables=selected_base_variables,
    )
    verification_summary = _build_carry_forward_verification_summary(
        feature_set_name=feature_set_name,
        ventilator_summary=ventilator_summary,
    )
    return Chapter1CarryForwardResult(
        feature_frame=feature_frame,
        feature_summary=feature_summary,
        ventilator_summary=ventilator_summary,
        missingness_by_hospital_and_family=missingness_by_hospital_and_family,
        verification_summary=verification_summary,
    )
