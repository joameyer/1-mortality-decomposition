from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from chapter1_mortality_decomposition.carry_forward import INSTANCE_KEY_COLUMNS
from chapter1_mortality_decomposition.config import chapter1_group_definitions
from chapter1_mortality_decomposition.utils import require_columns


@dataclass(frozen=True)
class ObservationGroupSpec:
    group_name: str
    label: str
    block_indicator_column: str
    time_since_last_column: str
    candidate_columns: tuple[str, ...]


@dataclass(frozen=True)
class Chapter1ObservationProcessResult:
    block_features: pd.DataFrame
    qc_summary: pd.DataFrame
    verification_summary: pd.DataFrame
    spot_check_examples: pd.DataFrame
    implementation_note_markdown: str


OBSERVATION_GROUP_SPECS = (
    ObservationGroupSpec(
        group_name="cardiac_rate",
        label="HR group",
        block_indicator_column="obs_hr_grp_block",
        time_since_last_column="tsl_hr_grp_h",
        candidate_columns=chapter1_group_definitions()["cardiac_rate"],
    ),
    ObservationGroupSpec(
        group_name="blood_pressure",
        label="BP group",
        block_indicator_column="obs_bp_grp_block",
        time_since_last_column="tsl_bp_grp_h",
        candidate_columns=chapter1_group_definitions()["blood_pressure"],
    ),
    ObservationGroupSpec(
        group_name="respiratory",
        label="Respiratory group",
        block_indicator_column="obs_resp_grp_block",
        time_since_last_column="tsl_resp_grp_h",
        candidate_columns=chapter1_group_definitions()["respiratory"],
    ),
    ObservationGroupSpec(
        group_name="oxygenation",
        label="Oxygenation group",
        block_indicator_column="obs_oxy_grp_block",
        time_since_last_column="tsl_oxy_grp_h",
        candidate_columns=chapter1_group_definitions()["oxygenation"],
    ),
)

OBSERVATION_PROCESS_FEATURE_COLUMNS = [
    spec.block_indicator_column for spec in OBSERVATION_GROUP_SPECS
] + [
    "n_core_grps_obs_block",
] + [
    spec.time_since_last_column for spec in OBSERVATION_GROUP_SPECS
]

SPOT_CHECK_COLUMNS = [
    "group_name",
    "scenario",
    *INSTANCE_KEY_COLUMNS,
    "obs_indicator_column",
    "obs_indicator_value",
    "time_since_last_column",
    "time_since_last_hours",
    "latest_raw_observed_h",
]


def _empty_block_features() -> pd.DataFrame:
    return pd.DataFrame(columns=[*INSTANCE_KEY_COLUMNS, *OBSERVATION_PROCESS_FEATURE_COLUMNS])


def _empty_qc_frame() -> pd.DataFrame:
    hidden_columns = []
    for spec in OBSERVATION_GROUP_SPECS:
        hidden_columns.extend(
            [
                f"_{spec.group_name}_latest_observed_h",
                f"_{spec.group_name}_ever_observed_by_prediction",
            ]
        )
    return pd.DataFrame(
        columns=[*INSTANCE_KEY_COLUMNS, *OBSERVATION_PROCESS_FEATURE_COLUMNS, *hidden_columns]
    )


def _normalize_instance_index(instance_index: pd.DataFrame) -> pd.DataFrame:
    normalized = instance_index[INSTANCE_KEY_COLUMNS].drop_duplicates().copy()
    normalized["stay_id_global"] = normalized["stay_id_global"].astype("string")
    normalized["hospital_id"] = normalized["hospital_id"].astype("string")
    for column in ("block_index", "block_start_h", "block_end_h", "prediction_time_h"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype("int64")
    return normalized.sort_values(
        ["hospital_id", "stay_id_global", "prediction_time_h", "block_index"],
        kind="stable",
    ).reset_index(drop=True)


def _resolve_group_columns(dynamic_columns: set[str]) -> dict[str, tuple[str, ...]]:
    resolved: dict[str, tuple[str, ...]] = {}
    missing_groups: list[str] = []
    for spec in OBSERVATION_GROUP_SPECS:
        present_columns = tuple(
            column for column in spec.candidate_columns if column in dynamic_columns
        )
        if not present_columns:
            missing_groups.append(f"{spec.label} ({', '.join(spec.candidate_columns)})")
            continue
        resolved[spec.group_name] = present_columns
    if missing_groups:
        raise KeyError(
            "dynamic_harmonized is missing required raw observation columns for: "
            + "; ".join(missing_groups)
        )
    return resolved


def _prepare_dynamic_history(
    dynamic_harmonized: pd.DataFrame,
    *,
    resolved_group_columns: dict[str, tuple[str, ...]],
    stay_index: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {"stay_id_global", "hospital_id", "minutes_since_admit"}
    require_columns(dynamic_harmonized, required_columns, "dynamic_harmonized")

    selected_columns = {
        "stay_id_global",
        "hospital_id",
        "minutes_since_admit",
        *[column for columns in resolved_group_columns.values() for column in columns],
    }
    history = dynamic_harmonized[[column for column in dynamic_harmonized.columns if column in selected_columns]].copy()
    history["stay_id_global"] = history["stay_id_global"].astype("string")
    history["hospital_id"] = history["hospital_id"].astype("string")
    history["minutes_since_admit"] = pd.to_numeric(history["minutes_since_admit"], errors="coerce")
    history = history.dropna(subset=["minutes_since_admit"]).reset_index(drop=True)
    return history.merge(stay_index, on=["stay_id_global", "hospital_id"], how="inner")


def _observed_time_array(stay_history: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    observed = stay_history.loc[
        stay_history[list(columns)].notna().any(axis=1),
        "minutes_since_admit",
    ]
    if observed.empty:
        return np.array([], dtype=float)
    return np.sort(observed.dropna().astype(float).drop_duplicates().to_numpy())


def _derive_block_features(
    instance_index: pd.DataFrame,
    dynamic_history: pd.DataFrame,
    *,
    resolved_group_columns: dict[str, tuple[str, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if instance_index.empty:
        return _empty_block_features(), _empty_qc_frame()

    qc_rows: list[pd.DataFrame] = []

    for (_, _), block_df in instance_index.groupby(
        ["stay_id_global", "hospital_id"],
        dropna=False,
        sort=False,
    ):
        stay_blocks = block_df.sort_values(["prediction_time_h", "block_index"], kind="stable").reset_index(
            drop=True
        )
        stay_history = dynamic_history[
            dynamic_history["stay_id_global"].eq(stay_blocks.at[0, "stay_id_global"])
            & dynamic_history["hospital_id"].eq(stay_blocks.at[0, "hospital_id"])
        ].sort_values("minutes_since_admit", kind="stable")

        block_start_minutes = stay_blocks["block_start_h"].to_numpy(dtype=float) * 60.0
        prediction_minutes = stay_blocks["prediction_time_h"].to_numpy(dtype=float) * 60.0

        derived = stay_blocks.copy()
        for spec in OBSERVATION_GROUP_SPECS:
            observed_times = _observed_time_array(
                stay_history,
                resolved_group_columns[spec.group_name],
            )

            if observed_times.size == 0:
                derived[spec.block_indicator_column] = pd.Series(
                    np.zeros(stay_blocks.shape[0], dtype="int64"),
                    index=stay_blocks.index,
                )
                derived[spec.time_since_last_column] = pd.Series(
                    np.repeat(np.nan, stay_blocks.shape[0]),
                    index=stay_blocks.index,
                    dtype="Float64",
                )
                derived[f"_{spec.group_name}_latest_observed_h"] = pd.Series(
                    np.repeat(np.nan, stay_blocks.shape[0]),
                    index=stay_blocks.index,
                    dtype="Float64",
                )
                derived[f"_{spec.group_name}_ever_observed_by_prediction"] = pd.Series(
                    np.repeat(False, stay_blocks.shape[0]),
                    index=stay_blocks.index,
                    dtype="boolean",
                )
                continue

            current_block_start_idx = np.searchsorted(observed_times, block_start_minutes, side="left")
            current_block_end_idx = np.searchsorted(observed_times, prediction_minutes, side="left")
            last_observed_idx = current_block_end_idx - 1
            ever_observed_by_prediction = last_observed_idx >= 0

            latest_observed_minutes = np.full(stay_blocks.shape[0], np.nan)
            latest_observed_minutes[ever_observed_by_prediction] = observed_times[
                last_observed_idx[ever_observed_by_prediction]
            ]

            time_since_last_hours = np.full(stay_blocks.shape[0], np.nan)
            time_since_last_hours[ever_observed_by_prediction] = (
                prediction_minutes[ever_observed_by_prediction]
                - latest_observed_minutes[ever_observed_by_prediction]
            ) / 60.0

            derived[spec.block_indicator_column] = pd.Series(
                (current_block_end_idx - current_block_start_idx > 0).astype("int64"),
                index=stay_blocks.index,
            )
            derived[spec.time_since_last_column] = pd.Series(
                time_since_last_hours,
                index=stay_blocks.index,
                dtype="Float64",
            )
            derived[f"_{spec.group_name}_latest_observed_h"] = pd.Series(
                latest_observed_minutes / 60.0,
                index=stay_blocks.index,
                dtype="Float64",
            )
            derived[f"_{spec.group_name}_ever_observed_by_prediction"] = pd.Series(
                ever_observed_by_prediction,
                index=stay_blocks.index,
                dtype="boolean",
            )

        derived["n_core_grps_obs_block"] = (
            derived[[spec.block_indicator_column for spec in OBSERVATION_GROUP_SPECS]]
            .sum(axis=1)
            .astype("int64")
        )
        qc_rows.append(derived)

    qc_frame = pd.concat(qc_rows, ignore_index=True)
    block_features = qc_frame[[*INSTANCE_KEY_COLUMNS, *OBSERVATION_PROCESS_FEATURE_COLUMNS]].copy()
    block_features = block_features.sort_values(
        ["hospital_id", "stay_id_global", "prediction_time_h", "block_index"],
        kind="stable",
    ).reset_index(drop=True)
    qc_frame = qc_frame.sort_values(
        ["hospital_id", "stay_id_global", "prediction_time_h", "block_index"],
        kind="stable",
    ).reset_index(drop=True)
    return block_features, qc_frame


def _build_qc_summary(block_features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {
            "section": "dataset",
            "variable": pd.NA,
            "metric": "block_rows_total",
            "value": int(block_features.shape[0]),
            "detail": "Unique usable 8-hour blocks before horizon duplication.",
        }
    ]

    for column in block_features.columns:
        rows.append(
            {
                "section": "schema",
                "variable": column,
                "metric": "dtype",
                "value": str(block_features[column].dtype),
                "detail": "Column dtype in the block-level observation-process export.",
            }
        )

    row_total = int(block_features.shape[0])
    for column in OBSERVATION_PROCESS_FEATURE_COLUMNS:
        non_missing_count = int(block_features[column].notna().sum()) if column in block_features else 0
        rows.extend(
            [
                {
                    "section": "non_missingness",
                    "variable": column,
                    "metric": "non_missing_count",
                    "value": non_missing_count,
                    "detail": "Rows with a non-missing value in the block-level feature table.",
                },
                {
                    "section": "non_missingness",
                    "variable": column,
                    "metric": "non_missing_proportion",
                    "value": float(non_missing_count / row_total) if row_total else pd.NA,
                    "detail": "Non-missing proportion in the block-level feature table.",
                },
            ]
        )

    n_core_distribution = (
        block_features["n_core_grps_obs_block"].value_counts(dropna=False).reindex(range(5), fill_value=0)
        if "n_core_grps_obs_block" in block_features
        else pd.Series([0, 0, 0, 0, 0], index=range(5))
    )
    for core_count, instance_count in n_core_distribution.items():
        rows.extend(
            [
                {
                    "section": "n_core_grps_obs_block_distribution",
                    "variable": "n_core_grps_obs_block",
                    "metric": f"value_{int(core_count)}_count",
                    "value": int(instance_count),
                    "detail": "Distribution of observed core-group counts within the current block.",
                },
                {
                    "section": "n_core_grps_obs_block_distribution",
                    "variable": "n_core_grps_obs_block",
                    "metric": f"value_{int(core_count)}_proportion",
                    "value": float(instance_count / row_total) if row_total else pd.NA,
                    "detail": "Distribution of observed core-group counts within the current block.",
                },
            ]
        )

    for spec in OBSERVATION_GROUP_SPECS:
        series = pd.to_numeric(block_features[spec.time_since_last_column], errors="coerce")
        description = series.describe(percentiles=[0.25, 0.5, 0.75])
        summary_values = {
            "non_missing_count": int(series.notna().sum()),
            "mean_h": float(description["mean"]) if pd.notna(description.get("mean")) else pd.NA,
            "std_h": float(description["std"]) if pd.notna(description.get("std")) else pd.NA,
            "min_h": float(description["min"]) if pd.notna(description.get("min")) else pd.NA,
            "p25_h": float(description["25%"]) if pd.notna(description.get("25%")) else pd.NA,
            "median_h": float(description["50%"]) if pd.notna(description.get("50%")) else pd.NA,
            "p75_h": float(description["75%"]) if pd.notna(description.get("75%")) else pd.NA,
            "max_h": float(description["max"]) if pd.notna(description.get("max")) else pd.NA,
            "never_observed_missing_count": int(series.isna().sum()),
        }
        for metric, value in summary_values.items():
            rows.append(
                {
                    "section": "tsl_summary",
                    "variable": spec.time_since_last_column,
                    "metric": metric,
                    "value": value,
                    "detail": "Hours from prediction time to the most recent raw group observation.",
                }
            )

    return pd.DataFrame(rows)


def _build_verification_summary(qc_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for spec in OBSERVATION_GROUP_SPECS:
        numeric_values = pd.to_numeric(qc_frame[spec.block_indicator_column], errors="coerce")
        unique_values = sorted(
            numeric_values.dropna().astype(int).drop_duplicates().tolist()
        )
        rows.append(
            {
                "check_id": f"{spec.block_indicator_column}_binary_values_only",
                "passed": set(unique_values).issubset({0, 1}),
                "value": ", ".join(str(value) for value in unique_values),
                "detail": "Observed-in-block indicators must be binary {0, 1}.",
            }
        )

    expected_group_sum = qc_frame[[spec.block_indicator_column for spec in OBSERVATION_GROUP_SPECS]].sum(axis=1)
    rows.append(
        {
            "check_id": "n_core_grps_obs_block_matches_group_sum",
            "passed": bool(qc_frame["n_core_grps_obs_block"].eq(expected_group_sum).all()),
            "value": int(qc_frame["n_core_grps_obs_block"].eq(expected_group_sum).sum())
            if not qc_frame.empty
            else 0,
            "detail": "n_core_grps_obs_block must equal the row-wise sum of the four group indicators.",
        }
    )
    rows.append(
        {
            "check_id": "n_core_grps_obs_block_within_expected_range",
            "passed": bool(qc_frame["n_core_grps_obs_block"].between(0, 4).all()),
            "value": int(qc_frame["n_core_grps_obs_block"].between(0, 4).sum()) if not qc_frame.empty else 0,
            "detail": "n_core_grps_obs_block must always fall between 0 and 4 inclusive.",
        }
    )

    for spec in OBSERVATION_GROUP_SPECS:
        time_since_last = pd.to_numeric(qc_frame[spec.time_since_last_column], errors="coerce")
        ever_observed = qc_frame[f"_{spec.group_name}_ever_observed_by_prediction"].astype("boolean")
        recent_examples = int(
            (
                pd.to_numeric(qc_frame[spec.block_indicator_column], errors="coerce").eq(1)
                & time_since_last.lt(8)
            ).sum()
        )

        rows.extend(
            [
                {
                    "check_id": f"{spec.time_since_last_column}_non_negative",
                    "passed": bool(time_since_last.dropna().ge(0).all()),
                    "value": int(time_since_last.dropna().ge(0).sum()),
                    "detail": "Time-since-last values must be non-negative.",
                },
                {
                    "check_id": f"{spec.time_since_last_column}_never_observed_rows_remain_missing",
                    "passed": bool(time_since_last[ever_observed.ne(True)].isna().all()),
                    "value": int(time_since_last[ever_observed.ne(True)].isna().sum()),
                    "detail": (
                        "Rows with no raw observation history in the group up to prediction time "
                        "must remain missing."
                    ),
                },
                {
                    "check_id": f"{spec.time_since_last_column}_current_block_recent_examples_present",
                    "passed": bool(recent_examples > 0)
                    if bool(pd.to_numeric(qc_frame[spec.block_indicator_column], errors="coerce").eq(1).any())
                    else True,
                    "value": recent_examples,
                    "detail": (
                        "At least one row should show an in-block observation with time-since-last < 8h "
                        "when that group is observed in the current block."
                    ),
                },
            ]
        )

    return pd.DataFrame(rows)


def _build_spot_check_examples(qc_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scenario_specs = (
        (
            "observed_in_current_block_recent_measurement",
            lambda frame, spec: pd.to_numeric(frame[spec.block_indicator_column], errors="coerce").eq(1)
            & pd.to_numeric(frame[spec.time_since_last_column], errors="coerce").lt(8),
        ),
        (
            "historical_measurement_before_current_block",
            lambda frame, spec: pd.to_numeric(frame[spec.block_indicator_column], errors="coerce").eq(0)
            & pd.to_numeric(frame[spec.time_since_last_column], errors="coerce").notna(),
        ),
        (
            "never_observed_by_prediction_time",
            lambda frame, spec: frame[f"_{spec.group_name}_ever_observed_by_prediction"].ne(True)
            & pd.to_numeric(frame[spec.time_since_last_column], errors="coerce").isna(),
        ),
    )

    for spec in OBSERVATION_GROUP_SPECS:
        for scenario_name, predicate in scenario_specs:
            matching_rows = qc_frame[predicate(qc_frame, spec)]
            if matching_rows.empty:
                continue
            row = matching_rows.iloc[0]
            rows.append(
                {
                    "group_name": spec.group_name,
                    "scenario": scenario_name,
                    "stay_id_global": row["stay_id_global"],
                    "hospital_id": row["hospital_id"],
                    "block_index": int(row["block_index"]),
                    "block_start_h": int(row["block_start_h"]),
                    "block_end_h": int(row["block_end_h"]),
                    "prediction_time_h": int(row["prediction_time_h"]),
                    "obs_indicator_column": spec.block_indicator_column,
                    "obs_indicator_value": int(row[spec.block_indicator_column]),
                    "time_since_last_column": spec.time_since_last_column,
                    "time_since_last_hours": row[spec.time_since_last_column],
                    "latest_raw_observed_h": row[f"_{spec.group_name}_latest_observed_h"],
                }
            )

    if rows:
        return pd.DataFrame(rows)[SPOT_CHECK_COLUMNS]
    return pd.DataFrame(columns=SPOT_CHECK_COLUMNS)


def _build_implementation_note(resolved_group_columns: dict[str, tuple[str, ...]]) -> str:
    mapping_lines = []
    for spec in OBSERVATION_GROUP_SPECS:
        mapping_lines.append(
            f"- `{spec.label}`: {', '.join(f'`{column}`' for column in resolved_group_columns[spec.group_name])}"
        )

    return "\n".join(
        [
            "# Chapter 1 Observation-Process Variable Note",
            "",
            "## Final Group Mapping Used",
            *mapping_lines,
            "",
            "## Derivation Rules",
            "- Input source: raw `dynamic/harmonized` ASIC measurements plus unique usable 8-hour blocks.",
            "- Block membership uses the existing upstream ASIC blocked-data contract: raw measurements with `minutes_since_admit` in `[block_start_h, block_end_h)`.",
            "- `obs_*_grp_block` equals 1 when any raw group measurement is observed inside the current 8-hour block; otherwise 0.",
            "- `n_core_grps_obs_block` is the row-wise sum of the four binary group indicators.",
            "- `tsl_*` equals `prediction_time_h - latest_raw_observed_time_h` within stay and group, using all raw history with `minutes_since_admit < prediction_time_h * 60`.",
            "- If a group has never been observed up to prediction time, the corresponding `tsl_*` stays missing.",
            "",
            "## Deviations From Requested Design",
            "- None in the derived variables themselves.",
            "- The block-level export is unique per usable 8-hour block before horizon duplication; separate optional merged model-ready artifacts duplicate the block features across horizon-specific prediction-instance rows for convenience.",
            "",
            "## Limitations / Ambiguities",
            "- The raw-history derivation depends on `minutes_since_admit` in the harmonized ASIC dynamic table being aligned with the blocked 8-hour artifacts.",
            "- Exact boundary handling was chosen to match the upstream blocked `*_obs_count` contract and verified empirically against the blocked feature table.",
        ]
    )


def merge_observation_process_into_model_ready(
    model_ready: pd.DataFrame,
    block_features: pd.DataFrame,
) -> pd.DataFrame:
    """Append the optional observation-process block without changing the base model-ready export."""

    return model_ready.merge(
        block_features,
        on=INSTANCE_KEY_COLUMNS,
        how="left",
    )


def build_chapter1_observation_process_features(
    instance_index: pd.DataFrame,
    dynamic_harmonized: pd.DataFrame,
) -> Chapter1ObservationProcessResult:
    """Derive grouped block-centered observation-process variables from raw harmonized ASIC time series."""

    require_columns(instance_index, set(INSTANCE_KEY_COLUMNS), "instance_index")
    normalized_index = _normalize_instance_index(instance_index)
    resolved_group_columns = _resolve_group_columns(set(dynamic_harmonized.columns))
    stay_index = normalized_index[["stay_id_global", "hospital_id"]].drop_duplicates()
    dynamic_history = _prepare_dynamic_history(
        dynamic_harmonized,
        resolved_group_columns=resolved_group_columns,
        stay_index=stay_index,
    )
    block_features, qc_frame = _derive_block_features(
        normalized_index,
        dynamic_history,
        resolved_group_columns=resolved_group_columns,
    )

    return Chapter1ObservationProcessResult(
        block_features=block_features,
        qc_summary=_build_qc_summary(block_features),
        verification_summary=_build_verification_summary(qc_frame),
        spot_check_examples=_build_spot_check_examples(qc_frame),
        implementation_note_markdown=_build_implementation_note(resolved_group_columns),
    )
