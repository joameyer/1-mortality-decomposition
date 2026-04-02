from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from chapter1_mortality_decomposition.utils import require_columns, write_dataframe


BLOCK_INDEX_COLUMNS = [
    "stay_id_global",
    "hospital_id",
    "block_index",
    "block_start_h",
    "block_end_h",
    "prediction_time_h",
]
BLOCKED_DYNAMIC_EXCLUDED_COLUMNS = {
    "hospital_id",
    "stay_id_global",
    "stay_id_local",
    "time",
    "time_h",
    "minutes_since_admit",
    "source_row_order",
}


@dataclass(frozen=True)
class ASICTemporalBlockArtifacts:
    block_hours: int
    artifact_prefix: str
    block_index: pd.DataFrame
    blocked_dynamic_features: pd.DataFrame
    stay_block_counts: pd.DataFrame


def _require_positive_block_hours(block_hours: int) -> int:
    normalized = int(block_hours)
    if normalized <= 0:
        raise ValueError("ASIC temporal block size must be a positive integer number of hours.")
    return normalized


def _artifact_prefix(block_hours: int) -> str:
    return f"asic_{int(block_hours)}h"


def _ends_exactly_column_name(block_hours: int) -> str:
    return f"ends_exactly_on_{int(block_hours)}h_boundary"


def _prepare_reference_stay_block_counts(reference_stay_block_counts: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        reference_stay_block_counts,
        {
            "stay_id_global",
            "hospital_id",
            "icu_admission_time",
            "icu_end_time_proxy",
            "icu_end_time_proxy_hours",
        },
        "reference_stay_block_counts",
    )

    stays = reference_stay_block_counts[
        [
            "stay_id_global",
            "hospital_id",
            "icu_admission_time",
            "icu_end_time_proxy",
            "icu_end_time_proxy_hours",
        ]
    ].copy()
    stays["stay_id_global"] = stays["stay_id_global"].astype("string")
    stays["hospital_id"] = stays["hospital_id"].astype("string")
    stays["icu_end_time_proxy_hours"] = pd.to_numeric(
        stays["icu_end_time_proxy_hours"],
        errors="coerce",
    )
    if stays[["stay_id_global", "hospital_id"]].duplicated().any():
        duplicate_rows = (
            stays.loc[
                stays[["stay_id_global", "hospital_id"]].duplicated(keep=False),
                ["stay_id_global", "hospital_id"],
            ]
            .drop_duplicates()
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(
            "reference_stay_block_counts must contain one row per stay/hospital pair. "
            f"Duplicate examples: {duplicate_rows}"
        )
    return stays.sort_values(["hospital_id", "stay_id_global"]).reset_index(drop=True)


def _prepare_dynamic_harmonized(
    dynamic_harmonized: pd.DataFrame,
    stay_block_counts: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        dynamic_harmonized,
        {"stay_id_global", "hospital_id", "time"},
        "dynamic_harmonized",
    )

    stay_index = stay_block_counts[["stay_id_global", "hospital_id"]].copy()
    dynamic = dynamic_harmonized.copy()
    dynamic["stay_id_global"] = dynamic["stay_id_global"].astype("string")
    dynamic["hospital_id"] = dynamic["hospital_id"].astype("string")
    dynamic = dynamic.merge(stay_index, on=["stay_id_global", "hospital_id"], how="inner")
    dynamic["time_h"] = (
        pd.to_timedelta(dynamic["time"], errors="coerce").dt.total_seconds() / 3600.0
    )
    dynamic["source_row_order"] = pd.Series(
        range(dynamic.shape[0]),
        index=dynamic.index,
        dtype="Int64",
    )
    return dynamic


def _build_stay_block_counts(
    reference_stay_block_counts: pd.DataFrame,
    *,
    block_hours: int,
) -> pd.DataFrame:
    prepared = _prepare_reference_stay_block_counts(reference_stay_block_counts)
    ends_exactly_column = _ends_exactly_column_name(block_hours)
    prepared["completed_block_count"] = pd.Series(0, index=prepared.index, dtype="Int64")

    non_negative_proxy = (
        prepared["icu_end_time_proxy_hours"].notna()
        & prepared["icu_end_time_proxy_hours"].ge(0)
    )
    prepared.loc[non_negative_proxy, "completed_block_count"] = (
        prepared.loc[non_negative_proxy, "icu_end_time_proxy_hours"] // block_hours
    ).astype("Int64")
    prepared["has_completed_block"] = prepared["completed_block_count"].ge(1)
    prepared[ends_exactly_column] = (
        non_negative_proxy
        & prepared["icu_end_time_proxy_hours"].mod(block_hours).eq(0)
    )
    prepared["ends_exactly_on_block_boundary"] = prepared[ends_exactly_column]
    prepared["terminal_block_end_h"] = prepared["completed_block_count"] * block_hours

    ordered_columns = [
        "stay_id_global",
        "hospital_id",
        "icu_admission_time",
        "icu_end_time_proxy",
        "icu_end_time_proxy_hours",
        "completed_block_count",
        "has_completed_block",
        ends_exactly_column,
        "ends_exactly_on_block_boundary",
        "terminal_block_end_h",
    ]
    return prepared[ordered_columns].reset_index(drop=True)


def _build_block_index(
    stay_block_counts: pd.DataFrame,
    *,
    block_hours: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in stay_block_counts.itertuples(index=False):
        completed_block_count = (
            0 if pd.isna(row.completed_block_count) else int(row.completed_block_count)
        )
        for block_index in range(completed_block_count):
            block_start_h = block_index * block_hours
            block_end_h = block_start_h + block_hours
            rows.append(
                {
                    "stay_id_global": row.stay_id_global,
                    "hospital_id": row.hospital_id,
                    "block_index": block_index,
                    "block_start_h": block_start_h,
                    "block_end_h": block_end_h,
                    "prediction_time_h": block_end_h,
                }
            )

    if not rows:
        return pd.DataFrame(columns=BLOCK_INDEX_COLUMNS)

    return pd.DataFrame(rows)[BLOCK_INDEX_COLUMNS].sort_values(
        ["hospital_id", "stay_id_global", "block_index"]
    ).reset_index(drop=True)


def _blocked_dynamic_feature_columns(dynamic_harmonized: pd.DataFrame) -> list[str]:
    return [
        column
        for column in dynamic_harmonized.columns
        if column not in BLOCKED_DYNAMIC_EXCLUDED_COLUMNS
    ]


def _blocked_dynamic_output_columns(feature_columns: list[str]) -> list[str]:
    columns = [
        *BLOCK_INDEX_COLUMNS,
        "dynamic_row_count",
        "non_missing_measurements_in_block",
        "observed_variables_in_block",
    ]
    for column in feature_columns:
        columns.extend(
            [
                f"{column}_obs_count",
                f"{column}_mean",
                f"{column}_median",
                f"{column}_min",
                f"{column}_max",
                f"{column}_last",
            ]
        )
    return columns


def _build_blocked_dynamic_features(
    block_index: pd.DataFrame,
    dynamic_harmonized: pd.DataFrame,
    *,
    block_hours: int,
) -> pd.DataFrame:
    feature_columns = _blocked_dynamic_feature_columns(dynamic_harmonized)
    output_columns = _blocked_dynamic_output_columns(feature_columns)

    if block_index.empty:
        return pd.DataFrame(columns=output_columns)

    assignable = dynamic_harmonized[
        dynamic_harmonized["time_h"].notna() & dynamic_harmonized["time_h"].ge(0)
    ].copy()
    assignable["block_index"] = (assignable["time_h"] // block_hours).astype("Int64")

    assigned = assignable.merge(
        block_index[BLOCK_INDEX_COLUMNS],
        on=["stay_id_global", "hospital_id", "block_index"],
        how="inner",
    ).sort_values(
        ["hospital_id", "stay_id_global", "block_index", "time_h", "source_row_order"],
        kind="stable",
    )

    aggregated = pd.DataFrame(columns=output_columns)
    if not assigned.empty:
        group_columns = BLOCK_INDEX_COLUMNS
        grouped = assigned.groupby(group_columns, dropna=False, sort=False)
        dynamic_row_count = grouped.size().rename("dynamic_row_count")

        assigned.loc[:, feature_columns] = assigned[feature_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        assigned["row_non_missing_measurements"] = assigned[feature_columns].notna().sum(axis=1)
        grouped = assigned.groupby(group_columns, dropna=False, sort=False)
        counts = grouped[feature_columns].count().rename(
            columns=lambda column: f"{column}_obs_count"
        )

        aggregated = pd.concat(
            [
                dynamic_row_count,
                grouped["row_non_missing_measurements"]
                .sum()
                .rename("non_missing_measurements_in_block"),
                counts.gt(0).sum(axis=1).rename("observed_variables_in_block"),
                counts,
                grouped[feature_columns].mean().rename(columns=lambda column: f"{column}_mean"),
                grouped[feature_columns].median().rename(
                    columns=lambda column: f"{column}_median"
                ),
                grouped[feature_columns].min().rename(columns=lambda column: f"{column}_min"),
                grouped[feature_columns].max().rename(columns=lambda column: f"{column}_max"),
                grouped[feature_columns].agg("last").rename(
                    columns=lambda column: f"{column}_last"
                ),
            ],
            axis=1,
        ).reset_index()

    blocked_dynamic = block_index.merge(
        aggregated,
        on=BLOCK_INDEX_COLUMNS,
        how="left",
    )

    count_columns = [
        "dynamic_row_count",
        "non_missing_measurements_in_block",
        "observed_variables_in_block",
        *[f"{column}_obs_count" for column in feature_columns],
    ]
    summary_columns = [
        column
        for column in output_columns
        if column not in BLOCK_INDEX_COLUMNS and column not in count_columns
    ]

    for column in count_columns:
        if column not in blocked_dynamic.columns:
            blocked_dynamic[column] = pd.Series(0, index=blocked_dynamic.index, dtype="Int64")
        else:
            blocked_dynamic[column] = (
                pd.to_numeric(blocked_dynamic[column], errors="coerce").fillna(0).astype("Int64")
            )

    for column in summary_columns:
        if column not in blocked_dynamic.columns:
            blocked_dynamic[column] = pd.Series(pd.NA, index=blocked_dynamic.index, dtype="Float64")
        else:
            blocked_dynamic[column] = pd.to_numeric(blocked_dynamic[column], errors="coerce")

    return blocked_dynamic[output_columns].sort_values(
        ["hospital_id", "stay_id_global", "block_index"]
    ).reset_index(drop=True)


def build_asic_temporal_block_artifacts(
    *,
    dynamic_harmonized: pd.DataFrame,
    reference_stay_block_counts: pd.DataFrame,
    block_hours: int,
) -> ASICTemporalBlockArtifacts:
    normalized_block_hours = _require_positive_block_hours(block_hours)
    stay_block_counts = _build_stay_block_counts(
        reference_stay_block_counts,
        block_hours=normalized_block_hours,
    )
    prepared_dynamic = _prepare_dynamic_harmonized(dynamic_harmonized, stay_block_counts)
    block_index = _build_block_index(
        stay_block_counts,
        block_hours=normalized_block_hours,
    )
    blocked_dynamic_features = _build_blocked_dynamic_features(
        block_index,
        prepared_dynamic,
        block_hours=normalized_block_hours,
    )
    return ASICTemporalBlockArtifacts(
        block_hours=normalized_block_hours,
        artifact_prefix=_artifact_prefix(normalized_block_hours),
        block_index=block_index,
        blocked_dynamic_features=blocked_dynamic_features,
        stay_block_counts=stay_block_counts,
    )


def write_asic_temporal_block_artifacts(
    artifacts: ASICTemporalBlockArtifacts,
    *,
    output_dir: Path,
    output_format: str = "csv",
) -> dict[str, Path]:
    extension = "csv" if output_format == "csv" else "parquet"
    blocked_dir = Path(output_dir)
    return {
        "block_index": write_dataframe(
            artifacts.block_index,
            blocked_dir / f"{artifacts.artifact_prefix}_block_index.{extension}",
            output_format=output_format,
        ),
        "blocked_dynamic_features": write_dataframe(
            artifacts.blocked_dynamic_features,
            blocked_dir / f"{artifacts.artifact_prefix}_blocked_dynamic_features.{extension}",
            output_format=output_format,
        ),
        "stay_block_counts": write_dataframe(
            artifacts.stay_block_counts,
            blocked_dir / f"{artifacts.artifact_prefix}_stay_block_counts.{extension}",
            output_format=output_format,
        ),
    }
