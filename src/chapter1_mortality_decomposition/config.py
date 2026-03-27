from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_CHAPTER1_FEATURE_SET_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "ch1_feature_sets.json"
)


@dataclass(frozen=True)
class Chapter1Config:
    core_vital_variables: tuple[str, ...]
    optional_variables: tuple[str, ...]
    feature_statistics: tuple[str, ...]
    horizons_hours: tuple[int, ...]
    min_required_core_groups: int = 3
    split_random_seed: int = 20260327
    feature_set_config_path: Path = DEFAULT_CHAPTER1_FEATURE_SET_CONFIG_PATH


@dataclass(frozen=True)
class Chapter1FeatureSetConfig:
    version: str
    primary_features: tuple[str, ...]
    extended_additional_features: tuple[str, ...]

    @property
    def extended_features(self) -> tuple[str, ...]:
        return (*self.primary_features, *self.extended_additional_features)

    @property
    def feature_sets(self) -> dict[str, tuple[str, ...]]:
        return {
            "primary": self.primary_features,
            "extended": self.extended_features,
        }


def normalize_horizons_hours(horizons_hours: Iterable[int] | None) -> tuple[int, ...]:
    if horizons_hours is None:
        return (8, 16, 24, 48, 72)

    normalized: list[int] = []
    seen: set[int] = set()
    for value in horizons_hours:
        horizon = int(value)
        if horizon <= 0:
            raise ValueError("Chapter 1 horizons must be positive integers in hours.")
        if horizon not in seen:
            normalized.append(horizon)
            seen.add(horizon)

    if not normalized:
        raise ValueError("At least one Chapter 1 horizon must be configured.")
    return tuple(normalized)


def default_chapter1_config() -> Chapter1Config:
    return Chapter1Config(
        core_vital_variables=(
            "heart_rate",
            "map",
            "sbp",
            "dbp",
            "resp_rate",
            "spo2",
            "sao2",
        ),
        optional_variables=("core_temp",),
        feature_statistics=("obs_count", "mean", "median", "min", "max", "last"),
        horizons_hours=normalize_horizons_hours(None),
    )


def updated_chapter1_config(
    config: Chapter1Config | None = None,
    *,
    horizons_hours: Iterable[int] | None = None,
    min_required_core_groups: int | None = None,
    split_random_seed: int | None = None,
    feature_set_config_path: Path | None = None,
) -> Chapter1Config:
    config = config or default_chapter1_config()
    replacements: dict[str, object] = {}
    if horizons_hours is not None:
        replacements["horizons_hours"] = normalize_horizons_hours(horizons_hours)
    if min_required_core_groups is not None:
        replacements["min_required_core_groups"] = int(min_required_core_groups)
    if split_random_seed is not None:
        replacements["split_random_seed"] = int(split_random_seed)
    if feature_set_config_path is not None:
        replacements["feature_set_config_path"] = Path(feature_set_config_path)
    return replace(config, **replacements)


def chapter1_group_definitions() -> dict[str, tuple[str, ...]]:
    return {
        "cardiac_rate": ("heart_rate",),
        "blood_pressure": ("map", "sbp", "dbp"),
        "respiratory": ("resp_rate",),
        "oxygenation": ("spo2", "sao2"),
    }


def _normalize_feature_names(feature_names: Iterable[str], *, label: str) -> tuple[str, ...]:
    normalized: list[str] = []
    duplicates: list[str] = []
    seen: set[str] = set()

    for feature_name in feature_names:
        name = str(feature_name).strip()
        if not name:
            raise ValueError(f"{label} contains an empty feature name.")
        if name in seen:
            duplicates.append(name)
            continue
        normalized.append(name)
        seen.add(name)

    if duplicates:
        duplicate_names = sorted(set(duplicates))
        raise ValueError(f"{label} contains duplicate feature names: {duplicate_names}")
    return tuple(normalized)


def load_chapter1_feature_set_config(
    path: Path = DEFAULT_CHAPTER1_FEATURE_SET_CONFIG_PATH,
) -> Chapter1FeatureSetConfig:
    config_path = Path(path)
    payload = json.loads(config_path.read_text())

    version = str(payload.get("version", "")).strip()
    if not version:
        raise ValueError("Chapter 1 feature-set config is missing a non-empty version.")

    primary_features = _normalize_feature_names(
        payload.get("primary_features", []),
        label="primary_features",
    )
    extended_additional_features = _normalize_feature_names(
        payload.get("extended_additional_features", []),
        label="extended_additional_features",
    )

    overlap = sorted(set(primary_features) & set(extended_additional_features))
    if overlap:
        raise ValueError(
            "extended_additional_features overlaps with primary_features: "
            f"{overlap}"
        )

    return Chapter1FeatureSetConfig(
        version=version,
        primary_features=primary_features,
        extended_additional_features=extended_additional_features,
    )


def _retained_blocked_dynamic_features(
    blocked_dynamic_features: pd.DataFrame,
    retained_stays: pd.DataFrame,
) -> pd.DataFrame:
    if retained_stays.empty:
        return blocked_dynamic_features.iloc[0:0].copy()

    retained_index = retained_stays[["stay_id_global", "hospital_id"]].copy()
    retained_index["stay_id_global"] = retained_index["stay_id_global"].astype("string")
    retained_index["hospital_id"] = retained_index["hospital_id"].astype("string")

    blocked = blocked_dynamic_features.copy()
    blocked["stay_id_global"] = blocked["stay_id_global"].astype("string")
    blocked["hospital_id"] = blocked["hospital_id"].astype("string")
    return blocked.merge(retained_index, on=["stay_id_global", "hospital_id"], how="inner")


def _base_feature_columns(
    base_variable: str,
    feature_statistics: tuple[str, ...],
    available_columns: set[str],
) -> list[str]:
    return [
        f"{base_variable}_{statistic}"
        for statistic in feature_statistics
        if f"{base_variable}_{statistic}" in available_columns
    ]


def _base_feature_has_retained_data(
    base_variable: str,
    retained_blocked_dynamic_features: pd.DataFrame,
    feature_statistics: tuple[str, ...],
) -> bool:
    available_columns = set(retained_blocked_dynamic_features.columns)
    obs_count_column = f"{base_variable}_obs_count"
    if obs_count_column in available_columns:
        obs_counts = pd.to_numeric(
            retained_blocked_dynamic_features[obs_count_column],
            errors="coerce",
        ).fillna(0)
        if obs_counts.gt(0).any():
            return True

    fallback_columns = [
        column
        for column in _base_feature_columns(base_variable, feature_statistics, available_columns)
        if column != obs_count_column
    ]
    return any(
        retained_blocked_dynamic_features[column].notna().any()
        for column in fallback_columns
    )


def build_chapter1_feature_set_definition(
    blocked_dynamic_features: pd.DataFrame,
    retained_stays: pd.DataFrame,
    config: Chapter1Config,
    feature_set_config: Chapter1FeatureSetConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_set_config = feature_set_config or load_chapter1_feature_set_config(
        config.feature_set_config_path
    )
    group_lookup: dict[str, str] = {}
    for group_name, variables in chapter1_group_definitions().items():
        for variable in variables:
            group_lookup[variable] = group_name

    available_columns = set(blocked_dynamic_features.columns)
    retained_blocked_dynamic_features = _retained_blocked_dynamic_features(
        blocked_dynamic_features,
        retained_stays,
    )

    all_base_variables = feature_set_config.extended_features
    base_feature_available_lookup = {
        base_variable: bool(
            _base_feature_columns(base_variable, config.feature_statistics, available_columns)
        )
        for base_variable in all_base_variables
    }
    base_feature_retained_data_lookup = {
        base_variable: (
            _base_feature_has_retained_data(
                base_variable,
                retained_blocked_dynamic_features,
                config.feature_statistics,
            )
            if base_feature_available_lookup[base_variable]
            else False
        )
        for base_variable in all_base_variables
    }

    rows: list[dict[str, object]] = []
    for feature_set_name, base_variables in feature_set_config.feature_sets.items():
        for base_variable in base_variables:
            for statistic in config.feature_statistics:
                feature_name = f"{base_variable}_{statistic}"
                available_in_blocked_schema = feature_name in available_columns
                if available_in_blocked_schema:
                    if statistic == "obs_count":
                        non_missing_in_retained_data = bool(
                            pd.to_numeric(
                                retained_blocked_dynamic_features[feature_name],
                                errors="coerce",
                            )
                            .fillna(0)
                            .gt(0)
                            .any()
                        )
                    else:
                        non_missing_in_retained_data = bool(
                            retained_blocked_dynamic_features[feature_name].notna().any()
                        )
                else:
                    non_missing_in_retained_data = False

                rows.append(
                    {
                        "feature_config_version": feature_set_config.version,
                        "feature_set_name": feature_set_name,
                        "feature_source_group": (
                            "primary"
                            if base_variable in feature_set_config.primary_features
                            else "extended_additional"
                        ),
                        "feature_name": feature_name,
                        "base_variable": base_variable,
                        "statistic": statistic,
                        "physiologic_group": group_lookup.get(base_variable, pd.NA),
                        "required_for_site_inclusion": base_variable in config.core_vital_variables,
                        "available_in_blocked_schema": available_in_blocked_schema,
                        "non_missing_in_retained_data": non_missing_in_retained_data,
                        "selected_for_model": available_in_blocked_schema,
                        "base_feature_available_in_blocked_schema": base_feature_available_lookup[
                            base_variable
                        ],
                        "base_feature_present_in_retained_data": (
                            base_feature_retained_data_lookup[base_variable]
                        ),
                    }
                )

    feature_set_definition = pd.DataFrame(rows)

    validation_rows: list[dict[str, object]] = []
    primary_feature_count = len(feature_set_config.primary_features)
    extended_only_feature_count = len(feature_set_config.extended_additional_features)
    total_extended_feature_count = len(feature_set_config.extended_features)
    for feature_set_name, base_variables in feature_set_config.feature_sets.items():
        missing_from_blocked_schema = [
            base_variable
            for base_variable in base_variables
            if not base_feature_available_lookup[base_variable]
        ]
        absent_from_retained_data = [
            base_variable
            for base_variable in base_variables
            if not base_feature_retained_data_lookup[base_variable]
        ]
        validation_rows.append(
            {
                "feature_config_version": feature_set_config.version,
                "feature_set_name": feature_set_name,
                "primary_feature_count": primary_feature_count,
                "extended_only_feature_count": extended_only_feature_count,
                "total_extended_feature_count": total_extended_feature_count,
                "configured_base_feature_count": len(base_variables),
                "available_in_blocked_schema_count": (
                    len(base_variables) - len(missing_from_blocked_schema)
                ),
                "missing_from_blocked_schema_count": len(missing_from_blocked_schema),
                "missing_from_blocked_schema_features": missing_from_blocked_schema,
                "present_in_retained_data_count": (
                    len(base_variables) - len(absent_from_retained_data)
                ),
                "absent_from_retained_data_count": len(absent_from_retained_data),
                "absent_from_retained_data_features": absent_from_retained_data,
                "duplicate_primary_feature_count": 0,
                "duplicate_extended_additional_feature_count": 0,
                "overlap_primary_extended_additional_count": 0,
            }
        )

    feature_set_validation_summary = pd.DataFrame(validation_rows)
    return feature_set_definition, feature_set_validation_summary
