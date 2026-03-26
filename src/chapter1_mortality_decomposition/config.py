from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class Chapter1Config:
    core_vital_variables: tuple[str, ...]
    optional_variables: tuple[str, ...]
    feature_statistics: tuple[str, ...]
    horizons_hours: tuple[int, ...]
    min_required_core_groups: int = 3


def normalize_horizons_hours(horizons_hours: Iterable[int] | None) -> tuple[int, ...]:
    if horizons_hours is None:
        return (8, 16, 24, 48)

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
) -> Chapter1Config:
    config = config or default_chapter1_config()
    replacements: dict[str, object] = {}
    if horizons_hours is not None:
        replacements["horizons_hours"] = normalize_horizons_hours(horizons_hours)
    if min_required_core_groups is not None:
        replacements["min_required_core_groups"] = int(min_required_core_groups)
    return replace(config, **replacements)


def chapter1_group_definitions() -> dict[str, tuple[str, ...]]:
    return {
        "cardiac_rate": ("heart_rate",),
        "blood_pressure": ("map", "sbp", "dbp"),
        "respiratory": ("resp_rate",),
        "oxygenation": ("spo2", "sao2"),
        "core_temp_optional": ("core_temp",),
    }


def selected_chapter1_feature_columns(
    blocked_dynamic_features: pd.DataFrame,
    config: Chapter1Config,
) -> list[str]:
    selected: list[str] = []
    variables = (*config.core_vital_variables, *config.optional_variables)
    available_columns = set(blocked_dynamic_features.columns)
    for variable in variables:
        for statistic in config.feature_statistics:
            candidate = f"{variable}_{statistic}"
            if candidate in available_columns:
                selected.append(candidate)
    return selected


def build_chapter1_feature_set_definition(
    blocked_dynamic_features: pd.DataFrame,
    config: Chapter1Config,
) -> pd.DataFrame:
    group_lookup: dict[str, str] = {}
    for group_name, variables in chapter1_group_definitions().items():
        for variable in variables:
            group_lookup[variable] = group_name

    rows = []
    available_columns = set(blocked_dynamic_features.columns)
    for variable in (*config.core_vital_variables, *config.optional_variables):
        for statistic in config.feature_statistics:
            feature_name = f"{variable}_{statistic}"
            rows.append(
                {
                    "feature_name": feature_name,
                    "base_variable": variable,
                    "statistic": statistic,
                    "physiologic_group": group_lookup.get(variable, pd.NA),
                    "required_for_site_inclusion": variable in config.core_vital_variables,
                    "selected_for_model": feature_name in available_columns,
                }
            )

    return pd.DataFrame(rows)
