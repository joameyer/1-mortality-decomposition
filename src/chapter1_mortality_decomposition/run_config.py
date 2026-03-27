from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from chapter1_mortality_decomposition.artifacts import DEFAULT_CHAPTER1_OUTPUT_DIR
from chapter1_mortality_decomposition.config import (
    DEFAULT_CHAPTER1_FEATURE_SET_CONFIG_PATH,
    Chapter1Config,
    default_chapter1_config,
    normalize_horizons_hours,
    updated_chapter1_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHAPTER1_RUN_CONFIG_PATH = REPO_ROOT / "config" / "ch1_run_config.json"
SUPPORTED_ARTIFACT_FORMATS = {"csv", "parquet"}


@dataclass(frozen=True)
class Chapter1RunConfig:
    input_dir: Path
    input_format: str
    output_dir: Path
    output_format: str
    horizons_hours: tuple[int, ...]
    min_required_core_groups: int
    split_random_seed: int
    feature_set_config_path: Path
    run_config_path: Path

    def to_chapter1_config(self) -> Chapter1Config:
        return updated_chapter1_config(
            default_chapter1_config(),
            horizons_hours=self.horizons_hours,
            min_required_core_groups=self.min_required_core_groups,
            split_random_seed=self.split_random_seed,
            feature_set_config_path=self.feature_set_config_path,
        )


def _resolve_repo_path(path_value: str | Path) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _normalize_artifact_format(
    value: object | None,
    *,
    label: str,
    default: str,
) -> str:
    if value is None:
        return default

    normalized = str(value).strip().lower()
    if normalized not in SUPPORTED_ARTIFACT_FORMATS:
        raise ValueError(
            f"{label} must be one of {sorted(SUPPORTED_ARTIFACT_FORMATS)}, got {value!r}."
        )
    return normalized


def load_chapter1_run_config(
    path: Path = DEFAULT_CHAPTER1_RUN_CONFIG_PATH,
) -> Chapter1RunConfig:
    config_path = Path(path)
    payload = json.loads(config_path.read_text())

    input_dir_value = payload.get("input_dir")
    if not input_dir_value:
        raise ValueError("Chapter 1 run config must define a non-empty input_dir.")

    min_required_core_groups = int(payload.get("min_required_core_groups", 3))
    if min_required_core_groups <= 0:
        raise ValueError("min_required_core_groups must be a positive integer.")
    split_random_seed = int(payload.get("split_random_seed", 20260327))

    return Chapter1RunConfig(
        input_dir=_resolve_repo_path(input_dir_value),
        input_format=_normalize_artifact_format(
            payload.get("input_format"),
            label="input_format",
            default="csv",
        ),
        output_dir=_resolve_repo_path(payload.get("output_dir", DEFAULT_CHAPTER1_OUTPUT_DIR)),
        output_format=_normalize_artifact_format(
            payload.get("output_format"),
            label="output_format",
            default="csv",
        ),
        horizons_hours=normalize_horizons_hours(payload.get("horizons_hours")),
        min_required_core_groups=min_required_core_groups,
        split_random_seed=split_random_seed,
        feature_set_config_path=_resolve_repo_path(
            payload.get("feature_set_config_path", DEFAULT_CHAPTER1_FEATURE_SET_CONFIG_PATH)
        ),
        run_config_path=config_path.resolve(),
    )
