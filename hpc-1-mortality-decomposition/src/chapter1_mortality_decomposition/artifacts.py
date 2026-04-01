from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from chapter1_mortality_decomposition.utils import read_dataframe


DEFAULT_CHAPTER1_OUTPUT_DIR = Path("artifacts") / "chapter1"


@dataclass(frozen=True)
class Chapter1InputTables:
    static_harmonized: pd.DataFrame
    dynamic_harmonized: pd.DataFrame
    block_index: pd.DataFrame
    blocked_dynamic_features: pd.DataFrame
    stay_block_counts: pd.DataFrame
    mech_vent_stay_level_qc: pd.DataFrame
    mech_vent_episode_level: pd.DataFrame


def load_chapter1_inputs(
    input_dir: Path,
    *,
    input_format: str = "csv",
) -> Chapter1InputTables:
    extension = "csv" if input_format == "csv" else "parquet"
    required_paths = {
        "static_harmonized": input_dir / "static" / f"harmonized.{extension}",
        "dynamic_harmonized": input_dir / "dynamic" / f"harmonized.{extension}",
        "block_index": input_dir / "blocked" / f"asic_8h_block_index.{extension}",
        "blocked_dynamic_features": (
            input_dir / "blocked" / f"asic_8h_blocked_dynamic_features.{extension}"
        ),
        "stay_block_counts": input_dir / "blocked" / f"asic_8h_stay_block_counts.{extension}",
        "mech_vent_stay_level_qc": (
            input_dir / "qc" / f"mech_vent_ge_24h_stay_level.{extension}"
        ),
        "mech_vent_episode_level": (
            input_dir / "qc" / f"mech_vent_ge_24h_episode_level.{extension}"
        ),
    }

    missing_paths = [str(path) for path in required_paths.values() if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing standardized ASIC input artifacts for Chapter 1 preprocessing: "
            + ", ".join(missing_paths)
        )

    return Chapter1InputTables(
        static_harmonized=read_dataframe(required_paths["static_harmonized"]),
        dynamic_harmonized=read_dataframe(required_paths["dynamic_harmonized"]),
        block_index=read_dataframe(required_paths["block_index"]),
        blocked_dynamic_features=read_dataframe(required_paths["blocked_dynamic_features"]),
        stay_block_counts=read_dataframe(required_paths["stay_block_counts"]),
        mech_vent_stay_level_qc=read_dataframe(required_paths["mech_vent_stay_level_qc"]),
        mech_vent_episode_level=read_dataframe(required_paths["mech_vent_episode_level"]),
    )
