from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from chapter1_mortality_decomposition.utils import read_dataframe


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHAPTER1_OUTPUT_DIR = Path("artifacts") / "chapter1"
DEFAULT_CLUSTER_RESULTS_DIR = Path("cluster-results") / "chapter1_true_results"
RESULT_ROOT_KIND_CLUSTER_EXPORT = "cluster_export"
RESULT_ROOT_KIND_SYNTHETIC_LOCAL = "synthetic_local"


@dataclass(frozen=True)
class Chapter1InputTables:
    static_harmonized: pd.DataFrame
    dynamic_harmonized: pd.DataFrame
    block_index: pd.DataFrame
    blocked_dynamic_features: pd.DataFrame
    stay_block_counts: pd.DataFrame
    mech_vent_stay_level_qc: pd.DataFrame
    mech_vent_episode_level: pd.DataFrame


@dataclass(frozen=True)
class Chapter1ResultRoot:
    path: Path
    kind: str
    exists: bool


@dataclass(frozen=True)
class Chapter1ResolvedArtifactPath:
    path: Path
    resolution: dict[str, object]


def _resolved_repo_root(repo_root: Path | None = None) -> Path:
    return Path(repo_root).resolve() if repo_root is not None else REPO_ROOT


def chapter1_synthetic_output_root(repo_root: Path | None = None) -> Path:
    return (_resolved_repo_root(repo_root) / DEFAULT_CHAPTER1_OUTPUT_DIR).resolve()


def chapter1_cluster_results_root(repo_root: Path | None = None) -> Path:
    return (_resolved_repo_root(repo_root) / DEFAULT_CLUSTER_RESULTS_DIR).resolve()


def chapter1_result_root_candidates(
    repo_root: Path | None = None,
) -> tuple[Chapter1ResultRoot, ...]:
    synthetic_root = chapter1_synthetic_output_root(repo_root)
    cluster_root = chapter1_cluster_results_root(repo_root)
    return (
        Chapter1ResultRoot(
            path=cluster_root,
            kind=RESULT_ROOT_KIND_CLUSTER_EXPORT,
            exists=cluster_root.exists(),
        ),
        Chapter1ResultRoot(
            path=synthetic_root,
            kind=RESULT_ROOT_KIND_SYNTHETIC_LOCAL,
            exists=synthetic_root.exists(),
        ),
    )


def resolve_chapter1_result_root(
    *,
    preferred_kind: str = RESULT_ROOT_KIND_CLUSTER_EXPORT,
    repo_root: Path | None = None,
    require_exists: bool = False,
    allow_fallback: bool = True,
) -> Chapter1ResultRoot:
    candidates = chapter1_result_root_candidates(repo_root=repo_root)
    candidates_by_kind = {candidate.kind: candidate for candidate in candidates}
    if preferred_kind not in candidates_by_kind:
        raise ValueError(
            "preferred_kind must be one of "
            f"{sorted(candidates_by_kind)}, got {preferred_kind!r}."
        )

    preferred = candidates_by_kind[preferred_kind]
    if not require_exists or preferred.exists:
        return preferred

    if allow_fallback:
        for candidate in candidates:
            if candidate.kind != preferred_kind and candidate.exists:
                return candidate

    checked_paths = ", ".join(str(candidate.path) for candidate in candidates)
    raise FileNotFoundError(
        "Could not resolve an existing Chapter 1 result root. "
        f"Checked: {checked_paths}"
    )


def resolve_chapter1_result_subdir(
    relative_path: Path | str,
    *,
    preferred_kind: str = RESULT_ROOT_KIND_CLUSTER_EXPORT,
    repo_root: Path | None = None,
    require_exists: bool = False,
    allow_fallback: bool = True,
) -> Chapter1ResolvedArtifactPath:
    relative_subdir = Path(relative_path)
    candidates = chapter1_result_root_candidates(repo_root=repo_root)
    candidates_by_kind = {candidate.kind: candidate for candidate in candidates}
    if preferred_kind not in candidates_by_kind:
        raise ValueError(
            "preferred_kind must be one of "
            f"{sorted(candidates_by_kind)}, got {preferred_kind!r}."
        )

    ordered_candidates = [candidates_by_kind[preferred_kind]]
    if allow_fallback:
        ordered_candidates.extend(
            candidate for candidate in candidates if candidate.kind != preferred_kind
        )

    checked_paths: list[str] = []
    for candidate in ordered_candidates:
        resolved_subdir = (candidate.path / relative_subdir).resolve()
        checked_paths.append(str(resolved_subdir))
        if not require_exists or resolved_subdir.exists():
            return Chapter1ResolvedArtifactPath(
                path=resolved_subdir,
                resolution={
                    "resolution_strategy": "default_result_root_search",
                    "preferred_result_kind": preferred_kind,
                    "selected_result_kind": candidate.kind,
                    "selected_result_root": str(candidate.path),
                    "checked_paths": checked_paths,
                },
            )

    raise FileNotFoundError(
        "Could not resolve an existing Chapter 1 artifact subdirectory. "
        f"Checked: {', '.join(checked_paths)}"
    )


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
