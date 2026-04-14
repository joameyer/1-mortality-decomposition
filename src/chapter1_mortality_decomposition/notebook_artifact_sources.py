from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from chapter1_mortality_decomposition.artifacts import (
    DEFAULT_CHAPTER1_OUTPUT_DIR,
    RESULT_ROOT_KIND_CLUSTER_EXPORT,
    RESULT_ROOT_KIND_SYNTHETIC_LOCAL,
    resolve_chapter1_result_subdir,
)


@dataclass(frozen=True)
class NotebookArtifactSource:
    label: str
    path: Path
    resolution: dict[str, object]


def find_notebook_repo_root(start: Path | None = None) -> Path:
    candidate_start = Path.cwd().resolve() if start is None else Path(start).resolve()
    for candidate in (candidate_start, *candidate_start.parents):
        if (candidate / "pyproject.toml").exists() and (
            candidate / "src" / "chapter1_mortality_decomposition"
        ).exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate the repository root from {candidate_start}."
    )


def chapter1_result_relative_path(path: Path | str) -> Path:
    normalized = Path(path)
    root_parts = DEFAULT_CHAPTER1_OUTPUT_DIR.parts
    if normalized.parts[: len(root_parts)] != root_parts:
        raise ValueError(
            "Expected a path rooted under "
            f"{DEFAULT_CHAPTER1_OUTPUT_DIR}, got {normalized}."
        )
    return Path(*normalized.parts[len(root_parts) :])


def notebook_artifact_source(
    label: str,
    *,
    path: Path,
    resolution: dict[str, object],
) -> NotebookArtifactSource:
    return NotebookArtifactSource(
        label=label,
        path=Path(path),
        resolution=dict(resolution),
    )


def resolve_notebook_artifact_source(
    relative_path: Path | str,
    *,
    label: str,
    preferred_result_kind: str = RESULT_ROOT_KIND_CLUSTER_EXPORT,
    repo_root: Path | None = None,
    allow_fallback: bool = True,
    require_exists: bool = True,
) -> NotebookArtifactSource:
    resolved = resolve_chapter1_result_subdir(
        relative_path,
        preferred_kind=preferred_result_kind,
        repo_root=repo_root,
        require_exists=require_exists,
        allow_fallback=allow_fallback,
    )
    return NotebookArtifactSource(
        label=label,
        path=resolved.path,
        resolution=resolved.resolution,
    )


def _display_path(path: Path, *, display_root: Path | None = None) -> str:
    resolved = Path(path).resolve()
    if display_root is not None:
        try:
            return str(resolved.relative_to(Path(display_root).resolve()))
        except ValueError:
            pass
    return str(resolved)


def _result_kind_label(result_kind: str | None) -> str:
    if result_kind == RESULT_ROOT_KIND_CLUSTER_EXPORT:
        return "cluster export"
    if result_kind == RESULT_ROOT_KIND_SYNTHETIC_LOCAL:
        return "synthetic local"
    return "explicit path"


def build_artifact_source_markdown(
    sources: Sequence[NotebookArtifactSource],
    *,
    display_root: Path | None = None,
) -> str:
    normalized_sources = list(sources)
    if not normalized_sources:
        raise ValueError("At least one notebook artifact source is required.")

    selected_kinds = {
        source.resolution.get("selected_result_kind")
        for source in normalized_sources
    }
    if selected_kinds == {RESULT_ROOT_KIND_CLUSTER_EXPORT}:
        summary = (
            "All notebook inputs currently resolve to `cluster export` artifacts. "
            "These are the authoritative local mirror of approved HPC outputs."
        )
    elif selected_kinds == {RESULT_ROOT_KIND_SYNTHETIC_LOCAL}:
        summary = (
            "All notebook inputs currently resolve to `synthetic local` artifacts. "
            "These are development and smoke-test outputs only, so embedded results are "
            "not scientifically interpretable."
        )
    else:
        summary = (
            "Notebook inputs currently resolve from mixed tiers or explicit paths. "
            "Check the table below before interpreting any outputs."
        )

    lines = [
        "## Artifact Source Mode",
        "",
        summary,
        "",
        "| input | tier | path |",
        "| --- | --- | --- |",
    ]
    for source in normalized_sources:
        lines.append(
            "| {label} | `{tier}` | `{path}` |".format(
                label=source.label,
                tier=_result_kind_label(source.resolution.get("selected_result_kind")),
                path=_display_path(source.path, display_root=display_root),
            )
        )

    if RESULT_ROOT_KIND_SYNTHETIC_LOCAL in selected_kinds:
        lines.extend(
            [
                "",
                "Synthetic-local inputs are useful for implementation checks, but they "
                "should not be used for scientific interpretation.",
            ]
        )
    return "\n".join(lines)
