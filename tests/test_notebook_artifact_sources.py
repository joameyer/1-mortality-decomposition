from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from chapter1_mortality_decomposition.artifacts import (
    RESULT_ROOT_KIND_CLUSTER_EXPORT,
    RESULT_ROOT_KIND_SYNTHETIC_LOCAL,
)
from chapter1_mortality_decomposition.notebook_artifact_sources import (
    build_artifact_source_markdown,
    chapter1_result_relative_path,
    notebook_artifact_source,
    resolve_notebook_artifact_source,
)


class NotebookArtifactSourceTests(TestCase):
    def test_chapter1_result_relative_path_strips_artifact_root_prefix(self) -> None:
        relative = chapter1_result_relative_path(
            Path("artifacts")
            / "chapter1"
            / "evaluation"
            / "asic"
            / "baselines"
            / "primary_medians"
        )

        self.assertEqual(
            relative,
            Path("evaluation") / "asic" / "baselines" / "primary_medians",
        )

    def test_resolve_notebook_artifact_source_prefers_cluster_export(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cluster_root = (
                tmp_path
                / "cluster-results"
                / "chapter1_true_results"
                / "evaluation"
                / "asic"
                / "baselines"
                / "primary_medians"
            )
            synthetic_root = (
                tmp_path
                / "artifacts"
                / "chapter1"
                / "evaluation"
                / "asic"
                / "baselines"
                / "primary_medians"
            )
            cluster_root.mkdir(parents=True)
            synthetic_root.mkdir(parents=True)

            source = resolve_notebook_artifact_source(
                Path("evaluation") / "asic" / "baselines" / "primary_medians",
                label="Baseline evaluation package",
                repo_root=tmp_path,
            )

            self.assertEqual(source.path, cluster_root.resolve())
            self.assertEqual(
                source.resolution["selected_result_kind"],
                RESULT_ROOT_KIND_CLUSTER_EXPORT,
            )

    def test_build_artifact_source_markdown_warns_on_synthetic_local(self) -> None:
        source = notebook_artifact_source(
            "Hard-case review",
            path=Path("/tmp/artifacts/chapter1/evaluation/asic/hard_cases"),
            resolution={
                "resolution_strategy": "default_result_root_search",
                "preferred_result_kind": RESULT_ROOT_KIND_CLUSTER_EXPORT,
                "selected_result_kind": RESULT_ROOT_KIND_SYNTHETIC_LOCAL,
                "selected_result_root": "/tmp/artifacts/chapter1",
                "checked_paths": ["/tmp/cluster-results/chapter1_true_results/evaluation/asic/hard_cases"],
            },
        )

        markdown = build_artifact_source_markdown([source])

        self.assertIn("synthetic local", markdown)
        self.assertIn("not scientifically interpretable", markdown)
