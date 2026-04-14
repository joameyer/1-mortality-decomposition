from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chapter1_mortality_decomposition.artifacts import (
    RESULT_ROOT_KIND_CLUSTER_EXPORT,
    RESULT_ROOT_KIND_SYNTHETIC_LOCAL,
    chapter1_cluster_results_root,
    chapter1_result_root_candidates,
    chapter1_synthetic_output_root,
    resolve_chapter1_result_root,
)


class ArtifactRootResolutionTests(TestCase):
    def test_candidate_roots_expose_cluster_then_synthetic(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            candidates = chapter1_result_root_candidates(repo_root=repo_root)

            self.assertEqual(
                [candidate.kind for candidate in candidates],
                [
                    RESULT_ROOT_KIND_CLUSTER_EXPORT,
                    RESULT_ROOT_KIND_SYNTHETIC_LOCAL,
                ],
            )
            self.assertEqual(
                candidates[0].path,
                (repo_root / "cluster-results" / "chapter1_true_results").resolve(),
            )
            self.assertEqual(
                candidates[1].path,
                (repo_root / "artifacts" / "chapter1").resolve(),
            )
            self.assertFalse(any(candidate.exists for candidate in candidates))

    def test_resolve_prefers_cluster_export_when_present(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            chapter1_cluster_results_root(repo_root).mkdir(parents=True)
            chapter1_synthetic_output_root(repo_root).mkdir(parents=True)

            resolved = resolve_chapter1_result_root(
                repo_root=repo_root,
                require_exists=True,
            )

            self.assertEqual(resolved.kind, RESULT_ROOT_KIND_CLUSTER_EXPORT)
            self.assertTrue(resolved.exists)

    def test_resolve_falls_back_to_synthetic_when_cluster_export_missing(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            chapter1_synthetic_output_root(repo_root).mkdir(parents=True)

            resolved = resolve_chapter1_result_root(
                repo_root=repo_root,
                require_exists=True,
            )

            self.assertEqual(resolved.kind, RESULT_ROOT_KIND_SYNTHETIC_LOCAL)
            self.assertTrue(resolved.exists)

    def test_resolve_without_fallback_raises_when_preferred_root_is_missing(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            chapter1_synthetic_output_root(repo_root).mkdir(parents=True)

            with self.assertRaises(FileNotFoundError):
                resolve_chapter1_result_root(
                    repo_root=repo_root,
                    require_exists=True,
                    allow_fallback=False,
                )
