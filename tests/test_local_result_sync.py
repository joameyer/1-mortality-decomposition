from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chapter1_mortality_decomposition.local_result_sync import (  # noqa: E402
    import_staged_chapter1_exports,
)


class LocalResultSyncTests(TestCase):
    def test_import_staged_exports_copies_manifest_listed_files_only(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            source_root = repo_root / "export-staging" / "chapter1_true_results"
            staged_dir = (
                source_root
                / "evaluation"
                / "asic"
                / "hard_cases"
                / "primary_medians"
                / "logistic_regression"
                / "asic_hard_case_comparison"
            )
            staged_dir.mkdir(parents=True, exist_ok=True)
            (staged_dir / "comparison_table.csv").write_text("variable,value\nmap,1.2\n")
            (staged_dir / "summary.md").write_text("# Summary\n")
            (staged_dir / "unexpected_extra.txt").write_text("do not copy\n")
            (staged_dir / "export_manifest.json").write_text(
                json.dumps(
                    {
                        "export_name": "asic_hard_case_comparison_local_review_aggregate_bundle",
                        "copied_relative_paths": [
                            "comparison_table.csv",
                            "summary.md",
                        ],
                    },
                    indent=2,
                )
            )

            destination_root = repo_root / "cluster-results" / "chapter1_true_results"
            result = import_staged_chapter1_exports(
                source_root=source_root,
                destination_root=destination_root,
            )

            imported_dir = destination_root / staged_dir.relative_to(source_root)
            self.assertEqual(len(result.manifest_paths), 1)
            self.assertTrue((imported_dir / "comparison_table.csv").exists())
            self.assertTrue((imported_dir / "summary.md").exists())
            self.assertTrue((imported_dir / "export_manifest.json").exists())
            self.assertFalse((imported_dir / "unexpected_extra.txt").exists())

    def test_import_staged_exports_refuses_overwrite_without_flag(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            source_root = repo_root / "export-staging" / "chapter1_true_results"
            staged_dir = source_root / "evaluation" / "asic" / "example_bundle"
            staged_dir.mkdir(parents=True, exist_ok=True)
            (staged_dir / "summary.md").write_text("# Summary\n")
            (staged_dir / "export_manifest.json").write_text(
                json.dumps(
                    {
                        "export_name": "example_bundle",
                        "copied_relative_paths": ["summary.md"],
                    },
                    indent=2,
                )
            )

            destination_root = repo_root / "cluster-results" / "chapter1_true_results"
            first_result = import_staged_chapter1_exports(
                source_root=source_root,
                destination_root=destination_root,
            )
            self.assertEqual(len(first_result.copied_paths), 2)

            with self.assertRaises(FileExistsError):
                import_staged_chapter1_exports(
                    source_root=source_root,
                    destination_root=destination_root,
                )
