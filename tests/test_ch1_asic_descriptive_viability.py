from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chapter1_mortality_decomposition.ch1_asic_descriptive_viability import (  # noqa: E402
    generate_ch1_asic_descriptive_viability,
)


class Sprint3AsicViabilityReviewTests(TestCase):
    def test_generate_writes_notebook_and_reports_from_existing_artifacts(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)

            hard_case_dir = (
                repo_root
                / "artifacts"
                / "chapter1"
                / "evaluation"
                / "asic"
                / "hard_cases"
                / "primary_medians"
                / "logistic_regression"
            )
            comparison_dir = hard_case_dir / "asic_hard_case_comparison"
            variable_audit_dir = hard_case_dir / "asic_hard_case_comparison_variable_audit"
            sofa_dir = hard_case_dir / "asic_sofa_feasibility_audit"
            foundation_dir = (
                repo_root
                / "artifacts"
                / "chapter1"
                / "evaluation"
                / "asic"
                / "horizon_dependence"
                / "foundation"
            )
            overlap_dir = foundation_dir.parent / "overlap"
            final_dir = foundation_dir.parent / "final"

            for directory in (
                hard_case_dir,
                comparison_dir,
                variable_audit_dir,
                sofa_dir,
                foundation_dir,
                overlap_dir,
                final_dir,
            ):
                directory.mkdir(parents=True, exist_ok=True)

            (hard_case_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                    },
                    indent=2,
                )
            )
            pd.DataFrame(
                [
                    {
                        "horizon_h": 24,
                        "n_nonfatal_last_points": 4696,
                        "n_fatal_last_points": 1682,
                        "nonfatal_q75_threshold": 0.014598,
                        "n_hard_cases": 346,
                        "pct_fatal_hard_cases": 0.206,
                    }
                ]
            ).to_csv(hard_case_dir / "horizon_hard_case_summary.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "stay_id_global": "s1",
                        "hospital_id": "H1",
                        "horizon_h": 24,
                        "label_value": 1,
                        "hard_case_flag": True,
                    }
                ]
            ).to_csv(hard_case_dir / "stay_level_hard_case_flags.csv", index=False)

            (comparison_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                        "target_horizon_h": 24,
                        "group_counts": {
                            "low-predicted fatal stays": 346,
                            "other fatal stays": 1336,
                            "total_fatal_stays": 1682,
                        },
                    },
                    indent=2,
                )
            )
            pd.DataFrame(
                [
                    {
                        "variable_label": "Hospital",
                        "absolute_standardized_difference": 2.1,
                    },
                    {
                        "variable_label": "Disease group",
                        "absolute_standardized_difference": 1.4,
                    },
                ]
            ).to_csv(comparison_dir / "comparison_table.csv", index=False)
            (comparison_dir / "summary.md").write_text(
                "# Summary\n\n"
                "- Low-predicted fatal stays were more common at one site.\n"
                "- Low-predicted fatal stays were enriched in respiratory / pulmonary disease-group assignments.\n"
                "- Local values here are synthetic implementation-test outputs only.\n"
            )

            (variable_audit_dir / "asic_hard_case_comparison_variable_audit_memo.md").write_text(
                "# Variable audit\n\n"
                "- Local outputs here are synthetic stand-in artifacts.\n\n"
                "- Overall judgement: `ISSUE 3.2 VARIABLE PACKAGE NOT YET READY`.\n"
                "- Blocking variable family: `age`.\n"
            )
            (sofa_dir / "sofa_feasibility_memo.md").write_text(
                "# SOFA feasibility\n\n"
                "- Local outputs are synthetic implementation-test outputs only.\n"
                "- Final classification: `NOT FEASIBLE`.\n"
            )

            pd.DataFrame(
                [
                    {
                        "horizon": "24h",
                        "nonfatal_last_n": 4696,
                        "fatal_last_n": 1682,
                        "nonfatal_q75_threshold": 0.014598,
                        "hard_case_n": 346,
                        "hard_case_share_among_fatal": 0.206,
                    },
                    {
                        "horizon": "48h",
                        "nonfatal_last_n": 4542,
                        "fatal_last_n": 1697,
                        "nonfatal_q75_threshold": 0.032415,
                        "hard_case_n": 352,
                        "hard_case_share_among_fatal": 0.207,
                    },
                ]
            ).to_csv(foundation_dir / "horizon_summary.csv", index=False)
            (foundation_dir / "artifact_foundation_note.md").write_text(
                "Synthetic stand-in data only.\n"
            )
            pd.DataFrame(
                [
                    {
                        "horizon_a": "24h",
                        "horizon_b": "48h",
                        "matched_fatal_n": 1600,
                        "hard_n_horizon_a": 346,
                        "hard_n_horizon_b": 352,
                        "intersection_n": 200,
                        "union_n": 498,
                        "jaccard_index": 0.402,
                    }
                ]
            ).to_csv(overlap_dir / "pairwise_overlap.csv", index=False)
            (overlap_dir / "overlap_note.md").write_text("Synthetic overlap note.\n")
            (final_dir / "run_manifest.json").write_text(
                json.dumps({"interpretation_label": "change form"}, indent=2)
            )
            (final_dir / "horizon_interpretation_memo.md").write_text(
                "# Horizon memo\n\n"
                "- Hard-case share is present across horizons.\n"
                "- Mortality-vs-risk shape shifts enough to count as change form in this synthetic run.\n"
            )
            (final_dir / "final_horizon_summary.md").write_text("Synthetic final summary.\n")
            (final_dir / "mortality_risk_horizon_comparison.png").write_bytes(b"png")

            notebook_path = repo_root / "notebooks" / "ch1_asic_descriptive_viability_review.ipynb"
            evidence_pack_path = repo_root / "reports" / "ch1_asic_descriptive_viability_evidence_pack.md"
            memo_path = repo_root / "reports" / "ch1_asic_descriptive_viability_memo_draft.md"

            context = generate_ch1_asic_descriptive_viability(
                repo_root=repo_root,
                notebook_path=notebook_path,
                evidence_pack_path=evidence_pack_path,
                memo_path=memo_path,
            )

            self.assertTrue(notebook_path.exists())
            self.assertTrue(evidence_pack_path.exists())
            self.assertTrue(memo_path.exists())

            evidence_text = evidence_pack_path.read_text()
            self.assertIn("## Hard-case definition summary", evidence_text)
            self.assertIn("asic_logistic_last_eligible_nonfatal_q75_v1", evidence_text)
            self.assertIn("346", evidence_text)
            self.assertIn("synthetic", evidence_text.lower())

            memo_text = memo_path.read_text()
            self.assertIn("## Decomposition decision", memo_text)
            self.assertIn("GO, but secondary only", memo_text)
            self.assertIn("MIMIC", memo_text)

            self.assertTrue(context.synthetic_local_outputs)
            self.assertIn("treatment-limitation", "\n".join(context.open_evidence_gaps).lower())
