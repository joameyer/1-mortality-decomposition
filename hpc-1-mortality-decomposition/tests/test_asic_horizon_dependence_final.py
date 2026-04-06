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

from chapter1_mortality_decomposition.asic_horizon_dependence_final import (
    BUBBLE_AREA_MAX,
    BUBBLE_AREA_MIN,
    _compute_bubble_areas,
    run_asic_horizon_dependence_final,
)


class AsicHorizonDependenceFinalTests(TestCase):
    def test_compute_bubble_areas_stays_bounded_and_monotone(self) -> None:
        sample_fractions = pd.Series([0.00, 0.01, 0.05, 0.20, 0.75], dtype=float)

        bubble_areas = _compute_bubble_areas(sample_fractions)

        self.assertEqual(len(bubble_areas), len(sample_fractions))
        self.assertGreaterEqual(float(bubble_areas.min()), BUBBLE_AREA_MIN)
        self.assertLessEqual(float(bubble_areas.max()), BUBBLE_AREA_MAX)
        self.assertTrue((pd.Series(bubble_areas).diff().fillna(0.0) >= 0.0).all())

    def test_run_writes_final_outputs_and_reports_consistency(self) -> None:
        horizons = [8, 16, 24, 48, 72]

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            hard_case_dir = tmp_path / "hard_cases"
            foundation_dir = tmp_path / "foundation"
            overlap_dir = tmp_path / "overlap"
            hard_case_dir.mkdir(parents=True)
            foundation_dir.mkdir(parents=True)
            overlap_dir.mkdir(parents=True)

            stay_rows = []
            for horizon in horizons:
                rows_for_horizon = [
                    ("n1", "H1", 0, 0.01 + horizon / 1000, False, 0.04 + horizon / 1000),
                    ("n2", "H1", 0, 0.02 + horizon / 1000, False, 0.04 + horizon / 1000),
                    ("f1", "H2", 1, 0.03 + horizon / 1000, True if horizon in {8, 16, 24, 48} else False, 0.04 + horizon / 1000),
                    ("f2", "H3", 1, 0.12 + horizon / 1000, False, 0.04 + horizon / 1000),
                ]
                if horizon >= 48:
                    rows_for_horizon.append(
                        ("f3", "H4", 1, 0.035 + horizon / 1000, True if horizon == 72 else False, 0.04 + horizon / 1000)
                    )
                for stay_id, hospital_id, label_value, probability, hard_flag, threshold in rows_for_horizon:
                    stay_rows.append(
                        {
                            "stay_id_global": stay_id,
                            "hospital_id": hospital_id,
                            "horizon_h": horizon,
                            "label_value": label_value,
                            "predicted_probability": probability,
                            "nonfatal_q75_threshold": threshold,
                            "hard_case_flag": hard_flag,
                        }
                    )

            stay_level_path = hard_case_dir / "stay_level_hard_case_flags.csv"
            pd.DataFrame(stay_rows).to_csv(stay_level_path, index=False)
            manifest = {
                "stay_level_artifact": str(stay_level_path.resolve()),
                "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
            }
            (hard_case_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

            foundation_rows = []
            for horizon in horizons:
                horizon_df = pd.DataFrame(stay_rows)
                horizon_df = horizon_df[horizon_df["horizon_h"] == horizon].copy()
                fatal_n = int((horizon_df["label_value"] == 1).sum())
                hard_n = int(pd.Series(horizon_df["hard_case_flag"]).astype(bool).sum())
                foundation_rows.append(
                    {
                        "horizon": f"{horizon}h",
                        "nonfatal_last_n": int((horizon_df["label_value"] == 0).sum()),
                        "fatal_last_n": fatal_n,
                        "nonfatal_q75_threshold": float(horizon_df["nonfatal_q75_threshold"].iloc[0]),
                        "hard_case_n": hard_n,
                        "hard_case_share_among_fatal": hard_n / fatal_n,
                    }
                )
            pd.DataFrame(foundation_rows).to_csv(foundation_dir / "horizon_summary.csv", index=False)
            (foundation_dir / "artifact_foundation_note.md").write_text("synthetic foundation note\n")

            pairwise_denominator_rows = []
            pairwise_overlap_rows = []
            directional_rows = []
            for horizon_a, horizon_b in [(8, 16), (8, 24), (8, 48), (8, 72), (16, 24), (16, 48), (16, 72), (24, 48), (24, 72), (48, 72)]:
                fatal_ids_a = {
                    row["stay_id_global"]
                    for row in stay_rows
                    if row["horizon_h"] == horizon_a and row["label_value"] == 1
                }
                fatal_ids_b = {
                    row["stay_id_global"]
                    for row in stay_rows
                    if row["horizon_h"] == horizon_b and row["label_value"] == 1
                }
                matched = sorted(fatal_ids_a & fatal_ids_b)
                hard_ids_a = {
                    row["stay_id_global"]
                    for row in stay_rows
                    if row["horizon_h"] == horizon_a and row["hard_case_flag"]
                } & set(matched)
                hard_ids_b = {
                    row["stay_id_global"]
                    for row in stay_rows
                    if row["horizon_h"] == horizon_b and row["hard_case_flag"]
                } & set(matched)
                intersection = hard_ids_a & hard_ids_b
                union = hard_ids_a | hard_ids_b

                pairwise_denominator_rows.append(
                    {
                        "horizon_a": f"{horizon_a}h",
                        "horizon_b": f"{horizon_b}h",
                        "fatal_n_horizon_a": len(fatal_ids_a),
                        "fatal_n_horizon_b": len(fatal_ids_b),
                        "matched_fatal_n": len(matched),
                    }
                )
                pairwise_overlap_rows.append(
                    {
                        "horizon_a": f"{horizon_a}h",
                        "horizon_b": f"{horizon_b}h",
                        "matched_fatal_n": len(matched),
                        "hard_n_horizon_a": len(hard_ids_a),
                        "hard_n_horizon_b": len(hard_ids_b),
                        "intersection_n": len(intersection),
                        "union_n": len(union),
                        "jaccard_index": len(intersection) / len(union) if union else 0.0,
                    }
                )
                directional_rows.extend(
                    [
                        {
                            "horizon_from": f"{horizon_a}h",
                            "horizon_to": f"{horizon_b}h",
                            "matched_fatal_n": len(matched),
                            "hard_n_from": len(hard_ids_a),
                            "hard_n_to": len(hard_ids_b),
                            "intersection_n": len(intersection),
                            "overlap_from_A_to_B": len(intersection) / len(hard_ids_a) if hard_ids_a else 0.0,
                        },
                        {
                            "horizon_from": f"{horizon_b}h",
                            "horizon_to": f"{horizon_a}h",
                            "matched_fatal_n": len(matched),
                            "hard_n_from": len(hard_ids_b),
                            "hard_n_to": len(hard_ids_a),
                            "intersection_n": len(intersection),
                            "overlap_from_A_to_B": len(intersection) / len(hard_ids_b) if hard_ids_b else 0.0,
                        },
                    ]
                )

            pd.DataFrame(pairwise_denominator_rows).to_csv(overlap_dir / "pairwise_denominators.csv", index=False)
            pd.DataFrame(pairwise_overlap_rows).to_csv(overlap_dir / "pairwise_overlap.csv", index=False)
            pd.DataFrame(directional_rows).to_csv(overlap_dir / "directional_overlap.csv", index=False)
            pd.DataFrame(
                [
                    {"hard_case_horizon_n": 0, "fatal_stay_count": 1, "fatal_stay_share": 0.33},
                    {"hard_case_horizon_n": 1, "fatal_stay_count": 0, "fatal_stay_share": 0.00},
                    {"hard_case_horizon_n": 2, "fatal_stay_count": 1, "fatal_stay_share": 0.33},
                    {"hard_case_horizon_n": 3, "fatal_stay_count": 0, "fatal_stay_share": 0.00},
                    {"hard_case_horizon_n": 4, "fatal_stay_count": 1, "fatal_stay_share": 0.33},
                    {"hard_case_horizon_n": 5, "fatal_stay_count": 0, "fatal_stay_share": 0.00},
                ]
            ).to_csv(overlap_dir / "persistence_distribution.csv", index=False)
            (overlap_dir / "overlap_note.md").write_text("synthetic overlap note\n")

            result = run_asic_horizon_dependence_final(
                hard_case_dir=hard_case_dir,
                foundation_dir=foundation_dir,
                overlap_dir=overlap_dir,
                output_dir=tmp_path / "final",
            )

            self.assertEqual(result.consistency_issues, ())
            self.assertTrue(result.artifacts.figure_path.exists())
            self.assertTrue(result.artifacts.interpretation_memo_path.exists())
            self.assertTrue(result.artifacts.final_summary_path.exists())
            self.assertTrue(result.artifacts.binned_summary_path.exists())
            self.assertIn(result.interpretation_label, {"persist", "change form", "shrink"})

            memo_text = result.artifacts.interpretation_memo_path.read_text()
            self.assertIn("24h", memo_text)
            self.assertIn("48h", memo_text)
            self.assertIn("synthetic", memo_text.lower())

            summary_text = result.artifacts.final_summary_path.read_text()
            self.assertIn("No consistency mismatches were detected", summary_text)
            self.assertIn("mortality_risk_horizon_comparison.png", summary_text)

    def test_run_records_consistency_issue_when_foundation_summary_disagrees(self) -> None:
        horizons = [8, 16, 24, 48, 72]

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            hard_case_dir = tmp_path / "hard_cases"
            foundation_dir = tmp_path / "foundation"
            overlap_dir = tmp_path / "overlap"
            hard_case_dir.mkdir(parents=True)
            foundation_dir.mkdir(parents=True)
            overlap_dir.mkdir(parents=True)

            stay_rows = []
            for horizon in horizons:
                stay_rows.extend(
                    [
                        {
                            "stay_id_global": "n1",
                            "hospital_id": "H1",
                            "horizon_h": horizon,
                            "label_value": 0,
                            "predicted_probability": 0.01,
                            "nonfatal_q75_threshold": 0.02,
                            "hard_case_flag": False,
                        },
                        {
                            "stay_id_global": "f1",
                            "hospital_id": "H2",
                            "horizon_h": horizon,
                            "label_value": 1,
                            "predicted_probability": 0.015,
                            "nonfatal_q75_threshold": 0.02,
                            "hard_case_flag": True,
                        },
                    ]
                )

            stay_level_path = hard_case_dir / "stay_level_hard_case_flags.csv"
            pd.DataFrame(stay_rows).to_csv(stay_level_path, index=False)
            (hard_case_dir / "run_manifest.json").write_text(
                json.dumps({"stay_level_artifact": str(stay_level_path.resolve())}, indent=2)
            )
            pd.DataFrame(
                [
                    {
                        "horizon": f"{horizon}h",
                        "nonfatal_last_n": 1,
                        "fatal_last_n": 2 if horizon == 24 else 1,
                        "nonfatal_q75_threshold": 0.02,
                        "hard_case_n": 1,
                        "hard_case_share_among_fatal": 1.0,
                    }
                    for horizon in horizons
                ]
            ).to_csv(foundation_dir / "horizon_summary.csv", index=False)
            (foundation_dir / "artifact_foundation_note.md").write_text("synthetic foundation note\n")

            pairwise_denominator_rows = []
            pairwise_overlap_rows = []
            directional_rows = []
            all_pairs = [
                (8, 16),
                (8, 24),
                (8, 48),
                (8, 72),
                (16, 24),
                (16, 48),
                (16, 72),
                (24, 48),
                (24, 72),
                (48, 72),
            ]
            for horizon_a, horizon_b in all_pairs:
                pairwise_denominator_rows.append(
                    {
                        "horizon_a": f"{horizon_a}h",
                        "horizon_b": f"{horizon_b}h",
                        "fatal_n_horizon_a": 1,
                        "fatal_n_horizon_b": 1,
                        "matched_fatal_n": 1,
                    }
                )
                pairwise_overlap_rows.append(
                    {
                        "horizon_a": f"{horizon_a}h",
                        "horizon_b": f"{horizon_b}h",
                        "matched_fatal_n": 1,
                        "hard_n_horizon_a": 1,
                        "hard_n_horizon_b": 1,
                        "intersection_n": 1,
                        "union_n": 1,
                        "jaccard_index": 1.0,
                    }
                )
                directional_rows.extend(
                    [
                        {
                            "horizon_from": f"{horizon_a}h",
                            "horizon_to": f"{horizon_b}h",
                            "matched_fatal_n": 1,
                            "hard_n_from": 1,
                            "hard_n_to": 1,
                            "intersection_n": 1,
                            "overlap_from_A_to_B": 1.0,
                        },
                        {
                            "horizon_from": f"{horizon_b}h",
                            "horizon_to": f"{horizon_a}h",
                            "matched_fatal_n": 1,
                            "hard_n_from": 1,
                            "hard_n_to": 1,
                            "intersection_n": 1,
                            "overlap_from_A_to_B": 1.0,
                        },
                    ]
                )
            pd.DataFrame(pairwise_denominator_rows).to_csv(overlap_dir / "pairwise_denominators.csv", index=False)
            pd.DataFrame(pairwise_overlap_rows).to_csv(overlap_dir / "pairwise_overlap.csv", index=False)
            pd.DataFrame(directional_rows).to_csv(overlap_dir / "directional_overlap.csv", index=False)
            pd.DataFrame(
                [
                    {"hard_case_horizon_n": i, "fatal_stay_count": 0 if i else 1, "fatal_stay_share": 0.0}
                    for i in range(6)
                ]
            ).to_csv(overlap_dir / "persistence_distribution.csv", index=False)
            (overlap_dir / "overlap_note.md").write_text("synthetic overlap note\n")

            result = run_asic_horizon_dependence_final(
                hard_case_dir=hard_case_dir,
                foundation_dir=foundation_dir,
                overlap_dir=overlap_dir,
                output_dir=tmp_path / "final",
            )

            self.assertTrue(result.consistency_issues)
            self.assertIn("24h", "\n".join(result.consistency_issues))
            final_summary_text = result.artifacts.final_summary_path.read_text()
            self.assertIn("Consistency mismatches were detected", final_summary_text)
