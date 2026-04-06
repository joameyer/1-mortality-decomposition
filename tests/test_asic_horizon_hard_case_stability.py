from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from chapter1_mortality_decomposition.asic_horizon_hard_case_stability import (
    run_asic_horizon_hard_case_stability,
)


class AsicHorizonHardCaseStabilityTests(TestCase):
    def test_run_writes_overlap_and_persistence_outputs(self) -> None:
        horizons = [8, 16, 24, 48, 72]

        stay_specs = {
            "A": {"hospital_id": "H1", "fatal_horizons": {8, 16, 24, 48, 72}, "hard_horizons": {8, 16}},
            "B": {"hospital_id": "H2", "fatal_horizons": {8, 16, 24}, "hard_horizons": {8}},
            "C": {"hospital_id": "H3", "fatal_horizons": {24, 48, 72}, "hard_horizons": {48, 72}},
            "D": {"hospital_id": "H4", "fatal_horizons": {48, 72}, "hard_horizons": {72}},
            "E": {"hospital_id": "H5", "fatal_horizons": {8, 16, 24, 48, 72}, "hard_horizons": set()},
            "N": {"hospital_id": "H6", "fatal_horizons": set(), "hard_horizons": set()},
        }

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            hard_case_dir = tmp_path / "hard_cases"
            hard_case_dir.mkdir(parents=True)

            rows: list[dict[str, object]] = []
            for horizon in horizons:
                for stay_id, spec in stay_specs.items():
                    fatal = horizon in spec["fatal_horizons"]
                    hard = horizon in spec["hard_horizons"]
                    rows.append(
                        {
                            "stay_id_global": stay_id,
                            "hospital_id": spec["hospital_id"],
                            "horizon_h": horizon,
                            "label_value": 1 if fatal else 0,
                            "hard_case_flag": hard,
                        }
                    )

            stay_level_path = hard_case_dir / "stay_level_hard_case_flags.csv"
            pd.DataFrame(rows).to_csv(stay_level_path, index=False)
            manifest = {
                "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                "stay_level_artifact": str(stay_level_path.resolve()),
            }
            (hard_case_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

            result = run_asic_horizon_hard_case_stability(
                hard_case_dir=hard_case_dir,
                output_dir=tmp_path / "overlap",
            )

            self.assertTrue(result.package3_ready)
            self.assertTrue(result.artifacts.pairwise_denominators_path.exists())
            self.assertTrue(result.artifacts.pairwise_overlap_path.exists())
            self.assertTrue(result.artifacts.directional_overlap_path.exists())
            self.assertTrue(result.artifacts.jaccard_heatmap_path.exists())
            self.assertTrue(result.artifacts.directional_overlap_heatmap_path.exists())
            self.assertTrue(result.artifacts.hard_case_persistence_path.exists())
            self.assertTrue(result.artifacts.persistence_distribution_path.exists())
            self.assertTrue(result.artifacts.persistence_barplot_path.exists())
            self.assertTrue(result.artifacts.overlap_note_path.exists())

            pairwise_denominators = pd.read_csv(result.artifacts.pairwise_denominators_path)
            row_8_24 = pairwise_denominators[
                pairwise_denominators["horizon_a"].eq("8h") & pairwise_denominators["horizon_b"].eq("24h")
            ].iloc[0]
            self.assertEqual(int(row_8_24["fatal_n_horizon_a"]), 3)
            self.assertEqual(int(row_8_24["fatal_n_horizon_b"]), 4)
            self.assertEqual(int(row_8_24["matched_fatal_n"]), 3)

            pairwise_overlap = pd.read_csv(result.artifacts.pairwise_overlap_path)
            row_8_16 = pairwise_overlap[
                pairwise_overlap["horizon_a"].eq("8h") & pairwise_overlap["horizon_b"].eq("16h")
            ].iloc[0]
            self.assertEqual(int(row_8_16["matched_fatal_n"]), 3)
            self.assertEqual(int(row_8_16["hard_n_horizon_a"]), 2)
            self.assertEqual(int(row_8_16["hard_n_horizon_b"]), 1)
            self.assertEqual(int(row_8_16["intersection_n"]), 1)
            self.assertEqual(int(row_8_16["union_n"]), 2)
            self.assertAlmostEqual(float(row_8_16["jaccard_index"]), 0.5)

            directional_overlap = pd.read_csv(result.artifacts.directional_overlap_path)
            row_16_to_8 = directional_overlap[
                directional_overlap["horizon_from"].eq("16h")
                & directional_overlap["horizon_to"].eq("8h")
            ].iloc[0]
            self.assertEqual(int(row_16_to_8["hard_n_from"]), 1)
            self.assertEqual(int(row_16_to_8["intersection_n"]), 1)
            self.assertAlmostEqual(float(row_16_to_8["overlap_from_A_to_B"]), 1.0)

            persistence = pd.read_csv(result.artifacts.hard_case_persistence_path)
            persistence = persistence.set_index("stay_id")
            self.assertEqual(int(persistence.at["A", "hard_case_horizon_n"]), 2)
            self.assertEqual(int(persistence.at["B", "hard_case_horizon_n"]), 1)
            self.assertEqual(int(persistence.at["C", "hard_case_horizon_n"]), 2)
            self.assertEqual(int(persistence.at["D", "hard_case_horizon_n"]), 1)
            self.assertEqual(int(persistence.at["E", "hard_case_horizon_n"]), 0)

            distribution = pd.read_csv(result.artifacts.persistence_distribution_path)
            distribution = distribution.set_index("hard_case_horizon_n")
            self.assertEqual(int(distribution.at[0, "fatal_stay_count"]), 1)
            self.assertEqual(int(distribution.at[1, "fatal_stay_count"]), 2)
            self.assertEqual(int(distribution.at[2, "fatal_stay_count"]), 2)
            self.assertEqual(int(distribution["fatal_stay_count"].sum()), 5)

            note_text = result.artifacts.overlap_note_path.read_text()
            self.assertIn("intersection of fatal stay IDs present in both horizons", note_text)
            self.assertIn("synthetic", note_text.lower())

    def test_run_fails_on_hospital_id_mismatch_for_same_stay(self) -> None:
        horizons = [8, 16, 24, 48, 72]

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            hard_case_dir = tmp_path / "hard_cases"
            hard_case_dir.mkdir(parents=True)

            rows: list[dict[str, object]] = []
            for horizon in horizons:
                rows.append(
                    {
                        "stay_id_global": "stay_1",
                        "hospital_id": "H1" if horizon != 48 else "H9",
                        "horizon_h": horizon,
                        "label_value": 1,
                        "hard_case_flag": False,
                    }
                )

            stay_level_path = hard_case_dir / "stay_level_hard_case_flags.csv"
            pd.DataFrame(rows).to_csv(stay_level_path, index=False)
            manifest = {
                "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                "stay_level_artifact": str(stay_level_path.resolve()),
            }
            (hard_case_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

            with self.assertRaisesRegex(ValueError, "hospital_id changed across horizons"):
                run_asic_horizon_hard_case_stability(
                    hard_case_dir=hard_case_dir,
                    output_dir=tmp_path / "overlap",
                )
