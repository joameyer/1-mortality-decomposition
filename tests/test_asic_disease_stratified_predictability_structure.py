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

from chapter1_mortality_decomposition.asic_disease_stratified_predictability_structure import (  # noqa: E402
    run_asic_disease_stratified_predictability_structure,
)
from chapter1_mortality_decomposition.icd10_disease_groups import (  # noqa: E402
    FROZEN_DISEASE_GROUP_HIERARCHY,
)


class ASICDiseaseStratifiedPredictabilityStructureTests(TestCase):
    def test_run_asic_disease_stratified_predictability_structure_writes_artifacts(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            comparison_dataset_path = tmp_path / "stay_level_comparison_dataset.csv"
            hard_case_path = tmp_path / "stay_level_hard_case_flags.csv"
            group_counts_path = tmp_path / "final_group_counts.csv"
            missing_disease_group_path = tmp_path / "missing_assignments.csv"

            pd.DataFrame(
                [
                    {
                        "stay_id_global": "s1",
                        "instance_id": "s1__b1__h24",
                        "hard_case_flag": True,
                        "hard_case_group": "low-predicted fatal stays",
                        "disease_group": "surgical / postoperative / trauma-related",
                        "prediction_time_h": 16,
                        "icu_end_time_proxy_hours": 80.0,
                        "hospital_id": "asic_A",
                        "pf_ratio_last": 300.0,
                        "map_last": 80.0,
                        "peep_last": 8.0,
                    },
                    {
                        "stay_id_global": "s2",
                        "instance_id": "s2__b2__h24",
                        "hard_case_flag": False,
                        "hard_case_group": "other fatal stays",
                        "disease_group": "surgical / postoperative / trauma-related",
                        "prediction_time_h": 24,
                        "icu_end_time_proxy_hours": 88.0,
                        "hospital_id": "asic_A",
                        "pf_ratio_last": 180.0,
                        "map_last": 55.0,
                        "peep_last": 10.0,
                    },
                    {
                        "stay_id_global": "s3",
                        "instance_id": "s3__b1__h24",
                        "hard_case_flag": True,
                        "hard_case_group": "low-predicted fatal stays",
                        "disease_group": "respiratory / pulmonary",
                        "prediction_time_h": 32,
                        "icu_end_time_proxy_hours": 96.0,
                        "hospital_id": "asic_B",
                        "pf_ratio_last": 250.0,
                        "map_last": 78.0,
                        "peep_last": 9.0,
                    },
                    {
                        "stay_id_global": "s4",
                        "instance_id": "s4__b2__h24",
                        "hard_case_flag": False,
                        "hard_case_group": "other fatal stays",
                        "disease_group": "respiratory / pulmonary",
                        "prediction_time_h": 40,
                        "icu_end_time_proxy_hours": 104.0,
                        "hospital_id": "asic_B",
                        "pf_ratio_last": 160.0,
                        "map_last": 58.0,
                        "peep_last": 11.0,
                    },
                ]
            ).to_csv(comparison_dataset_path, index=False)

            pd.DataFrame(
                [
                    {
                        "stay_id_global": "s1",
                        "hospital_id": "asic_A",
                        "horizon_h": 24,
                        "label_value": 1,
                        "instance_id": "s1__b1__h24",
                        "block_index": 1,
                        "prediction_time_h": 16,
                        "hard_case_flag": True,
                        "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                    },
                    {
                        "stay_id_global": "s2",
                        "hospital_id": "asic_A",
                        "horizon_h": 24,
                        "label_value": 1,
                        "instance_id": "s2__b2__h24",
                        "block_index": 2,
                        "prediction_time_h": 24,
                        "hard_case_flag": False,
                        "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                    },
                    {
                        "stay_id_global": "s3",
                        "hospital_id": "asic_B",
                        "horizon_h": 24,
                        "label_value": 1,
                        "instance_id": "s3__b1__h24",
                        "block_index": 1,
                        "prediction_time_h": 32,
                        "hard_case_flag": True,
                        "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                    },
                    {
                        "stay_id_global": "s4",
                        "hospital_id": "asic_B",
                        "horizon_h": 24,
                        "label_value": 1,
                        "instance_id": "s4__b2__h24",
                        "block_index": 2,
                        "prediction_time_h": 40,
                        "hard_case_flag": False,
                        "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                    },
                ]
            ).to_csv(hard_case_path, index=False)

            pd.DataFrame(
                [
                    {"final_disease_group": group, "stay_count": count}
                    for group, count in zip(
                        FROZEN_DISEASE_GROUP_HIERARCHY,
                        [20, 18, 5, 4, 3, 2],
                        strict=True,
                    )
                ]
            ).to_csv(group_counts_path, index=False)

            result = run_asic_disease_stratified_predictability_structure(
                comparison_dataset_path=comparison_dataset_path,
                hard_case_path=hard_case_path,
                disease_group_path=missing_disease_group_path,
                disease_group_counts_path=group_counts_path,
                output_dir=tmp_path / "output",
            )

            self.assertEqual(result.final_judgment, "uninterpretable")
            self.assertTrue(result.artifacts.assignment_qc_path.exists())
            self.assertTrue(result.artifacts.disease_group_summary_path.exists())
            self.assertTrue(result.artifacts.hardcase_summary_path.exists())
            self.assertTrue(result.artifacts.contrast_panel_path.exists())
            self.assertTrue(result.artifacts.figure_path.exists())
            self.assertTrue(result.artifacts.memo_path.exists())

            hardcase_summary = pd.read_csv(result.artifacts.hardcase_summary_path)
            self.assertEqual(int(hardcase_summary.shape[0]), 6)
            surgical_row = hardcase_summary[
                hardcase_summary["disease_group"].eq("surgical / postoperative / trauma-related")
            ].iloc[0]
            self.assertEqual(int(surgical_row["fatal_stays"]), 2)
            self.assertEqual(int(surgical_row["low_predicted_fatal_stays"]), 1)
            self.assertAlmostEqual(float(surgical_row["hard_case_share_among_fatal"]), 0.5)

            manifest = json.loads(result.artifacts.manifest_path.read_text())
            self.assertEqual(
                manifest["disease_group_source_mode"],
                "comparison_dataset_embedded_assignments_with_validation_counts",
            )
            self.assertEqual(manifest["final_judgment"], "uninterpretable")
