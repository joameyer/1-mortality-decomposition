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

from chapter1_mortality_decomposition.asic_hard_case_comparison import (  # noqa: E402
    LOW_PREDICTED_FATAL_GROUP,
    OTHER_FATAL_GROUP,
    build_stay_level_comparison_dataset,
    run_asic_hard_case_comparison,
)


class ASICHardCaseComparisonTests(TestCase):
    def test_run_asic_hard_case_comparison_writes_package(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            hard_case_path = tmp_path / "stay_level_hard_case_flags.csv"
            model_ready_path = tmp_path / "chapter1_primary_model_ready_dataset.csv"
            asic_input_root = tmp_path / "asic_harmonized"
            static_dir = asic_input_root / "static"
            static_dir.mkdir(parents=True)

            pd.DataFrame(
                [
                    {
                        "instance_id": "s1__b0__h24",
                        "stay_id_global": "s1",
                        "hospital_id": "H2",
                        "block_index": 0,
                        "prediction_time_h": 1,
                        "horizon_h": 24,
                        "label_value": 1,
                        "hard_case_flag": True,
                        "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                    },
                    {
                        "instance_id": "s2__b1__h24",
                        "stay_id_global": "s2",
                        "hospital_id": "H2",
                        "block_index": 1,
                        "prediction_time_h": 3,
                        "horizon_h": 24,
                        "label_value": 1,
                        "hard_case_flag": True,
                        "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                    },
                    {
                        "instance_id": "s3__b0__h24",
                        "stay_id_global": "s3",
                        "hospital_id": "H1",
                        "block_index": 0,
                        "prediction_time_h": 2,
                        "horizon_h": 24,
                        "label_value": 1,
                        "hard_case_flag": False,
                        "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                    },
                    {
                        "instance_id": "s4__b1__h24",
                        "stay_id_global": "s4",
                        "hospital_id": "H1",
                        "block_index": 1,
                        "prediction_time_h": 4,
                        "horizon_h": 24,
                        "label_value": 1,
                        "hard_case_flag": False,
                        "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                    },
                    {
                        "instance_id": "ignore__b0__h24",
                        "stay_id_global": "ignore",
                        "hospital_id": "HX",
                        "block_index": 0,
                        "prediction_time_h": 1,
                        "horizon_h": 24,
                        "label_value": 0,
                        "hard_case_flag": False,
                        "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                    },
                    {
                        "instance_id": "ignore__b0__h48",
                        "stay_id_global": "ignore48",
                        "hospital_id": "HX",
                        "block_index": 0,
                        "prediction_time_h": 1,
                        "horizon_h": 48,
                        "label_value": 1,
                        "hard_case_flag": True,
                        "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                    },
                ]
            ).to_csv(hard_case_path, index=False)

            pd.DataFrame(
                [
                    {
                        "instance_id": "s1__b0__h24",
                        "stay_id_global": "s1",
                        "hospital_id": "H2",
                        "block_index": 0,
                        "prediction_time_h": 1,
                        "horizon_h": 24,
                        "label_value": 1,
                        "pf_ratio_last": 10.0,
                        "map_last": 50.0,
                        "creatinine_last": 1.0,
                        "peep_last": 8.0,
                    },
                    {
                        "instance_id": "s2__b1__h24",
                        "stay_id_global": "s2",
                        "hospital_id": "H2",
                        "block_index": 1,
                        "prediction_time_h": 3,
                        "horizon_h": 24,
                        "label_value": 1,
                        "pf_ratio_last": 30.0,
                        "map_last": 70.0,
                        "creatinine_last": 3.0,
                        "peep_last": 10.0,
                    },
                    {
                        "instance_id": "s3__b0__h24",
                        "stay_id_global": "s3",
                        "hospital_id": "H1",
                        "block_index": 0,
                        "prediction_time_h": 2,
                        "horizon_h": 24,
                        "label_value": 1,
                        "pf_ratio_last": 20.0,
                        "map_last": 60.0,
                        "creatinine_last": 2.0,
                        "peep_last": 9.0,
                    },
                    {
                        "instance_id": "s4__b1__h24",
                        "stay_id_global": "s4",
                        "hospital_id": "H1",
                        "block_index": 1,
                        "prediction_time_h": 4,
                        "horizon_h": 24,
                        "label_value": 1,
                        "pf_ratio_last": 40.0,
                        "map_last": 80.0,
                        "creatinine_last": 4.0,
                        "peep_last": 11.0,
                    },
                ]
            ).to_csv(model_ready_path, index=False)

            pd.DataFrame(
                [
                    {
                        "stay_id_global": "s1",
                        "hospital_id": "H2",
                        "age_group": "<70",
                        "sex": "M",
                        "icd10_codes": "J18",
                    },
                    {
                        "stay_id_global": "s2",
                        "hospital_id": "H2",
                        "age_group": "70-79",
                        "sex": "M",
                        "icd10_codes": "T81, I50",
                    },
                    {
                        "stay_id_global": "s3",
                        "hospital_id": "H1",
                        "age_group": "70-79",
                        "sex": "F",
                        "icd10_codes": "I50",
                    },
                    {
                        "stay_id_global": "s4",
                        "hospital_id": "H1",
                        "age_group": "80-130",
                        "sex": "M",
                        "icd10_codes": "A41",
                    },
                ]
            ).to_csv(static_dir / "harmonized.csv", index=False)

            result = run_asic_hard_case_comparison(
                hard_case_path=hard_case_path,
                model_ready_path=model_ready_path,
                asic_input_root=asic_input_root,
                output_dir=tmp_path / "asic_hard_case_comparison_output",
            )

            self.assertEqual(result.comparison_dataset.shape[0], 4)
            self.assertEqual(
                result.comparison_dataset["hard_case_group"].value_counts().to_dict(),
                {
                    LOW_PREDICTED_FATAL_GROUP: 2,
                    OTHER_FATAL_GROUP: 2,
                },
            )
            self.assertEqual(
                result.comparison_dataset["disease_group"].tolist(),
                [
                    "respiratory / pulmonary",
                    "surgical / postoperative / trauma-related",
                    "cardiovascular",
                    "infection / sepsis non-pulmonary",
                ],
            )
            self.assertEqual(
                result.comparison_table["variable"].tolist(),
                [
                    "age_group",
                    "sex",
                    "disease_group",
                    "hospital_id",
                    "prediction_time_h",
                    "pf_ratio_last",
                    "map_last",
                    "creatinine_last",
                    "peep_last",
                ],
            )

            prediction_row = result.comparison_table[
                result.comparison_table["variable"].eq("prediction_time_h")
            ].iloc[0]
            self.assertAlmostEqual(float(prediction_row["standardized_difference"]), -1.0, places=3)

            sex_row = result.comparison_table[result.comparison_table["variable"].eq("sex")].iloc[0]
            self.assertAlmostEqual(
                float(sex_row["absolute_standardized_difference"]),
                1.414,
                places=3,
            )

            self.assertTrue(result.artifacts.comparison_dataset_path.exists())
            self.assertTrue(result.artifacts.comparison_table_path.exists())
            self.assertTrue(result.artifacts.effect_size_plot_data_path.exists())
            self.assertTrue(result.artifacts.standardized_difference_details_path.exists())
            self.assertTrue(result.artifacts.figure_path.exists())
            self.assertTrue(result.artifacts.summary_path.exists())
            self.assertTrue(result.artifacts.manifest_path.exists())

            manifest = json.loads(result.artifacts.manifest_path.read_text())
            self.assertEqual(manifest["group_counts"]["total_fatal_stays"], 4)
            self.assertEqual(manifest["group_counts"][LOW_PREDICTED_FATAL_GROUP], 2)
            self.assertEqual(manifest["group_counts"][OTHER_FATAL_GROUP], 2)
            self.assertIn("join_logic", manifest)

    def test_build_stay_level_comparison_dataset_fails_when_model_ready_row_is_missing(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            hard_case_path = tmp_path / "stay_level_hard_case_flags.csv"
            model_ready_path = tmp_path / "chapter1_primary_model_ready_dataset.csv"
            asic_input_root = tmp_path / "asic_harmonized"
            static_dir = asic_input_root / "static"
            static_dir.mkdir(parents=True)

            pd.DataFrame(
                [
                    {
                        "instance_id": "s1__b0__h24",
                        "stay_id_global": "s1",
                        "hospital_id": "H1",
                        "block_index": 0,
                        "prediction_time_h": 1,
                        "horizon_h": 24,
                        "label_value": 1,
                        "hard_case_flag": True,
                        "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                    }
                ]
            ).to_csv(hard_case_path, index=False)

            pd.DataFrame(
                columns=[
                    "instance_id",
                    "stay_id_global",
                    "hospital_id",
                    "block_index",
                    "prediction_time_h",
                    "horizon_h",
                    "label_value",
                    "pf_ratio_last",
                    "map_last",
                    "creatinine_last",
                    "peep_last",
                ]
            ).to_csv(model_ready_path, index=False)

            pd.DataFrame(
                [
                    {
                        "stay_id_global": "s1",
                        "hospital_id": "H1",
                        "age_group": "<70",
                        "sex": "M",
                        "icd10_codes": "J18",
                    }
                ]
            ).to_csv(static_dir / "harmonized.csv", index=False)

            with self.assertRaisesRegex(ValueError, "could not be linked to the Chapter 1 model-ready dataset"):
                build_stay_level_comparison_dataset(
                    hard_case_path=hard_case_path,
                    model_ready_path=model_ready_path,
                    asic_input_root=asic_input_root,
                )
