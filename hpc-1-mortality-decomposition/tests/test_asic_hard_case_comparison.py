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
                        "icu_end_time_proxy_hours": 24.0,
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
                        "icu_end_time_proxy_hours": 36.0,
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
                        "icu_end_time_proxy_hours": 60.0,
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
                        "icu_end_time_proxy_hours": 72.0,
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
            self.assertTrue(result.artifacts.early_vs_late_fatal_timing.summary_path.exists())
            self.assertTrue(result.artifacts.early_vs_late_fatal_timing.figure_path.exists())
            self.assertTrue(result.artifacts.early_vs_late_fatal_timing.note_path.exists())

            manifest = json.loads(result.artifacts.manifest_path.read_text())
            self.assertEqual(manifest["group_counts"]["total_fatal_stays"], 4)
            self.assertEqual(manifest["group_counts"][LOW_PREDICTED_FATAL_GROUP], 2)
            self.assertEqual(manifest["group_counts"][OTHER_FATAL_GROUP], 2)
            self.assertIn("join_logic", manifest)
            self.assertIn("secondary_outputs", manifest)
            self.assertFalse(
                manifest["secondary_outputs"]["early_vs_late_fatal_timing_split"][
                    "comparison_performed"
                ]
            )

            timing_summary = pd.read_csv(result.artifacts.early_vs_late_fatal_timing.summary_path)
            low_predicted_row = timing_summary[
                timing_summary["variable"].eq("low_predicted_fatal")
            ].iloc[0]
            self.assertEqual(low_predicted_row["early_icu_death"], "2 (100%)")
            self.assertEqual(low_predicted_row["late_icu_death"], "0 (0%)")

            timing_note = result.artifacts.early_vs_late_fatal_timing.note_path.read_text()
            self.assertIn("too sparse", timing_note)
            self.assertIn("decorative", timing_note)

    def test_run_asic_hard_case_comparison_writes_non_sparse_early_vs_late_comparison(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            hard_case_path = tmp_path / "stay_level_hard_case_flags.csv"
            model_ready_path = tmp_path / "chapter1_primary_model_ready_dataset.csv"
            asic_input_root = tmp_path / "asic_harmonized"
            static_dir = asic_input_root / "static"
            static_dir.mkdir(parents=True)

            hard_case_rows: list[dict[str, object]] = []
            model_ready_rows: list[dict[str, object]] = []
            static_rows: list[dict[str, object]] = []
            disease_cycle = ["J18", "T81", "I50", "A41"]

            for stay_index in range(20):
                stay_id = f"s{stay_index + 1}"
                hospital_id = "H1" if stay_index < 10 else "H2"
                block_index = stay_index % 3
                early_group = stay_index < 10
                hard_case_flag = stay_index in {0, 1, 2, 3, 4, 10, 11, 12, 13, 14}
                prediction_time_h = 8 + (stay_index % 4) * 8 if early_group else 56 + (stay_index % 4) * 8
                icu_end_time_proxy_hours = 24.0 + stay_index if early_group else 96.0 + stay_index
                instance_id = f"{stay_id}__b{block_index}__h24"

                hard_case_rows.append(
                    {
                        "instance_id": instance_id,
                        "stay_id_global": stay_id,
                        "hospital_id": hospital_id,
                        "block_index": block_index,
                        "prediction_time_h": prediction_time_h,
                        "horizon_h": 24,
                        "label_value": 1,
                        "hard_case_flag": hard_case_flag,
                        "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                    }
                )
                model_ready_rows.append(
                    {
                        "instance_id": instance_id,
                        "stay_id_global": stay_id,
                        "hospital_id": hospital_id,
                        "block_index": block_index,
                        "prediction_time_h": prediction_time_h,
                        "horizon_h": 24,
                        "label_value": 1,
                        "icu_end_time_proxy_hours": icu_end_time_proxy_hours,
                        "pf_ratio_last": 180.0 - stay_index,
                        "map_last": 65.0 + stay_index,
                        "creatinine_last": 1.2 + 0.2 * stay_index,
                        "peep_last": 8.0 + (stay_index % 5),
                    }
                )
                static_rows.append(
                    {
                        "stay_id_global": stay_id,
                        "hospital_id": hospital_id,
                        "age_group": ["<70", "70-79", "80-130"][stay_index % 3],
                        "sex": "F" if stay_index % 2 == 0 else "M",
                        "icd10_codes": disease_cycle[stay_index % len(disease_cycle)],
                    }
                )

            pd.DataFrame(hard_case_rows).to_csv(hard_case_path, index=False)
            pd.DataFrame(model_ready_rows).to_csv(model_ready_path, index=False)
            pd.DataFrame(static_rows).to_csv(static_dir / "harmonized.csv", index=False)

            result = run_asic_hard_case_comparison(
                hard_case_path=hard_case_path,
                model_ready_path=model_ready_path,
                asic_input_root=asic_input_root,
                output_dir=tmp_path / "asic_hard_case_comparison_output",
            )

            timing_summary = pd.read_csv(result.artifacts.early_vs_late_fatal_timing.summary_path)
            self.assertIn("compact_descriptive_comparison", timing_summary["section"].tolist())
            low_predicted_row = timing_summary[
                timing_summary["variable"].eq("low_predicted_fatal")
            ].iloc[0]
            self.assertEqual(low_predicted_row["early_icu_death"], "5 (50%)")
            self.assertEqual(low_predicted_row["late_icu_death"], "5 (50%)")

            timing_note = result.artifacts.early_vs_late_fatal_timing.note_path.read_text()
            self.assertNotIn("too sparse", timing_note)
            self.assertIn("does not materially alter", timing_note)

            manifest = json.loads(result.artifacts.manifest_path.read_text())
            secondary_output = manifest["secondary_outputs"]["early_vs_late_fatal_timing_split"]
            self.assertTrue(secondary_output["comparison_performed"])
            self.assertEqual(secondary_output["group_counts"]["early ICU death (<=48h)"], 10)
            self.assertEqual(secondary_output["group_counts"]["late ICU death (>48h)"], 10)

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
                    "icu_end_time_proxy_hours",
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
