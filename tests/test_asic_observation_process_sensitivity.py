from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chapter1_mortality_decomposition.asic_observation_process_sensitivity import (  # noqa: E402
    LOW_PREDICTED_FATAL_GROUP,
    OTHER_FATAL_GROUP,
    build_observation_process_comparison_outputs,
    derive_observation_process_sensitivity_dataset,
    load_authoritative_observation_process_anchor,
    run_asic_observation_process_sensitivity,
)


class ASICObservationProcessSensitivityTests(unittest.TestCase):
    def test_derive_observation_process_sensitivity_dataset_builds_expected_columns(self) -> None:
        anchor = pd.DataFrame(
            [
                {
                    "stay_id_global": "stay_hard",
                    "instance_id": "stay_hard__b1__h24",
                    "hospital_id": "asic_A",
                    "block_index": 1,
                    "block_start_h": 8,
                    "block_end_h": 16,
                    "prediction_time_h": 16,
                    "icu_end_time_proxy_hours": 20.0,
                    "hard_case_flag": True,
                    "hard_case_group": LOW_PREDICTED_FATAL_GROUP,
                    "predicted_probability": 0.11,
                    "nonfatal_q75_threshold": 0.20,
                    "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                    "obs_hr_grp_block": 1,
                    "obs_bp_grp_block": 0,
                    "obs_resp_grp_block": 1,
                    "obs_oxy_grp_block": 1,
                    "n_core_grps_obs_block": 3,
                    "tsl_hr_grp_h": 0.25,
                    "tsl_bp_grp_h": 10.0,
                    "tsl_resp_grp_h": 0.5,
                    "tsl_oxy_grp_h": None,
                    "pf_ratio_last": None,
                    "map_last": 70.0,
                    "creatinine_last": 120.0,
                    "peep_last": None,
                },
                {
                    "stay_id_global": "stay_other",
                    "instance_id": "stay_other__b2__h24",
                    "hospital_id": "asic_A",
                    "block_index": 2,
                    "block_start_h": 16,
                    "block_end_h": 24,
                    "prediction_time_h": 24,
                    "icu_end_time_proxy_hours": 30.0,
                    "hard_case_flag": False,
                    "hard_case_group": OTHER_FATAL_GROUP,
                    "predicted_probability": 0.35,
                    "nonfatal_q75_threshold": 0.20,
                    "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                    "obs_hr_grp_block": 1,
                    "obs_bp_grp_block": 1,
                    "obs_resp_grp_block": 1,
                    "obs_oxy_grp_block": 1,
                    "n_core_grps_obs_block": 4,
                    "tsl_hr_grp_h": 0.25,
                    "tsl_bp_grp_h": 0.25,
                    "tsl_resp_grp_h": 0.25,
                    "tsl_oxy_grp_h": 0.25,
                    "pf_ratio_last": 180.0,
                    "map_last": 72.0,
                    "creatinine_last": 110.0,
                    "peep_last": 8.0,
                },
            ]
        )

        derived = derive_observation_process_sensitivity_dataset(anchor)
        hard_row = derived.loc[derived["stay_id_global"].eq("stay_hard")].iloc[0]

        self.assertEqual(int(hard_row["n_core_groups_fresh_block"]), 3)
        self.assertEqual(int(hard_row["n_core_groups_historical_only"]), 1)
        self.assertEqual(int(hard_row["n_core_groups_never_observed"]), 1)
        self.assertEqual(int(hard_row["n_core_groups_stale_ge_8h"]), 1)
        self.assertAlmostEqual(float(hard_row["time_since_last_any_core_h"]), 0.25)
        self.assertAlmostEqual(float(hard_row["max_time_since_last_core_h"]), 10.0)
        self.assertTrue(bool(hard_row["core_block_incomplete_any"]))
        self.assertTrue(bool(hard_row["any_stale_core_ge_8h_flag"]))
        self.assertEqual(int(hard_row["n_frozen_proxy_missing"]), 2)

    def test_build_observation_process_comparison_outputs_reports_expected_variables(self) -> None:
        dataset = pd.DataFrame(
            [
                {
                    "hard_case_flag": True,
                    "n_core_groups_fresh_block": 3,
                    "core_block_complete_all4": False,
                    "n_core_groups_historical_only": 1,
                    "n_core_groups_never_observed": 1,
                    "n_frozen_proxy_missing": 2,
                    "time_since_last_any_core_h": 0.25,
                    "max_time_since_last_core_h": 10.0,
                    "any_stale_core_ge_8h_flag": True,
                },
                {
                    "hard_case_flag": True,
                    "n_core_groups_fresh_block": 4,
                    "core_block_complete_all4": True,
                    "n_core_groups_historical_only": 0,
                    "n_core_groups_never_observed": 0,
                    "n_frozen_proxy_missing": 1,
                    "time_since_last_any_core_h": 0.25,
                    "max_time_since_last_core_h": 8.0,
                    "any_stale_core_ge_8h_flag": True,
                },
                {
                    "hard_case_flag": False,
                    "n_core_groups_fresh_block": 4,
                    "core_block_complete_all4": True,
                    "n_core_groups_historical_only": 0,
                    "n_core_groups_never_observed": 0,
                    "n_frozen_proxy_missing": 0,
                    "time_since_last_any_core_h": 0.25,
                    "max_time_since_last_core_h": 0.5,
                    "any_stale_core_ge_8h_flag": False,
                },
                {
                    "hard_case_flag": False,
                    "n_core_groups_fresh_block": 4,
                    "core_block_complete_all4": True,
                    "n_core_groups_historical_only": 0,
                    "n_core_groups_never_observed": 0,
                    "n_frozen_proxy_missing": 0,
                    "time_since_last_any_core_h": 0.5,
                    "max_time_since_last_core_h": 9.0,
                    "any_stale_core_ge_8h_flag": True,
                },
            ]
        )

        comparison_table, effect_details = build_observation_process_comparison_outputs(dataset)

        self.assertEqual(int(comparison_table.shape[0]), 8)
        self.assertIn("All 4 core groups observed in anchor block", comparison_table["variable_label"].tolist())
        stale_row = effect_details.loc[effect_details["variable"].eq("any_stale_core_ge_8h_flag")].iloc[0]
        self.assertGreater(float(stale_row["absolute_standardized_difference"]), 0.5)

    def test_run_asic_observation_process_sensitivity_writes_requested_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            comparison_dataset_path = tmp_path / "stay_level_comparison_dataset.csv"
            hard_case_path = tmp_path / "stay_level_hard_case_flags.csv"
            observation_process_path = tmp_path / "chapter1_observation_process_block_features.csv"

            pd.DataFrame(
                [
                    {
                        "stay_id_global": "stay_hard",
                        "instance_id": "stay_hard__b1__h24",
                        "hard_case_flag": True,
                        "hard_case_group": LOW_PREDICTED_FATAL_GROUP,
                        "age_group": "<70",
                        "sex": "M",
                        "disease_group": "respiratory / pulmonary",
                        "prediction_time_h": 16,
                        "icu_end_time_proxy_hours": 20.0,
                        "hospital_id": "asic_A",
                        "pf_ratio_last": None,
                        "map_last": 70.0,
                        "creatinine_last": 120.0,
                        "peep_last": None,
                    },
                    {
                        "stay_id_global": "stay_other",
                        "instance_id": "stay_other__b2__h24",
                        "hard_case_flag": False,
                        "hard_case_group": OTHER_FATAL_GROUP,
                        "age_group": "70-79",
                        "sex": "F",
                        "disease_group": "cardiovascular",
                        "prediction_time_h": 24,
                        "icu_end_time_proxy_hours": 30.0,
                        "hospital_id": "asic_A",
                        "pf_ratio_last": 180.0,
                        "map_last": 72.0,
                        "creatinine_last": 110.0,
                        "peep_last": 8.0,
                    },
                ]
            ).to_csv(comparison_dataset_path, index=False)

            pd.DataFrame(
                [
                    {
                        "stay_id_global": "stay_hard",
                        "hospital_id": "asic_A",
                        "horizon_h": 24,
                        "label_value": 1,
                        "instance_id": "stay_hard__b1__h24",
                        "block_index": 1,
                        "prediction_time_h": 16,
                        "predicted_probability": 0.11,
                        "nonfatal_q75_threshold": 0.20,
                        "hard_case_flag": True,
                        "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                    },
                    {
                        "stay_id_global": "stay_other",
                        "hospital_id": "asic_A",
                        "horizon_h": 24,
                        "label_value": 1,
                        "instance_id": "stay_other__b2__h24",
                        "block_index": 2,
                        "prediction_time_h": 24,
                        "predicted_probability": 0.35,
                        "nonfatal_q75_threshold": 0.20,
                        "hard_case_flag": False,
                        "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                    },
                ]
            ).to_csv(hard_case_path, index=False)

            pd.DataFrame(
                [
                    {
                        "stay_id_global": "stay_hard",
                        "hospital_id": "asic_A",
                        "block_index": 1,
                        "block_start_h": 8,
                        "block_end_h": 16,
                        "prediction_time_h": 16,
                        "obs_hr_grp_block": 1,
                        "obs_bp_grp_block": 0,
                        "obs_resp_grp_block": 1,
                        "obs_oxy_grp_block": 1,
                        "n_core_grps_obs_block": 3,
                        "tsl_hr_grp_h": 0.25,
                        "tsl_bp_grp_h": 10.0,
                        "tsl_resp_grp_h": 0.5,
                        "tsl_oxy_grp_h": None,
                    },
                    {
                        "stay_id_global": "stay_other",
                        "hospital_id": "asic_A",
                        "block_index": 2,
                        "block_start_h": 16,
                        "block_end_h": 24,
                        "prediction_time_h": 24,
                        "obs_hr_grp_block": 1,
                        "obs_bp_grp_block": 1,
                        "obs_resp_grp_block": 1,
                        "obs_oxy_grp_block": 1,
                        "n_core_grps_obs_block": 4,
                        "tsl_hr_grp_h": 0.25,
                        "tsl_bp_grp_h": 0.25,
                        "tsl_resp_grp_h": 0.25,
                        "tsl_oxy_grp_h": 0.25,
                    },
                ]
            ).to_csv(observation_process_path, index=False)

            result = run_asic_observation_process_sensitivity(
                comparison_dataset_path=comparison_dataset_path,
                hard_case_path=hard_case_path,
                observation_process_path=observation_process_path,
                output_dataset_path=tmp_path
                / "artifacts"
                / "chapter1"
                / "evaluation"
                / "asic"
                / "hard_cases"
                / "primary_medians"
                / "logistic_regression"
                / "asic_observation_process_sensitivity"
                / "stay_level_observation_process_dataset.csv",
                output_comparison_table_path=tmp_path
                / "artifacts"
                / "chapter1"
                / "evaluation"
                / "asic"
                / "hard_cases"
                / "primary_medians"
                / "logistic_regression"
                / "asic_observation_process_sensitivity"
                / "comparison_table.csv",
                output_effect_details_path=tmp_path
                / "artifacts"
                / "chapter1"
                / "evaluation"
                / "asic"
                / "hard_cases"
                / "primary_medians"
                / "logistic_regression"
                / "asic_observation_process_sensitivity"
                / "effect_size_details.csv",
                output_figure_path=tmp_path
                / "artifacts"
                / "chapter1"
                / "evaluation"
                / "asic"
                / "hard_cases"
                / "primary_medians"
                / "logistic_regression"
                / "asic_observation_process_sensitivity"
                / "effect_size_figure.png",
                output_memo_path=tmp_path
                / "artifacts"
                / "chapter1"
                / "evaluation"
                / "asic"
                / "hard_cases"
                / "primary_medians"
                / "logistic_regression"
                / "asic_observation_process_sensitivity"
                / "memo.md",
                output_manifest_path=tmp_path
                / "artifacts"
                / "chapter1"
                / "evaluation"
                / "asic"
                / "hard_cases"
                / "primary_medians"
                / "logistic_regression"
                / "asic_observation_process_sensitivity"
                / "run_manifest.json",
                promoted_memo_path=tmp_path
                / "reports"
                / "chapter1"
                / "asic_observation_process_sensitivity_memo.md",
            )

            self.assertTrue(result.artifacts.dataset_path.exists())
            self.assertTrue(result.artifacts.comparison_table_path.exists())
            self.assertTrue(result.artifacts.effect_details_path.exists())
            self.assertTrue(result.artifacts.figure_path.exists())
            self.assertTrue(result.artifacts.memo_path.exists())
            self.assertTrue(result.artifacts.manifest_path.exists())
            self.assertIsNotNone(result.artifacts.promoted_memo_path)
            assert result.artifacts.promoted_memo_path is not None
            self.assertTrue(result.artifacts.promoted_memo_path.exists())
            self.assertEqual(
                result.artifacts.memo_path.read_text(),
                result.artifacts.promoted_memo_path.read_text(),
            )

            loaded_anchor, metadata = load_authoritative_observation_process_anchor(
                comparison_dataset_path=comparison_dataset_path,
                hard_case_path=hard_case_path,
                observation_process_path=observation_process_path,
            )
            self.assertEqual(int(loaded_anchor.shape[0]), 2)
            self.assertEqual(int(metadata["group_counts"][LOW_PREDICTED_FATAL_GROUP]), 1)


if __name__ == "__main__":
    unittest.main()
