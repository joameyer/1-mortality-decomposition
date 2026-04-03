from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from chapter1_mortality_decomposition.hard_case_agreement import (
    run_asic_hard_case_agreement_sensitivity,
)


class HardCaseAgreementTests(TestCase):
    def test_run_asic_hard_case_agreement_writes_outputs_and_reports_unmatched_fatal_stays(self) -> None:
        logistic_prediction_rows = [
            {
                "instance_id": "n1__b0__h24",
                "stay_id_global": "n1",
                "hospital_id": "H1",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "train",
                "label_value": 0,
                "predicted_probability": 0.10,
                "model_name": "logistic_regression",
            },
            {
                "instance_id": "n2__b0__h24",
                "stay_id_global": "n2",
                "hospital_id": "H1",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "validation",
                "label_value": 0,
                "predicted_probability": 0.20,
                "model_name": "logistic_regression",
            },
            {
                "instance_id": "n3__b0__h24",
                "stay_id_global": "n3",
                "hospital_id": "H2",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "test",
                "label_value": 0,
                "predicted_probability": 0.30,
                "model_name": "logistic_regression",
            },
            {
                "instance_id": "n4__b0__h24",
                "stay_id_global": "n4",
                "hospital_id": "H2",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "test",
                "label_value": 0,
                "predicted_probability": 0.40,
                "model_name": "logistic_regression",
            },
            {
                "instance_id": "f1__b0__h24",
                "stay_id_global": "f1",
                "hospital_id": "H1",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "train",
                "label_value": 1,
                "predicted_probability": 0.20,
                "model_name": "logistic_regression",
            },
            {
                "instance_id": "f2__b0__h24",
                "stay_id_global": "f2",
                "hospital_id": "H1",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "validation",
                "label_value": 1,
                "predicted_probability": 0.50,
                "model_name": "logistic_regression",
            },
            {
                "instance_id": "f3__b0__h24",
                "stay_id_global": "f3",
                "hospital_id": "H2",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "test",
                "label_value": 1,
                "predicted_probability": 0.10,
                "model_name": "logistic_regression",
            },
            {
                "instance_id": "f5__b0__h24",
                "stay_id_global": "f5",
                "hospital_id": "H2",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "test",
                "label_value": 1,
                "predicted_probability": 0.30,
                "model_name": "logistic_regression",
            },
        ]
        xgb_platt_prediction_rows = [
            {
                "instance_id": "n1__b0__h24",
                "stay_id_global": "n1",
                "hospital_id": "H1",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "train",
                "label_value": 0,
                "model_name": "xgboost",
                "raw_predicted_probability": 0.01,
                "recalibrated_probability": 0.05,
                "recalibration_method": "platt",
                "recalibration_status": "fit",
                "recalibration_notes": "fitted_on_validation_split_only",
            },
            {
                "instance_id": "n2__b0__h24",
                "stay_id_global": "n2",
                "hospital_id": "H1",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "validation",
                "label_value": 0,
                "model_name": "xgboost",
                "raw_predicted_probability": 0.02,
                "recalibrated_probability": 0.10,
                "recalibration_method": "platt",
                "recalibration_status": "fit",
                "recalibration_notes": "fitted_on_validation_split_only",
            },
            {
                "instance_id": "n3__b0__h24",
                "stay_id_global": "n3",
                "hospital_id": "H2",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "test",
                "label_value": 0,
                "model_name": "xgboost",
                "raw_predicted_probability": 0.03,
                "recalibrated_probability": 0.20,
                "recalibration_method": "platt",
                "recalibration_status": "fit",
                "recalibration_notes": "fitted_on_validation_split_only",
            },
            {
                "instance_id": "n4__b0__h24",
                "stay_id_global": "n4",
                "hospital_id": "H2",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "test",
                "label_value": 0,
                "model_name": "xgboost",
                "raw_predicted_probability": 0.04,
                "recalibrated_probability": 0.30,
                "recalibration_method": "platt",
                "recalibration_status": "fit",
                "recalibration_notes": "fitted_on_validation_split_only",
            },
            {
                "instance_id": "f1__b0__h24",
                "stay_id_global": "f1",
                "hospital_id": "H1",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "train",
                "label_value": 1,
                "model_name": "xgboost",
                "raw_predicted_probability": 0.05,
                "recalibrated_probability": 0.20,
                "recalibration_method": "platt",
                "recalibration_status": "fit",
                "recalibration_notes": "fitted_on_validation_split_only",
            },
            {
                "instance_id": "f2__b0__h24",
                "stay_id_global": "f2",
                "hospital_id": "H1",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "validation",
                "label_value": 1,
                "model_name": "xgboost",
                "raw_predicted_probability": 0.06,
                "recalibrated_probability": 0.15,
                "recalibration_method": "platt",
                "recalibration_status": "fit",
                "recalibration_notes": "fitted_on_validation_split_only",
            },
            {
                "instance_id": "f4__b0__h24",
                "stay_id_global": "f4",
                "hospital_id": "H2",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "test",
                "label_value": 1,
                "model_name": "xgboost",
                "raw_predicted_probability": 0.07,
                "recalibrated_probability": 0.10,
                "recalibration_method": "platt",
                "recalibration_status": "fit",
                "recalibration_notes": "fitted_on_validation_split_only",
            },
            {
                "instance_id": "f5__b0__h24",
                "stay_id_global": "f5",
                "hospital_id": "H2",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "test",
                "label_value": 1,
                "model_name": "xgboost",
                "raw_predicted_probability": 0.08,
                "recalibrated_probability": 0.40,
                "recalibration_method": "platt",
                "recalibration_status": "fit",
                "recalibration_notes": "fitted_on_validation_split_only",
            },
        ]

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            logistic_root = tmp_path / "baselines"
            logistic_dir = logistic_root / "logistic_regression" / "horizon_24h"
            logistic_dir.mkdir(parents=True)
            pd.DataFrame(logistic_prediction_rows).to_csv(logistic_dir / "predictions.csv", index=False)

            recalibration_root = tmp_path / "recalibration" / "xgboost" / "horizon_24h"
            recalibration_root.mkdir(parents=True)
            pd.DataFrame(xgb_platt_prediction_rows).to_csv(
                recalibration_root / "platt_predictions.csv",
                index=False,
            )

            result = run_asic_hard_case_agreement_sensitivity(
                logistic_input_root=logistic_root,
                xgb_recalibration_root=tmp_path / "recalibration",
                xgb_recalibration_method="platt",
                output_dir=tmp_path / "agreement",
                horizons=[24],
            )

            self.assertTrue(result.artifacts.stay_level_path.exists())
            self.assertTrue(result.artifacts.horizon_summary_path.exists())
            self.assertTrue(result.artifacts.manifest_path.exists())

            stay_level = pd.read_csv(result.artifacts.stay_level_path)
            self.assertEqual(
                stay_level.columns.tolist(),
                [
                    "stay_id_global",
                    "hospital_id",
                    "horizon_h",
                    "logistic_model_name",
                    "logistic_predicted_probability",
                    "logistic_nonfatal_q75_threshold",
                    "logistic_hard_case_flag",
                    "logistic_instance_id",
                    "logistic_block_index",
                    "logistic_prediction_time_h",
                    "logistic_hard_case_rule",
                    "xgb_recal_model_name",
                    "xgb_recal_predicted_probability",
                    "xgb_recal_nonfatal_q75_threshold",
                    "xgb_recal_hard_case_flag",
                    "xgb_recal_instance_id",
                    "xgb_recal_block_index",
                    "xgb_recal_prediction_time_h",
                    "xgb_recal_hard_case_rule",
                    "hard_case_agreement_flag",
                    "hard_case_logistic_only_flag",
                    "hard_case_xgb_only_flag",
                    "agreement_rule",
                ],
            )
            self.assertEqual(stay_level["stay_id_global"].tolist(), ["f1", "f2", "f5"])

            by_stay = stay_level.set_index("stay_id_global")
            self.assertTrue(bool(by_stay.at["f1", "hard_case_agreement_flag"]))
            self.assertTrue(bool(by_stay.at["f2", "hard_case_xgb_only_flag"]))
            self.assertTrue(bool(by_stay.at["f5", "hard_case_logistic_only_flag"]))

            summary = pd.read_csv(result.artifacts.horizon_summary_path)
            self.assertEqual(int(summary.at[0, "n_fatal_with_both_models_available"]), 3)
            self.assertEqual(int(summary.at[0, "n_fatal_logistic_only_available"]), 1)
            self.assertEqual(int(summary.at[0, "n_fatal_xgb_recal_only_available"]), 1)
            self.assertEqual(int(summary.at[0, "n_fatal_dropped_unmatched"]), 2)
            self.assertEqual(int(summary.at[0, "n_logistic_hard"]), 2)
            self.assertEqual(int(summary.at[0, "n_xgb_recal_hard"]), 2)
            self.assertEqual(int(summary.at[0, "n_both_hard"]), 1)
            self.assertEqual(int(summary.at[0, "n_logistic_only"]), 1)
            self.assertEqual(int(summary.at[0, "n_xgb_recal_only"]), 1)
            self.assertAlmostEqual(float(summary.at[0, "jaccard_hard_case_overlap"]), 1 / 3)
            self.assertTrue(bool(summary.at[0, "agreement_subgroup_warning"]))
            self.assertIn("n_both_hard_lt_20", str(summary.at[0, "warning_reason"]))

    def test_run_asic_hard_case_agreement_fails_when_recalibrated_xgboost_predictions_missing(self) -> None:
        logistic_prediction_rows = [
            {
                "instance_id": "stay_a__b0__h24",
                "stay_id_global": "stay_a",
                "hospital_id": "H1",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "train",
                "label_value": 1,
                "predicted_probability": 0.20,
                "model_name": "logistic_regression",
            },
            {
                "instance_id": "stay_b__b0__h24",
                "stay_id_global": "stay_b",
                "hospital_id": "H1",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "train",
                "label_value": 0,
                "predicted_probability": 0.10,
                "model_name": "logistic_regression",
            },
        ]

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            logistic_dir = tmp_path / "baselines" / "logistic_regression" / "horizon_24h"
            logistic_dir.mkdir(parents=True)
            pd.DataFrame(logistic_prediction_rows).to_csv(logistic_dir / "predictions.csv", index=False)
            (tmp_path / "recalibration" / "xgboost" / "horizon_24h").mkdir(parents=True)

            with self.assertRaisesRegex(FileNotFoundError, "Expected saved recalibrated XGBoost predictions"):
                run_asic_hard_case_agreement_sensitivity(
                    logistic_input_root=tmp_path / "baselines",
                    xgb_recalibration_root=tmp_path / "recalibration",
                    xgb_recalibration_method="platt",
                    output_dir=tmp_path / "agreement",
                    horizons=[24],
                )
