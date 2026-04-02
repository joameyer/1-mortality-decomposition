from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from chapter1_mortality_decomposition.xgboost_recalibration import (
    run_asic_xgboost_recalibration,
)


class XGBoostRecalibrationTests(TestCase):
    def test_run_asic_xgboost_recalibration_writes_outputs(self) -> None:
        rows = [
            {
                "instance_id": "stay_a__b0__h24",
                "stay_id_global": "stay_a",
                "hospital_id": "H1",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "train",
                "label_value": 1,
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
            },
            {
                "instance_id": "stay_c__b0__h24",
                "stay_id_global": "stay_c",
                "hospital_id": "H2",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "validation",
                "label_value": 1,
            },
            {
                "instance_id": "stay_d__b0__h24",
                "stay_id_global": "stay_d",
                "hospital_id": "H2",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "validation",
                "label_value": 0,
            },
            {
                "instance_id": "stay_e__b0__h24",
                "stay_id_global": "stay_e",
                "hospital_id": "H3",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "validation",
                "label_value": 1,
            },
            {
                "instance_id": "stay_f__b0__h24",
                "stay_id_global": "stay_f",
                "hospital_id": "H3",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "validation",
                "label_value": 0,
            },
            {
                "instance_id": "stay_g__b0__h24",
                "stay_id_global": "stay_g",
                "hospital_id": "H4",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "test",
                "label_value": 1,
            },
            {
                "instance_id": "stay_h__b0__h24",
                "stay_id_global": "stay_h",
                "hospital_id": "H4",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "test",
                "label_value": 0,
            },
            {
                "instance_id": "stay_i__b0__h24",
                "stay_id_global": "stay_i",
                "hospital_id": "H5",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "test",
                "label_value": 1,
            },
            {
                "instance_id": "stay_j__b0__h24",
                "stay_id_global": "stay_j",
                "hospital_id": "H5",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "test",
                "label_value": 0,
            },
        ]
        logistic_predictions = pd.DataFrame(rows).assign(
            predicted_probability=[0.85, 0.15, 0.75, 0.20, 0.65, 0.25, 0.70, 0.18, 0.80, 0.22],
            model_name="logistic_regression",
        )
        xgboost_predictions = pd.DataFrame(rows).assign(
            predicted_probability=[0.98, 0.05, 0.92, 0.08, 0.88, 0.12, 0.90, 0.10, 0.95, 0.15],
            model_name="xgboost",
        )

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_root = tmp_path / "baselines"
            output_dir = tmp_path / "recalibration"
            logistic_dir = input_root / "logistic_regression" / "horizon_24h"
            xgboost_dir = input_root / "xgboost" / "horizon_24h"
            logistic_dir.mkdir(parents=True)
            xgboost_dir.mkdir(parents=True)
            logistic_predictions.to_csv(logistic_dir / "predictions.csv", index=False)
            xgboost_predictions.to_csv(xgboost_dir / "predictions.csv", index=False)

            result = run_asic_xgboost_recalibration(
                input_root=input_root,
                output_dir=output_dir,
                horizons=[24],
            )

            self.assertEqual(result.horizons_processed, (24,))
            self.assertTrue(result.combined_comparison_metrics_path.exists())
            self.assertTrue(result.combined_test_reliability_summary_path.exists())
            self.assertTrue(result.summary_figure_path.exists())
            self.assertTrue(result.interpretation_note_path.exists())
            self.assertTrue(result.manifest_path.exists())

            horizon_dir = output_dir / "horizon_24h"
            self.assertTrue((horizon_dir / "comparison_metrics.csv").exists())
            self.assertTrue((horizon_dir / "logistic_reference_metrics_by_split.csv").exists())
            self.assertTrue((horizon_dir / "xgboost_raw_metrics_by_split.csv").exists())
            self.assertTrue((horizon_dir / "platt_predictions.csv").exists())
            self.assertTrue((horizon_dir / "platt_metrics_by_split.csv").exists())
            self.assertTrue((horizon_dir / "isotonic_predictions.csv").exists())
            self.assertTrue((horizon_dir / "isotonic_metrics_by_split.csv").exists())
            self.assertTrue((horizon_dir / "test_reliability_comparison.png").exists())
            self.assertTrue((horizon_dir / "test_probability_distribution.png").exists())

            comparison_metrics = pd.read_csv(result.combined_comparison_metrics_path)
            self.assertEqual(
                set(comparison_metrics["model_variant"].tolist()),
                {
                    "logistic_regression",
                    "xgboost_raw",
                    "xgboost_platt",
                    "xgboost_isotonic",
                },
            )
            self.assertEqual(set(comparison_metrics["split"].tolist()), {"train", "validation", "test"})

            platt_predictions = pd.read_csv(horizon_dir / "platt_predictions.csv")
            self.assertIn("raw_predicted_probability", platt_predictions.columns)
            self.assertIn("recalibrated_probability", platt_predictions.columns)
            self.assertEqual(set(platt_predictions["recalibration_method"].tolist()), {"platt"})

            platt_fit_summary = json.loads((horizon_dir / "platt_fit_summary.json").read_text())
            isotonic_fit_summary = json.loads((horizon_dir / "isotonic_fit_summary.json").read_text())
            self.assertEqual(platt_fit_summary["fit_split"], "validation")
            self.assertEqual(platt_fit_summary["status"], "fit")
            self.assertEqual(isotonic_fit_summary["fit_split"], "validation")
            self.assertEqual(isotonic_fit_summary["status"], "fit")
