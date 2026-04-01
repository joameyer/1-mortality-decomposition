from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from chapter1_mortality_decomposition.baseline_xgboost import (
    DEFAULT_XGBOOST_PARAMETERS,
    run_asic_primary_xgboost,
)


class XGBoostBaselineTests(TestCase):
    def test_run_asic_primary_xgboost_writes_outputs(self) -> None:
        model_ready = pd.DataFrame(
            [
                {
                    "instance_id": "stay_a__b0__h8",
                    "stay_id_global": "stay_a",
                    "hospital_id": "H1",
                    "block_index": 0,
                    "prediction_time_h": 8,
                    "horizon_h": 8,
                    "split": "train",
                    "label_value": 1,
                    "heart_rate_median": 90,
                    "creatinine_median": 1.2,
                    "heart_rate_mean": 91,
                },
                {
                    "instance_id": "stay_b__b0__h8",
                    "stay_id_global": "stay_b",
                    "hospital_id": "H1",
                    "block_index": 0,
                    "prediction_time_h": 8,
                    "horizon_h": 8,
                    "split": "train",
                    "label_value": 0,
                    "heart_rate_median": 70,
                    "creatinine_median": 0.8,
                    "heart_rate_mean": 72,
                },
                {
                    "instance_id": "stay_c__b0__h8",
                    "stay_id_global": "stay_c",
                    "hospital_id": "H1",
                    "block_index": 0,
                    "prediction_time_h": 8,
                    "horizon_h": 8,
                    "split": "validation",
                    "label_value": 1,
                    "heart_rate_median": 95,
                    "creatinine_median": 1.5,
                    "heart_rate_mean": 96,
                },
                {
                    "instance_id": "stay_d__b0__h8",
                    "stay_id_global": "stay_d",
                    "hospital_id": "H2",
                    "block_index": 0,
                    "prediction_time_h": 8,
                    "horizon_h": 8,
                    "split": "test",
                    "label_value": 0,
                    "heart_rate_median": 65,
                    "creatinine_median": 0.7,
                    "heart_rate_mean": 66,
                },
                {
                    "instance_id": "stay_a__b0__h24",
                    "stay_id_global": "stay_a",
                    "hospital_id": "H1",
                    "block_index": 0,
                    "prediction_time_h": 8,
                    "horizon_h": 24,
                    "split": "train",
                    "label_value": 1,
                    "heart_rate_median": 92,
                    "creatinine_median": 1.1,
                    "heart_rate_mean": 93,
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
                    "heart_rate_median": 68,
                    "creatinine_median": 0.9,
                    "heart_rate_mean": 69,
                },
                {
                    "instance_id": "stay_c__b0__h24",
                    "stay_id_global": "stay_c",
                    "hospital_id": "H1",
                    "block_index": 0,
                    "prediction_time_h": 8,
                    "horizon_h": 24,
                    "split": "validation",
                    "label_value": 0,
                    "heart_rate_median": 85,
                    "creatinine_median": 1.0,
                    "heart_rate_mean": 86,
                },
                {
                    "instance_id": "stay_d__b0__h24",
                    "stay_id_global": "stay_d",
                    "hospital_id": "H2",
                    "block_index": 0,
                    "prediction_time_h": 8,
                    "horizon_h": 24,
                    "split": "test",
                    "label_value": 0,
                    "heart_rate_median": 66,
                    "creatinine_median": 0.8,
                    "heart_rate_mean": 67,
                },
            ]
        )
        feature_set_definition = pd.DataFrame(
            [
                {
                    "feature_set_name": "primary",
                    "feature_name": "heart_rate_median",
                    "base_variable": "heart_rate",
                    "statistic": "median",
                    "selected_for_model": True,
                },
                {
                    "feature_set_name": "primary",
                    "feature_name": "creatinine_median",
                    "base_variable": "creatinine",
                    "statistic": "median",
                    "selected_for_model": True,
                },
                {
                    "feature_set_name": "primary",
                    "feature_name": "heart_rate_mean",
                    "base_variable": "heart_rate",
                    "statistic": "mean",
                    "selected_for_model": True,
                },
                {
                    "feature_set_name": "extended",
                    "feature_name": "pct_median",
                    "base_variable": "pct",
                    "statistic": "median",
                    "selected_for_model": True,
                },
            ]
        )

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "chapter1_primary_model_ready_dataset.csv"
            feature_path = tmp_path / "chapter1_feature_set_definition.csv"
            output_dir = tmp_path / "baselines"
            model_ready.to_csv(input_path, index=False)
            feature_set_definition.to_csv(feature_path, index=False)

            result = run_asic_primary_xgboost(
                input_dataset_path=input_path,
                feature_set_definition_path=feature_path,
                output_dir=output_dir,
                horizons=[8, 24],
            )

            self.assertEqual(result.horizons_processed, (8, 24))
            self.assertEqual(
                result.selected_feature_columns,
                ("heart_rate_median", "creatinine_median"),
            )

            summary = pd.read_csv(result.summary_path)
            self.assertEqual(set(summary["horizon_h"].tolist()), {8, 24})

            manifest = json.loads(result.manifest_path.read_text())
            self.assertEqual(manifest["selected_feature_columns"], ["heart_rate_median", "creatinine_median"])
            self.assertEqual(manifest["xgboost_parameters"], DEFAULT_XGBOOST_PARAMETERS)

            for horizon_h in (8, 24):
                horizon_dir = output_dir / f"horizon_{horizon_h}h"
                self.assertTrue((horizon_dir / "predictions.csv").exists())
                self.assertTrue((horizon_dir / "metrics.csv").exists())
                self.assertTrue((horizon_dir / "metadata.json").exists())
                self.assertTrue((horizon_dir / "selected_feature_columns.json").exists())
                self.assertTrue((horizon_dir / "preprocessing.pkl").exists())
                self.assertTrue((horizon_dir / "xgboost_model.pkl").exists())
                self.assertTrue((horizon_dir / "pipeline.pkl").exists())

                predictions = pd.read_csv(horizon_dir / "predictions.csv")
                self.assertIn("predicted_probability", predictions.columns)
                self.assertEqual(set(predictions["model_name"].tolist()), {"xgboost"})

                metrics = pd.read_csv(horizon_dir / "metrics.csv")
                self.assertEqual(set(metrics["split"].tolist()), {"train", "validation", "test"})
