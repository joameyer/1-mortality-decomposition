from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from chapter1_mortality_decomposition.baseline_logistic import (
    compute_binary_classification_metrics,
    run_asic_primary_logistic_regression,
    select_primary_logistic_feature_columns,
)
from tests.ch1_baseline_test_utils import write_primary_baseline_fixture


class LogisticBaselineTests(TestCase):
    def test_select_primary_logistic_feature_columns_uses_only_primary_medians_and_static(self) -> None:
        model_ready = pd.DataFrame(
            columns=[
                "instance_id",
                "stay_id_global",
                "hospital_id",
                "block_index",
                "prediction_time_h",
                "horizon_h",
                "split",
                "label_value",
                "heart_rate_median",
                "heart_rate_mean",
                "creatinine_median",
                "age_years",
                "pct_median",
                "heart_rate_missing_after_locf",
            ]
        )
        feature_set_definition = pd.DataFrame(
            [
                {
                    "feature_set_name": "primary",
                    "feature_name": "heart_rate_obs_count",
                    "base_variable": "heart_rate",
                    "statistic": "obs_count",
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
                    "feature_name": "age_years",
                    "base_variable": "age_years",
                    "statistic": pd.NA,
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

        selected_columns, mapping_report = select_primary_logistic_feature_columns(
            model_ready,
            feature_set_definition,
        )

        self.assertEqual(
            selected_columns,
            ["heart_rate_median", "creatinine_median", "age_years"],
        )
        self.assertEqual(mapping_report["primary_static_context_feature_columns"], ["age_years"])
        self.assertNotIn("heart_rate_mean", selected_columns)
        self.assertNotIn("pct_median", selected_columns)

    def test_compute_binary_classification_metrics_handles_single_class_split(self) -> None:
        metrics = compute_binary_classification_metrics(
            y_true=[0, 0, 0],
            y_prob=[0.1, 0.2, 0.3],
        )

        self.assertEqual(metrics["sample_count"], 3)
        self.assertEqual(metrics["event_count"], 0)
        self.assertTrue(math.isnan(float(metrics["auroc"])))
        self.assertTrue(math.isnan(float(metrics["auprc"])))
        self.assertTrue(math.isnan(float(metrics["calibration_intercept"])))
        self.assertTrue(math.isnan(float(metrics["calibration_slope"])))
        self.assertIn("single_class", str(metrics["metric_notes"]))

    def test_run_asic_primary_logistic_regression_writes_outputs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fixture = write_primary_baseline_fixture(tmp_path)
            input_path = fixture["input_dataset_path"]
            feature_path = fixture["feature_set_definition_path"]
            output_dir = fixture["output_dir"]
            standardized_input_dir = fixture["standardized_input_dir"]

            result = run_asic_primary_logistic_regression(
                input_dataset_path=input_path,
                feature_set_definition_path=feature_path,
                output_dir=output_dir,
                horizons=[8, 24],
                standardized_input_dir=standardized_input_dir,
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

            for horizon_h in (8, 24):
                horizon_dir = output_dir / f"horizon_{horizon_h}h"
                self.assertTrue((horizon_dir / "predictions.csv").exists())
                self.assertTrue((horizon_dir / "all_valid_predictions.csv").exists())
                self.assertTrue((horizon_dir / "all_valid_prediction_qc.csv").exists())
                self.assertTrue((horizon_dir / "metrics.csv").exists())
                self.assertTrue((horizon_dir / "metadata.json").exists())
                self.assertTrue((horizon_dir / "selected_feature_columns.json").exists())
                self.assertTrue((horizon_dir / "preprocessing.pkl").exists())
                self.assertTrue((horizon_dir / "logistic_regression_model.pkl").exists())
                self.assertTrue((horizon_dir / "pipeline.pkl").exists())

                predictions = pd.read_csv(horizon_dir / "predictions.csv")
                all_valid_predictions = pd.read_csv(horizon_dir / "all_valid_predictions.csv")
                qc = pd.read_csv(horizon_dir / "all_valid_prediction_qc.csv")
                self.assertIn("predicted_probability", predictions.columns)
                self.assertEqual(set(predictions["model_name"].tolist()), {"logistic_regression"})
                self.assertEqual(set(all_valid_predictions["model_name"].tolist()), {"logistic_regression"})
                self.assertGreater(all_valid_predictions.shape[0], predictions.shape[0])
                self.assertTrue(
                    set(predictions["instance_id"].astype("string")).issubset(
                        set(all_valid_predictions["instance_id"].astype("string"))
                    )
                )
                self.assertEqual(
                    int(all_valid_predictions["is_labelable"].fillna(False).astype(bool).sum()),
                    predictions.shape[0],
                )
                self.assertEqual(
                    int(qc.loc[0, "evaluation_prediction_count"]),
                    predictions.shape[0],
                )
                self.assertEqual(
                    int(qc.loc[0, "all_valid_prediction_count"]),
                    all_valid_predictions.shape[0],
                )
                self.assertTrue(bool(qc.loc[0, "evaluation_subset_of_all_valid"]))
                self.assertTrue(
                    all_valid_predictions["unlabeled_reason"]
                    .astype("string")
                    .eq("non_survivor_proxy_end_not_within_horizon")
                    .any()
                )

                metrics = pd.read_csv(horizon_dir / "metrics.csv")
                self.assertEqual(set(metrics["split"].tolist()), {"train", "validation", "test"})

    def test_run_asic_primary_logistic_regression_rejects_unexpected_split_labels(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fixture = write_primary_baseline_fixture(tmp_path)
            input_path = fixture["input_dataset_path"]
            feature_path = fixture["feature_set_definition_path"]
            output_dir = fixture["output_dir"]
            standardized_input_dir = fixture["standardized_input_dir"]
            model_ready = pd.read_csv(input_path)
            model_ready.loc[model_ready["stay_id_global"].eq("stay_b"), "split"] = "holdout"
            model_ready.to_csv(input_path, index=False)

            with self.assertRaisesRegex(ValueError, "unexpected split labels"):
                run_asic_primary_logistic_regression(
                    input_dataset_path=input_path,
                    feature_set_definition_path=feature_path,
                    output_dir=output_dir,
                    horizons=[24],
                    standardized_input_dir=standardized_input_dir,
                )
