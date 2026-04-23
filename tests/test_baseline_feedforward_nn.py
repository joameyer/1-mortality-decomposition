from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from chapter1_mortality_decomposition.baseline_feedforward_nn import (
    run_asic_primary_feedforward_nn,
)
from tests.ch1_baseline_test_utils import write_primary_baseline_fixture


class FeedforwardNNBaselineTests(TestCase):
    def test_run_asic_primary_feedforward_nn_writes_outputs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fixture = write_primary_baseline_fixture(tmp_path)
            input_path = fixture["input_dataset_path"]
            feature_path = fixture["feature_set_definition_path"]
            output_dir = tmp_path / "feedforward_nn_baseline"
            standardized_input_dir = fixture["standardized_input_dir"]

            result = run_asic_primary_feedforward_nn(
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
            self.assertEqual(
                manifest["selected_feature_columns"],
                ["heart_rate_median", "creatinine_median"],
            )

            for horizon_h in (8, 24):
                horizon_dir = output_dir / f"horizon_{horizon_h}h"
                self.assertTrue((horizon_dir / "predictions.csv").exists())
                self.assertTrue((horizon_dir / "all_valid_predictions.csv").exists())
                self.assertTrue((horizon_dir / "all_valid_prediction_qc.csv").exists())
                self.assertTrue((horizon_dir / "metrics.csv").exists())
                self.assertTrue((horizon_dir / "metadata.json").exists())
                self.assertTrue((horizon_dir / "selected_feature_columns.json").exists())
                self.assertTrue((horizon_dir / "preprocessing.pkl").exists())
                self.assertTrue((horizon_dir / "feedforward_nn_model.pkl").exists())
                self.assertTrue((horizon_dir / "pipeline.pkl").exists())

                predictions = pd.read_csv(horizon_dir / "predictions.csv")
                all_valid_predictions = pd.read_csv(horizon_dir / "all_valid_predictions.csv")
                qc = pd.read_csv(horizon_dir / "all_valid_prediction_qc.csv")
                metadata = json.loads((horizon_dir / "metadata.json").read_text())
                self.assertEqual(set(predictions["model_name"].tolist()), {"feedforward_nn"})
                self.assertEqual(set(all_valid_predictions["model_name"].tolist()), {"feedforward_nn"})
                self.assertGreater(all_valid_predictions.shape[0], predictions.shape[0])
                self.assertEqual(
                    int(qc.loc[0, "evaluation_prediction_count"]),
                    predictions.shape[0],
                )
                self.assertIn(
                    metadata["model"]["fit_metadata"]["monitor_split"],
                    {"train", "validation"},
                )
                self.assertGreaterEqual(
                    int(metadata["model"]["fit_metadata"]["epochs_completed"]),
                    1,
                )

                metrics = pd.read_csv(horizon_dir / "metrics.csv")
                self.assertEqual(set(metrics["split"].tolist()), {"train", "validation", "test"})
