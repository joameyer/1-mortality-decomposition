from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from chapter1_mortality_decomposition.feedforward_nn_hard_case_sensitivity import (
    run_feedforward_nn_hard_case_sensitivity,
)


def _write_predictions(
    root: Path,
    *,
    model_name: str,
    horizon_h: int,
    rows: list[dict[str, object]],
) -> None:
    horizon_dir = root / model_name / f"horizon_{int(horizon_h)}h"
    horizon_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).assign(model_name=model_name).to_csv(horizon_dir / "predictions.csv", index=False)


class FeedforwardNNHardCaseSensitivityTests(TestCase):
    def test_run_feedforward_nn_hard_case_sensitivity_reuses_saved_predictions_and_writes_outputs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            baseline_root = tmp_path / "baselines"
            evaluation_output_dir = tmp_path / "evaluation"
            hard_case_output_dir = tmp_path / "hard_cases" / "feedforward_nn"
            sensitivity_output_dir = tmp_path / "hard_cases" / "agreement" / "feedforward_nn_sensitivity"

            rows_24 = [
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
                    "predicted_probability": 0.18,
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
                    "predicted_probability": 0.34,
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
                    "predicted_probability": 0.44,
                },
            ]
            rows_48 = [
                {
                    "instance_id": "n1__b0__h48",
                    "stay_id_global": "n1",
                    "hospital_id": "H1",
                    "block_index": 0,
                    "prediction_time_h": 8,
                    "horizon_h": 48,
                    "split": "train",
                    "label_value": 0,
                    "predicted_probability": 0.15,
                },
                {
                    "instance_id": "n2__b0__h48",
                    "stay_id_global": "n2",
                    "hospital_id": "H1",
                    "block_index": 0,
                    "prediction_time_h": 8,
                    "horizon_h": 48,
                    "split": "validation",
                    "label_value": 0,
                    "predicted_probability": 0.25,
                },
                {
                    "instance_id": "n3__b0__h48",
                    "stay_id_global": "n3",
                    "hospital_id": "H2",
                    "block_index": 0,
                    "prediction_time_h": 8,
                    "horizon_h": 48,
                    "split": "test",
                    "label_value": 0,
                    "predicted_probability": 0.35,
                },
                {
                    "instance_id": "n4__b0__h48",
                    "stay_id_global": "n4",
                    "hospital_id": "H2",
                    "block_index": 0,
                    "prediction_time_h": 8,
                    "horizon_h": 48,
                    "split": "test",
                    "label_value": 0,
                    "predicted_probability": 0.45,
                },
                {
                    "instance_id": "f1__b0__h48",
                    "stay_id_global": "f1",
                    "hospital_id": "H1",
                    "block_index": 0,
                    "prediction_time_h": 8,
                    "horizon_h": 48,
                    "split": "train",
                    "label_value": 1,
                    "predicted_probability": 0.22,
                },
                {
                    "instance_id": "f2__b0__h48",
                    "stay_id_global": "f2",
                    "hospital_id": "H1",
                    "block_index": 0,
                    "prediction_time_h": 8,
                    "horizon_h": 48,
                    "split": "validation",
                    "label_value": 1,
                    "predicted_probability": 0.36,
                },
                {
                    "instance_id": "f3__b0__h48",
                    "stay_id_global": "f3",
                    "hospital_id": "H2",
                    "block_index": 0,
                    "prediction_time_h": 8,
                    "horizon_h": 48,
                    "split": "test",
                    "label_value": 1,
                    "predicted_probability": 0.50,
                },
            ]

            logistic_24 = rows_24
            logistic_48 = rows_48
            xgboost_24 = [{**row, "predicted_probability": value} for row, value in zip(
                rows_24,
                [0.08, 0.18, 0.28, 0.38, 0.22, 0.26, 0.48],
            )]
            xgboost_48 = [{**row, "predicted_probability": value} for row, value in zip(
                rows_48,
                [0.12, 0.22, 0.32, 0.42, 0.24, 0.30, 0.52],
            )]
            nn_24 = [{**row, "predicted_probability": value} for row, value in zip(
                rows_24,
                [0.09, 0.19, 0.29, 0.39, 0.19, 0.29, 0.45],
            )]
            nn_48 = [{**row, "predicted_probability": value} for row, value in zip(
                rows_48,
                [0.13, 0.23, 0.33, 0.43, 0.23, 0.33, 0.49],
            )]

            for horizon_h, logistic_rows, xgboost_rows, nn_rows in (
                (24, logistic_24, xgboost_24, nn_24),
                (48, logistic_48, xgboost_48, nn_48),
            ):
                _write_predictions(baseline_root, model_name="logistic_regression", horizon_h=horizon_h, rows=logistic_rows)
                _write_predictions(baseline_root, model_name="xgboost", horizon_h=horizon_h, rows=xgboost_rows)
                _write_predictions(baseline_root, model_name="feedforward_nn", horizon_h=horizon_h, rows=nn_rows)

            result = run_feedforward_nn_hard_case_sensitivity(
                baseline_input_root=baseline_root,
                evaluation_output_dir=evaluation_output_dir,
                hard_case_output_dir=hard_case_output_dir,
                sensitivity_output_dir=sensitivity_output_dir,
                horizons=[24, 48],
            )

            self.assertTrue(result.artifacts.nn_hard_case_stay_level_path.exists())
            self.assertTrue(result.artifacts.nn_hard_case_summary_path.exists())
            self.assertTrue(result.artifacts.performance_summary_path.exists())
            self.assertTrue(result.artifacts.overlap_summary_path.exists())
            self.assertTrue(result.artifacts.boundary_summary_path.exists())
            self.assertTrue(result.artifacts.figure_path.exists())
            self.assertTrue(result.artifacts.memo_path.exists())
            self.assertTrue(result.artifacts.manifest_path.exists())
            self.assertIsNotNone(result.artifacts.three_way_overlap_24h_path)
            assert result.artifacts.three_way_overlap_24h_path is not None
            self.assertTrue(result.artifacts.three_way_overlap_24h_path.exists())

            overlap_summary = pd.read_csv(result.artifacts.overlap_summary_path)
            self.assertEqual(
                set(overlap_summary["pair_name"].tolist()),
                {"feedforward_nn_vs_logistic_regression", "feedforward_nn_vs_xgboost"},
            )
            self.assertEqual(set(overlap_summary["horizon_h"].tolist()), {24, 48})
            self.assertIn("jaccard_agreement", overlap_summary.columns)
            self.assertIn("overlap_share_of_left_hard", overlap_summary.columns)
            self.assertIn("overlap_share_of_right_hard", overlap_summary.columns)

            performance_summary = pd.read_csv(result.artifacts.performance_summary_path)
            self.assertEqual(set(performance_summary["horizon_h"].tolist()), {24, 48})
            self.assertTrue(
                performance_summary["reliability_plot_path"].map(lambda value: Path(value).exists()).all()
            )

            memo_text = result.artifacts.memo_path.read_text()
            self.assertIn("Final judgment:", memo_text)
            self.assertIn("Feedforward NN", memo_text)

    def test_run_feedforward_nn_hard_case_sensitivity_regenerates_missing_nn_evaluation_slice(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            baseline_root = tmp_path / "baselines"
            evaluation_output_dir = tmp_path / "evaluation"
            hard_case_output_dir = tmp_path / "hard_cases" / "feedforward_nn"
            sensitivity_output_dir = tmp_path / "hard_cases" / "agreement" / "feedforward_nn_sensitivity"

            rows_24 = [
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
                    "predicted_probability": 0.18,
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
                    "predicted_probability": 0.34,
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
                    "predicted_probability": 0.44,
                },
            ]

            logistic_24 = rows_24
            xgboost_24 = [{**row, "predicted_probability": value} for row, value in zip(
                rows_24,
                [0.08, 0.18, 0.28, 0.38, 0.22, 0.26, 0.48],
            )]
            nn_24 = [{**row, "predicted_probability": value} for row, value in zip(
                rows_24,
                [0.09, 0.19, 0.29, 0.39, 0.19, 0.29, 0.45],
            )]

            _write_predictions(baseline_root, model_name="logistic_regression", horizon_h=24, rows=logistic_24)
            _write_predictions(baseline_root, model_name="xgboost", horizon_h=24, rows=xgboost_24)
            _write_predictions(baseline_root, model_name="feedforward_nn", horizon_h=24, rows=nn_24)

            evaluation_output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "model_name": "logistic_regression",
                        "horizon_h": 24,
                        "split": "validation",
                        "sample_count": 2,
                        "event_count": 1,
                        "non_event_count": 1,
                        "event_rate": 0.5,
                        "auroc": 1.0,
                        "auprc": 1.0,
                        "calibration_intercept": 0.0,
                        "calibration_slope": 1.0,
                        "brier_score": 0.1,
                        "metric_notes": pd.NA,
                    }
                ]
            ).to_csv(evaluation_output_dir / "combined_metrics.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "model_name": "logistic_regression",
                        "horizon_h": 24,
                        "selected_split": "validation",
                        "selected_split_evaluable": True,
                        "selection_reason": "seeded_fixture",
                    }
                ]
            ).to_csv(evaluation_output_dir / "reporting_split_summary.csv", index=False)

            result = run_feedforward_nn_hard_case_sensitivity(
                baseline_input_root=baseline_root,
                evaluation_output_dir=evaluation_output_dir,
                hard_case_output_dir=hard_case_output_dir,
                sensitivity_output_dir=sensitivity_output_dir,
                horizons=[24],
            )

            performance_summary = pd.read_csv(result.artifacts.performance_summary_path)
            self.assertEqual(set(performance_summary["model_name"].tolist()), {"feedforward_nn"})
            self.assertEqual(set(performance_summary["horizon_h"].tolist()), {24})
            self.assertTrue(result.artifacts.memo_path.read_text().count("AUROC") >= 1)
