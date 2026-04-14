from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from chapter1_mortality_decomposition.artifacts import (
    RESULT_ROOT_KIND_CLUSTER_EXPORT,
    RESULT_ROOT_KIND_SYNTHETIC_LOCAL,
)
from chapter1_mortality_decomposition.baseline_evaluation import (
    _resolve_default_baseline_input_root,
    run_asic_baseline_evaluation,
)


class BaselineEvaluationTests(TestCase):
    def test_default_input_root_resolution_prefers_cluster_export_predictions(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cluster_root = (
                tmp_path
                / "cluster-results"
                / "chapter1_true_results"
                / "baselines"
                / "asic"
                / "primary_medians"
            )
            synthetic_root = (
                tmp_path
                / "artifacts"
                / "chapter1"
                / "baselines"
                / "asic"
                / "primary_medians"
            )
            cluster_root.mkdir(parents=True)
            synthetic_root.mkdir(parents=True)

            resolved_root, resolution = _resolve_default_baseline_input_root(
                preferred_result_kind=RESULT_ROOT_KIND_CLUSTER_EXPORT,
                repo_root=tmp_path,
            )

            self.assertEqual(resolved_root, cluster_root.resolve())
            self.assertEqual(resolution["selected_result_kind"], RESULT_ROOT_KIND_CLUSTER_EXPORT)

    def test_default_input_root_resolution_falls_back_to_synthetic_predictions(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            synthetic_root = (
                tmp_path
                / "artifacts"
                / "chapter1"
                / "baselines"
                / "asic"
                / "primary_medians"
            )
            synthetic_root.mkdir(parents=True)

            resolved_root, resolution = _resolve_default_baseline_input_root(
                preferred_result_kind=RESULT_ROOT_KIND_CLUSTER_EXPORT,
                repo_root=tmp_path,
            )

            self.assertEqual(resolved_root, synthetic_root.resolve())
            self.assertEqual(resolution["selected_result_kind"], RESULT_ROOT_KIND_SYNTHETIC_LOCAL)

    def test_run_asic_baseline_evaluation_writes_outputs_and_falls_back_from_degenerate_test(self) -> None:
        prediction_rows = [
            {
                "instance_id": "stay_a__b0__h24",
                "stay_id_global": "stay_a",
                "hospital_id": "H1",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "train",
                "label_value": 1,
                "predicted_probability": 0.80,
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
                "predicted_probability": 0.20,
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
                "predicted_probability": 0.70,
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
                "predicted_probability": 0.30,
            },
            {
                "instance_id": "stay_e__b0__h24",
                "stay_id_global": "stay_e",
                "hospital_id": "H1",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "test",
                "label_value": 0,
                "predicted_probability": 0.10,
            },
            {
                "instance_id": "stay_f__b0__h24",
                "stay_id_global": "stay_f",
                "hospital_id": "H2",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "test",
                "label_value": 0,
                "predicted_probability": 0.15,
            },
        ]

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_root = tmp_path / "baselines"
            output_dir = tmp_path / "evaluation"
            for model_name in ("logistic_regression", "xgboost"):
                horizon_dir = input_root / model_name / "horizon_24h"
                horizon_dir.mkdir(parents=True)
                predictions = pd.DataFrame(prediction_rows).assign(model_name=model_name)
                predictions.to_csv(horizon_dir / "predictions.csv", index=False)

            result = run_asic_baseline_evaluation(
                input_root=input_root,
                output_dir=output_dir,
                horizons=[24],
            )

            combined_metrics = pd.read_csv(result.combined_metrics_path)
            self.assertEqual(set(combined_metrics["model_name"].tolist()), {"logistic_regression", "xgboost"})
            self.assertEqual(set(combined_metrics["split"].tolist()), {"train", "validation", "test"})

            reporting_summary = pd.read_csv(result.reporting_split_summary_path)
            self.assertTrue(reporting_summary["selected_split"].eq("validation").all())
            self.assertTrue(reporting_summary["selected_split_evaluable"].astype(bool).all())

            for model_name in ("logistic_regression", "xgboost"):
                model_dir = output_dir / model_name
                self.assertTrue((model_dir / "horizon_24h" / "metrics_by_split.csv").exists())
                self.assertTrue((model_dir / "horizon_24h" / "risk_binned_summary.csv").exists())
                self.assertTrue((model_dir / "horizon_24h" / "reliability_plot.png").exists())
                self.assertTrue((model_dir / "horizon_24h" / "mortality_vs_risk_plot.png").exists())
                self.assertTrue((model_dir / "horizon_comparison_plot.png").exists())
                self.assertTrue((model_dir / "horizon_risk_structure_grid.png").exists())
                self.assertTrue((model_dir / "primary_24h_site_summary.csv").exists())
                self.assertTrue((model_dir / "primary_24h_site_risk_structure.png").exists())

            self.assertTrue(result.interpretation_note_path.exists())
