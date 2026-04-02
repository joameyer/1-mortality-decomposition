from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from chapter1_mortality_decomposition.temporal_blocks import (
    build_asic_temporal_block_artifacts,
)
from chapter1_mortality_decomposition.temporal_preview import (
    DEFAULT_NOTEBOOK_PATH,
    DEFAULT_PREVIEW_OUTPUT_ROOT,
    _build_comparison_package,
    _notebook_payload,
)


def _risk_binned_summary(*, model_name: str, event_pattern: list[int]) -> pd.DataFrame:
    rows = []
    total_events = sum(event_pattern)
    total_samples = len(event_pattern) * 10
    for bin_index, event_count in enumerate(event_pattern, start=1):
        sample_count = 10
        rows.append(
            {
                "model_name": model_name,
                "horizon_h": 24,
                "split": "validation",
                "sample_scope": "overall",
                "group_id": "overall",
                "bin_index": bin_index,
                "bin_label": f"Q{bin_index:02d}",
                "sample_count": sample_count,
                "event_count": event_count,
                "non_event_count": sample_count - event_count,
                "sample_fraction": sample_count / total_samples,
                "event_fraction_of_events": (event_count / total_events) if total_events else 0.0,
                "predicted_probability_mean": 0.05 * bin_index,
                "predicted_probability_min": 0.05 * bin_index - 0.01,
                "predicted_probability_max": 0.05 * bin_index + 0.01,
                "observed_mortality": event_count / sample_count,
            }
        )
    return pd.DataFrame(rows)


def _write_minimal_evaluation_root(
    evaluation_root: Path,
    *,
    logistic_auroc: float,
    xgboost_auroc: float,
) -> None:
    evaluation_root.mkdir(parents=True, exist_ok=True)

    combined_metrics = pd.DataFrame(
        [
            {
                "model_name": "logistic_regression",
                "horizon_h": 24,
                "split": "validation",
                "sample_count": 40,
                "event_count": 4,
                "non_event_count": 36,
                "event_rate": 0.10,
                "auroc": logistic_auroc,
                "auprc": 0.22,
                "calibration_intercept": -0.10,
                "calibration_slope": 0.95,
                "metric_notes": pd.NA,
                "brier_score": 0.08,
                "binary_metrics_evaluable": True,
                "finite_prediction_count": 40,
            },
            {
                "model_name": "xgboost",
                "horizon_h": 24,
                "split": "validation",
                "sample_count": 40,
                "event_count": 4,
                "non_event_count": 36,
                "event_rate": 0.10,
                "auroc": xgboost_auroc,
                "auprc": 0.27,
                "calibration_intercept": 0.05,
                "calibration_slope": 1.10,
                "metric_notes": pd.NA,
                "brier_score": 0.07,
                "binary_metrics_evaluable": True,
                "finite_prediction_count": 40,
            },
        ]
    )
    combined_metrics.to_csv(evaluation_root / "combined_metrics.csv", index=False)

    reporting_summary = pd.DataFrame(
        [
            {
                "model_name": "logistic_regression",
                "horizon_h": 24,
                "selected_split": "validation",
                "selected_split_evaluable": True,
                "selection_reason": "first_binary_evaluable_split_in_priority_order",
                "sample_count": 40,
                "event_count": 4,
                "non_event_count": 36,
            },
            {
                "model_name": "xgboost",
                "horizon_h": 24,
                "selected_split": "validation",
                "selected_split_evaluable": True,
                "selection_reason": "first_binary_evaluable_split_in_priority_order",
                "sample_count": 40,
                "event_count": 4,
                "non_event_count": 36,
            },
        ]
    )
    reporting_summary.to_csv(evaluation_root / "reporting_split_summary.csv", index=False)

    for model_name, event_pattern in {
        "logistic_regression": [0, 1, 1, 2],
        "xgboost": [0, 0, 1, 3],
    }.items():
        model_dir = evaluation_root / model_name / "horizon_24h"
        model_dir.mkdir(parents=True, exist_ok=True)
        _risk_binned_summary(model_name=model_name, event_pattern=event_pattern).to_csv(
            model_dir / "risk_binned_summary.csv",
            index=False,
        )


class TemporalBlockArtifactTests(TestCase):
    def test_build_asic_temporal_block_artifacts_constructs_16h_blocks_from_harmonized_dynamic(self) -> None:
        reference_stay_block_counts = pd.DataFrame(
            [
                {
                    "stay_id_global": "stay_a",
                    "hospital_id": "H1",
                    "icu_admission_time": 0,
                    "icu_end_time_proxy": "1 days 10:00:00",
                    "icu_end_time_proxy_hours": 34,
                },
                {
                    "stay_id_global": "stay_b",
                    "hospital_id": "H1",
                    "icu_admission_time": 0,
                    "icu_end_time_proxy": "0 days 10:00:00",
                    "icu_end_time_proxy_hours": 10,
                },
            ]
        )
        dynamic_harmonized = pd.DataFrame(
            [
                {
                    "stay_id_global": "stay_a",
                    "hospital_id": "H1",
                    "time": "0 days 00:30:00",
                    "minutes_since_admit": 30,
                    "heart_rate": 80,
                    "map": 70,
                },
                {
                    "stay_id_global": "stay_a",
                    "hospital_id": "H1",
                    "time": "0 days 07:00:00",
                    "minutes_since_admit": 420,
                    "heart_rate": 100,
                    "map": 75,
                },
                {
                    "stay_id_global": "stay_a",
                    "hospital_id": "H1",
                    "time": "0 days 15:45:00",
                    "minutes_since_admit": 945,
                    "heart_rate": 90,
                    "map": 65,
                },
                {
                    "stay_id_global": "stay_a",
                    "hospital_id": "H1",
                    "time": "0 days 16:00:00",
                    "minutes_since_admit": 960,
                    "heart_rate": 120,
                    "map": 80,
                },
                {
                    "stay_id_global": "stay_a",
                    "hospital_id": "H1",
                    "time": "0 days 20:00:00",
                    "minutes_since_admit": 1200,
                    "heart_rate": 110,
                    "map": pd.NA,
                },
                {
                    "stay_id_global": "stay_b",
                    "hospital_id": "H1",
                    "time": "0 days 01:00:00",
                    "minutes_since_admit": 60,
                    "heart_rate": 60,
                    "map": 55,
                },
            ]
        )

        result = build_asic_temporal_block_artifacts(
            dynamic_harmonized=dynamic_harmonized,
            reference_stay_block_counts=reference_stay_block_counts,
            block_hours=16,
        )

        self.assertEqual(result.artifact_prefix, "asic_16h")
        self.assertEqual(result.stay_block_counts["completed_block_count"].tolist(), [2, 0])
        self.assertEqual(result.block_index["prediction_time_h"].tolist(), [16, 32])
        self.assertEqual(result.blocked_dynamic_features.shape[0], 2)

        first_block = result.blocked_dynamic_features.iloc[0]
        self.assertEqual(int(first_block["dynamic_row_count"]), 3)
        self.assertEqual(int(first_block["heart_rate_obs_count"]), 3)
        self.assertAlmostEqual(float(first_block["heart_rate_mean"]), 90.0)
        self.assertAlmostEqual(float(first_block["heart_rate_median"]), 90.0)
        self.assertEqual(float(first_block["heart_rate_last"]), 90.0)
        self.assertEqual(int(first_block["map_obs_count"]), 3)
        self.assertAlmostEqual(float(first_block["map_median"]), 70.0)
        self.assertEqual(float(first_block["map_last"]), 65.0)

        second_block = result.blocked_dynamic_features.iloc[1]
        self.assertEqual(int(second_block["dynamic_row_count"]), 2)
        self.assertEqual(int(second_block["heart_rate_obs_count"]), 2)
        self.assertAlmostEqual(float(second_block["heart_rate_median"]), 115.0)
        self.assertEqual(float(second_block["heart_rate_last"]), 110.0)
        self.assertEqual(int(second_block["map_obs_count"]), 1)
        self.assertEqual(float(second_block["map_last"]), 80.0)


class TemporalComparisonPackageTests(TestCase):
    def test_build_comparison_package_writes_table_figures_note_and_notebook(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            eight_hour_eval_root = tmp_path / "evaluation_8h"
            sixteen_hour_eval_root = tmp_path / "evaluation_16h"
            comparison_root = tmp_path / "comparison"
            notebook_path = tmp_path / "preview_review.ipynb"

            _write_minimal_evaluation_root(
                eight_hour_eval_root,
                logistic_auroc=0.72,
                xgboost_auroc=0.81,
            )
            _write_minimal_evaluation_root(
                sixteen_hour_eval_root,
                logistic_auroc=0.70,
                xgboost_auroc=0.79,
            )

            result = _build_comparison_package(
                eight_hour_evaluation_root=eight_hour_eval_root,
                sixteen_hour_evaluation_root=sixteen_hour_eval_root,
                comparison_output_dir=comparison_root,
                notebook_path=notebook_path,
            )

            comparison_table = pd.read_csv(result.comparison_table_path)
            self.assertEqual(set(comparison_table["aggregation"].tolist()), {"8h", "16h"})
            self.assertEqual(
                set(comparison_table["model_name"].tolist()),
                {"logistic_regression", "xgboost"},
            )
            self.assertTrue(result.note_path.exists())
            self.assertTrue(result.notebook_path.exists())
            self.assertEqual(len(result.figure_paths), 4)
            for figure_path in result.figure_paths:
                self.assertTrue(figure_path.exists())

    def test_default_notebook_path_lives_under_preview_output_root(self) -> None:
        self.assertTrue(str(DEFAULT_NOTEBOOK_PATH).startswith(str(DEFAULT_PREVIEW_OUTPUT_ROOT)))

    def test_notebook_payload_uses_repo_relative_specs_when_available(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        payload = _notebook_payload(
            eight_hour_evaluation_root=repo_root / "artifacts/chapter1/evaluation/asic/baselines/primary_medians",
            sixteen_hour_evaluation_root=repo_root / "artifacts/chapter1/temporal_preview/asic/aggregation_16h/evaluation/asic/baselines/primary_medians",
            comparison_table_path=repo_root / "artifacts/chapter1/temporal_preview/asic/aggregation_16h/comparison/aggregation_comparison_metrics.csv",
            note_path=repo_root / "artifacts/chapter1/temporal_preview/asic/aggregation_16h/comparison/preview_note.md",
            figure_paths=[
                repo_root / "artifacts/chapter1/temporal_preview/asic/aggregation_16h/comparison/logistic_regression_24h_reliability_8h_vs_16h.png",
            ],
        )
        source = "".join(payload["cells"][1]["source"])
        self.assertIn("REPO_ROOT = find_project_root", source)
        self.assertIn('"repo_relative"', source)
        self.assertNotIn(str(repo_root), source)
