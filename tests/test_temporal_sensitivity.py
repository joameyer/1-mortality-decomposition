from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from chapter1_mortality_decomposition.temporal_sensitivity import (
    _build_aggregation_directional_overlap,
    _build_aggregation_pairwise_tables,
    _build_aggregation_persistence_tables,
    _build_frozen_split_alignment_summary,
    write_temporal_sensitivity_interpretation_memo,
)


class TemporalSensitivityTests(TestCase):
    def test_build_frozen_split_alignment_summary_allows_unused_frozen_assignments(self) -> None:
        retained = pd.DataFrame(
            [
                {"stay_id_global": "stay_a", "hospital_id": "H1"},
                {"stay_id_global": "stay_b", "hospital_id": "H1"},
            ]
        )
        frozen = pd.DataFrame(
            [
                {"stay_id_global": "stay_a", "hospital_id": "H1", "split": "train"},
                {"stay_id_global": "stay_b", "hospital_id": "H1", "split": "validation"},
                {"stay_id_global": "stay_c", "hospital_id": "H2", "split": "test"},
            ]
        )

        summary, aligned = _build_frozen_split_alignment_summary(
            retained,
            frozen,
            aggregation_label="24h",
        )

        unused_row = summary[summary["check_id"].eq("unused_frozen_split_assignments_after_alignment")]
        self.assertEqual(int(unused_row["count"].iloc[0]), 1)
        self.assertFalse(bool(unused_row["passed"].iloc[0]))
        self.assertEqual(set(aligned["stay_id_global"].astype(str)), {"stay_a", "stay_b"})

    def test_build_aggregation_overlap_outputs(self) -> None:
        harmonized = pd.DataFrame(
            [
                {
                    "aggregation": "8h",
                    "stay_id": "stay_1",
                    "hospital_id": "H1",
                    "fatal_flag": True,
                    "hard_case_flag": True,
                    "available_flag": True,
                },
                {
                    "aggregation": "8h",
                    "stay_id": "stay_2",
                    "hospital_id": "H1",
                    "fatal_flag": True,
                    "hard_case_flag": False,
                    "available_flag": True,
                },
                {
                    "aggregation": "16h",
                    "stay_id": "stay_1",
                    "hospital_id": "H1",
                    "fatal_flag": True,
                    "hard_case_flag": True,
                    "available_flag": True,
                },
                {
                    "aggregation": "16h",
                    "stay_id": "stay_2",
                    "hospital_id": "H1",
                    "fatal_flag": True,
                    "hard_case_flag": True,
                    "available_flag": True,
                },
                {
                    "aggregation": "24h",
                    "stay_id": "stay_1",
                    "hospital_id": "H1",
                    "fatal_flag": True,
                    "hard_case_flag": False,
                    "available_flag": True,
                },
                {
                    "aggregation": "24h",
                    "stay_id": "stay_2",
                    "hospital_id": "H1",
                    "fatal_flag": True,
                    "hard_case_flag": True,
                    "available_flag": True,
                },
            ]
        )
        aggregation_labels = ["8h", "16h", "24h"]

        pairwise_denominators, pairwise_overlap = _build_aggregation_pairwise_tables(
            harmonized,
            aggregation_labels=aggregation_labels,
        )
        directional_overlap = _build_aggregation_directional_overlap(
            pairwise_overlap,
            aggregation_labels=aggregation_labels,
        )
        persistence, persistence_distribution = _build_aggregation_persistence_tables(
            harmonized,
            aggregation_labels=aggregation_labels,
        )

        pair_8h_16h = pairwise_overlap[
            pairwise_overlap["aggregation_a"].eq("8h")
            & pairwise_overlap["aggregation_b"].eq("16h")
        ].iloc[0]
        self.assertEqual(int(pair_8h_16h["matched_fatal_n"]), 2)
        self.assertEqual(int(pair_8h_16h["hard_n_aggregation_a"]), 1)
        self.assertEqual(int(pair_8h_16h["hard_n_aggregation_b"]), 2)
        self.assertEqual(int(pair_8h_16h["intersection_n"]), 1)
        self.assertEqual(int(pair_8h_16h["union_n"]), 2)
        self.assertAlmostEqual(float(pair_8h_16h["jaccard_index"]), 0.5)

        overlap_8_to_16 = directional_overlap[
            directional_overlap["aggregation_from"].eq("8h")
            & directional_overlap["aggregation_to"].eq("16h")
        ].iloc[0]
        overlap_16_to_8 = directional_overlap[
            directional_overlap["aggregation_from"].eq("16h")
            & directional_overlap["aggregation_to"].eq("8h")
        ].iloc[0]
        self.assertAlmostEqual(float(overlap_8_to_16["overlap_from_A_to_B"]), 1.0)
        self.assertAlmostEqual(float(overlap_16_to_8["overlap_from_A_to_B"]), 0.5)

        self.assertEqual(int(persistence["hard_case_aggregation_n"].sum()), 4)
        two_aggregation_row = persistence_distribution[
            persistence_distribution["hard_case_aggregation_n"].eq(2)
        ].iloc[0]
        self.assertEqual(int(two_aggregation_row["fatal_stay_count"]), 2)
        self.assertEqual(int(pairwise_denominators["matched_fatal_n"].min()), 2)

    def test_write_temporal_sensitivity_interpretation_memo_writes_result_file(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            comparison_root = Path(tmp_dir)
            (comparison_root / "run_manifest.json").write_text(
                json.dumps({"artifact_paths": {}}, indent=2)
            )

            pd.DataFrame(
                [
                    {
                        "model_name": "logistic_regression",
                        "horizon_h": 24,
                        "aggregation": "8h",
                        "selected_split": "test",
                        "selected_split_evaluable": True,
                        "selection_reason": "first_binary_evaluable_split_in_priority_order",
                        "sample_count": 100,
                        "event_count": 20,
                        "non_event_count": 80,
                        "event_rate": 0.20,
                        "auroc": 0.82,
                        "auprc": 0.27,
                        "calibration_intercept": -0.10,
                        "calibration_slope": 0.97,
                        "brier_score": 0.02,
                        "binary_metrics_evaluable": True,
                        "finite_prediction_count": 100,
                        "metric_notes": pd.NA,
                        "sample_count_reference_8h": 100,
                        "event_count_reference_8h": 20,
                        "non_event_count_reference_8h": 80,
                        "event_rate_reference_8h": 0.20,
                        "auroc_reference_8h": 0.82,
                        "auprc_reference_8h": 0.27,
                        "calibration_intercept_reference_8h": -0.10,
                        "calibration_slope_reference_8h": 0.97,
                        "brier_score_reference_8h": 0.02,
                        "sample_count_delta_vs_8h": 0,
                        "event_count_delta_vs_8h": 0,
                        "non_event_count_delta_vs_8h": 0,
                        "event_rate_delta_vs_8h": 0.0,
                        "auroc_delta_vs_8h": 0.0,
                        "auprc_delta_vs_8h": 0.0,
                        "calibration_intercept_delta_vs_8h": 0.0,
                        "calibration_slope_delta_vs_8h": 0.0,
                        "brier_score_delta_vs_8h": 0.0,
                    },
                    {
                        "model_name": "logistic_regression",
                        "horizon_h": 24,
                        "aggregation": "16h",
                        "selected_split": "test",
                        "selected_split_evaluable": True,
                        "selection_reason": "first_binary_evaluable_split_in_priority_order",
                        "sample_count": 60,
                        "event_count": 12,
                        "non_event_count": 48,
                        "event_rate": 0.20,
                        "auroc": 0.815,
                        "auprc": 0.245,
                        "calibration_intercept": -0.12,
                        "calibration_slope": 0.96,
                        "brier_score": 0.021,
                        "binary_metrics_evaluable": True,
                        "finite_prediction_count": 60,
                        "metric_notes": pd.NA,
                        "sample_count_reference_8h": 100,
                        "event_count_reference_8h": 20,
                        "non_event_count_reference_8h": 80,
                        "event_rate_reference_8h": 0.20,
                        "auroc_reference_8h": 0.82,
                        "auprc_reference_8h": 0.27,
                        "calibration_intercept_reference_8h": -0.10,
                        "calibration_slope_reference_8h": 0.97,
                        "brier_score_reference_8h": 0.02,
                        "sample_count_delta_vs_8h": -40,
                        "event_count_delta_vs_8h": -8,
                        "non_event_count_delta_vs_8h": -32,
                        "event_rate_delta_vs_8h": 0.0,
                        "auroc_delta_vs_8h": -0.005,
                        "auprc_delta_vs_8h": -0.025,
                        "calibration_intercept_delta_vs_8h": -0.02,
                        "calibration_slope_delta_vs_8h": -0.01,
                        "brier_score_delta_vs_8h": 0.001,
                    },
                    {
                        "model_name": "logistic_regression",
                        "horizon_h": 24,
                        "aggregation": "24h",
                        "selected_split": "test",
                        "selected_split_evaluable": True,
                        "selection_reason": "first_binary_evaluable_split_in_priority_order",
                        "sample_count": 40,
                        "event_count": 8,
                        "non_event_count": 32,
                        "event_rate": 0.20,
                        "auroc": 0.810,
                        "auprc": 0.230,
                        "calibration_intercept": -0.13,
                        "calibration_slope": 0.95,
                        "brier_score": 0.0215,
                        "binary_metrics_evaluable": True,
                        "finite_prediction_count": 40,
                        "metric_notes": pd.NA,
                        "sample_count_reference_8h": 100,
                        "event_count_reference_8h": 20,
                        "non_event_count_reference_8h": 80,
                        "event_rate_reference_8h": 0.20,
                        "auroc_reference_8h": 0.82,
                        "auprc_reference_8h": 0.27,
                        "calibration_intercept_reference_8h": -0.10,
                        "calibration_slope_reference_8h": 0.97,
                        "brier_score_reference_8h": 0.02,
                        "sample_count_delta_vs_8h": -60,
                        "event_count_delta_vs_8h": -12,
                        "non_event_count_delta_vs_8h": -48,
                        "event_rate_delta_vs_8h": 0.0,
                        "auroc_delta_vs_8h": -0.01,
                        "auprc_delta_vs_8h": -0.04,
                        "calibration_intercept_delta_vs_8h": -0.03,
                        "calibration_slope_delta_vs_8h": -0.02,
                        "brier_score_delta_vs_8h": 0.0015,
                    },
                    {
                        "model_name": "xgboost",
                        "horizon_h": 24,
                        "aggregation": "8h",
                        "selected_split": "test",
                        "selected_split_evaluable": True,
                        "selection_reason": "first_binary_evaluable_split_in_priority_order",
                        "sample_count": 100,
                        "event_count": 20,
                        "non_event_count": 80,
                        "event_rate": 0.20,
                        "auroc": 0.85,
                        "auprc": 0.32,
                        "calibration_intercept": -3.70,
                        "calibration_slope": 1.16,
                        "brier_score": 0.13,
                        "binary_metrics_evaluable": True,
                        "finite_prediction_count": 100,
                        "metric_notes": pd.NA,
                        "sample_count_reference_8h": 100,
                        "event_count_reference_8h": 20,
                        "non_event_count_reference_8h": 80,
                        "event_rate_reference_8h": 0.20,
                        "auroc_reference_8h": 0.85,
                        "auprc_reference_8h": 0.32,
                        "calibration_intercept_reference_8h": -3.70,
                        "calibration_slope_reference_8h": 1.16,
                        "brier_score_reference_8h": 0.13,
                        "sample_count_delta_vs_8h": 0,
                        "event_count_delta_vs_8h": 0,
                        "non_event_count_delta_vs_8h": 0,
                        "event_rate_delta_vs_8h": 0.0,
                        "auroc_delta_vs_8h": 0.0,
                        "auprc_delta_vs_8h": 0.0,
                        "calibration_intercept_delta_vs_8h": 0.0,
                        "calibration_slope_delta_vs_8h": 0.0,
                        "brier_score_delta_vs_8h": 0.0,
                    },
                    {
                        "model_name": "xgboost",
                        "horizon_h": 24,
                        "aggregation": "16h",
                        "selected_split": "test",
                        "selected_split_evaluable": True,
                        "selection_reason": "first_binary_evaluable_split_in_priority_order",
                        "sample_count": 60,
                        "event_count": 12,
                        "non_event_count": 48,
                        "event_rate": 0.20,
                        "auroc": 0.845,
                        "auprc": 0.29,
                        "calibration_intercept": -3.68,
                        "calibration_slope": 1.17,
                        "brier_score": 0.131,
                        "binary_metrics_evaluable": True,
                        "finite_prediction_count": 60,
                        "metric_notes": pd.NA,
                        "sample_count_reference_8h": 100,
                        "event_count_reference_8h": 20,
                        "non_event_count_reference_8h": 80,
                        "event_rate_reference_8h": 0.20,
                        "auroc_reference_8h": 0.85,
                        "auprc_reference_8h": 0.32,
                        "calibration_intercept_reference_8h": -3.70,
                        "calibration_slope_reference_8h": 1.16,
                        "brier_score_reference_8h": 0.13,
                        "sample_count_delta_vs_8h": -40,
                        "event_count_delta_vs_8h": -8,
                        "non_event_count_delta_vs_8h": -32,
                        "event_rate_delta_vs_8h": 0.0,
                        "auroc_delta_vs_8h": -0.005,
                        "auprc_delta_vs_8h": -0.03,
                        "calibration_intercept_delta_vs_8h": 0.02,
                        "calibration_slope_delta_vs_8h": 0.01,
                        "brier_score_delta_vs_8h": 0.001,
                    },
                    {
                        "model_name": "xgboost",
                        "horizon_h": 24,
                        "aggregation": "24h",
                        "selected_split": "test",
                        "selected_split_evaluable": True,
                        "selection_reason": "first_binary_evaluable_split_in_priority_order",
                        "sample_count": 40,
                        "event_count": 8,
                        "non_event_count": 32,
                        "event_rate": 0.20,
                        "auroc": 0.836,
                        "auprc": 0.25,
                        "calibration_intercept": -3.60,
                        "calibration_slope": 1.13,
                        "brier_score": 0.132,
                        "binary_metrics_evaluable": True,
                        "finite_prediction_count": 40,
                        "metric_notes": pd.NA,
                        "sample_count_reference_8h": 100,
                        "event_count_reference_8h": 20,
                        "non_event_count_reference_8h": 80,
                        "event_rate_reference_8h": 0.20,
                        "auroc_reference_8h": 0.85,
                        "auprc_reference_8h": 0.32,
                        "calibration_intercept_reference_8h": -3.70,
                        "calibration_slope_reference_8h": 1.16,
                        "brier_score_reference_8h": 0.13,
                        "sample_count_delta_vs_8h": -60,
                        "event_count_delta_vs_8h": -12,
                        "non_event_count_delta_vs_8h": -48,
                        "event_rate_delta_vs_8h": 0.0,
                        "auroc_delta_vs_8h": -0.014,
                        "auprc_delta_vs_8h": -0.07,
                        "calibration_intercept_delta_vs_8h": 0.10,
                        "calibration_slope_delta_vs_8h": -0.03,
                        "brier_score_delta_vs_8h": 0.002,
                    },
                ]
            ).to_csv(comparison_root / "reporting_metric_summary.csv", index=False)

            pd.DataFrame(
                [
                    {
                        "model_name": "logistic_regression",
                        "horizon_h": 24,
                        "aggregation": "8h",
                        "selected_split": "test",
                        "selected_split_evaluable": True,
                        "calibration_intercept": -0.10,
                        "calibration_slope": 0.97,
                        "brier_score": 0.02,
                        "calibration_intercept_delta_vs_8h": 0.0,
                        "calibration_slope_delta_vs_8h": 0.0,
                        "brier_score_delta_vs_8h": 0.0,
                        "binary_metrics_evaluable": True,
                        "metric_notes": pd.NA,
                    },
                    {
                        "model_name": "logistic_regression",
                        "horizon_h": 24,
                        "aggregation": "16h",
                        "selected_split": "test",
                        "selected_split_evaluable": True,
                        "calibration_intercept": -0.12,
                        "calibration_slope": 0.96,
                        "brier_score": 0.021,
                        "calibration_intercept_delta_vs_8h": -0.02,
                        "calibration_slope_delta_vs_8h": -0.01,
                        "brier_score_delta_vs_8h": 0.001,
                        "binary_metrics_evaluable": True,
                        "metric_notes": pd.NA,
                    },
                    {
                        "model_name": "logistic_regression",
                        "horizon_h": 24,
                        "aggregation": "24h",
                        "selected_split": "test",
                        "selected_split_evaluable": True,
                        "calibration_intercept": -0.13,
                        "calibration_slope": 0.95,
                        "brier_score": 0.0215,
                        "calibration_intercept_delta_vs_8h": -0.03,
                        "calibration_slope_delta_vs_8h": -0.02,
                        "brier_score_delta_vs_8h": 0.0015,
                        "binary_metrics_evaluable": True,
                        "metric_notes": pd.NA,
                    },
                ]
            ).to_csv(comparison_root / "calibration_summary.csv", index=False)

            pd.DataFrame(
                [
                    {
                        "aggregation": "8h",
                        "model_name": "logistic_regression",
                        "horizon_h": 24,
                        "selected_split": "test",
                        "sample_count": 100,
                        "event_count": 20,
                        "lower_half_event_rate": 0.01,
                        "upper_half_event_rate": 0.04,
                        "upper_half_event_share": 0.86,
                        "top_bin_observed_mortality": 0.12,
                        "bottom_bin_observed_mortality": 0.003,
                        "structure_ordered": True,
                    },
                    {
                        "aggregation": "16h",
                        "model_name": "logistic_regression",
                        "horizon_h": 24,
                        "selected_split": "test",
                        "sample_count": 60,
                        "event_count": 12,
                        "lower_half_event_rate": 0.011,
                        "upper_half_event_rate": 0.039,
                        "upper_half_event_share": 0.85,
                        "top_bin_observed_mortality": 0.118,
                        "bottom_bin_observed_mortality": 0.0035,
                        "structure_ordered": True,
                    },
                    {
                        "aggregation": "24h",
                        "model_name": "logistic_regression",
                        "horizon_h": 24,
                        "selected_split": "test",
                        "sample_count": 40,
                        "event_count": 8,
                        "lower_half_event_rate": 0.012,
                        "upper_half_event_rate": 0.040,
                        "upper_half_event_share": 0.87,
                        "top_bin_observed_mortality": 0.117,
                        "bottom_bin_observed_mortality": 0.0036,
                        "structure_ordered": True,
                    },
                    {
                        "aggregation": "8h",
                        "model_name": "xgboost",
                        "horizon_h": 24,
                        "selected_split": "test",
                        "sample_count": 100,
                        "event_count": 20,
                        "lower_half_event_rate": 0.005,
                        "upper_half_event_rate": 0.040,
                        "upper_half_event_share": 0.89,
                        "top_bin_observed_mortality": 0.136,
                        "bottom_bin_observed_mortality": 0.0015,
                        "structure_ordered": True,
                    },
                    {
                        "aggregation": "16h",
                        "model_name": "xgboost",
                        "horizon_h": 24,
                        "selected_split": "test",
                        "sample_count": 60,
                        "event_count": 12,
                        "lower_half_event_rate": 0.004,
                        "upper_half_event_rate": 0.041,
                        "upper_half_event_share": 0.91,
                        "top_bin_observed_mortality": 0.132,
                        "bottom_bin_observed_mortality": 0.0018,
                        "structure_ordered": True,
                    },
                    {
                        "aggregation": "24h",
                        "model_name": "xgboost",
                        "horizon_h": 24,
                        "selected_split": "test",
                        "sample_count": 40,
                        "event_count": 8,
                        "lower_half_event_rate": 0.006,
                        "upper_half_event_rate": 0.039,
                        "upper_half_event_share": 0.88,
                        "top_bin_observed_mortality": 0.127,
                        "bottom_bin_observed_mortality": 0.0009,
                        "structure_ordered": True,
                    },
                ]
            ).to_csv(comparison_root / "mortality_risk_structure_summary.csv", index=False)

            pd.DataFrame(
                [
                    {
                        "horizon_h": 24,
                        "model_name": "logistic_regression",
                        "aggregation": "8h",
                        "n_fatal_last_points": 100,
                        "n_hard_cases": 20,
                        "pct_fatal_hard_cases": 0.20,
                        "nonfatal_q75_threshold": 0.015,
                        "n_fatal_last_points_reference_8h": 100,
                        "n_hard_cases_reference_8h": 20,
                        "pct_fatal_hard_cases_reference_8h": 0.20,
                        "nonfatal_q75_threshold_reference_8h": 0.015,
                        "n_fatal_last_points_delta_vs_8h": 0,
                        "n_hard_cases_delta_vs_8h": 0,
                        "pct_fatal_hard_cases_delta_vs_8h": 0.0,
                        "nonfatal_q75_threshold_delta_vs_8h": 0.0,
                    },
                    {
                        "horizon_h": 24,
                        "model_name": "logistic_regression",
                        "aggregation": "16h",
                        "n_fatal_last_points": 100,
                        "n_hard_cases": 24,
                        "pct_fatal_hard_cases": 0.24,
                        "nonfatal_q75_threshold": 0.016,
                        "n_fatal_last_points_reference_8h": 100,
                        "n_hard_cases_reference_8h": 20,
                        "pct_fatal_hard_cases_reference_8h": 0.20,
                        "nonfatal_q75_threshold_reference_8h": 0.015,
                        "n_fatal_last_points_delta_vs_8h": 0,
                        "n_hard_cases_delta_vs_8h": 4,
                        "pct_fatal_hard_cases_delta_vs_8h": 0.04,
                        "nonfatal_q75_threshold_delta_vs_8h": 0.001,
                    },
                    {
                        "horizon_h": 24,
                        "model_name": "logistic_regression",
                        "aggregation": "24h",
                        "n_fatal_last_points": 100,
                        "n_hard_cases": 25,
                        "pct_fatal_hard_cases": 0.25,
                        "nonfatal_q75_threshold": 0.0165,
                        "n_fatal_last_points_reference_8h": 100,
                        "n_hard_cases_reference_8h": 20,
                        "pct_fatal_hard_cases_reference_8h": 0.20,
                        "nonfatal_q75_threshold_reference_8h": 0.015,
                        "n_fatal_last_points_delta_vs_8h": 0,
                        "n_hard_cases_delta_vs_8h": 5,
                        "pct_fatal_hard_cases_delta_vs_8h": 0.05,
                        "nonfatal_q75_threshold_delta_vs_8h": 0.0015,
                    },
                ]
            ).to_csv(comparison_root / "hard_case_prevalence_summary.csv", index=False)

            pd.DataFrame(
                [
                    {
                        "aggregation_a": "8h",
                        "aggregation_b": "16h",
                        "matched_fatal_n": 100,
                        "hard_n_aggregation_a": 20,
                        "hard_n_aggregation_b": 24,
                        "intersection_n": 16,
                        "union_n": 28,
                        "jaccard_index": 16 / 28,
                    },
                    {
                        "aggregation_a": "8h",
                        "aggregation_b": "24h",
                        "matched_fatal_n": 100,
                        "hard_n_aggregation_a": 20,
                        "hard_n_aggregation_b": 25,
                        "intersection_n": 14,
                        "union_n": 31,
                        "jaccard_index": 14 / 31,
                    },
                    {
                        "aggregation_a": "16h",
                        "aggregation_b": "24h",
                        "matched_fatal_n": 100,
                        "hard_n_aggregation_a": 24,
                        "hard_n_aggregation_b": 25,
                        "intersection_n": 19,
                        "union_n": 30,
                        "jaccard_index": 19 / 30,
                    },
                ]
            ).to_csv(comparison_root / "logistic_24h_hard_case_pairwise_overlap.csv", index=False)

            pd.DataFrame(
                [
                    {
                        "aggregation_from": "8h",
                        "aggregation_to": "16h",
                        "matched_fatal_n": 100,
                        "hard_n_from": 20,
                        "hard_n_to": 24,
                        "intersection_n": 16,
                        "overlap_from_A_to_B": 0.80,
                    },
                    {
                        "aggregation_from": "16h",
                        "aggregation_to": "8h",
                        "matched_fatal_n": 100,
                        "hard_n_from": 24,
                        "hard_n_to": 20,
                        "intersection_n": 16,
                        "overlap_from_A_to_B": 16 / 24,
                    },
                    {
                        "aggregation_from": "8h",
                        "aggregation_to": "24h",
                        "matched_fatal_n": 100,
                        "hard_n_from": 20,
                        "hard_n_to": 25,
                        "intersection_n": 14,
                        "overlap_from_A_to_B": 0.70,
                    },
                    {
                        "aggregation_from": "24h",
                        "aggregation_to": "8h",
                        "matched_fatal_n": 100,
                        "hard_n_from": 25,
                        "hard_n_to": 20,
                        "intersection_n": 14,
                        "overlap_from_A_to_B": 14 / 25,
                    },
                    {
                        "aggregation_from": "16h",
                        "aggregation_to": "24h",
                        "matched_fatal_n": 100,
                        "hard_n_from": 24,
                        "hard_n_to": 25,
                        "intersection_n": 19,
                        "overlap_from_A_to_B": 19 / 24,
                    },
                    {
                        "aggregation_from": "24h",
                        "aggregation_to": "16h",
                        "matched_fatal_n": 100,
                        "hard_n_from": 25,
                        "hard_n_to": 24,
                        "intersection_n": 19,
                        "overlap_from_A_to_B": 19 / 25,
                    },
                ]
            ).to_csv(comparison_root / "logistic_24h_hard_case_directional_overlap.csv", index=False)

            pd.DataFrame(
                [
                    {"hard_case_aggregation_n": 0, "fatal_stay_count": 50, "fatal_stay_share": 0.50},
                    {"hard_case_aggregation_n": 1, "fatal_stay_count": 10, "fatal_stay_share": 0.10},
                    {"hard_case_aggregation_n": 2, "fatal_stay_count": 15, "fatal_stay_share": 0.15},
                    {"hard_case_aggregation_n": 3, "fatal_stay_count": 25, "fatal_stay_share": 0.25},
                ]
            ).to_csv(
                comparison_root / "logistic_24h_hard_case_persistence_distribution.csv",
                index=False,
            )

            output_path = write_temporal_sensitivity_interpretation_memo(
                comparison_root=comparison_root,
            )

            self.assertEqual(
                output_path,
                comparison_root / "temporal_aggregation_sensitivity_interpretation.md",
            )
            text = output_path.read_text()
            self.assertIn("# ASIC Temporal Aggregation Sensitivity Interpretation", text)
            self.assertIn("Classification: `partially weakened under coarsening`", text)
            self.assertIn("Logistic 24h discrimination", text)
            manifest_payload = json.loads((comparison_root / "run_manifest.json").read_text())
            self.assertEqual(
                manifest_payload["artifact_paths"]["temporal_aggregation_sensitivity_interpretation"],
                str(output_path.resolve()),
            )
