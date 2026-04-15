from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from chapter1_mortality_decomposition.temporal_blocks import (
    build_asic_temporal_block_artifacts,
)
from chapter1_mortality_decomposition.temporal_preview import (
    DEFAULT_NOTEBOOK_PATH,
    DEFAULT_PREVIEW_OUTPUT_ROOT,
    _build_comparison_package,
    _notebook_payload,
    run_asic_temporal_aggregation_preview,
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
        self.assertIn("cluster_export_relative", source)
        self.assertIn("resolve_optional_artifact_path", source)
        self.assertIn("cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians", source)
        self.assertIn("cluster-results/chapter1_true_results/temporal_preview/asic/aggregation_16h/comparison/aggregation_comparison_metrics.csv", source)
        self.assertIn('"repo_relative"', source)
        self.assertNotIn(str(repo_root), source)


class TemporalPreviewRunWiringTests(TestCase):
    def test_temporal_preview_writes_proxy_labels_and_passes_preprocessing_root_to_baselines(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_root = tmp_path / "temporal_preview_output"
            standardized_input_dir = tmp_path / "standardized_inputs"
            frozen_chapter1_dir = tmp_path / "frozen_chapter1"
            eight_hour_evaluation_root = tmp_path / "evaluation_8h"
            notebook_path = output_root / "comparison" / "preview_review.ipynb"

            (frozen_chapter1_dir / "splits").mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"stay_id_global": "stay_a", "split": "train"}]).to_csv(
                frozen_chapter1_dir / "splits" / "chapter1_stay_split_assignments.csv",
                index=False,
            )

            config = SimpleNamespace(
                input_dir=standardized_input_dir,
                input_format="csv",
                min_required_core_groups=3,
                split_random_seed=13,
                feature_set_config_path=tmp_path / "feature_set_config.json",
            )

            cohort_result = SimpleNamespace(
                table=pd.DataFrame(
                    [
                        {
                            "stay_id_global": "stay_a",
                            "hospital_id": "H1",
                            "icu_mortality": 1,
                        }
                    ]
                ),
                retained_stays=pd.DataFrame([{"stay_id_global": "stay_a"}]),
            )
            valid_instances_result = SimpleNamespace(
                valid_instances=pd.DataFrame(
                    [
                        {
                            "stay_id_global": "stay_a",
                            "hospital_id": "H1",
                            "instance_id": "stay_a__24h__1",
                            "block_index": 1,
                            "prediction_time_h": 16,
                            "future_window_end_h": 40,
                            "horizon_h": 24,
                            "icu_end_time_proxy_hours": 30,
                        }
                    ]
                ),
                counts_by_horizon=pd.DataFrame(
                    [{"horizon_h": 24, "instance_count": 1}]
                ),
            )
            labels_result = SimpleNamespace(
                labels=pd.DataFrame(
                    [
                        {
                            "stay_id_global": "stay_a",
                            "hospital_id": "H1",
                            "instance_id": "stay_a__24h__1",
                            "block_index": 1,
                            "prediction_time_h": 16,
                            "future_window_end_h": 40,
                            "horizon_h": 24,
                            "icu_end_time_proxy_hours": 30,
                            "label_name": "proxy_within_horizon_icu_mortality",
                            "label_definition_id": "proxy_within_horizon_icu_mortality_v1",
                            "label_definition_status": "approved_proxy",
                            "event_time_proxy_h": 30,
                            "proxy_horizon_labelable": True,
                            "label_value": 1,
                            "label_available": True,
                            "unlabeled_reason": pd.NA,
                            "label_semantics": "test",
                        }
                    ]
                ),
                usable_labels=pd.DataFrame(
                    [
                        {
                            "stay_id_global": "stay_a",
                            "hospital_id": "H1",
                            "instance_id": "stay_a__24h__1",
                            "block_index": 1,
                            "prediction_time_h": 16,
                            "future_window_end_h": 40,
                            "horizon_h": 24,
                            "icu_end_time_proxy_hours": 30,
                            "label_name": "proxy_within_horizon_icu_mortality",
                            "label_definition_id": "proxy_within_horizon_icu_mortality_v1",
                            "label_definition_status": "approved_proxy",
                            "event_time_proxy_h": 30,
                            "proxy_horizon_labelable": True,
                            "label_value": 1,
                            "label_available": True,
                            "unlabeled_reason": pd.NA,
                            "label_semantics": "test",
                        }
                    ]
                ),
                summary_by_horizon=pd.DataFrame(
                    [
                        {
                            "horizon_h": 24,
                            "total_valid_prediction_instances": 1,
                            "labelable_instances": 1,
                            "positive_labels": 1,
                            "negative_labels": 0,
                            "unlabeled_instances": 0,
                        }
                    ]
                ),
            )
            block_artifacts = SimpleNamespace(
                stay_block_counts=pd.DataFrame(
                    [
                        {
                            "stay_id_global": "stay_a",
                            "hospital_id": "H1",
                            "completed_block_count": 1,
                        }
                    ]
                ),
                block_index=pd.DataFrame(
                    [
                        {
                            "stay_id_global": "stay_a",
                            "hospital_id": "H1",
                            "block_index": 1,
                            "prediction_time_h": 16,
                        }
                    ]
                ),
                blocked_dynamic_features=pd.DataFrame(
                    [
                        {
                            "stay_id_global": "stay_a",
                            "hospital_id": "H1",
                            "block_index": 1,
                            "prediction_time_h": 16,
                            "heart_rate_mean": 80.0,
                        }
                    ]
                ),
            )
            feature_set_definition = pd.DataFrame(
                [{"feature_set_name": "primary", "feature_name": "heart_rate_mean"}]
            )
            model_ready_result = SimpleNamespace(
                table=pd.DataFrame(
                    [
                        {
                            "stay_id_global": "stay_a",
                            "hospital_id": "H1",
                            "horizon_h": 24,
                            "split": "train",
                            "label_value": 1,
                            "heart_rate_mean": 80.0,
                        }
                    ]
                ),
                readiness_summary=pd.DataFrame([{"summary": "ok"}]),
                feature_availability_by_horizon=pd.DataFrame([{"horizon_h": 24, "feature_count": 1}]),
                split_summary=pd.DataFrame([{"split": "train", "sample_count": 1}]),
                split_verification_summary=pd.DataFrame([{"check": "ok"}]),
                locf_feature_summary=pd.DataFrame([{"feature_name": "heart_rate_mean"}]),
                ventilator_locf_summary=pd.DataFrame([{"summary": "ok"}]),
                missingness_by_hospital_and_family=pd.DataFrame([{"hospital_id": "H1"}]),
                carry_forward_verification_summary=pd.DataFrame([{"check": "ok"}]),
            )

            def write_dataframe_stub(frame: pd.DataFrame, path: Path, output_format: str = "csv") -> Path:
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(path, index=False)
                return path

            def write_text_stub(text: str, path: Path) -> Path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
                return path

            def write_json_stub(payload: object, path: Path) -> Path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n")
                return path

            def write_block_artifacts_stub(*args, **kwargs) -> dict[str, Path]:
                blocked_dir = kwargs["output_dir"]
                blocked_dir.mkdir(parents=True, exist_ok=True)
                blocked_path = blocked_dir / "asic_16h_blocked_dynamic_features.csv"
                block_artifacts.blocked_dynamic_features.to_csv(blocked_path, index=False)
                return {
                    "blocked_dynamic_features": blocked_path,
                    "stay_block_counts": blocked_dir / "asic_16h_stay_block_counts.csv",
                    "block_index": blocked_dir / "asic_16h_block_index.csv",
                }

            with (
                patch(
                    "chapter1_mortality_decomposition.temporal_preview.load_chapter1_run_config",
                    return_value=config,
                ),
                patch(
                    "chapter1_mortality_decomposition.temporal_preview._load_standardized_asic_inputs",
                    return_value={
                        "dynamic_harmonized": pd.DataFrame([{"stay_id_global": "stay_a"}]),
                        "static_harmonized": pd.DataFrame([{"stay_id_global": "stay_a"}]),
                        "reference_stay_block_counts": pd.DataFrame([{"stay_id_global": "stay_a"}]),
                        "mech_vent_stay_level_qc": pd.DataFrame([{"stay_id_global": "stay_a"}]),
                        "mech_vent_episode_level": pd.DataFrame([{"stay_id_global": "stay_a"}]),
                    },
                ),
                patch(
                    "chapter1_mortality_decomposition.temporal_preview.build_asic_temporal_block_artifacts",
                    return_value=block_artifacts,
                ),
                patch(
                    "chapter1_mortality_decomposition.temporal_preview.write_asic_temporal_block_artifacts",
                    side_effect=write_block_artifacts_stub,
                ),
                patch(
                    "chapter1_mortality_decomposition.temporal_preview.build_chapter1_cohort",
                    return_value=cohort_result,
                ),
                patch(
                    "chapter1_mortality_decomposition.temporal_preview.build_chapter1_valid_instances",
                    return_value=valid_instances_result,
                ),
                patch(
                    "chapter1_mortality_decomposition.temporal_preview.build_chapter1_proxy_horizon_labels",
                    return_value=labels_result,
                ),
                patch(
                    "chapter1_mortality_decomposition.temporal_preview._build_chapter1_cohort_summary",
                    return_value=pd.DataFrame([{"summary": "ok"}]),
                ),
                patch(
                    "chapter1_mortality_decomposition.temporal_preview._build_chapter1_verification_summary",
                    return_value=pd.DataFrame([{"summary": "ok"}]),
                ),
                patch(
                    "chapter1_mortality_decomposition.temporal_preview._build_frozen_split_alignment_summary",
                    return_value=(
                        pd.DataFrame([{"summary": "ok"}]),
                        pd.DataFrame([{"stay_id_global": "stay_a", "split": "train"}]),
                    ),
                ),
                patch(
                    "chapter1_mortality_decomposition.temporal_preview.build_chapter1_feature_set_definition",
                    return_value=(feature_set_definition, pd.DataFrame([{"summary": "ok"}])),
                ),
                patch(
                    "chapter1_mortality_decomposition.temporal_preview.build_chapter1_model_ready_dataset",
                    return_value=model_ready_result,
                ),
                patch(
                    "chapter1_mortality_decomposition.temporal_preview.write_dataframe",
                    side_effect=write_dataframe_stub,
                ),
                patch(
                    "chapter1_mortality_decomposition.temporal_preview.write_text",
                    side_effect=write_text_stub,
                ),
                patch(
                    "chapter1_mortality_decomposition.temporal_preview._write_json",
                    side_effect=write_json_stub,
                ),
                patch(
                    "chapter1_mortality_decomposition.temporal_preview.run_asic_primary_logistic_regression",
                    return_value=SimpleNamespace(),
                ) as logistic_mock,
                patch(
                    "chapter1_mortality_decomposition.temporal_preview.run_asic_primary_xgboost",
                    return_value=SimpleNamespace(),
                ) as xgboost_mock,
                patch(
                    "chapter1_mortality_decomposition.temporal_preview.run_asic_baseline_evaluation",
                    return_value=None,
                ),
                patch(
                    "chapter1_mortality_decomposition.temporal_preview._build_comparison_package",
                    return_value=SimpleNamespace(
                        comparison_table_path=output_root / "comparison" / "aggregation_comparison_metrics.csv",
                        note_path=output_root / "comparison" / "preview_note.md",
                        notebook_path=notebook_path,
                        figure_paths=(),
                    ),
                ),
            ):
                run_asic_temporal_aggregation_preview(
                    output_root=output_root,
                    frozen_chapter1_dir=frozen_chapter1_dir,
                    eight_hour_evaluation_root=eight_hour_evaluation_root,
                    run_config_path=tmp_path / "ch1_run_config.json",
                )

            preprocessing_root = output_root / "preprocessing"
            proxy_labels_path = preprocessing_root / "labels" / "chapter1_proxy_horizon_labels.csv"
            self.assertTrue(proxy_labels_path.exists())
            self.assertEqual(
                logistic_mock.call_args.kwargs["preprocessing_root"],
                preprocessing_root,
            )
            self.assertEqual(
                xgboost_mock.call_args.kwargs["preprocessing_root"],
                preprocessing_root,
            )
