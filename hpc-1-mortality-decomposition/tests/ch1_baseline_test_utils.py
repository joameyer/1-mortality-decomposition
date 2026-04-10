from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_primary_baseline_fixture(root: Path) -> dict[str, Path]:
    artifact_root = root / "chapter1"
    model_ready_path = artifact_root / "model_ready" / "chapter1_primary_model_ready_dataset.csv"
    feature_set_definition_path = (
        artifact_root / "feature_sets" / "chapter1_feature_set_definition.csv"
    )
    labels_path = artifact_root / "labels" / "chapter1_proxy_horizon_labels.csv"
    stay_split_assignments_path = (
        artifact_root / "splits" / "chapter1_stay_split_assignments.csv"
    )
    standardized_input_dir = root / "standardized_inputs"
    blocked_dynamic_features_path = (
        standardized_input_dir / "blocked" / "asic_8h_blocked_dynamic_features.csv"
    )
    mech_vent_episode_level_path = (
        standardized_input_dir / "qc" / "mech_vent_ge_24h_episode_level.csv"
    )

    feature_set_definition = pd.DataFrame(
        [
            {
                "feature_set_name": "primary",
                "feature_name": "heart_rate_median",
                "base_variable": "heart_rate",
                "statistic": "median",
                "selected_for_model": True,
                "available_in_blocked_schema": True,
                "base_feature_available_in_blocked_schema": True,
                "base_feature_present_in_retained_data": True,
            },
            {
                "feature_set_name": "primary",
                "feature_name": "creatinine_median",
                "base_variable": "creatinine",
                "statistic": "median",
                "selected_for_model": True,
                "available_in_blocked_schema": True,
                "base_feature_available_in_blocked_schema": True,
                "base_feature_present_in_retained_data": True,
            },
            {
                "feature_set_name": "primary",
                "feature_name": "heart_rate_mean",
                "base_variable": "heart_rate",
                "statistic": "mean",
                "selected_for_model": True,
                "available_in_blocked_schema": True,
                "base_feature_available_in_blocked_schema": True,
                "base_feature_present_in_retained_data": True,
            },
            {
                "feature_set_name": "extended",
                "feature_name": "pct_median",
                "base_variable": "pct",
                "statistic": "median",
                "selected_for_model": True,
                "available_in_blocked_schema": True,
                "base_feature_available_in_blocked_schema": True,
                "base_feature_present_in_retained_data": True,
            },
        ]
    )

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

    proxy_horizon_labels = pd.DataFrame(
        [
            {
                "instance_id": "stay_a__b0__h8",
                "stay_id_global": "stay_a",
                "hospital_id": "H1",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
                "horizon_h": 8,
                "label_value": 1,
                "proxy_horizon_labelable": True,
                "unlabeled_reason": pd.NA,
            },
            {
                "instance_id": "stay_b__b0__h8",
                "stay_id_global": "stay_b",
                "hospital_id": "H1",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
                "horizon_h": 8,
                "label_value": 0,
                "proxy_horizon_labelable": True,
                "unlabeled_reason": pd.NA,
            },
            {
                "instance_id": "stay_c__b0__h8",
                "stay_id_global": "stay_c",
                "hospital_id": "H1",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
                "horizon_h": 8,
                "label_value": 1,
                "proxy_horizon_labelable": True,
                "unlabeled_reason": pd.NA,
            },
            {
                "instance_id": "stay_d__b0__h8",
                "stay_id_global": "stay_d",
                "hospital_id": "H2",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
                "horizon_h": 8,
                "label_value": 0,
                "proxy_horizon_labelable": True,
                "unlabeled_reason": pd.NA,
            },
            {
                "instance_id": "stay_e__b0__h8",
                "stay_id_global": "stay_e",
                "hospital_id": "H2",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
                "horizon_h": 8,
                "label_value": pd.NA,
                "proxy_horizon_labelable": False,
                "unlabeled_reason": "non_survivor_proxy_end_not_within_horizon",
            },
            {
                "instance_id": "stay_a__b0__h24",
                "stay_id_global": "stay_a",
                "hospital_id": "H1",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "label_value": 1,
                "proxy_horizon_labelable": True,
                "unlabeled_reason": pd.NA,
            },
            {
                "instance_id": "stay_b__b0__h24",
                "stay_id_global": "stay_b",
                "hospital_id": "H1",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "label_value": 0,
                "proxy_horizon_labelable": True,
                "unlabeled_reason": pd.NA,
            },
            {
                "instance_id": "stay_c__b0__h24",
                "stay_id_global": "stay_c",
                "hospital_id": "H1",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "label_value": 0,
                "proxy_horizon_labelable": True,
                "unlabeled_reason": pd.NA,
            },
            {
                "instance_id": "stay_d__b0__h24",
                "stay_id_global": "stay_d",
                "hospital_id": "H2",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "label_value": 0,
                "proxy_horizon_labelable": True,
                "unlabeled_reason": pd.NA,
            },
            {
                "instance_id": "stay_e__b0__h24",
                "stay_id_global": "stay_e",
                "hospital_id": "H2",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "label_value": pd.NA,
                "proxy_horizon_labelable": False,
                "unlabeled_reason": "non_survivor_proxy_end_not_within_horizon",
            },
        ]
    )

    stay_split_assignments = pd.DataFrame(
        [
            {"stay_id_global": "stay_a", "hospital_id": "H1", "split": "train"},
            {"stay_id_global": "stay_b", "hospital_id": "H1", "split": "train"},
            {"stay_id_global": "stay_c", "hospital_id": "H1", "split": "validation"},
            {"stay_id_global": "stay_d", "hospital_id": "H2", "split": "test"},
            {"stay_id_global": "stay_e", "hospital_id": "H2", "split": "test"},
        ]
    )

    blocked_dynamic_features = pd.DataFrame(
        [
            {
                "stay_id_global": "stay_a",
                "hospital_id": "H1",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
                "heart_rate_median": 90,
                "creatinine_median": 1.2,
                "heart_rate_mean": 91,
            },
            {
                "stay_id_global": "stay_b",
                "hospital_id": "H1",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
                "heart_rate_median": 70,
                "creatinine_median": 0.8,
                "heart_rate_mean": 72,
            },
            {
                "stay_id_global": "stay_c",
                "hospital_id": "H1",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
                "heart_rate_median": 95,
                "creatinine_median": 1.5,
                "heart_rate_mean": 96,
            },
            {
                "stay_id_global": "stay_d",
                "hospital_id": "H2",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
                "heart_rate_median": 65,
                "creatinine_median": 0.7,
                "heart_rate_mean": 66,
            },
            {
                "stay_id_global": "stay_e",
                "hospital_id": "H2",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
                "heart_rate_median": 88,
                "creatinine_median": 1.4,
                "heart_rate_mean": 89,
            },
        ]
    )

    mech_vent_episode_level = pd.DataFrame(
        columns=["stay_id_global", "hospital_id", "episode_start_time", "episode_end_time"]
    )

    for path, frame in (
        (model_ready_path, model_ready),
        (feature_set_definition_path, feature_set_definition),
        (labels_path, proxy_horizon_labels),
        (stay_split_assignments_path, stay_split_assignments),
        (blocked_dynamic_features_path, blocked_dynamic_features),
        (mech_vent_episode_level_path, mech_vent_episode_level),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    return {
        "artifact_root": artifact_root,
        "input_dataset_path": model_ready_path,
        "feature_set_definition_path": feature_set_definition_path,
        "output_dir": root / "baselines",
        "standardized_input_dir": standardized_input_dir,
    }
