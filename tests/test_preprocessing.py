from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chapter1_mortality_decomposition.artifacts import Chapter1InputTables
from chapter1_mortality_decomposition.pipeline import build_chapter1_dataset


def build_synthetic_inputs() -> Chapter1InputTables:
    static_harmonized = pd.DataFrame(
        [
            {
                "stay_id_global": "stay_a",
                "hospital_id": "H1",
                "icu_readmit": 0,
                "icu_mortality": 1,
                "icd10_codes": "A00",
            },
            {
                "stay_id_global": "stay_b",
                "hospital_id": "H1",
                "icu_readmit": 1,
                "icu_mortality": 0,
                "icd10_codes": "B00",
            },
            {
                "stay_id_global": "stay_c",
                "hospital_id": "H1",
                "icu_readmit": pd.NA,
                "icu_mortality": 0,
                "icd10_codes": "C00",
            },
            {
                "stay_id_global": "stay_e",
                "hospital_id": "H1",
                "icu_readmit": 0,
                "icu_mortality": 0,
                "icd10_codes": "E00",
            },
            {
                "stay_id_global": "stay_f",
                "hospital_id": "H1",
                "icu_readmit": 0,
                "icu_mortality": pd.NA,
                "icd10_codes": "F00",
            },
            {
                "stay_id_global": "stay_g",
                "hospital_id": "H1",
                "icu_readmit": 0,
                "icu_mortality": 0,
                "icd10_codes": "G00",
            },
            {
                "stay_id_global": "stay_d",
                "hospital_id": "H2",
                "icu_readmit": 0,
                "icu_mortality": 1,
                "icd10_codes": "D00",
            },
            {
                "stay_id_global": "stay_h",
                "hospital_id": "H1",
                "icu_readmit": 0,
                "icu_mortality": 0,
                "icd10_codes": "H00",
            },
        ]
    )

    dynamic_harmonized = pd.DataFrame(
        [
            {
                "stay_id_global": "stay_a",
                "hospital_id": "H1",
                "heart_rate": 88,
                "map": 72,
                "resp_rate": 18,
                "spo2": 97,
            },
            {
                "stay_id_global": "stay_b",
                "hospital_id": "H1",
                "heart_rate": 84,
                "map": 69,
                "resp_rate": 20,
                "spo2": 96,
            },
            {
                "stay_id_global": "stay_c",
                "hospital_id": "H1",
                "heart_rate": 90,
                "map": 74,
                "resp_rate": 21,
                "spo2": 95,
            },
            {
                "stay_id_global": "stay_f",
                "hospital_id": "H1",
                "heart_rate": 78,
                "map": 70,
                "resp_rate": 17,
                "spo2": 98,
            },
            {
                "stay_id_global": "stay_g",
                "hospital_id": "H1",
                "heart_rate": 76,
                "map": 68,
                "resp_rate": 16,
                "spo2": 99,
            },
            {
                "stay_id_global": "stay_d",
                "hospital_id": "H2",
                "heart_rate": 82,
                "map": pd.NA,
                "resp_rate": pd.NA,
                "spo2": pd.NA,
            },
            {
                "stay_id_global": "stay_h",
                "hospital_id": "H1",
                "heart_rate": 80,
                "map": 71,
                "resp_rate": 19,
                "spo2": 98,
            },
        ]
    )

    stay_block_counts = pd.DataFrame(
        [
            {
                "stay_id_global": "stay_a",
                "icu_admission_time": "2026-01-01 00:00:00",
                "icu_end_time_proxy": "2026-01-01 22:00:00",
                "icu_end_time_proxy_hours": 22,
            },
            {
                "stay_id_global": "stay_b",
                "icu_admission_time": "2026-01-01 00:00:00",
                "icu_end_time_proxy": "2026-01-02 06:00:00",
                "icu_end_time_proxy_hours": 30,
            },
            {
                "stay_id_global": "stay_c",
                "icu_admission_time": "2026-01-01 00:00:00",
                "icu_end_time_proxy": "2026-01-02 06:00:00",
                "icu_end_time_proxy_hours": 30,
            },
            {
                "stay_id_global": "stay_d",
                "icu_admission_time": "2026-01-01 00:00:00",
                "icu_end_time_proxy": "2026-01-02 06:00:00",
                "icu_end_time_proxy_hours": 30,
            },
            {
                "stay_id_global": "stay_e",
                "icu_admission_time": "2026-01-01 00:00:00",
                "icu_end_time_proxy": "2026-01-02 06:00:00",
                "icu_end_time_proxy_hours": 30,
            },
            {
                "stay_id_global": "stay_f",
                "icu_admission_time": "2026-01-01 00:00:00",
                "icu_end_time_proxy": "2026-01-02 06:00:00",
                "icu_end_time_proxy_hours": 30,
            },
            {
                "stay_id_global": "stay_g",
                "icu_admission_time": "2026-01-01 00:00:00",
                "icu_end_time_proxy": "2026-01-02 16:00:00",
                "icu_end_time_proxy_hours": 40,
            },
            {
                "stay_id_global": "stay_h",
                "icu_admission_time": "2026-01-01 00:00:00",
                "icu_end_time_proxy": "2026-01-03 22:00:00",
                "icu_end_time_proxy_hours": 70,
            },
        ]
    )

    block_index = pd.DataFrame(
        [
            {
                "stay_id_global": "stay_a",
                "hospital_id": "H1",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
            },
            {
                "stay_id_global": "stay_a",
                "hospital_id": "H1",
                "block_index": 1,
                "block_start_h": 8,
                "block_end_h": 16,
                "prediction_time_h": 16,
            },
            {
                "stay_id_global": "stay_b",
                "hospital_id": "H1",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
            },
            {
                "stay_id_global": "stay_c",
                "hospital_id": "H1",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
            },
            {
                "stay_id_global": "stay_d",
                "hospital_id": "H2",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
            },
            {
                "stay_id_global": "stay_f",
                "hospital_id": "H1",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
            },
            {
                "stay_id_global": "stay_g",
                "hospital_id": "H1",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
            },
            {
                "stay_id_global": "stay_g",
                "hospital_id": "H1",
                "block_index": 1,
                "block_start_h": 8,
                "block_end_h": 16,
                "prediction_time_h": 16,
            },
            {
                "stay_id_global": "stay_h",
                "hospital_id": "H1",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
            },
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
                "heart_rate_obs_count": 2,
                "heart_rate_mean": 87,
                "map_obs_count": 2,
                "map_mean": 72,
                "resp_rate_obs_count": 1,
                "resp_rate_mean": 18,
                "spo2_obs_count": 1,
                "spo2_mean": 97,
                "pct_obs_count": 1,
                "pct_mean": 0.5,
            },
            {
                "stay_id_global": "stay_a",
                "hospital_id": "H1",
                "block_index": 1,
                "block_start_h": 8,
                "block_end_h": 16,
                "prediction_time_h": 16,
                "heart_rate_obs_count": 1,
                "heart_rate_mean": 92,
                "map_obs_count": 1,
                "map_mean": 68,
                "resp_rate_obs_count": 1,
                "resp_rate_mean": 24,
                "spo2_obs_count": 1,
                "spo2_mean": 94,
                "pct_obs_count": 1,
                "pct_mean": 0.7,
            },
            {
                "stay_id_global": "stay_b",
                "hospital_id": "H1",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
                "heart_rate_obs_count": 1,
                "heart_rate_mean": 84,
                "map_obs_count": 1,
                "map_mean": 69,
                "resp_rate_obs_count": 1,
                "resp_rate_mean": 20,
                "spo2_obs_count": 1,
                "spo2_mean": 96,
            },
            {
                "stay_id_global": "stay_c",
                "hospital_id": "H1",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
                "heart_rate_obs_count": 1,
                "heart_rate_mean": 90,
                "map_obs_count": 1,
                "map_mean": 74,
                "resp_rate_obs_count": 1,
                "resp_rate_mean": 21,
                "spo2_obs_count": 1,
                "spo2_mean": 95,
            },
            {
                "stay_id_global": "stay_d",
                "hospital_id": "H2",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
                "heart_rate_obs_count": 1,
                "heart_rate_mean": 82,
                "map_obs_count": 0,
                "map_mean": pd.NA,
                "resp_rate_obs_count": 0,
                "resp_rate_mean": pd.NA,
                "spo2_obs_count": 0,
                "spo2_mean": pd.NA,
            },
            {
                "stay_id_global": "stay_f",
                "hospital_id": "H1",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
                "heart_rate_obs_count": 1,
                "heart_rate_mean": 78,
                "map_obs_count": 1,
                "map_mean": 70,
                "resp_rate_obs_count": 1,
                "resp_rate_mean": 17,
                "spo2_obs_count": 1,
                "spo2_mean": 98,
            },
            {
                "stay_id_global": "stay_g",
                "hospital_id": "H1",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
                "heart_rate_obs_count": 0,
                "heart_rate_mean": pd.NA,
                "map_obs_count": 0,
                "map_mean": pd.NA,
                "resp_rate_obs_count": 0,
                "resp_rate_mean": pd.NA,
                "spo2_obs_count": 0,
                "spo2_mean": pd.NA,
            },
            {
                "stay_id_global": "stay_g",
                "hospital_id": "H1",
                "block_index": 1,
                "block_start_h": 8,
                "block_end_h": 16,
                "prediction_time_h": 16,
                "heart_rate_obs_count": 1,
                "heart_rate_mean": 74,
                "map_obs_count": 1,
                "map_mean": 67,
                "resp_rate_obs_count": 1,
                "resp_rate_mean": 15,
                "spo2_obs_count": 1,
                "spo2_mean": 99,
            },
            {
                "stay_id_global": "stay_h",
                "hospital_id": "H1",
                "block_index": 0,
                "block_start_h": 0,
                "block_end_h": 8,
                "prediction_time_h": 8,
                "heart_rate_obs_count": 1,
                "heart_rate_mean": 80,
                "map_obs_count": 1,
                "map_mean": 71,
                "resp_rate_obs_count": 1,
                "resp_rate_mean": 19,
                "spo2_obs_count": 1,
                "spo2_mean": 98,
                "pct_obs_count": 1,
                "pct_mean": 0.3,
            },
        ]
    )

    mech_vent_stay_level_qc = pd.DataFrame(
        [
            {"stay_id_global": "stay_a", "hospital_id": "H1", "mech_vent_ge_24h_qc": True},
            {"stay_id_global": "stay_b", "hospital_id": "H1", "mech_vent_ge_24h_qc": True},
            {"stay_id_global": "stay_c", "hospital_id": "H1", "mech_vent_ge_24h_qc": True},
            {"stay_id_global": "stay_d", "hospital_id": "H2", "mech_vent_ge_24h_qc": True},
            {"stay_id_global": "stay_e", "hospital_id": "H1", "mech_vent_ge_24h_qc": True},
            {"stay_id_global": "stay_f", "hospital_id": "H1", "mech_vent_ge_24h_qc": True},
            {"stay_id_global": "stay_g", "hospital_id": "H1", "mech_vent_ge_24h_qc": True},
            {"stay_id_global": "stay_h", "hospital_id": "H1", "mech_vent_ge_24h_qc": False},
        ]
    )

    return Chapter1InputTables(
        static_harmonized=static_harmonized,
        dynamic_harmonized=dynamic_harmonized,
        block_index=block_index,
        blocked_dynamic_features=blocked_dynamic_features,
        stay_block_counts=stay_block_counts,
        mech_vent_stay_level_qc=mech_vent_stay_level_qc,
    )


def write_standardized_inputs(base_dir: Path, inputs: Chapter1InputTables) -> None:
    (base_dir / "static").mkdir(parents=True, exist_ok=True)
    (base_dir / "dynamic").mkdir(parents=True, exist_ok=True)
    (base_dir / "blocked").mkdir(parents=True, exist_ok=True)
    (base_dir / "qc").mkdir(parents=True, exist_ok=True)

    inputs.static_harmonized.to_csv(base_dir / "static" / "harmonized.csv", index=False)
    inputs.dynamic_harmonized.to_csv(base_dir / "dynamic" / "harmonized.csv", index=False)
    inputs.block_index.to_csv(base_dir / "blocked" / "asic_8h_block_index.csv", index=False)
    inputs.blocked_dynamic_features.to_csv(
        base_dir / "blocked" / "asic_8h_blocked_dynamic_features.csv",
        index=False,
    )
    inputs.stay_block_counts.to_csv(
        base_dir / "blocked" / "asic_8h_stay_block_counts.csv",
        index=False,
    )
    inputs.mech_vent_stay_level_qc.to_csv(
        base_dir / "qc" / "mech_vent_ge_24h_stay_level.csv",
        index=False,
    )


def write_run_config(path: Path, *, input_dir: Path, output_dir: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "input_format": "csv",
                "output_dir": str(output_dir),
                "output_format": "csv",
                "feature_set_config_path": "config/ch1_feature_sets.json",
                "horizons_hours": [8, 16, 24, 48, 72],
                "min_required_core_groups": 3,
            },
            indent=2,
        )
    )


class Chapter1PreprocessingTest(unittest.TestCase):
    def test_build_chapter1_dataset_end_to_end(self) -> None:
        dataset = build_chapter1_dataset(build_synthetic_inputs())

        retained_stays = set(dataset.cohort.table["stay_id_global"].tolist())
        self.assertEqual(retained_stays, {"stay_a", "stay_g"})

        site_flags = dataset.cohort.site_eligibility.set_index("hospital_id")["site_included_ch1"]
        self.assertIs(bool(site_flags["H1"]), True)
        self.assertIs(bool(site_flags["H2"]), False)

        stay_status = dataset.cohort.stay_exclusions.set_index("stay_id_global")
        self.assertIn("missing/unusable icu_mortality", stay_status.at["stay_f", "exclusion_reason"])
        self.assertIn("readmission flagged", stay_status.at["stay_b", "exclusion_reason"])
        self.assertIn("missing readmission", stay_status.at["stay_c", "exclusion_reason"])
        self.assertIn(
            "mechanical ventilation <24h by upstream QC",
            stay_status.at["stay_h", "exclusion_reason"],
        )

        self.assertEqual(
            set(dataset.valid_instances.counts_by_horizon["horizon_h"].tolist()),
            {8, 16, 24, 48, 72},
        )
        self.assertEqual(dataset.valid_instances.valid_instances.shape[0], 15)
        self.assertEqual(dataset.labels.usable_labels.shape[0], 12)
        self.assertEqual(dataset.feature_sets["primary"].model_ready.table.shape[0], 12)
        self.assertEqual(dataset.feature_sets["extended"].model_ready.table.shape[0], 12)
        self.assertNotIn("pct_obs_count", dataset.feature_sets["primary"].model_ready.table.columns)
        self.assertIn("pct_obs_count", dataset.feature_sets["extended"].model_ready.table.columns)

        validation_summary = dataset.feature_set_validation_summary.set_index("feature_set_name")
        self.assertEqual(validation_summary.at["primary", "primary_feature_count"], 31)
        self.assertEqual(validation_summary.at["primary", "extended_only_feature_count"], 15)
        self.assertEqual(validation_summary.at["primary", "total_extended_feature_count"], 46)
        self.assertIn(
            "fio2",
            validation_summary.at["primary", "missing_from_blocked_schema_features"],
        )
        self.assertNotIn(
            "pct",
            validation_summary.at["extended", "missing_from_blocked_schema_features"],
        )
        self.assertIn(
            "il6",
            validation_summary.at["extended", "missing_from_blocked_schema_features"],
        )

        stay_a_labels = dataset.labels.labels[dataset.labels.labels["stay_id_global"] == "stay_a"]
        label_lookup = {
            (row.block_index, row.horizon_h): (
                pd.NA if pd.isna(row.label_value) else int(row.label_value)
            )
            for row in stay_a_labels.itertuples(index=False)
        }
        self.assertTrue(pd.isna(label_lookup[(0, 8)]))
        self.assertEqual(label_lookup[(0, 16)], 1)
        self.assertEqual(label_lookup[(0, 24)], 1)
        self.assertEqual(label_lookup[(1, 8)], 1)
        self.assertEqual(label_lookup[(1, 48)], 1)
        self.assertEqual(label_lookup[(1, 72)], 1)

        stay_g_block0 = dataset.valid_instances.candidate_instances[
            (dataset.valid_instances.candidate_instances["stay_id_global"] == "stay_g")
            & (dataset.valid_instances.candidate_instances["block_index"] == 0)
        ]
        self.assertTrue(stay_g_block0["valid_instance"].eq(False).all())
        self.assertTrue(
            stay_g_block0["exclusion_reason"]
            .eq("insufficient_core_vital_group_coverage_in_block")
            .all()
        )
        self.assertTrue(stay_g_block0["covered_core_vital_group_count"].eq(0).all())

        summary_by_horizon = dataset.labels.summary_by_horizon.set_index("horizon_h")
        self.assertEqual(summary_by_horizon.at[8, "total_valid_prediction_instances"], 3)
        self.assertEqual(summary_by_horizon.at[8, "labelable_instances"], 2)
        self.assertEqual(summary_by_horizon.at[8, "positive_labels"], 1)
        self.assertEqual(summary_by_horizon.at[8, "negative_labels"], 1)
        self.assertEqual(summary_by_horizon.at[8, "unlabeled_instances"], 1)
        self.assertEqual(summary_by_horizon.at[72, "labelable_instances"], 2)

        reason_summary = dataset.labels.unlabeled_reason_summary
        self.assertEqual(
            int(
                reason_summary.loc[
                    (reason_summary["horizon_h"] == 48)
                    & (
                        reason_summary["unlabeled_reason"]
                        == "survivor_without_full_horizon_observation"
                    ),
                    "instance_count",
                ].iloc[0]
            ),
            1,
        )
        self.assertEqual(
            int(
                reason_summary.loc[
                    (reason_summary["horizon_h"] == 8)
                    & (
                        reason_summary["unlabeled_reason"]
                        == "non_survivor_proxy_end_not_within_horizon"
                    ),
                    "instance_count",
                ].iloc[0]
            ),
            1,
        )

        verification = dataset.verification_summary.set_index("check_id")["passed"]
        self.assertTrue(bool(verification["no_excluded_hospital_contributes_retained_stays"]))
        self.assertTrue(bool(verification["no_mech_vent_failed_stay_retained"]))
        self.assertTrue(bool(verification["no_missing_or_readmission_flagged_stay_retained"]))
        self.assertTrue(bool(verification["valid_instances_restricted_to_retained_stays"]))
        self.assertTrue(bool(verification["proxy_label_counts_consistent_with_valid_instances"]))

        cohort_summary = dataset.cohort_summary
        self.assertEqual(
            int(
                cohort_summary.loc[
                    cohort_summary["metric"] == "retained_stays",
                    "value",
                ].iloc[0]
            ),
            2,
        )
        self.assertEqual(
            int(
                cohort_summary.loc[
                    (cohort_summary["metric"] == "labelable_instances")
                    & (cohort_summary["horizon_h"] == 16),
                    "value",
                ].iloc[0]
            ),
            3,
        )

    def test_cli_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            input_dir = temp_path / "input"
            output_dir = temp_path / "output"
            write_standardized_inputs(input_dir, build_synthetic_inputs())

            env = os.environ.copy()
            existing_pythonpath = env.get("PYTHONPATH")
            env["PYTHONPATH"] = (
                str(SRC)
                if not existing_pythonpath
                else f"{SRC}{os.pathsep}{existing_pythonpath}"
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "chapter1_mortality_decomposition",
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

            self.assertIn("Wrote ", result.stdout)
            self.assertTrue((output_dir / "cohort" / "chapter1_retained_stay_table.csv").exists())
            self.assertTrue((output_dir / "cohort" / "chapter1_cohort_summary.csv").exists())
            self.assertTrue(
                (output_dir / "cohort" / "chapter1_verification_summary.csv").exists()
            )
            self.assertTrue(
                (
                    output_dir / "feature_sets" / "chapter1_feature_set_validation_summary.csv"
                ).exists()
            )
            self.assertTrue((output_dir / "instances" / "chapter1_valid_instances.csv").exists())
            self.assertTrue(
                (
                    output_dir / "labels" / "chapter1_proxy_horizon_labels.csv"
                ).exists()
            )
            self.assertTrue(
                (
                    output_dir / "labels" / "chapter1_proxy_unlabeled_reason_summary.csv"
                ).exists()
            )
            self.assertTrue(
                (output_dir / "model_ready" / "chapter1_primary_model_ready_dataset.csv").exists()
            )
            self.assertTrue(
                (output_dir / "model_ready" / "chapter1_extended_model_ready_dataset.csv").exists()
            )

            label_summary = pd.read_csv(
                output_dir / "labels" / "chapter1_proxy_label_summary_by_horizon.csv"
            )
            self.assertEqual(set(label_summary["horizon_h"].tolist()), {8, 16, 24, 48, 72})
            self.assertIn("Valid prediction instances:", result.stdout)
            self.assertIn("Usable proxy labels:", result.stdout)
            self.assertIn("primary model-ready rows:", result.stdout)
            self.assertIn("extended model-ready rows:", result.stdout)

    def test_cli_reads_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            input_dir = temp_path / "input"
            output_dir = temp_path / "output"
            run_config_path = temp_path / "ch1_run_config.json"
            write_standardized_inputs(input_dir, build_synthetic_inputs())
            write_run_config(run_config_path, input_dir=input_dir, output_dir=output_dir)

            env = os.environ.copy()
            existing_pythonpath = env.get("PYTHONPATH")
            env["PYTHONPATH"] = (
                str(SRC)
                if not existing_pythonpath
                else f"{SRC}{os.pathsep}{existing_pythonpath}"
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "chapter1_mortality_decomposition",
                    "--run-config",
                    str(run_config_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

            self.assertIn("Wrote ", result.stdout)
            self.assertTrue((output_dir / "cohort" / "chapter1_retained_stay_table.csv").exists())
            self.assertTrue((output_dir / "instances" / "chapter1_valid_instances.csv").exists())
            self.assertTrue(
                (output_dir / "labels" / "chapter1_proxy_horizon_labels.csv").exists()
            )


if __name__ == "__main__":
    unittest.main()
