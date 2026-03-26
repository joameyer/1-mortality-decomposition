from __future__ import annotations

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
        ]
    )

    return Chapter1InputTables(
        static_harmonized=static_harmonized,
        dynamic_harmonized=dynamic_harmonized,
        block_index=block_index,
        blocked_dynamic_features=blocked_dynamic_features,
        stay_block_counts=stay_block_counts,
    )


def write_standardized_inputs(base_dir: Path, inputs: Chapter1InputTables) -> None:
    (base_dir / "static").mkdir(parents=True, exist_ok=True)
    (base_dir / "dynamic").mkdir(parents=True, exist_ok=True)
    (base_dir / "blocked").mkdir(parents=True, exist_ok=True)

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
        self.assertIn("no dynamic data", stay_status.at["stay_e", "exclusion_reason"])

        self.assertEqual(
            set(dataset.valid_instances.counts_by_horizon["horizon_h"].tolist()),
            {8, 16, 24, 48},
        )
        self.assertEqual(dataset.valid_instances.valid_instances.shape[0], 12)
        self.assertEqual(dataset.model_ready.table.shape[0], 12)

        stay_a_labels = dataset.labels.usable_labels[
            dataset.labels.usable_labels["stay_id_global"] == "stay_a"
        ]
        label_lookup = {
            (row.block_index, row.horizon_h): int(row.label_value)
            for row in stay_a_labels.itertuples(index=False)
        }
        self.assertEqual(label_lookup[(0, 8)], 0)
        self.assertEqual(label_lookup[(0, 16)], 1)
        self.assertEqual(label_lookup[(0, 24)], 1)
        self.assertEqual(label_lookup[(1, 8)], 1)
        self.assertEqual(label_lookup[(1, 48)], 1)

        stay_g_block0 = dataset.valid_instances.candidate_instances[
            (dataset.valid_instances.candidate_instances["stay_id_global"] == "stay_g")
            & (dataset.valid_instances.candidate_instances["block_index"] == 0)
        ]
        self.assertTrue(stay_g_block0["valid_instance"].eq(False).all())
        self.assertTrue(
            stay_g_block0["exclusion_reason"].eq("no_chapter1_feature_data_in_block").all()
        )

        self.assertEqual(dataset.labels.notes.loc[0, "note_id"], "proxy_horizon_label")
        self.assertIn("icu_end_time_proxy_hours", dataset.labels.notes.loc[0, "note"])

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

            self.assertIn("Wrote 22 Chapter 1 tables", result.stdout)
            self.assertTrue((output_dir / "cohort" / "chapter1_retained_stay_table.csv").exists())
            self.assertTrue((output_dir / "labels" / "chapter1_horizon_labels.csv").exists())
            self.assertTrue(
                (output_dir / "model_ready" / "chapter1_model_ready_dataset.csv").exists()
            )

            label_summary = pd.read_csv(
                output_dir / "labels" / "chapter1_label_summary_by_horizon.csv"
            )
            self.assertEqual(set(label_summary["horizon_h"].tolist()), {8, 16, 24, 48})


if __name__ == "__main__":
    unittest.main()
