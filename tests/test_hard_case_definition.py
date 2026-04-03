from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from chapter1_mortality_decomposition.hard_case_definition import (
    HARD_CASE_RULE,
    run_asic_logistic_hard_case_definition,
)


class HardCaseDefinitionTests(TestCase):
    def test_run_asic_logistic_hard_case_definition_selects_last_points_and_writes_outputs(self) -> None:
        prediction_rows = [
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
                "model_name": "logistic_regression",
            },
            {
                "instance_id": "n1__b1__h24",
                "stay_id_global": "n1",
                "hospital_id": "H1",
                "block_index": 1,
                "prediction_time_h": 16,
                "horizon_h": 24,
                "split": "train",
                "label_value": 0,
                "predicted_probability": 0.20,
                "model_name": "logistic_regression",
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
                "predicted_probability": 0.40,
                "model_name": "logistic_regression",
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
                "predicted_probability": 0.80,
                "model_name": "logistic_regression",
            },
            {
                "instance_id": "n3__b1__h24",
                "stay_id_global": "n3",
                "hospital_id": "H2",
                "block_index": 1,
                "prediction_time_h": 16,
                "horizon_h": 24,
                "split": "test",
                "label_value": 0,
                "predicted_probability": 0.60,
                "model_name": "logistic_regression",
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
                "predicted_probability": 0.30,
                "model_name": "logistic_regression",
            },
            {
                "instance_id": "f1__b1__h24",
                "stay_id_global": "f1",
                "hospital_id": "H1",
                "block_index": 1,
                "prediction_time_h": 16,
                "horizon_h": 24,
                "split": "train",
                "label_value": 1,
                "predicted_probability": 0.45,
                "model_name": "logistic_regression",
            },
            {
                "instance_id": "f2__b0__h24",
                "stay_id_global": "f2",
                "hospital_id": "H2",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "validation",
                "label_value": 1,
                "predicted_probability": 0.70,
                "model_name": "logistic_regression",
            },
            {
                "instance_id": "f2__b1__h24",
                "stay_id_global": "f2",
                "hospital_id": "H2",
                "block_index": 1,
                "prediction_time_h": 16,
                "horizon_h": 24,
                "split": "validation",
                "label_value": 1,
                "predicted_probability": 0.90,
                "model_name": "logistic_regression",
            },
        ]

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_root = tmp_path / "baselines"
            output_dir = tmp_path / "hard_cases"
            horizon_dir = input_root / "logistic_regression" / "horizon_24h"
            horizon_dir.mkdir(parents=True)
            pd.DataFrame(prediction_rows).to_csv(horizon_dir / "predictions.csv", index=False)

            result = run_asic_logistic_hard_case_definition(
                input_root=input_root,
                output_dir=output_dir,
                horizons=[24],
            )

            self.assertTrue(result.artifacts.stay_level_path.exists())
            self.assertTrue(result.artifacts.horizon_summary_path.exists())
            self.assertTrue(result.artifacts.manifest_path.exists())
            self.assertEqual(result.hard_case_rule, HARD_CASE_RULE)

            stay_level = pd.read_csv(result.artifacts.stay_level_path)
            self.assertEqual(stay_level.shape[0], 5)
            self.assertEqual(
                stay_level["instance_id"].tolist(),
                ["f1__b1__h24", "n1__b1__h24", "n2__b0__h24", "f2__b1__h24", "n3__b1__h24"],
            )
            self.assertTrue(stay_level["hard_case_rule"].eq(HARD_CASE_RULE).all())

            fatal_flags = stay_level.set_index("stay_id_global")["hard_case_flag"].astype(bool).to_dict()
            self.assertTrue(fatal_flags["f1"])
            self.assertFalse(fatal_flags["f2"])
            self.assertFalse(fatal_flags["n1"])

            summary = pd.read_csv(result.artifacts.horizon_summary_path)
            self.assertEqual(int(summary.at[0, "n_nonfatal_last_points"]), 3)
            self.assertEqual(int(summary.at[0, "n_fatal_last_points"]), 2)
            self.assertAlmostEqual(float(summary.at[0, "nonfatal_q75_threshold"]), 0.5)
            self.assertEqual(int(summary.at[0, "n_hard_cases"]), 1)
            self.assertAlmostEqual(float(summary.at[0, "pct_fatal_hard_cases"]), 0.5)
            self.assertTrue(bool(summary.at[0, "subgroup_size_warning"]))
            self.assertIn("n_hard_cases_lt_20", str(summary.at[0, "warning_reason"]))

    def test_run_asic_logistic_hard_case_definition_fails_on_missing_probability(self) -> None:
        prediction_rows = [
            {
                "instance_id": "f1__b0__h24",
                "stay_id_global": "f1",
                "hospital_id": "H1",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "train",
                "label_value": 1,
                "predicted_probability": 0.20,
                "model_name": "logistic_regression",
            },
            {
                "instance_id": "n1__b0__h24",
                "stay_id_global": "n1",
                "hospital_id": "H1",
                "block_index": 0,
                "prediction_time_h": 8,
                "horizon_h": 24,
                "split": "train",
                "label_value": 0,
                "predicted_probability": pd.NA,
                "model_name": "logistic_regression",
            },
        ]

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_root = tmp_path / "baselines"
            horizon_dir = input_root / "logistic_regression" / "horizon_24h"
            horizon_dir.mkdir(parents=True)
            pd.DataFrame(prediction_rows).to_csv(horizon_dir / "predictions.csv", index=False)

            with self.assertRaisesRegex(ValueError, "missing predicted_probability"):
                run_asic_logistic_hard_case_definition(
                    input_root=input_root,
                    output_dir=tmp_path / "hard_cases",
                    horizons=[24],
                )
