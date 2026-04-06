from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chapter1_mortality_decomposition.asic_horizon_dependence_foundation import (
    run_asic_horizon_dependence_foundation,
)


class AsicHorizonDependenceFoundationTests(TestCase):
    def test_run_writes_requested_outputs_and_reports_matching_ready(self) -> None:
        horizons = [8, 16, 24, 48, 72]

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            hard_case_dir = tmp_path / "hard_cases"
            baseline_root = tmp_path / "baselines" / "logistic_regression"
            hard_case_dir.mkdir(parents=True)
            baseline_root.mkdir(parents=True)

            stay_rows: list[dict[str, object]] = []
            summary_rows: list[dict[str, object]] = []
            prediction_paths: dict[str, str] = {}

            stay_templates = [
                ("n_keep", "H1", 0),
                ("n_drop48", "H1", 0),
                ("n_drop72", "H2", 0),
                ("f_hard", "H2", 1),
                ("f_other", "H3", 1),
            ]

            threshold_map = {
                8: 0.020,
                16: 0.030,
                24: 0.040,
                48: 0.050,
                72: 0.060,
            }
            probabilities = {
                "n_keep": 0.010,
                "n_drop48": 0.020,
                "n_drop72": 0.030,
                "f_hard": 0.015,
                "f_other": 0.090,
            }

            for horizon in horizons:
                horizon_dir = baseline_root / f"horizon_{horizon}h"
                horizon_dir.mkdir(parents=True)
                prediction_paths[str(horizon)] = str((horizon_dir / "predictions.csv").resolve())

                present_stays = list(stay_templates)
                if horizon >= 48:
                    present_stays = [item for item in present_stays if item[0] != "n_drop48"]
                if horizon >= 72:
                    present_stays = [item for item in present_stays if item[0] != "n_drop72"]

                prediction_rows: list[dict[str, object]] = []
                for index, (stay_id, hospital_id, label_value) in enumerate(present_stays):
                    prediction_rows.extend(
                        [
                            {
                                "instance_id": f"{stay_id}__b0__h{horizon}",
                                "stay_id_global": stay_id,
                                "hospital_id": hospital_id,
                                "block_index": 0,
                                "prediction_time_h": 8,
                                "horizon_h": horizon,
                                "split": "train",
                                "label_value": label_value,
                                "predicted_probability": probabilities[stay_id] / 2,
                                "model_name": "logistic_regression",
                            },
                            {
                                "instance_id": f"{stay_id}__b1__h{horizon}",
                                "stay_id_global": stay_id,
                                "hospital_id": hospital_id,
                                "block_index": 1,
                                "prediction_time_h": 16,
                                "horizon_h": horizon,
                                "split": "test" if label_value == 1 else "train",
                                "label_value": label_value,
                                "predicted_probability": probabilities[stay_id],
                                "model_name": "logistic_regression",
                            },
                        ]
                    )

                    stay_rows.append(
                        {
                            "stay_id_global": stay_id,
                            "hospital_id": hospital_id,
                            "horizon_h": horizon,
                            "split": "test" if label_value == 1 else "train",
                            "label_value": label_value,
                            "instance_id": f"{stay_id}__b1__h{horizon}",
                            "block_index": 1,
                            "prediction_time_h": 16,
                            "predicted_probability": probabilities[stay_id],
                            "model_name": "logistic_regression",
                            "nonfatal_q75_threshold": threshold_map[horizon],
                            "hard_case_flag": label_value == 1 and probabilities[stay_id] <= threshold_map[horizon],
                            "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                        }
                    )

                pd.DataFrame(prediction_rows).to_csv(horizon_dir / "predictions.csv", index=False)

                fatal_last_n = sum(int(label_value == 1) for _, _, label_value in present_stays)
                hard_case_n = sum(
                    int(label_value == 1 and probabilities[stay_id] <= threshold_map[horizon])
                    for stay_id, _, label_value in present_stays
                )
                nonfatal_last_n = sum(int(label_value == 0) for _, _, label_value in present_stays)
                summary_rows.append(
                    {
                        "horizon_h": horizon,
                        "model_name": "logistic_regression",
                        "n_prediction_rows_loaded": len(prediction_rows),
                        "n_rows_collapsed_before_last_point": len(prediction_rows) - len(present_stays),
                        "n_selected_last_points": len(present_stays),
                        "n_nonfatal_last_points": nonfatal_last_n,
                        "n_fatal_last_points": fatal_last_n,
                        "nonfatal_q75_threshold": threshold_map[horizon],
                        "n_hard_cases": hard_case_n,
                        "pct_fatal_hard_cases": hard_case_n / fatal_last_n,
                        "subgroup_size_warning": True,
                        "warning_reason": "n_hard_cases_lt_20",
                    }
                )

            stay_level_path = hard_case_dir / "stay_level_hard_case_flags.csv"
            saved_summary_path = hard_case_dir / "horizon_hard_case_summary.csv"
            pd.DataFrame(stay_rows).to_csv(stay_level_path, index=False)
            pd.DataFrame(summary_rows).to_csv(saved_summary_path, index=False)

            manifest = {
                "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                "horizon_summary_artifact": str(saved_summary_path.resolve()),
                "horizons_processed": horizons,
                "prediction_paths_by_horizon": prediction_paths,
                "stay_level_artifact": str(stay_level_path.resolve()),
            }
            (hard_case_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

            result = run_asic_horizon_dependence_foundation(
                hard_case_dir=hard_case_dir,
                output_dir=tmp_path / "foundation",
            )

            self.assertTrue(result.cross_horizon_matching_ready)
            self.assertFalse(result.schema_harmonization_required)
            self.assertEqual(result.mismatches_vs_saved_summary, ())
            self.assertTrue(result.artifacts.horizon_summary_csv_path.exists())
            self.assertTrue(result.artifacts.horizon_summary_markdown_path.exists())
            self.assertTrue(result.artifacts.note_path.exists())

            summary = pd.read_csv(result.artifacts.horizon_summary_csv_path)
            self.assertEqual(
                summary["horizon"].tolist(),
                ["8h", "16h", "24h", "48h", "72h"],
            )
            self.assertEqual(summary["nonfatal_last_n"].tolist(), [3, 3, 3, 2, 1])
            self.assertEqual(summary["fatal_last_n"].tolist(), [2, 2, 2, 2, 2])
            self.assertEqual(summary["hard_case_n"].tolist(), [1, 1, 1, 1, 1])

            note_text = result.artifacts.note_path.read_text()
            self.assertIn("stay_id_global", note_text)
            self.assertIn("Package 2 can proceed cleanly", note_text)
            self.assertIn("missing_vs_8h", note_text)
            self.assertIn("synthetic", note_text.lower())

    def test_run_fails_when_threshold_is_not_constant_within_horizon(self) -> None:
        horizons = [8, 16, 24, 48, 72]

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            hard_case_dir = tmp_path / "hard_cases"
            baseline_root = tmp_path / "baselines" / "logistic_regression"
            hard_case_dir.mkdir(parents=True)
            baseline_root.mkdir(parents=True)

            stay_rows = []
            summary_rows = []
            prediction_paths: dict[str, str] = {}

            for horizon in horizons:
                horizon_dir = baseline_root / f"horizon_{horizon}h"
                horizon_dir.mkdir(parents=True)
                prediction_path = horizon_dir / "predictions.csv"
                prediction_paths[str(horizon)] = str(prediction_path.resolve())
                prediction_df = pd.DataFrame(
                    [
                        {
                            "instance_id": f"n{horizon}__b0__h{horizon}",
                            "stay_id_global": f"n{horizon}",
                            "hospital_id": "H1",
                            "block_index": 0,
                            "prediction_time_h": 8,
                            "horizon_h": horizon,
                            "split": "train",
                            "label_value": 0,
                            "predicted_probability": 0.01,
                            "model_name": "logistic_regression",
                        },
                        {
                            "instance_id": f"f{horizon}__b0__h{horizon}",
                            "stay_id_global": f"f{horizon}",
                            "hospital_id": "H2",
                            "block_index": 0,
                            "prediction_time_h": 8,
                            "horizon_h": horizon,
                            "split": "test",
                            "label_value": 1,
                            "predicted_probability": 0.02,
                            "model_name": "logistic_regression",
                        },
                    ]
                )
                prediction_df.to_csv(prediction_path, index=False)

                stay_rows.extend(
                    [
                        {
                            "stay_id_global": f"n{horizon}",
                            "hospital_id": "H1",
                            "horizon_h": horizon,
                            "split": "train",
                            "label_value": 0,
                            "instance_id": f"n{horizon}__b0__h{horizon}",
                            "block_index": 0,
                            "prediction_time_h": 8,
                            "predicted_probability": 0.01,
                            "model_name": "logistic_regression",
                            "nonfatal_q75_threshold": 0.01,
                            "hard_case_flag": False,
                            "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                        },
                        {
                            "stay_id_global": f"f{horizon}",
                            "hospital_id": "H2",
                            "horizon_h": horizon,
                            "split": "test",
                            "label_value": 1,
                            "instance_id": f"f{horizon}__b0__h{horizon}",
                            "block_index": 0,
                            "prediction_time_h": 8,
                            "predicted_probability": 0.02,
                            "model_name": "logistic_regression",
                            "nonfatal_q75_threshold": 0.02 if horizon == 24 else 0.01,
                            "hard_case_flag": False,
                            "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                        },
                    ]
                )

                summary_rows.append(
                    {
                        "horizon_h": horizon,
                        "n_nonfatal_last_points": 1,
                        "n_fatal_last_points": 1,
                        "nonfatal_q75_threshold": 0.01,
                        "n_hard_cases": 0,
                        "pct_fatal_hard_cases": 0.0,
                    }
                )

            stay_level_path = hard_case_dir / "stay_level_hard_case_flags.csv"
            saved_summary_path = hard_case_dir / "horizon_hard_case_summary.csv"
            pd.DataFrame(stay_rows).to_csv(stay_level_path, index=False)
            pd.DataFrame(summary_rows).to_csv(saved_summary_path, index=False)
            manifest = {
                "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                "horizon_summary_artifact": str(saved_summary_path.resolve()),
                "horizons_processed": horizons,
                "prediction_paths_by_horizon": prediction_paths,
                "stay_level_artifact": str(stay_level_path.resolve()),
            }
            (hard_case_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

            with self.assertRaisesRegex(ValueError, "single nonfatal_q75_threshold per horizon"):
                run_asic_horizon_dependence_foundation(
                    hard_case_dir=hard_case_dir,
                    output_dir=tmp_path / "foundation",
                )
