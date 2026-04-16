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

from chapter1_mortality_decomposition.asic_hard_case_comparison import (  # noqa: E402
    LOW_PREDICTED_FATAL_GROUP,
    OTHER_FATAL_GROUP,
)
from chapter1_mortality_decomposition.asic_uk04_observation_process_followup import (  # noqa: E402
    ANALYSIS_POPULATION_ALL_FATAL,
    ANALYSIS_POPULATION_HARD_CASES,
    build_reference_hard_case_effects,
    build_uk04_vs_non_uk04_summary,
    run_asic_uk04_observation_process_followup,
)


def _build_derived_dataset() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for index in range(2):
        rows.append(
            {
                "stay_id_global": f"uk04_hard_{index}",
                "instance_id": f"uk04_hard_{index}__b1__h24",
                "hospital_id": "asic_UK04",
                "hard_case_flag": True,
                "hard_case_group": LOW_PREDICTED_FATAL_GROUP,
                "n_core_groups_fresh_block": 4,
                "core_block_complete_all4": True,
                "n_core_groups_historical_only": 0,
                "n_core_groups_never_observed": 0,
                "n_frozen_proxy_missing": 0,
                "time_since_last_any_core_h": 0.25,
                "max_time_since_last_core_h": 0.50,
                "any_stale_core_ge_8h_flag": False,
            }
        )
    for index in range(2):
        rows.append(
            {
                "stay_id_global": f"uk04_other_{index}",
                "instance_id": f"uk04_other_{index}__b1__h24",
                "hospital_id": "asic_UK04",
                "hard_case_flag": False,
                "hard_case_group": OTHER_FATAL_GROUP,
                "n_core_groups_fresh_block": 4,
                "core_block_complete_all4": True,
                "n_core_groups_historical_only": 0,
                "n_core_groups_never_observed": 0,
                "n_frozen_proxy_missing": 0,
                "time_since_last_any_core_h": 0.25,
                "max_time_since_last_core_h": 0.50,
                "any_stale_core_ge_8h_flag": False,
            }
        )
    for index in range(2):
        rows.append(
            {
                "stay_id_global": f"non_hard_{index}",
                "instance_id": f"non_hard_{index}__b1__h24",
                "hospital_id": "asic_UK07",
                "hard_case_flag": True,
                "hard_case_group": LOW_PREDICTED_FATAL_GROUP,
                "n_core_groups_fresh_block": 3,
                "core_block_complete_all4": False,
                "n_core_groups_historical_only": 1,
                "n_core_groups_never_observed": 0,
                "n_frozen_proxy_missing": 1,
                "time_since_last_any_core_h": 0.50,
                "max_time_since_last_core_h": 10.0,
                "any_stale_core_ge_8h_flag": True,
            }
        )
    for index in range(2):
        rows.append(
            {
                "stay_id_global": f"non_other_{index}",
                "instance_id": f"non_other_{index}__b1__h24",
                "hospital_id": "asic_UK07",
                "hard_case_flag": False,
                "hard_case_group": OTHER_FATAL_GROUP,
                "n_core_groups_fresh_block": 4,
                "core_block_complete_all4": True,
                "n_core_groups_historical_only": 0,
                "n_core_groups_never_observed": 0,
                "n_frozen_proxy_missing": 0,
                "time_since_last_any_core_h": 0.25,
                "max_time_since_last_core_h": 1.0,
                "any_stale_core_ge_8h_flag": False,
            }
        )

    return pd.DataFrame(rows)


class ASICUK04ObservationProcessFollowupTests(TestCase):
    def test_build_uk04_vs_non_uk04_summary_reports_direction_against_reference(self) -> None:
        dataset = _build_derived_dataset()
        reference_effects = build_reference_hard_case_effects(dataset)

        summary = build_uk04_vs_non_uk04_summary(
            dataset,
            reference_effects=reference_effects,
            analysis_population=ANALYSIS_POPULATION_ALL_FATAL,
        )

        self.assertEqual(int(summary.shape[0]), 8)
        complete_row = summary[summary["variable"].eq("core_block_complete_all4")].iloc[0]
        self.assertEqual(complete_row["direction_vs_hard_case_pattern"], "opposite_direction")
        self.assertEqual(complete_row["uk04_summary"], "4 (100.0%)")
        self.assertEqual(complete_row["non_uk04_summary"], "2 (50.0%)")

    def test_run_asic_uk04_observation_process_followup_writes_outputs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            comparison_dataset_path = tmp_path / "stay_level_comparison_dataset.csv"
            hard_case_path = tmp_path / "stay_level_hard_case_flags.csv"
            observation_process_path = tmp_path / "chapter1_observation_process_block_features.csv"
            output_dir = tmp_path / "site_sensitivity"

            comparison_rows: list[dict[str, object]] = []
            hard_case_rows: list[dict[str, object]] = []
            observation_rows: list[dict[str, object]] = []

            for site, hard_flag, prediction_time_h, pf_ratio, map_last, peep_last, fresh, stale, proxy_missing, max_gap in [
                ("asic_UK04", True, 24.0, 250.0, 80.0, 8.0, 4, False, 0, 0.5),
                ("asic_UK04", True, 32.0, 240.0, 82.0, 8.0, 4, False, 0, 0.5),
                ("asic_UK04", False, 40.0, 200.0, 70.0, 9.0, 4, False, 0, 0.5),
                ("asic_UK04", False, 48.0, 210.0, 72.0, 9.0, 4, False, 0, 0.5),
                ("asic_UK07", True, 24.0, 150.0, 58.0, 11.0, 3, True, 1, 10.0),
                ("asic_UK07", True, 32.0, 160.0, 60.0, 11.0, 3, True, 1, 10.0),
                ("asic_UK07", False, 40.0, 190.0, 65.0, 10.0, 4, False, 0, 1.0),
                ("asic_UK07", False, 48.0, 195.0, 66.0, 10.0, 4, False, 0, 1.0),
            ]:
                stay_id = f"{site}_{'hard' if hard_flag else 'other'}_{int(prediction_time_h)}"
                instance_id = f"{stay_id}__b1__h24"
                comparison_rows.append(
                    {
                        "stay_id_global": stay_id,
                        "instance_id": instance_id,
                        "hard_case_flag": hard_flag,
                        "hard_case_group": (
                            LOW_PREDICTED_FATAL_GROUP if hard_flag else OTHER_FATAL_GROUP
                        ),
                        "prediction_time_h": prediction_time_h,
                        "icu_end_time_proxy_hours": prediction_time_h + 24.0,
                        "hospital_id": site,
                        "pf_ratio_last": pf_ratio if proxy_missing == 0 else None,
                        "map_last": map_last,
                        "creatinine_last": 100.0,
                        "peep_last": peep_last if proxy_missing == 0 else None,
                    }
                )
                hard_case_rows.append(
                    {
                        "stay_id_global": stay_id,
                        "instance_id": instance_id,
                        "hospital_id": site,
                        "horizon_h": 24,
                        "label_value": 1,
                        "block_index": 1,
                        "prediction_time_h": prediction_time_h,
                        "predicted_probability": 0.10 if hard_flag else 0.40,
                        "nonfatal_q75_threshold": 0.20,
                        "hard_case_flag": hard_flag,
                        "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                    }
                )
                observation_rows.append(
                    {
                        "stay_id_global": stay_id,
                        "hospital_id": site,
                        "block_index": 1,
                        "block_start_h": prediction_time_h - 8.0,
                        "block_end_h": prediction_time_h,
                        "prediction_time_h": prediction_time_h,
                        "obs_hr_grp_block": 1,
                        "obs_bp_grp_block": 1,
                        "obs_resp_grp_block": 1,
                        "obs_oxy_grp_block": 1 if fresh == 4 else 0,
                        "n_core_grps_obs_block": fresh,
                        "tsl_hr_grp_h": 0.25,
                        "tsl_bp_grp_h": 0.25,
                        "tsl_resp_grp_h": 0.25,
                        "tsl_oxy_grp_h": max_gap if stale else 0.5,
                    }
                )

            pd.DataFrame(comparison_rows).to_csv(comparison_dataset_path, index=False)
            pd.DataFrame(hard_case_rows).to_csv(hard_case_path, index=False)
            pd.DataFrame(observation_rows).to_csv(observation_process_path, index=False)

            result = run_asic_uk04_observation_process_followup(
                comparison_dataset_path=comparison_dataset_path,
                hard_case_path=hard_case_path,
                observation_process_path=observation_process_path,
                output_dir=output_dir,
            )

            self.assertEqual(
                result.interpretation_category,
                "modest_enrichment_not_clearly_explained_by_measured_observation_process",
            )
            self.assertTrue(result.artifacts.summary_path.exists())
            self.assertTrue(result.artifacts.hard_case_followup_path.exists())
            self.assertTrue(result.artifacts.memo_path.exists())
            self.assertTrue(result.artifacts.manifest_path.exists())

            summary = pd.read_csv(result.artifacts.summary_path)
            self.assertEqual(summary["analysis_population"].unique().tolist(), [ANALYSIS_POPULATION_ALL_FATAL])
            hard_followup = pd.read_csv(result.artifacts.hard_case_followup_path)
            self.assertEqual(
                hard_followup["analysis_population"].unique().tolist(),
                [ANALYSIS_POPULATION_HARD_CASES],
            )

            manifest = json.loads(result.artifacts.manifest_path.read_text())
            self.assertEqual(
                manifest["interpretation_category"],
                "modest_enrichment_not_clearly_explained_by_measured_observation_process",
            )
