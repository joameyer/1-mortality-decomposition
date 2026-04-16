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
from chapter1_mortality_decomposition.asic_site_enrichment_decision import (  # noqa: E402
    build_site_hard_case_comparison,
    build_site_hard_case_summary,
    build_site_persistence_check,
    run_asic_site_enrichment_decision,
)


def _build_four_site_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    site_specs = {
        "asic_A": {"hard": 8, "other": 10},
        "asic_B": {"hard": 6, "other": 14},
        "asic_C": {"hard": 6, "other": 14},
        "asic_D": {"hard": 5, "other": 13},
    }

    comparison_rows: list[dict[str, object]] = []
    hard_case_rows: list[dict[str, object]] = []

    site_offsets = {
        "asic_A": 0.0,
        "asic_B": 6.0,
        "asic_C": 12.0,
        "asic_D": 18.0,
    }

    for site, counts in site_specs.items():
        offset = site_offsets[site]
        for index in range(counts["hard"]):
            stay_id = f"{site}_hard_{index}"
            instance_id = f"{stay_id}__b1__h24"
            comparison_rows.append(
                {
                    "stay_id_global": stay_id,
                    "instance_id": instance_id,
                    "hard_case_flag": True,
                    "hard_case_group": LOW_PREDICTED_FATAL_GROUP,
                    "hospital_id": site,
                    "prediction_time_h": 120.0 + offset + index,
                    "map_last": 82.0 + offset * 0.1 + index * 0.2,
                    "pf_ratio_last": 260.0 + offset + index * 3.0,
                    "peep_last": 8.0 + index * 0.1,
                }
            )
            hard_case_rows.append(
                {
                    "stay_id_global": stay_id,
                    "instance_id": instance_id,
                    "hospital_id": site,
                    "horizon_h": 24,
                    "label_value": 1,
                    "hard_case_flag": True,
                    "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                }
            )

        for index in range(counts["other"]):
            stay_id = f"{site}_other_{index}"
            instance_id = f"{stay_id}__b1__h24"
            comparison_rows.append(
                {
                    "stay_id_global": stay_id,
                    "instance_id": instance_id,
                    "hard_case_flag": False,
                    "hard_case_group": OTHER_FATAL_GROUP,
                    "hospital_id": site,
                    "prediction_time_h": 180.0 + offset + index,
                    "map_last": 60.0 + offset * 0.1 + index * 0.2,
                    "pf_ratio_last": 180.0 + offset + index * 2.0,
                    "peep_last": 11.0 + index * 0.1,
                }
            )
            hard_case_rows.append(
                {
                    "stay_id_global": stay_id,
                    "instance_id": instance_id,
                    "hospital_id": site,
                    "horizon_h": 24,
                    "label_value": 1,
                    "hard_case_flag": False,
                    "hard_case_rule": "asic_logistic_last_eligible_nonfatal_q75_v1",
                }
            )

    return pd.DataFrame(comparison_rows), pd.DataFrame(hard_case_rows)


class ASICSiteEnrichmentDecisionTests(TestCase):
    def test_build_site_hard_case_summary_reports_requested_columns(self) -> None:
        dataset = pd.DataFrame(
            [
                {
                    "stay_id_global": "s1",
                    "instance_id": "s1__b1__h24",
                    "hard_case_flag": True,
                    "hard_case_group": LOW_PREDICTED_FATAL_GROUP,
                    "hospital_id": "asic_A",
                    "prediction_time_h": 100.0,
                    "map_last": 80.0,
                    "pf_ratio_last": 250.0,
                    "peep_last": 8.0,
                },
                {
                    "stay_id_global": "s2",
                    "instance_id": "s2__b1__h24",
                    "hard_case_flag": False,
                    "hard_case_group": OTHER_FATAL_GROUP,
                    "hospital_id": "asic_A",
                    "prediction_time_h": 150.0,
                    "map_last": 60.0,
                    "pf_ratio_last": 180.0,
                    "peep_last": 11.0,
                },
                {
                    "stay_id_global": "s3",
                    "instance_id": "s3__b1__h24",
                    "hard_case_flag": False,
                    "hard_case_group": OTHER_FATAL_GROUP,
                    "hospital_id": "asic_B",
                    "prediction_time_h": 160.0,
                    "map_last": 61.0,
                    "pf_ratio_last": 175.0,
                    "peep_last": 10.5,
                },
            ]
        )

        summary = build_site_hard_case_summary(dataset)

        self.assertEqual(
            summary.columns.tolist(),
            [
                "site",
                "fatal_stays",
                "hard_cases",
                "other_fatal_cases",
                "within_site_hard_case_share",
                "share_of_all_hard_cases",
                "share_of_all_fatal_stays",
                "within_site_minus_overall_hard_case_share_pp",
                "hard_case_share_minus_fatal_share_pp",
            ],
        )
        first_row = summary.iloc[0]
        self.assertEqual(first_row["site"], "asic_A")
        self.assertEqual(int(first_row["fatal_stays"]), 2)
        self.assertEqual(int(first_row["hard_cases"]), 1)
        self.assertAlmostEqual(float(first_row["within_site_hard_case_share"]), 0.5)

    def test_build_site_persistence_check_reports_same_direction_sites(self) -> None:
        comparison_dataset, _ = _build_four_site_dataset()

        persistence = build_site_persistence_check(comparison_dataset)

        self.assertEqual(int(persistence.shape[0]), 16)
        self.assertTrue(persistence["assessable"].all())
        self.assertTrue(
            persistence["direction_classification"].eq("same_direction").all()
        )

    def test_run_asic_site_enrichment_decision_writes_outputs_and_modest_decision(self) -> None:
        comparison_dataset, hard_case_flags = _build_four_site_dataset()

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            comparison_dataset_path = tmp_path / "stay_level_comparison_dataset.csv"
            hard_case_path = tmp_path / "stay_level_hard_case_flags.csv"
            output_dir = tmp_path / "site_sensitivity"

            comparison_dataset.to_csv(comparison_dataset_path, index=False)
            hard_case_flags.to_csv(hard_case_path, index=False)

            result = run_asic_site_enrichment_decision(
                comparison_dataset_path=comparison_dataset_path,
                hard_case_path=hard_case_path,
                output_dir=output_dir,
            )

            self.assertEqual(result.decision_category, "Some site enrichment, but clearly modest")
            self.assertEqual(result.primary_enriched_site, "asic_A")
            self.assertFalse(result.package2_justified)
            self.assertTrue(result.artifacts.site_hard_case_summary_path.exists())
            self.assertTrue(result.artifacts.site_hard_case_comparison_path.exists())
            self.assertTrue(result.artifacts.site_persistence_check_path.exists())
            self.assertTrue(result.artifacts.memo_path.exists())
            self.assertTrue(result.artifacts.manifest_path.exists())

            comparison = pd.read_csv(result.artifacts.site_hard_case_comparison_path)
            self.assertEqual(int(comparison.shape[0]), 4)
            self.assertEqual(comparison.iloc[0]["site"], "asic_A")

            persistence = pd.read_csv(result.artifacts.site_persistence_check_path)
            self.assertEqual(int(persistence.shape[0]), 16)
            self.assertTrue(persistence["direction_classification"].eq("same_direction").all())

            manifest = json.loads(result.artifacts.manifest_path.read_text())
            self.assertEqual(
                manifest["decision_category"],
                "Some site enrichment, but clearly modest",
            )
            self.assertEqual(manifest["primary_enriched_site"], "asic_A")
            self.assertFalse(bool(manifest["package2_justified"]))

    def test_build_site_hard_case_comparison_reports_global_metrics(self) -> None:
        comparison_dataset, _ = _build_four_site_dataset()

        comparison, metrics = build_site_hard_case_comparison(comparison_dataset)

        self.assertEqual(int(comparison.shape[0]), 4)
        self.assertGreater(float(metrics["cramers_v"]), 0.10)
        self.assertLess(float(metrics["cramers_v"]), 0.20)
        self.assertEqual(comparison.iloc[0]["site"], "asic_A")
