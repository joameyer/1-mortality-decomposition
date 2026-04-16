from __future__ import annotations

from unittest import TestCase

import pandas as pd

from chapter1_mortality_decomposition.temporal_sensitivity import (
    _build_aggregation_directional_overlap,
    _build_aggregation_pairwise_tables,
    _build_aggregation_persistence_tables,
    _build_frozen_split_alignment_summary,
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
