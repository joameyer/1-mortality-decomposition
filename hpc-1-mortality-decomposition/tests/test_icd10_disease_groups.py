from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chapter1_mortality_decomposition.icd10_disease_groups import (  # noqa: E402
    CARDIOVASCULAR_GROUP,
    INFECTION_SEPSIS_NON_PULMONARY_GROUP,
    NEUROLOGIC_GROUP,
    OTHER_MIXED_UNCATEGORIZED_GROUP,
    RESPIRATORY_PULMONARY_GROUP,
    SURGICAL_POSTOPERATIVE_TRAUMA_GROUP,
    derive_icd10_disease_group,
    normalize_icd10_stems,
)


class ICD10DiseaseGroupTests(TestCase):
    def test_normalize_icd10_stems_collapses_decimals_strips_bang_and_deduplicates(self) -> None:
        stems = normalize_icd10_stems(" j80.03 , U07.1! , j80.01, J95 , J95 ")
        self.assertEqual(stems, ("J80", "J95", "U07"))

    def test_surgical_group_wins_over_other_matches(self) -> None:
        match = derive_icd10_disease_group("J18, A41, T81, I50")
        self.assertEqual(
            match.candidate_groups,
            (
                SURGICAL_POSTOPERATIVE_TRAUMA_GROUP,
                RESPIRATORY_PULMONARY_GROUP,
                INFECTION_SEPSIS_NON_PULMONARY_GROUP,
                CARDIOVASCULAR_GROUP,
            ),
        )
        self.assertEqual(match.final_group, SURGICAL_POSTOPERATIVE_TRAUMA_GROUP)

    def test_respiratory_group_keeps_j95_reserved_for_surgical_rule(self) -> None:
        match = derive_icd10_disease_group("J95")
        self.assertEqual(
            match.candidate_groups,
            (SURGICAL_POSTOPERATIVE_TRAUMA_GROUP,),
        )
        self.assertEqual(match.final_group, SURGICAL_POSTOPERATIVE_TRAUMA_GROUP)

    def test_infection_group_uses_u07_and_not_auxiliary_u_codes(self) -> None:
        infection_match = derive_icd10_disease_group("U07.1!, Z22, U81")
        self.assertEqual(
            infection_match.candidate_groups,
            (INFECTION_SEPSIS_NON_PULMONARY_GROUP,),
        )
        self.assertEqual(infection_match.final_group, INFECTION_SEPSIS_NON_PULMONARY_GROUP)

        auxiliary_only_match = derive_icd10_disease_group("U80, U81, Z22, Z29")
        self.assertEqual(auxiliary_only_match.candidate_groups, ())
        self.assertEqual(auxiliary_only_match.final_group, OTHER_MIXED_UNCATEGORIZED_GROUP)

    def test_cardiovascular_excludes_stroke_but_includes_r57(self) -> None:
        stroke_match = derive_icd10_disease_group("I61")
        self.assertEqual(stroke_match.candidate_groups, (NEUROLOGIC_GROUP,))
        self.assertEqual(stroke_match.final_group, NEUROLOGIC_GROUP)

        cardiovascular_match = derive_icd10_disease_group("R57, I50")
        self.assertEqual(cardiovascular_match.candidate_groups, (CARDIOVASCULAR_GROUP,))
        self.assertEqual(cardiovascular_match.final_group, CARDIOVASCULAR_GROUP)

    def test_neurologic_includes_cns_neoplasm_stems(self) -> None:
        match = derive_icd10_disease_group("D43")
        self.assertEqual(match.candidate_groups, (NEUROLOGIC_GROUP,))
        self.assertEqual(match.final_group, NEUROLOGIC_GROUP)

    def test_assignment_uses_normalized_set_not_raw_order(self) -> None:
        first = derive_icd10_disease_group("A41, J18, I50")
        second = derive_icd10_disease_group("I50, J18, A41, A41")
        self.assertEqual(first.candidate_groups, second.candidate_groups)
        self.assertEqual(first.final_group, second.final_group)
