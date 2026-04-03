from __future__ import annotations

from dataclasses import dataclass
import re


SURGICAL_POSTOPERATIVE_TRAUMA_GROUP = "surgical / postoperative / trauma-related"
RESPIRATORY_PULMONARY_GROUP = "respiratory / pulmonary"
INFECTION_SEPSIS_NON_PULMONARY_GROUP = "infection / sepsis non-pulmonary"
CARDIOVASCULAR_GROUP = "cardiovascular"
NEUROLOGIC_GROUP = "neurologic"
OTHER_MIXED_UNCATEGORIZED_GROUP = "other / mixed / uncategorized"

FROZEN_DISEASE_GROUP_HIERARCHY = (
    SURGICAL_POSTOPERATIVE_TRAUMA_GROUP,
    RESPIRATORY_PULMONARY_GROUP,
    INFECTION_SEPSIS_NON_PULMONARY_GROUP,
    CARDIOVASCULAR_GROUP,
    NEUROLOGIC_GROUP,
    OTHER_MIXED_UNCATEGORIZED_GROUP,
)

_POSTOPERATIVE_COMPLICATION_STEMS = frozenset({"I97", "J95", "K91", "M96", "N99"})
_CNS_NEOPLASM_STEMS = frozenset({"C70", "C71", "C72", "D32", "D33", "D42", "D43"})


@dataclass(frozen=True)
class ICD10DiseaseGroupMatch:
    normalized_stems: tuple[str, ...]
    candidate_groups: tuple[str, ...]
    final_group: str


def normalize_icd10_token(raw_token: object) -> str | None:
    if raw_token is None:
        return None

    token = str(raw_token).strip().upper().replace(" ", "")
    if not token or token in {"<NA>", "NAN", "NONE"}:
        return None

    token = token.rstrip("!")
    match = re.match(r"^([A-Z])(\d{2})", token)
    if match is None:
        return None
    return f"{match.group(1)}{match.group(2)}"


def normalize_icd10_stems(raw_codes: object) -> tuple[str, ...]:
    if raw_codes is None:
        return ()

    text = str(raw_codes).strip()
    if not text or text in {"<NA>", "nan", "None"}:
        return ()

    stems = {
        normalized
        for token in text.split(",")
        if (normalized := normalize_icd10_token(token)) is not None
    }
    return tuple(sorted(stems))


def match_candidate_disease_groups(normalized_stems: tuple[str, ...]) -> tuple[str, ...]:
    candidates: list[str] = []

    if any(_matches_surgical_or_postoperative(stem) for stem in normalized_stems):
        candidates.append(SURGICAL_POSTOPERATIVE_TRAUMA_GROUP)
    if any(_matches_respiratory(stem) for stem in normalized_stems):
        candidates.append(RESPIRATORY_PULMONARY_GROUP)
    if any(_matches_infection(stem) for stem in normalized_stems):
        candidates.append(INFECTION_SEPSIS_NON_PULMONARY_GROUP)
    if any(_matches_cardiovascular(stem) for stem in normalized_stems):
        candidates.append(CARDIOVASCULAR_GROUP)
    if any(_matches_neurologic(stem) for stem in normalized_stems):
        candidates.append(NEUROLOGIC_GROUP)

    return tuple(candidates)


def derive_icd10_disease_group(raw_codes: object) -> ICD10DiseaseGroupMatch:
    normalized_stems = normalize_icd10_stems(raw_codes)
    candidate_groups = match_candidate_disease_groups(normalized_stems)
    final_group = (
        candidate_groups[0] if candidate_groups else OTHER_MIXED_UNCATEGORIZED_GROUP
    )
    return ICD10DiseaseGroupMatch(
        normalized_stems=normalized_stems,
        candidate_groups=candidate_groups,
        final_group=final_group,
    )


def _matches_surgical_or_postoperative(stem: str) -> bool:
    return stem[:1] in {"S", "T", "V", "W", "X", "Y"} or stem in _POSTOPERATIVE_COMPLICATION_STEMS


def _matches_respiratory(stem: str) -> bool:
    return stem.startswith("J") and stem != "J95"


def _matches_infection(stem: str) -> bool:
    return stem[:1] in {"A", "B"} or stem in {"R65", "U07"}


def _matches_cardiovascular(stem: str) -> bool:
    return stem == "R57" or (stem.startswith("I") and not stem.startswith("I6") and stem != "I97")


def _matches_neurologic(stem: str) -> bool:
    return stem.startswith("G") or stem.startswith("I6") or stem in _CNS_NEOPLASM_STEMS
