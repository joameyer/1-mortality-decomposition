from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from chapter1_mortality_decomposition.config import Chapter1Config, default_chapter1_config
from chapter1_mortality_decomposition.utils import require_columns


SPLIT_NAMES = ("train", "validation", "test")
SPLIT_PROPORTIONS = (0.70, 0.15, 0.15)
SPLIT_PROPORTION_LOOKUP = dict(zip(SPLIT_NAMES, SPLIT_PROPORTIONS))


@dataclass(frozen=True)
class Chapter1StaySplitResult:
    stay_assignments: pd.DataFrame
    stay_summary: pd.DataFrame
    verification_summary: pd.DataFrame


def _allocate_split_counts(total_count: int) -> dict[str, int]:
    raw_counts = [total_count * proportion for proportion in SPLIT_PROPORTIONS]
    allocated = [int(raw_count) for raw_count in raw_counts]
    remainder = total_count - sum(allocated)

    if remainder > 0:
        fractional_order = sorted(
            range(len(SPLIT_NAMES)),
            key=lambda index: (raw_counts[index] - allocated[index], -index),
            reverse=True,
        )
        for index in fractional_order[:remainder]:
            allocated[index] += 1

    return dict(zip(SPLIT_NAMES, allocated))


def _summarize_stay_assignments(stay_assignments: pd.DataFrame) -> pd.DataFrame:
    hospitals = sorted(stay_assignments["hospital_id"].dropna().astype("string").unique().tolist())
    rows: list[dict[str, object]] = []

    def append_summary_rows(summary_level: str, hospital_id: object, group_df: pd.DataFrame) -> None:
        total_stays = int(group_df["stay_id_global"].nunique(dropna=True))
        for split in SPLIT_NAMES:
            split_df = group_df[group_df["split"] == split]
            stay_count = int(split_df["stay_id_global"].nunique(dropna=True))
            positive_stays = int(split_df["icu_mortality"].eq(1).sum())
            negative_stays = int(split_df["icu_mortality"].eq(0).sum())
            rows.append(
                {
                    "summary_level": summary_level,
                    "hospital_id": hospital_id,
                    "split": split,
                    "stay_count": stay_count,
                    "positive_stays": positive_stays,
                    "negative_stays": negative_stays,
                    "label_prevalence": (
                        float(positive_stays / stay_count) if stay_count else pd.NA
                    ),
                    "actual_proportion": (
                        float(stay_count / total_stays) if total_stays else pd.NA
                    ),
                    "target_proportion": SPLIT_PROPORTION_LOOKUP[split],
                }
            )

    append_summary_rows("overall", pd.NA, stay_assignments)
    for hospital_id in hospitals:
        append_summary_rows(
            "hospital",
            hospital_id,
            stay_assignments[stay_assignments["hospital_id"].astype("string") == hospital_id],
        )

    return pd.DataFrame(rows)


def _build_split_verification_summary(
    retained_cohort: pd.DataFrame,
    stay_assignments: pd.DataFrame,
) -> pd.DataFrame:
    retained_stays = retained_cohort["stay_id_global"].astype("string")
    assignment_stays = stay_assignments["stay_id_global"].astype("string")
    duplicate_assignment_counts = stay_assignments.groupby("stay_id_global", dropna=False).size()

    rows = [
        {
            "check_id": "no_stay_in_more_than_one_split",
            "passed": bool(duplicate_assignment_counts.eq(1).all()) if not stay_assignments.empty else True,
            "detail": "Each retained stay_id_global is assigned to exactly one split.",
        },
        {
            "check_id": "all_retained_stays_assigned_to_split",
            "passed": set(assignment_stays.tolist()) == set(retained_stays.tolist()),
            "detail": "Every retained stay from the canonical Chapter 1 cohort has a split assignment.",
        },
    ]
    return pd.DataFrame(rows)


def build_chapter1_stay_splits(
    retained_cohort: pd.DataFrame,
    config: Chapter1Config | None = None,
) -> Chapter1StaySplitResult:
    config = config or default_chapter1_config()

    require_columns(
        retained_cohort,
        {"stay_id_global", "hospital_id", "icu_mortality"},
        "retained_cohort",
    )

    cohort = retained_cohort[["stay_id_global", "hospital_id", "icu_mortality"]].copy()
    cohort["stay_id_global"] = cohort["stay_id_global"].astype("string")
    cohort["hospital_id"] = cohort["hospital_id"].astype("string")
    cohort["icu_mortality"] = pd.to_numeric(cohort["icu_mortality"], errors="coerce")

    if cohort["stay_id_global"].duplicated().any():
        duplicate_ids = (
            cohort.loc[cohort["stay_id_global"].duplicated(keep=False), "stay_id_global"]
            .dropna()
            .drop_duplicates()
            .tolist()[:10]
        )
        raise ValueError(
            "Chapter 1 split generation requires one retained cohort row per stay_id_global. "
            f"Duplicate IDs found: {duplicate_ids}"
        )

    assignment_frames: list[pd.DataFrame] = []
    hospitals = sorted(cohort["hospital_id"].dropna().astype("string").unique().tolist())
    for hospital_index, hospital_id in enumerate(hospitals):
        hospital_df = cohort[cohort["hospital_id"].astype("string") == hospital_id].copy()
        stratum_frames: list[pd.DataFrame] = []
        outcome_values = sorted(hospital_df["icu_mortality"].dropna().astype(int).unique().tolist(), reverse=True)
        for outcome_index, outcome_value in enumerate(outcome_values):
            stratum = hospital_df[hospital_df["icu_mortality"].eq(outcome_value)].copy()
            if stratum.empty:
                continue

            shuffled = stratum.sort_values("stay_id_global").sample(
                frac=1,
                random_state=(config.split_random_seed + (hospital_index * 100) + outcome_index),
            ).reset_index(drop=True)
            split_counts = _allocate_split_counts(int(shuffled.shape[0]))

            start = 0
            split_frames: list[pd.DataFrame] = []
            for split_name in SPLIT_NAMES:
                count = split_counts[split_name]
                split_frame = shuffled.iloc[start : start + count].copy()
                split_frame["split"] = split_name
                split_frames.append(split_frame)
                start += count
            stratum_frames.append(pd.concat(split_frames, ignore_index=True))

        if stratum_frames:
            assignment_frames.append(pd.concat(stratum_frames, ignore_index=True))

    if assignment_frames:
        stay_assignments = pd.concat(assignment_frames, ignore_index=True)
    else:
        stay_assignments = cohort.iloc[0:0].copy()
        stay_assignments["split"] = pd.Series(dtype="string")

    stay_assignments["split_random_seed"] = config.split_random_seed
    stay_assignments = stay_assignments[
        ["stay_id_global", "hospital_id", "icu_mortality", "split", "split_random_seed"]
    ].sort_values(["hospital_id", "split", "stay_id_global"]).reset_index(drop=True)

    stay_summary = _summarize_stay_assignments(stay_assignments)
    verification_summary = _build_split_verification_summary(cohort, stay_assignments)
    return Chapter1StaySplitResult(
        stay_assignments=stay_assignments,
        stay_summary=stay_summary,
        verification_summary=verification_summary,
    )
