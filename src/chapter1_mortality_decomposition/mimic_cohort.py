from __future__ import annotations

import argparse
import csv
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "ch1_mimic_cohort.yaml"

INVASIVE_VENT_ITEMID = 225792
NONINVASIVE_VENT_ITEMID = 225794


@dataclass(frozen=True)
class MimicCohortConfig:
    mimic_root: Path
    reports_dir: Path
    processed_dir: Path
    cohort_output_csv: str
    chunksize: int = 200_000

    @property
    def cohort_output_path(self) -> Path:
        return self.processed_dir / self.cohort_output_csv


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path) -> MimicCohortConfig:
    raw: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(f"Unsupported config line in {path}: {line!r}")
        key, value = stripped.split(":", 1)
        raw[key.strip()] = value.strip().strip("'\"")

    required = ["mimic_root", "reports_dir", "processed_dir", "cohort_output_csv"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"MIMIC cohort config {path} is missing keys: {missing}")

    return MimicCohortConfig(
        mimic_root=_resolve_path(raw["mimic_root"]),
        reports_dir=_resolve_path(raw["reports_dir"]),
        processed_dir=_resolve_path(raw["processed_dir"]),
        cohort_output_csv=raw["cohort_output_csv"],
        chunksize=int(raw.get("chunksize", "200000")),
    )


def table_path(mimic_root: Path, table: str) -> Path:
    module = {
        "icustays": "icu",
        "procedureevents": "icu",
        "patients": "hosp",
        "admissions": "hosp",
    }[table]
    return mimic_root / module / f"{table}.csv.gz"


def require_columns(path: Path, required_columns: Iterable[str], *, table_name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required MIMIC table for {table_name}: {path}")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(required_columns) - set(header))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def validate_inputs(config: MimicCohortConfig) -> None:
    require_columns(
        table_path(config.mimic_root, "icustays"),
        {
            "subject_id",
            "hadm_id",
            "stay_id",
            "intime",
            "outtime",
            "first_careunit",
            "last_careunit",
        },
        table_name="icu.icustays",
    )
    require_columns(
        table_path(config.mimic_root, "patients"),
        {"subject_id", "gender", "anchor_age", "anchor_year"},
        table_name="hosp.patients",
    )
    require_columns(
        table_path(config.mimic_root, "admissions"),
        {
            "subject_id",
            "hadm_id",
            "deathtime",
            "hospital_expire_flag",
            "discharge_location",
        },
        table_name="hosp.admissions",
    )
    require_columns(
        table_path(config.mimic_root, "procedureevents"),
        {"subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid"},
        table_name="icu.procedureevents",
    )


def _load_csv_to_sqlite(
    conn: sqlite3.Connection,
    path: Path,
    table_name: str,
    usecols: list[str],
    *,
    chunksize: int,
    itemid_filter: set[int] | None = None,
) -> None:
    first = True
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        if itemid_filter is not None:
            chunk = chunk[chunk["itemid"].isin(itemid_filter)].copy()
        if chunk.empty and not first:
            continue
        chunk.to_sql(table_name, conn, if_exists="replace" if first else "append", index=False)
        first = False
    if first:
        pd.DataFrame(columns=usecols).to_sql(table_name, conn, if_exists="replace", index=False)


def load_required_tables(conn: sqlite3.Connection, config: MimicCohortConfig) -> None:
    _load_csv_to_sqlite(
        conn,
        table_path(config.mimic_root, "icustays"),
        "icustays",
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            "first_careunit",
            "last_careunit",
            "intime",
            "outtime",
        ],
        chunksize=config.chunksize,
    )
    _load_csv_to_sqlite(
        conn,
        table_path(config.mimic_root, "patients"),
        "patients",
        ["subject_id", "gender", "anchor_age", "anchor_year"],
        chunksize=config.chunksize,
    )
    _load_csv_to_sqlite(
        conn,
        table_path(config.mimic_root, "admissions"),
        "admissions",
        ["subject_id", "hadm_id", "deathtime", "hospital_expire_flag", "discharge_location"],
        chunksize=config.chunksize,
    )
    _load_csv_to_sqlite(
        conn,
        table_path(config.mimic_root, "procedureevents"),
        "procedureevents",
        ["subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid"],
        chunksize=config.chunksize,
        itemid_filter={INVASIVE_VENT_ITEMID, NONINVASIVE_VENT_ITEMID},
    )


COHORT_SQL = f"""
DROP TABLE IF EXISTS vent_by_stay;
CREATE TEMP TABLE vent_by_stay AS
SELECT
    stay_id,
    SUM(
        CASE
            WHEN itemid = {INVASIVE_VENT_ITEMID}
             AND starttime IS NOT NULL
             AND endtime IS NOT NULL
            THEN (julianday(endtime) - julianday(starttime)) * 24.0
            ELSE 0.0
        END
    ) AS total_invasive_hours,
    SUM(CASE WHEN itemid = {INVASIVE_VENT_ITEMID} THEN 1 ELSE 0 END) AS invasive_episode_count,
    SUM(
        CASE
            WHEN itemid = {NONINVASIVE_VENT_ITEMID}
             AND starttime IS NOT NULL
             AND endtime IS NOT NULL
            THEN (julianday(endtime) - julianday(starttime)) * 24.0
            ELSE 0.0
        END
    ) AS total_noninvasive_hours,
    SUM(CASE WHEN itemid = {NONINVASIVE_VENT_ITEMID} THEN 1 ELSE 0 END) AS noninvasive_episode_count
FROM procedureevents
GROUP BY stay_id;

DROP TABLE IF EXISTS ranked_stays;
CREATE TEMP TABLE ranked_stays AS
SELECT
    i.subject_id,
    i.hadm_id,
    i.stay_id,
    i.intime,
    i.outtime,
    i.first_careunit,
    i.last_careunit,
    p.gender,
    p.anchor_age,
    p.anchor_year,
    CAST(p.anchor_age AS REAL)
        + (CAST(strftime('%Y', i.intime) AS INTEGER) - CAST(p.anchor_year AS INTEGER))
        AS age_at_icu_intime,
    ROW_NUMBER() OVER (
        PARTITION BY i.subject_id
        ORDER BY datetime(i.intime) ASC, CAST(i.stay_id AS INTEGER) ASC
    ) AS stay_rank,
    (julianday(i.outtime) - julianday(i.intime)) * 24.0 AS icu_los_hours,
    COALESCE(v.total_invasive_hours, 0.0) AS total_invasive_hours,
    COALESCE(v.invasive_episode_count, 0) AS invasive_episode_count,
    COALESCE(v.total_noninvasive_hours, 0.0) AS total_noninvasive_hours,
    COALESCE(v.noninvasive_episode_count, 0) AS noninvasive_episode_count,
    a.deathtime,
    a.hospital_expire_flag,
    a.discharge_location,
    CASE
        WHEN a.deathtime IS NOT NULL
         AND julianday(a.deathtime) <= julianday(i.outtime)
        THEN 1 ELSE 0
    END AS icu_mortality,
    CASE
        WHEN a.deathtime IS NOT NULL
         AND julianday(a.deathtime) > julianday(i.outtime)
        THEN 1 ELSE 0
    END AS death_after_icu_outtime,
    CASE
        WHEN CAST(p.anchor_age AS REAL)
             + (CAST(strftime('%Y', i.intime) AS INTEGER) - CAST(p.anchor_year AS INTEGER))
             >= 18
        THEN 1 ELSE 0
    END AS passed_adult_age,
    CASE
        WHEN ROW_NUMBER() OVER (
            PARTITION BY i.subject_id
            ORDER BY datetime(i.intime) ASC, CAST(i.stay_id AS INTEGER) ASC
        ) = 1
        THEN 1 ELSE 0
    END AS passed_first_icu_stay,
    CASE WHEN COALESCE(v.total_invasive_hours, 0.0) >= 24.0 THEN 1 ELSE 0 END
        AS passed_invasive_vent_ge_24h,
    CASE
        WHEN COALESCE(v.total_invasive_hours, 0.0)
             <= ((julianday(i.outtime) - julianday(i.intime)) * 24.0) + 4.0
        THEN 1 ELSE 0
    END AS passed_vent_los_qc
FROM icustays i
INNER JOIN patients p
    ON i.subject_id = p.subject_id
INNER JOIN admissions a
    ON i.hadm_id = a.hadm_id
   AND i.subject_id = a.subject_id
LEFT JOIN vent_by_stay v
    ON i.stay_id = v.stay_id;

DROP TABLE IF EXISTS cohort_stay_level;
CREATE TEMP TABLE cohort_stay_level AS
SELECT
    r.*,
    CASE
        WHEN EXISTS (
            SELECT 1
            FROM icustays later
            WHERE later.subject_id = r.subject_id
              AND later.hadm_id = r.hadm_id
              AND later.stay_id <> r.stay_id
              AND datetime(later.intime) > datetime(r.outtime)
        )
        THEN 1 ELSE 0
    END AS same_hadm_later_icu_stay,
    CASE
        WHEN passed_adult_age = 1
         AND passed_first_icu_stay = 1
         AND passed_invasive_vent_ge_24h = 1
         AND passed_vent_los_qc = 1
        THEN 1 ELSE 0
    END AS retained_stay_level_cohort
FROM ranked_stays r;
"""


def execute_cohort_sql(conn: sqlite3.Connection) -> pd.DataFrame:
    conn.executescript(COHORT_SQL)
    return pd.read_sql_query(
        """
        SELECT
            subject_id,
            hadm_id,
            stay_id,
            intime,
            outtime,
            age_at_icu_intime,
            gender,
            stay_rank,
            icu_los_hours,
            total_invasive_hours,
            total_noninvasive_hours,
            invasive_episode_count,
            noninvasive_episode_count,
            deathtime,
            hospital_expire_flag,
            discharge_location,
            icu_mortality,
            death_after_icu_outtime,
            first_careunit,
            last_careunit,
            same_hadm_later_icu_stay,
            passed_adult_age,
            passed_first_icu_stay,
            passed_invasive_vent_ge_24h,
            passed_vent_los_qc,
            retained_stay_level_cohort
        FROM cohort_stay_level
        ORDER BY subject_id, datetime(intime), stay_id
        """,
        conn,
    )


def percentile_summary(values: pd.Series) -> dict[str, float | int | str]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"count": 0, "min": "", "p25": "", "median": "", "p75": "", "max": ""}
    p25, median, p75 = np.percentile(numeric.to_numpy(dtype=float), [25, 50, 75])
    return {
        "count": int(numeric.shape[0]),
        "min": float(numeric.min()),
        "p25": float(p25),
        "median": float(median),
        "p75": float(p75),
        "max": float(numeric.max()),
    }


def build_flow(cohort: pd.DataFrame) -> pd.DataFrame:
    total = int(cohort.shape[0])
    adult = cohort[cohort["passed_adult_age"].eq(1)]
    first = adult[adult["passed_first_icu_stay"].eq(1)]
    vent = first[first["passed_invasive_vent_ge_24h"].eq(1)]
    qc = vent[vent["passed_vent_los_qc"].eq(1)]
    steps = [
        ("total_icu_stays_considered", total, "All ICU stays in icu.icustays joined to patients and admissions."),
        ("age_ge_18_retained", int(adult.shape[0]), "Retain age_at_icu_intime >= 18."),
        ("first_icu_stay_retained", int(first.shape[0]), "Retain stay_rank = 1 per subject."),
        ("invasive_vent_ge_24h_retained", int(vent.shape[0]), "Retain total invasive ventilation hours >= 24 using itemid 225792 only."),
        ("ventilation_qc_guard_retained", int(qc.shape[0]), "Retain total_invasive_hours <= icu_los_hours + 4."),
        ("final_retained_stay_level_cohort_count", int(qc.shape[0]), "Final 5.1.b1 stay-level cohort; valid prediction-instance eligibility deferred."),
    ]
    rows = []
    previous = None
    for order, (step, count, note) in enumerate(steps, start=1):
        rows.append(
            {
                "step_order": order,
                "flow_step": step,
                "retained_count": count,
                "excluded_from_previous": "" if previous is None else previous - count,
                "step_note": note,
            }
        )
        previous = count
    return pd.DataFrame(rows)


def build_qc_summary(cohort: pd.DataFrame) -> pd.DataFrame:
    retained = cohort[cohort["retained_stay_level_cohort"].eq(1)].copy()
    pre_qc = cohort[
        cohort["passed_adult_age"].eq(1)
        & cohort["passed_first_icu_stay"].eq(1)
        & cohort["passed_invasive_vent_ge_24h"].eq(1)
    ]
    rows: list[dict[str, object]] = []

    def add_stat(prefix: str, values: pd.Series) -> None:
        for metric, value in percentile_summary(values).items():
            rows.append({"metric": f"{prefix}_{metric}", "value": value, "scope": "retained_cohort"})

    add_stat("age_at_icu_intime", retained["age_at_icu_intime"])
    add_stat("icu_los_hours", retained["icu_los_hours"])
    add_stat("total_invasive_hours", retained["total_invasive_hours"])
    rows.extend(
        [
            {
                "metric": "icu_mortality_count",
                "value": int(retained["icu_mortality"].sum()),
                "scope": "retained_cohort",
            },
            {
                "metric": "death_after_icu_outtime_count",
                "value": int(retained["death_after_icu_outtime"].sum()),
                "scope": "retained_cohort",
            },
            {
                "metric": "missing_discharge_location_count",
                "value": int(retained["discharge_location"].isna().sum() + retained["discharge_location"].astype("string").str.strip().eq("").sum()),
                "scope": "retained_cohort",
            },
            {
                "metric": "vent_vs_los_qc_guard_fail_count",
                "value": int(pre_qc["passed_vent_los_qc"].eq(0).sum()),
                "scope": "adult_first_stay_with_invasive_vent_ge_24h",
            },
            {
                "metric": "noninvasive_vent_episode_stay_count",
                "value": int(retained["noninvasive_episode_count"].gt(0).sum()),
                "scope": "retained_cohort_context_only",
            },
            {
                "metric": "noninvasive_vent_total_hours_sum",
                "value": float(pd.to_numeric(retained["total_noninvasive_hours"], errors="coerce").fillna(0).sum()),
                "scope": "retained_cohort_context_only",
            },
            {
                "metric": "retained_icu_los_hours_lt_24_count",
                "value": int(pd.to_numeric(retained["icu_los_hours"], errors="coerce").lt(24).sum()),
                "scope": "retained_cohort_ventilation_timing_qc",
            },
            {
                "metric": "retained_total_invasive_hours_gt_icu_los_count",
                "value": int(
                    pd.to_numeric(retained["total_invasive_hours"], errors="coerce").gt(
                        pd.to_numeric(retained["icu_los_hours"], errors="coerce")
                    ).sum()
                ),
                "scope": "retained_cohort_ventilation_timing_qc",
            },
            {
                "metric": "retained_total_invasive_hours_gt_icu_los_plus_2_count",
                "value": int(
                    pd.to_numeric(retained["total_invasive_hours"], errors="coerce").gt(
                        pd.to_numeric(retained["icu_los_hours"], errors="coerce") + 2
                    ).sum()
                ),
                "scope": "retained_cohort_ventilation_timing_qc",
            },
            {
                "metric": "retained_total_invasive_hours_gt_icu_los_plus_4_count",
                "value": int(
                    pd.to_numeric(retained["total_invasive_hours"], errors="coerce").gt(
                        pd.to_numeric(retained["icu_los_hours"], errors="coerce") + 4
                    ).sum()
                ),
                "scope": "retained_cohort_ventilation_timing_qc",
            },
            {
                "metric": "retained_invasive_and_noninvasive_episode_count",
                "value": int(
                    (
                        pd.to_numeric(retained["invasive_episode_count"], errors="coerce").fillna(0).gt(0)
                        & pd.to_numeric(retained["noninvasive_episode_count"], errors="coerce").fillna(0).gt(0)
                    ).sum()
                ),
                "scope": "retained_cohort_context_only",
            },
            {
                "metric": "valid_prediction_instance_eligibility_enforced",
                "value": 0,
                "scope": "deferred_after_block_construction",
            },
        ]
    )
    return pd.DataFrame(rows)


def build_ventilation_qc_addendum(cohort: pd.DataFrame) -> pd.DataFrame:
    retained = cohort[cohort["retained_stay_level_cohort"].eq(1)].copy()
    los = pd.to_numeric(retained["icu_los_hours"], errors="coerce")
    invasive = pd.to_numeric(retained["total_invasive_hours"], errors="coerce")
    diff = invasive - los
    invasive_episode_count = pd.to_numeric(
        retained["invasive_episode_count"],
        errors="coerce",
    ).fillna(0)
    noninvasive_episode_count = pd.to_numeric(
        retained["noninvasive_episode_count"],
        errors="coerce",
    ).fillna(0)

    rows: list[dict[str, object]] = [
        {
            "summary_family": "edge_case_count",
            "metric": "retained_icu_los_hours_lt_24_count",
            "value": int(los.lt(24).sum()),
            "denominator": int(retained.shape[0]),
            "note": "Retained stays with ICU LOS under 24h despite invasive ventilation >=24h.",
        },
        {
            "summary_family": "edge_case_count",
            "metric": "retained_total_invasive_hours_gt_icu_los_count",
            "value": int(invasive.gt(los).sum()),
            "denominator": int(retained.shape[0]),
            "note": "Retained stays where summed invasive procedure duration exceeds ICU LOS by any amount.",
        },
        {
            "summary_family": "edge_case_count",
            "metric": "retained_total_invasive_hours_gt_icu_los_plus_2_count",
            "value": int(invasive.gt(los + 2).sum()),
            "denominator": int(retained.shape[0]),
            "note": "Retained stays where summed invasive procedure duration exceeds ICU LOS by more than 2 hours.",
        },
        {
            "summary_family": "edge_case_count",
            "metric": "retained_total_invasive_hours_gt_icu_los_plus_4_count",
            "value": int(invasive.gt(los + 4).sum()),
            "denominator": int(retained.shape[0]),
            "note": "Retained stays exceeding the current +4h QC guard; should be zero in the retained cohort.",
        },
        {
            "summary_family": "edge_case_count",
            "metric": "retained_invasive_and_noninvasive_episode_count",
            "value": int((invasive_episode_count.gt(0) & noninvasive_episode_count.gt(0)).sum()),
            "denominator": int(retained.shape[0]),
            "note": "Retained stays with both invasive and non-invasive procedure episodes present; non-invasive time is context only.",
        },
    ]
    for metric, value in percentile_summary(diff).items():
        rows.append(
            {
                "summary_family": "invasive_minus_los_hours_distribution",
                "metric": f"invasive_minus_los_hours_{metric}",
                "value": value,
                "denominator": int(retained.shape[0]),
                "note": "Distribution of total_invasive_hours - icu_los_hours in retained stays.",
            }
        )
    for episode_count, count in invasive_episode_count.value_counts().sort_index().items():
        rows.append(
            {
                "summary_family": "invasive_episode_count_distribution",
                "metric": f"invasive_episode_count_{int(episode_count)}",
                "value": int(count),
                "denominator": int(retained.shape[0]),
                "note": "Distribution of invasive ventilation procedure episode counts per retained stay.",
            }
        )
    return pd.DataFrame(rows)


def build_transfer_discharge_summary(cohort: pd.DataFrame) -> pd.DataFrame:
    retained = cohort[cohort["retained_stay_level_cohort"].eq(1)].copy()
    rows: list[dict[str, object]] = []
    discharge = retained["discharge_location"].fillna("").astype("string").replace("", "(missing)")
    for value, count in discharge.value_counts(dropna=False).sort_index().items():
        rows.append(
            {
                "summary_family": "discharge_location_distribution",
                "summary_value": str(value),
                "count": int(count),
                "note": "Distribution in retained 5.1.b1 stay-level cohort.",
            }
        )
    rows.append(
        {
            "summary_family": "same_hadm_later_icu_stay_count",
            "summary_value": "retained_first_stays_with_later_icu_stay_same_hadm",
            "count": int(retained["same_hadm_later_icu_stay"].sum()),
            "note": "Verifies stays are not merged; later ICU stays in the same hospitalization are not part of the retained first-stay cohort.",
        }
    )
    for column in ["first_careunit", "last_careunit"]:
        values = retained[column].fillna("").astype("string").replace("", "(missing)")
        for value, count in values.value_counts(dropna=False).sort_index().items():
            rows.append(
                {
                    "summary_family": f"{column}_distribution",
                    "summary_value": str(value),
                    "count": int(count),
                    "note": "Careunit context only; not used to redefine or merge ICU stays.",
                }
            )
    return pd.DataFrame(rows)


def ventilation_qc_interpretation(cohort: pd.DataFrame) -> str:
    addendum = build_ventilation_qc_addendum(cohort)

    def value(metric: str) -> int:
        selected = addendum.loc[addendum["metric"].eq(metric), "value"]
        if selected.empty:
            return 0
        return int(float(selected.iloc[0]))

    retained_count = int(cohort["retained_stay_level_cohort"].sum())
    los_lt_24 = value("retained_icu_los_hours_lt_24_count")
    gt_los = value("retained_total_invasive_hours_gt_icu_los_count")
    gt_los_plus_2 = value("retained_total_invasive_hours_gt_icu_los_plus_2_count")
    gt_los_plus_4 = value("retained_total_invasive_hours_gt_icu_los_plus_4_count")
    if retained_count == 0:
        return "No retained stays were available for ventilation-vs-LOS QC interpretation."
    return (
        f"Ventilation-vs-LOS edge cases are limited in the retained cohort: {los_lt_24} "
        f"of {retained_count} retained stays have ICU LOS <24h; {gt_los} have summed "
        f"invasive ventilation time greater than ICU LOS; {gt_los_plus_2} exceed ICU LOS "
        f"by more than 2h; and {gt_los_plus_4} exceed ICU LOS by more than 4h. The current "
        "+4h QC guard therefore appears sufficient for this stay-level gate. Short-LOS "
        "retained stays and small positive timing differences should be carried forward as "
        "documented procedure-timing limitations rather than triggering a cohort rule change."
    )


def write_note(config: MimicCohortConfig, cohort: pd.DataFrame) -> None:
    retained_count = int(cohort["retained_stay_level_cohort"].sum())
    total_count = int(cohort.shape[0])
    mortality_count = int(cohort.loc[cohort["retained_stay_level_cohort"].eq(1), "icu_mortality"].sum())
    lines = [
        "# Chapter 1 MIMIC Stay-Level Cohort Extraction Note",
        "",
        "## Purpose",
        "",
        "This report documents the 5.1.b1 stay-level MIMIC operationalization of the frozen Chapter 1 cohort. It is not 8h block construction, horizon-label generation, valid prediction-instance filtering, or model fitting.",
        "",
        "## Data Source",
        "",
        f"- MIMIC root: `{config.mimic_root}`",
        f"- ICU stays considered: {total_count}",
        f"- Retained stay-level cohort rows: {retained_count}",
        f"- Retained in-ICU mortality count: {mortality_count}",
        "",
        "## Implemented Gates",
        "",
        "- Adult inclusion: `age_at_icu_intime = anchor_age + (year(icustays.intime) - anchor_year)`, retain age >= 18.",
        "- First ICU stay: `ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY intime ASC, stay_id ASC)`, retain `stay_rank = 1`.",
        "- Mechanical ventilation: sum invasive ventilation procedure time for itemid `225792`; retain `total_invasive_hours >= 24`.",
        "- Ventilation QC guard: retain `total_invasive_hours <= icu_los_hours + 4`.",
        "- ICU mortality: `admissions.deathtime` non-null and `deathtime <= icustays.outtime`.",
        "- ICU entry/exit: `icustays.intime` and `icustays.outtime` are authoritative.",
        "- Transfer/discharge handling: each `stay_id` remains distinct; multiple ICU stays are not merged.",
        "",
        "Non-invasive ventilation itemid `225794` is summarized only for QC/context and is not counted toward the >=24h gate.",
        "",
        "No LOS >=48h, trauma, AMA, hospice, or discharge-location exclusions are applied in 5.1.b1.",
        "",
        "## Discharge Location Clarification",
        "",
        "`admissions.discharge_location` is a hospital-discharge disposition field, not an ICU-discharge outcome field. Therefore `discharge_location = DIED` is not the primary Chapter 1 ICU mortality definition. The ICU mortality definition remains `deathtime` non-null and `deathtime <= icustays.outtime`. Differences between discharge-location counts and ICU-mortality counts are expected because these fields describe different event levels.",
        "",
        "## Ventilation Timing QC Addendum",
        "",
        ventilation_qc_interpretation(cohort),
        "",
        "## Deferred Requirement",
        "",
        "Valid prediction-instance eligibility is required later for final Chapter 1 preprocessing, but is intentionally not enforced in 5.1.b1 because it depends on block construction and horizon-specific label availability.",
        "",
        "## Outputs",
        "",
        "- `reports/ch1_mimic_cohort_flow.csv`",
        "- `reports/ch1_mimic_cohort_qc_summary.csv`",
        "- `reports/ch1_mimic_transfer_discharge_summary.csv`",
        "- `reports/ch1_mimic_ventilation_qc_addendum.csv`",
        f"- `{config.cohort_output_path}`",
        "",
    ]
    (config.reports_dir / "ch1_mimic_cohort_note.md").write_text("\n".join(lines))


def run_cohort(config: MimicCohortConfig) -> pd.DataFrame:
    validate_inputs(config)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ch1_mimic_cohort_") as tmpdir:
        sqlite_path = Path(tmpdir) / "cohort.sqlite"
        conn = sqlite3.connect(sqlite_path)
        try:
            load_required_tables(conn, config)
            cohort = execute_cohort_sql(conn)
        finally:
            conn.close()

    retained = cohort[cohort["retained_stay_level_cohort"].eq(1)].copy()
    retained.to_csv(config.cohort_output_path, index=False)
    build_flow(cohort).to_csv(config.reports_dir / "ch1_mimic_cohort_flow.csv", index=False)
    build_qc_summary(cohort).to_csv(
        config.reports_dir / "ch1_mimic_cohort_qc_summary.csv",
        index=False,
    )
    build_transfer_discharge_summary(cohort).to_csv(
        config.reports_dir / "ch1_mimic_transfer_discharge_summary.csv",
        index=False,
    )
    build_ventilation_qc_addendum(cohort).to_csv(
        config.reports_dir / "ch1_mimic_ventilation_qc_addendum.csv",
        index=False,
    )
    write_note(config, cohort)
    return retained


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and verify the Chapter 1 MIMIC stay-level cohort."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mimic-root", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.mimic_root is not None:
        config = MimicCohortConfig(**{**config.__dict__, "mimic_root": _resolve_path(args.mimic_root)})
    if args.reports_dir is not None:
        config = MimicCohortConfig(**{**config.__dict__, "reports_dir": _resolve_path(args.reports_dir)})
    if args.processed_dir is not None:
        config = MimicCohortConfig(**{**config.__dict__, "processed_dir": _resolve_path(args.processed_dir)})
    retained = run_cohort(config)
    print(
        f"Wrote MIMIC stay-level cohort with {len(retained)} retained stays to "
        f"{config.cohort_output_path}"
    )


if __name__ == "__main__":
    main()
