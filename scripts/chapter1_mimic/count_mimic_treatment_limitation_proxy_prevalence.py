#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MIMIC_ROOT = REPO_ROOT / "mimic-iv-demo" / "data"
DEFAULT_INVENTORY_PATH = (
    REPO_ROOT
    / "analysis_artifacts"
    / "chapter1_mimic_treatment_limitation_proxies"
    / "mimic_treatment_limitation_proxy_inventory_schema_scan.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "analysis_artifacts" / "chapter1_mimic_treatment_limitation_proxies"
)
DEFAULT_COUNTS_NAME = "mimic_treatment_limitation_proxy_full_data_counts.csv"
DEFAULT_TIMING_NAME = "mimic_treatment_limitation_proxy_timing_summary.csv"
DEFAULT_DOMAIN_NAME = "mimic_treatment_limitation_proxy_domain_summary.csv"
DEFAULT_NOTE_NAME = "mimic_treatment_limitation_full_data_counts_note.md"
DEFAULT_MANIFEST_NAME = "manifest_full_data_counts.json"

DOMAIN_ORDER = [
    "code_status_dnr_dni",
    "comfort_measures_only",
    "withdrawal_or_withholding",
    "palliative_care",
    "hospice",
    "brain_death_or_organ_donation",
    "ama_or_nonstandard_discharge",
    "ambiguous_goals_of_care",
]

COUNTS_COLUMNS = [
    "proxy_domain",
    "source_schema",
    "source_table",
    "source_field",
    "itemid_or_code",
    "raw_label",
    "proxy_strength",
    "recommended_use",
    "n_events_or_records",
    "n_unique_subjects",
    "n_unique_hadm",
    "n_unique_stays",
    "n_in_chapter1_mimic_cohort",
    "n_proxy_positive_stays",
    "n_proxy_positive_fatal_stays",
    "n_proxy_positive_nonfatal_stays",
    "share_proxy_positive_stays",
    "share_proxy_positive_fatal_stays",
    "share_proxy_positive_nonfatal_stays",
    "timing_summary_available",
    "main_limitation",
    "notes",
]

TIMING_COLUMNS = [
    "proxy_domain",
    "source_schema",
    "source_table",
    "source_field",
    "itemid_or_code",
    "raw_label",
    "timestamp_field",
    "n_proxy_positive_stays_with_timestamp",
    "n_before_icu_intime",
    "n_after_icu_intime",
    "n_before_icu_outtime_or_discharge",
    "n_after_icu_outtime_or_discharge",
    "n_before_death_for_fatal_stays",
    "n_after_death_for_fatal_stays",
    "median_hours_from_icu_intime",
    "iqr_hours_from_icu_intime",
    "median_hours_before_death_fatal_stays",
    "iqr_hours_before_death_fatal_stays",
    "timing_interpretation",
    "notes",
]

DOMAIN_COLUMNS = [
    "proxy_domain",
    "proxy_strength_max",
    "n_sources",
    "n_proxy_positive_stays_any_source",
    "n_proxy_positive_fatal_stays_any_source",
    "share_proxy_positive_stays_any_source",
    "share_proxy_positive_fatal_stays_any_source",
    "best_timing_usability",
    "recommended_use_domain",
    "main_limitation_domain",
    "notes",
]

POSITIVE_CODE_STATUS_RE = re.compile(
    r"\b(?:dnr|dni|dnar)\b|do\s+not\s+resuscitate|do\s+not\s+intubate|no\s+cpr",
    flags=re.IGNORECASE,
)
FULL_CODE_RE = re.compile(r"full\s+code|resuscitate\s*\(\s*full\s*code\s*\)", flags=re.IGNORECASE)
AMA_RE = re.compile(
    r"against\s+medical\s+advice|\bama\b|elop|left\s+without\s+being\s+seen",
    flags=re.IGNORECASE,
)
HOSPICE_RE = re.compile(r"hospice", flags=re.IGNORECASE)


@dataclass(frozen=True)
class Candidate:
    proxy_domain: str
    source_schema: str
    source_table: str
    source_field: str
    itemid_or_code: str
    raw_label: str
    proxy_strength: str
    recommended_use: str
    main_limitation: str
    notes: str
    candidate_source_type: str = ""


@dataclass
class RunContext:
    denominator_stays: int
    denominator_fatal_stays: int
    denominator_nonfatal_stays: int
    cohort_path: Path
    mimic_root: Path
    tables_accessed: set[str]
    skipped_candidates: list[dict[str, str]]
    missing_timing_anchors: list[str]


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def table_path(mimic_root: Path, schema: str, table: str) -> Path:
    for suffix in (".csv.gz", ".csv"):
        path = mimic_root / schema / f"{table}{suffix}"
        if path.exists():
            return path
    return mimic_root / schema / f"{table}.csv.gz"


def read_columns(path: Path) -> list[str]:
    if not path.exists():
        return []
    return list(pd.read_csv(path, nrows=0).columns)


def read_table(
    mimic_root: Path,
    schema: str,
    table: str,
    *,
    usecols: Iterable[str] | None = None,
) -> pd.DataFrame | None:
    path = table_path(mimic_root, schema, table)
    if not path.exists():
        return None
    columns = read_columns(path)
    selected = None
    if usecols is not None:
        requested = set(usecols)
        selected = [column for column in columns if column in requested]
    return pd.read_csv(path, usecols=selected, low_memory=False)


def iter_table_chunks(
    mimic_root: Path,
    schema: str,
    table: str,
    *,
    usecols: Iterable[str],
    chunksize: int,
) -> Iterable[pd.DataFrame]:
    path = table_path(mimic_root, schema, table)
    if not path.exists():
        return
    columns = read_columns(path)
    selected = [column for column in columns if column in set(usecols)]
    if not selected:
        return
    yield from pd.read_csv(path, usecols=selected, chunksize=chunksize, low_memory=False)


def resolve_cohort_path(mimic_root: Path, explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            mimic_root
            / "1-mortality-decomposition"
            / "processed"
            / "ch1_mimic_stay_level_cohort.csv",
            mimic_root / "processed" / "ch1_mimic_stay_level_cohort.csv",
            REPO_ROOT / "mimic-iv-demo" / "data" / "processed" / "ch1_mimic_stay_level_cohort.csv",
        ]
    )
    existing = [path.resolve() for path in candidates if path.exists()]
    if not existing:
        checked = "\n".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "Could not identify the established Chapter 1 MIMIC stay-level cohort. "
            "Refusing to fall back to all MIMIC admissions. Checked:\n" + checked
        )
    if explicit is None:
        # Prefer the cohort under the selected MIMIC root.
        rooted = [
            path
            for path in existing
            if str(path).startswith(str(mimic_root.resolve()))
            and path.name == "ch1_mimic_stay_level_cohort.csv"
        ]
        if rooted:
            return rooted[0]
    return existing[0]


def load_cohort(path: Path) -> pd.DataFrame:
    cohort = pd.read_csv(path, low_memory=False)
    required = {
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime",
        "icu_mortality",
        "retained_stay_level_cohort",
    }
    missing = sorted(required - set(cohort.columns))
    if missing:
        raise KeyError(f"Chapter 1 cohort file {path} is missing required columns: {missing}")
    cohort = cohort[cohort["retained_stay_level_cohort"].eq(1)].copy()
    if cohort.empty:
        raise ValueError(f"Chapter 1 cohort file {path} has no retained stays.")
    for column in ["intime", "outtime", "deathtime"]:
        if column in cohort.columns:
            cohort[column] = pd.to_datetime(cohort[column], errors="coerce")
    cohort["icu_mortality"] = pd.to_numeric(cohort["icu_mortality"], errors="coerce").fillna(0).astype(int)
    for column in ["subject_id", "hadm_id", "stay_id"]:
        cohort[column] = pd.to_numeric(cohort[column], errors="coerce").astype("Int64")
    return cohort


def reviewed_candidates(inventory_path: Path) -> tuple[list[Candidate], list[dict[str, str]]]:
    inventory = pd.read_csv(inventory_path, dtype=str).fillna("")
    deferred: list[dict[str, str]] = []
    for row in inventory[inventory["decision_preliminary"].eq("needs_review")].to_dict("records"):
        deferred.append(
            {
                "raw_label": row.get("raw_label", ""),
                "reason": "5.2a candidate remained needs_review and was not approved by the 5.2b review instructions",
            }
        )
    excluded_count = int(inventory["decision_preliminary"].eq("exclude").sum())
    if excluded_count:
        deferred.append(
            {
                "raw_label": f"{excluded_count} inventory rows with decision_preliminary=exclude",
                "reason": "Rejected false positives or outcome fields were not counted",
            }
        )
    included = inventory[
        inventory["decision_preliminary"].isin(
            ["include_for_5_2b", "include_descriptive_only_for_5_2b"]
        )
    ].copy()

    candidates: list[Candidate] = []
    for row in included.to_dict("records"):
        domain = row["proxy_domain"]
        raw_label = row["raw_label"]
        normalized = normalize_text(raw_label)
        if domain not in DOMAIN_ORDER:
            deferred.append({"raw_label": raw_label, "reason": f"unsupported domain {domain}"})
            continue
        if domain == "code_status_dnr_dni":
            has_positive_value = bool(POSITIVE_CODE_STATUS_RE.search(raw_label))
            is_source_label = row["source_table"] in {"d_items", "poe"}
            if FULL_CODE_RE.search(raw_label):
                deferred.append(
                    {
                        "raw_label": raw_label,
                        "reason": "full-code value is not treatment-limitation-positive",
                    }
                )
                continue
            if row["source_table"] == "poe" and not has_positive_value:
                deferred.append(
                    {
                        "raw_label": raw_label,
                        "reason": "POE order source row lacks limitation-positive value; POE detail values are counted separately when available",
                    }
                )
                continue
            if not has_positive_value and not is_source_label:
                deferred.append(
                    {
                        "raw_label": raw_label,
                        "reason": "code-status candidate does not indicate limitation-positive value",
                    }
                )
                continue
        if domain in {"comfort_measures_only", "withdrawal_or_withholding", "ambiguous_goals_of_care"}:
            deferred.append({"raw_label": raw_label, "reason": "not approved by 5.2a human review"})
            continue
        use = {
            "code_status_dnr_dni": "code_status_limitation_positive",
            "palliative_care": "descriptive_supporting_context",
            "hospice": "discharge_end_of_life_context",
            "brain_death_or_organ_donation": "separate_context_domain",
            "ama_or_nonstandard_discharge": "discharge_process_context",
        }.get(domain, "descriptive_context")
        candidates.append(
            Candidate(
                proxy_domain=domain,
                source_schema=row["source_schema"],
                source_table=row["source_table"],
                source_field=row["source_field"],
                itemid_or_code=row["itemid_or_code"],
                raw_label=raw_label,
                proxy_strength=row["proxy_strength_preliminary"],
                recommended_use=use,
                main_limitation=row["main_limitation"],
                notes=row["notes"],
                candidate_source_type=row.get("candidate_source_type", ""),
            )
        )

    candidates.append(
        Candidate(
            proxy_domain="ama_or_nonstandard_discharge",
            source_schema="hosp",
            source_table="admissions",
            source_field="discharge_location",
            itemid_or_code="",
            raw_label="AMA / against medical advice / elopement discharge values",
            proxy_strength="weak",
            recommended_use="discharge_process_context",
            main_limitation=(
                "Discharge/care-process context only; not treatment-limitation or end-of-life proxy."
            ),
            notes=(
                "Added by 5.2b human review instruction because the demo scan had no AMA discharge row."
            ),
            candidate_source_type="admission_discharge_value",
        )
    )
    return candidates, deferred


def parse_icd_code(value: str) -> tuple[str, str]:
    match = re.match(r"ICD(\d+):(.*)", str(value))
    if not match:
        return "", str(value)
    return match.group(1), match.group(2)


def link_to_cohort(records: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return records.assign(stay_id=pd.Series(dtype="Int64"), icu_mortality=pd.Series(dtype=int))
    frame = records.copy()
    for column in ["subject_id", "hadm_id", "stay_id"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    if "stay_id" in frame.columns and frame["stay_id"].notna().any():
        keys = ["stay_id"]
    elif {"subject_id", "hadm_id"}.issubset(frame.columns):
        keys = ["subject_id", "hadm_id"]
    elif "hadm_id" in frame.columns:
        keys = ["hadm_id"]
    else:
        return frame.iloc[0:0].copy()
    keep = [
        column
        for column in ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "deathtime", "icu_mortality"]
        if column in cohort.columns
    ]
    cohort_link = cohort[keep].drop_duplicates(subset=keys)
    suffix = "_cohort" if "stay_id" in frame.columns else ""
    linked = frame.merge(cohort_link, on=keys, how="inner", suffixes=("", suffix))
    if "stay_id_cohort" in linked.columns:
        linked["stay_id"] = linked["stay_id_cohort"]
        linked = linked.drop(columns=["stay_id_cohort"])
    return linked


def share(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return round(float(numerator) / float(denominator), 6)


def source_counts(records: pd.DataFrame, cohort_records: pd.DataFrame, context: RunContext) -> dict[str, object]:
    n_proxy = int(cohort_records["stay_id"].nunique()) if "stay_id" in cohort_records.columns else 0
    if "icu_mortality" in cohort_records.columns and "stay_id" in cohort_records.columns:
        stay_mortality = cohort_records[["stay_id", "icu_mortality"]].drop_duplicates("stay_id")
        n_fatal = int(stay_mortality["icu_mortality"].eq(1).sum())
    else:
        n_fatal = 0
    n_nonfatal = n_proxy - n_fatal
    return {
        "n_events_or_records": int(len(records)),
        "n_unique_subjects": int(records["subject_id"].nunique()) if "subject_id" in records.columns else "",
        "n_unique_hadm": int(records["hadm_id"].nunique()) if "hadm_id" in records.columns else "",
        "n_unique_stays": int(records["stay_id"].nunique()) if "stay_id" in records.columns else "",
        "n_in_chapter1_mimic_cohort": context.denominator_stays,
        "n_proxy_positive_stays": n_proxy,
        "n_proxy_positive_fatal_stays": n_fatal,
        "n_proxy_positive_nonfatal_stays": n_nonfatal,
        "share_proxy_positive_stays": share(n_proxy, context.denominator_stays),
        "share_proxy_positive_fatal_stays": share(n_fatal, context.denominator_fatal_stays),
        "share_proxy_positive_nonfatal_stays": share(n_nonfatal, context.denominator_nonfatal_stays),
    }


def iqr_text(values: pd.Series) -> str:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return ""
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    return f"{q1:.2f}-{q3:.2f}"


def timing_row(
    candidate: Candidate,
    cohort_records: pd.DataFrame,
    timestamp_field: str,
    notes: str,
) -> dict[str, object]:
    if timestamp_field not in cohort_records.columns or cohort_records.empty:
        return {
            **candidate_identity(candidate),
            "timestamp_field": timestamp_field,
            "n_proxy_positive_stays_with_timestamp": 0,
            "n_before_icu_intime": "",
            "n_after_icu_intime": "",
            "n_before_icu_outtime_or_discharge": "",
            "n_after_icu_outtime_or_discharge": "",
            "n_before_death_for_fatal_stays": "",
            "n_after_death_for_fatal_stays": "",
            "median_hours_from_icu_intime": "",
            "iqr_hours_from_icu_intime": "",
            "median_hours_before_death_fatal_stays": "",
            "iqr_hours_before_death_fatal_stays": "",
            "timing_interpretation": "timing_not_available",
            "notes": notes,
        }

    timed = cohort_records.copy()
    timed[timestamp_field] = pd.to_datetime(timed[timestamp_field], errors="coerce")
    timed = timed.dropna(subset=[timestamp_field, "stay_id"]).copy()
    if timed.empty:
        return {
            **candidate_identity(candidate),
            "timestamp_field": timestamp_field,
            "n_proxy_positive_stays_with_timestamp": 0,
            "n_before_icu_intime": 0,
            "n_after_icu_intime": 0,
            "n_before_icu_outtime_or_discharge": 0,
            "n_after_icu_outtime_or_discharge": 0,
            "n_before_death_for_fatal_stays": 0,
            "n_after_death_for_fatal_stays": 0,
            "median_hours_from_icu_intime": "",
            "iqr_hours_from_icu_intime": "",
            "median_hours_before_death_fatal_stays": "",
            "iqr_hours_before_death_fatal_stays": "",
            "timing_interpretation": "timestamp field present but no usable timestamps after cohort linkage",
            "notes": notes,
        }

    timed = timed.sort_values(["stay_id", timestamp_field]).drop_duplicates("stay_id", keep="first")
    timed["hours_from_icu_intime"] = (
        timed[timestamp_field] - timed["intime"]
    ).dt.total_seconds() / 3600.0
    timed["hours_before_death"] = (
        timed["deathtime"] - timed[timestamp_field]
    ).dt.total_seconds() / 3600.0 if "deathtime" in timed.columns else pd.NA

    fatal = timed[timed["icu_mortality"].eq(1)].copy() if "icu_mortality" in timed.columns else timed.iloc[0:0]
    before_out = timed[timestamp_field].le(timed["outtime"]) if "outtime" in timed.columns else pd.Series(dtype=bool)
    after_out = timed[timestamp_field].gt(timed["outtime"]) if "outtime" in timed.columns else pd.Series(dtype=bool)
    before_death = fatal[timestamp_field].le(fatal["deathtime"]) if "deathtime" in fatal.columns else pd.Series(dtype=bool)
    after_death = fatal[timestamp_field].gt(fatal["deathtime"]) if "deathtime" in fatal.columns else pd.Series(dtype=bool)

    median_from_intime = pd.to_numeric(timed["hours_from_icu_intime"], errors="coerce").median()
    fatal_hours = pd.to_numeric(fatal.get("hours_before_death", pd.Series(dtype=float)), errors="coerce")
    median_before_death = fatal_hours.median()
    if candidate.proxy_domain in {"hospice", "ama_or_nonstandard_discharge"}:
        interpretation = "post_event_or_discharge_context"
    elif not before_out.empty and before_out.mean() >= 0.8:
        interpretation = "stay_timed_structured_proxy; horizon-specific prediction-time check still required"
    else:
        interpretation = "post_event_or_unclear"

    return {
        **candidate_identity(candidate),
        "timestamp_field": timestamp_field,
        "n_proxy_positive_stays_with_timestamp": int(timed["stay_id"].nunique()),
        "n_before_icu_intime": int(timed[timestamp_field].lt(timed["intime"]).sum()),
        "n_after_icu_intime": int(timed[timestamp_field].ge(timed["intime"]).sum()),
        "n_before_icu_outtime_or_discharge": int(before_out.sum()) if not before_out.empty else "",
        "n_after_icu_outtime_or_discharge": int(after_out.sum()) if not after_out.empty else "",
        "n_before_death_for_fatal_stays": int(before_death.sum()) if not before_death.empty else "",
        "n_after_death_for_fatal_stays": int(after_death.sum()) if not after_death.empty else "",
        "median_hours_from_icu_intime": "" if pd.isna(median_from_intime) else round(float(median_from_intime), 3),
        "iqr_hours_from_icu_intime": iqr_text(timed["hours_from_icu_intime"]),
        "median_hours_before_death_fatal_stays": "" if pd.isna(median_before_death) else round(float(median_before_death), 3),
        "iqr_hours_before_death_fatal_stays": iqr_text(fatal_hours),
        "timing_interpretation": interpretation,
        "notes": notes,
    }


def candidate_identity(candidate: Candidate) -> dict[str, object]:
    return {
        "proxy_domain": candidate.proxy_domain,
        "source_schema": candidate.source_schema,
        "source_table": candidate.source_table,
        "source_field": candidate.source_field,
        "itemid_or_code": candidate.itemid_or_code,
        "raw_label": candidate.raw_label,
    }


def empty_counts_row(candidate: Candidate, context: RunContext, note: str) -> dict[str, object]:
    return {
        **candidate_identity(candidate),
        "proxy_strength": candidate.proxy_strength,
        "recommended_use": candidate.recommended_use,
        "n_events_or_records": 0,
        "n_unique_subjects": 0,
        "n_unique_hadm": 0,
        "n_unique_stays": 0,
        "n_in_chapter1_mimic_cohort": context.denominator_stays,
        "n_proxy_positive_stays": 0,
        "n_proxy_positive_fatal_stays": 0,
        "n_proxy_positive_nonfatal_stays": 0,
        "share_proxy_positive_stays": share(0, context.denominator_stays),
        "share_proxy_positive_fatal_stays": share(0, context.denominator_fatal_stays),
        "share_proxy_positive_nonfatal_stays": share(0, context.denominator_nonfatal_stays),
        "timing_summary_available": "false",
        "main_limitation": candidate.main_limitation,
        "notes": note,
        "_positive_stay_ids": set(),
        "_positive_fatal_stay_ids": set(),
    }


def counts_row(
    candidate: Candidate,
    context: RunContext,
    records: pd.DataFrame,
    cohort_records: pd.DataFrame,
    timing_available: bool,
    notes: str,
) -> dict[str, object]:
    positive_stays: set[int] = set()
    positive_fatal_stays: set[int] = set()
    if "stay_id" in cohort_records.columns:
        stay_frame = cohort_records[["stay_id", "icu_mortality"]].dropna(subset=["stay_id"]).drop_duplicates("stay_id")
        positive_stays = {int(value) for value in stay_frame["stay_id"].astype(int).tolist()}
        positive_fatal_stays = {
            int(row.stay_id)
            for row in stay_frame.itertuples(index=False)
            if int(row.icu_mortality) == 1
        }
    return {
        **candidate_identity(candidate),
        "proxy_strength": candidate.proxy_strength,
        "recommended_use": candidate.recommended_use,
        **source_counts(records, cohort_records, context),
        "timing_summary_available": str(bool(timing_available)).lower(),
        "main_limitation": candidate.main_limitation,
        "notes": notes,
        "_positive_stay_ids": positive_stays,
        "_positive_fatal_stay_ids": positive_fatal_stays,
    }


def count_admissions_candidate(
    candidate: Candidate,
    cohort: pd.DataFrame,
    context: RunContext,
) -> tuple[dict[str, object], dict[str, object]]:
    admissions = read_table(
        context.mimic_root,
        "hosp",
        "admissions",
        usecols=["subject_id", "hadm_id", "dischtime", "discharge_location"],
    )
    context.tables_accessed.add("hosp.admissions")
    if admissions is None:
        note = "Skipped: hosp.admissions was unavailable."
        context.skipped_candidates.append({**candidate_identity(candidate), "reason": note})
        return empty_counts_row(candidate, context, note), timing_row(candidate, pd.DataFrame(), "dischtime", note)
    values = admissions["discharge_location"].fillna("").astype(str)
    if candidate.proxy_domain == "hospice":
        matched = admissions[values.str.contains(HOSPICE_RE, na=False)].copy()
    elif candidate.proxy_domain == "ama_or_nonstandard_discharge":
        matched = admissions[values.str.contains(AMA_RE, na=False)].copy()
    else:
        matched = admissions.iloc[0:0].copy()
    matched["dischtime"] = pd.to_datetime(matched.get("dischtime"), errors="coerce")
    cohort_records = link_to_cohort(matched, cohort)
    note = "Counted from full-data admissions discharge_location; discharge-context only."
    return (
        counts_row(candidate, context, matched, cohort_records, True, note),
        timing_row(candidate, cohort_records, "dischtime", note),
    )


def count_icd_candidate(
    candidate: Candidate,
    cohort: pd.DataFrame,
    context: RunContext,
) -> tuple[dict[str, object], dict[str, object]]:
    version, code = parse_icd_code(candidate.itemid_or_code)
    table = "diagnoses_icd" if candidate.source_table == "d_icd_diagnoses" else "procedures_icd"
    usecols = ["subject_id", "hadm_id", "icd_code", "icd_version"]
    if table == "procedures_icd":
        usecols.append("chartdate")
    source = read_table(context.mimic_root, "hosp", table, usecols=usecols)
    context.tables_accessed.add(f"hosp.{table}")
    if source is None:
        note = f"Skipped: hosp.{table} was unavailable."
        context.skipped_candidates.append({**candidate_identity(candidate), "reason": note})
        return empty_counts_row(candidate, context, note), timing_row(candidate, pd.DataFrame(), "", note)
    matched = source[
        source["icd_code"].astype(str).eq(code)
        & source["icd_version"].astype(str).eq(version)
    ].copy()
    cohort_records = link_to_cohort(matched, cohort)
    if "chartdate" in cohort_records.columns:
        cohort_records["chartdate"] = pd.to_datetime(cohort_records["chartdate"], errors="coerce")
        note = f"Counted via hosp.{table}; ICD chartdate is date-level only."
        return (
            counts_row(candidate, context, matched, cohort_records, True, note),
            timing_row(candidate, cohort_records, "chartdate", note),
        )
    note = f"Counted via hosp.{table}; no event timestamp available for ICD diagnosis rows."
    return (
        counts_row(candidate, context, matched, cohort_records, False, note),
        timing_row(candidate, pd.DataFrame(), "", note),
    )


def d_items_linksto(mimic_root: Path, itemid: int) -> str:
    d_items = read_table(mimic_root, "icu", "d_items", usecols=["itemid", "linksto"])
    if d_items is None:
        return ""
    row = d_items[pd.to_numeric(d_items["itemid"], errors="coerce").eq(itemid)]
    if row.empty:
        return ""
    return str(row["linksto"].iloc[0]).strip()


def positive_code_status_mask(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str)
    positive = values.str.contains(POSITIVE_CODE_STATUS_RE, na=False)
    full_code = values.str.contains(FULL_CODE_RE, na=False) & ~positive
    return positive & ~full_code


def count_d_items_candidate(
    candidate: Candidate,
    cohort: pd.DataFrame,
    context: RunContext,
    chunksize: int,
) -> tuple[dict[str, object], dict[str, object]]:
    try:
        itemid = int(float(candidate.itemid_or_code))
    except ValueError:
        note = "Skipped: d_items candidate had no numeric itemid."
        context.skipped_candidates.append({**candidate_identity(candidate), "reason": note})
        return empty_counts_row(candidate, context, note), timing_row(candidate, pd.DataFrame(), "", note)
    linksto = d_items_linksto(context.mimic_root, itemid)
    if not linksto or str(linksto).lower() == "nan":
        note = "Skipped: d_items linksto table unavailable in dictionary."
        context.skipped_candidates.append({**candidate_identity(candidate), "reason": note})
        return empty_counts_row(candidate, context, note), timing_row(candidate, pd.DataFrame(), "", note)
    table = normalize_text(linksto).replace(" ", "")
    path = table_path(context.mimic_root, "icu", table)
    if not path.exists():
        note = f"Skipped: linked ICU event table icu.{table} was unavailable."
        context.skipped_candidates.append({**candidate_identity(candidate), "reason": note})
        return empty_counts_row(candidate, context, note), timing_row(candidate, pd.DataFrame(), "", note)
    columns = read_columns(path)
    timestamp_field = "charttime" if "charttime" in columns else "starttime" if "starttime" in columns else "storetime" if "storetime" in columns else ""
    usecols = ["subject_id", "hadm_id", "stay_id", "itemid", "value"]
    if timestamp_field:
        usecols.append(timestamp_field)
    matched_chunks: list[pd.DataFrame] = []
    for chunk in iter_table_chunks(
        context.mimic_root,
        "icu",
        table,
        usecols=usecols,
        chunksize=chunksize,
    ):
        hit = chunk[pd.to_numeric(chunk["itemid"], errors="coerce").eq(itemid)].copy()
        if hit.empty:
            continue
        if candidate.proxy_domain == "code_status_dnr_dni":
            if "value" not in hit.columns:
                hit = hit.iloc[0:0].copy()
            else:
                hit = hit[positive_code_status_mask(hit["value"])].copy()
        matched_chunks.append(hit)
    context.tables_accessed.add(f"icu.{table}")
    matched = pd.concat(matched_chunks, ignore_index=True) if matched_chunks else pd.DataFrame(columns=usecols)
    if timestamp_field and timestamp_field in matched.columns:
        matched[timestamp_field] = pd.to_datetime(matched[timestamp_field], errors="coerce")
    cohort_records = link_to_cohort(matched, cohort)
    note = f"Counted from icu.{table} itemid {itemid}."
    if candidate.proxy_domain == "code_status_dnr_dni":
        note += " Full-code/resuscitate-only values were not counted as limitation-positive."
    return (
        counts_row(candidate, context, matched, cohort_records, bool(timestamp_field), note),
        timing_row(candidate, cohort_records, timestamp_field, note),
    )


def count_d_items_candidates(
    candidates: list[Candidate],
    cohort: pd.DataFrame,
    context: RunContext,
    chunksize: int,
) -> list[tuple[dict[str, object], dict[str, object]]]:
    if not candidates:
        return []
    d_items = read_table(context.mimic_root, "icu", "d_items", usecols=["itemid", "linksto"])
    if d_items is None:
        output = []
        for candidate in candidates:
            note = "Skipped: icu.d_items dictionary unavailable."
            context.skipped_candidates.append({**candidate_identity(candidate), "reason": note})
            output.append((empty_counts_row(candidate, context, note), timing_row(candidate, pd.DataFrame(), "", note)))
        return output

    itemid_to_candidates: dict[int, list[Candidate]] = {}
    for candidate in candidates:
        try:
            itemid = int(float(candidate.itemid_or_code))
        except ValueError:
            note = "Skipped: d_items candidate had no numeric itemid."
            context.skipped_candidates.append({**candidate_identity(candidate), "reason": note})
            itemid_to_candidates.setdefault(-1, []).append(candidate)
            continue
        itemid_to_candidates.setdefault(itemid, []).append(candidate)

    link_lookup = {
        int(row.itemid): str(row.linksto).strip()
        for row in d_items.itertuples(index=False)
        if not pd.isna(row.itemid)
    }
    table_to_itemids: dict[str, set[int]] = {}
    missing_link: list[Candidate] = []
    for itemid, item_candidates in itemid_to_candidates.items():
        if itemid < 0:
            continue
        linksto = link_lookup.get(itemid, "")
        if not linksto or linksto.lower() == "nan":
            missing_link.extend(item_candidates)
            continue
        table_to_itemids.setdefault(normalize_text(linksto).replace(" ", ""), set()).add(itemid)

    matched_by_itemid: dict[int, pd.DataFrame] = {}
    timestamp_by_table: dict[str, str] = {}
    for table, itemids in table_to_itemids.items():
        path = table_path(context.mimic_root, "icu", table)
        if not path.exists():
            continue
        columns = read_columns(path)
        timestamp_field = "charttime" if "charttime" in columns else "starttime" if "starttime" in columns else "storetime" if "storetime" in columns else ""
        timestamp_by_table[table] = timestamp_field
        usecols = ["subject_id", "hadm_id", "stay_id", "itemid", "value"]
        if timestamp_field:
            usecols.append(timestamp_field)
        chunks: list[pd.DataFrame] = []
        for chunk in iter_table_chunks(
            context.mimic_root,
            "icu",
            table,
            usecols=usecols,
            chunksize=chunksize,
        ):
            item_series = pd.to_numeric(chunk["itemid"], errors="coerce")
            hit = chunk[item_series.isin(itemids)].copy()
            if not hit.empty:
                chunks.append(hit)
        context.tables_accessed.add(f"icu.{table}")
        table_hits = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
        if timestamp_field and timestamp_field in table_hits.columns:
            table_hits[timestamp_field] = pd.to_datetime(table_hits[timestamp_field], errors="coerce")
        for itemid in itemids:
            matched_by_itemid[itemid] = table_hits[
                pd.to_numeric(table_hits.get("itemid"), errors="coerce").eq(itemid)
            ].copy()

    output: list[tuple[dict[str, object], dict[str, object]]] = []
    for candidate in candidates:
        try:
            itemid = int(float(candidate.itemid_or_code))
        except ValueError:
            note = "Skipped: d_items candidate had no numeric itemid."
            output.append((empty_counts_row(candidate, context, note), timing_row(candidate, pd.DataFrame(), "", note)))
            continue
        linksto = link_lookup.get(itemid, "")
        table = normalize_text(linksto).replace(" ", "")
        if not linksto or linksto.lower() == "nan":
            note = "Skipped: d_items linksto table unavailable in dictionary."
            context.skipped_candidates.append({**candidate_identity(candidate), "reason": note})
            output.append((empty_counts_row(candidate, context, note), timing_row(candidate, pd.DataFrame(), "", note)))
            continue
        path = table_path(context.mimic_root, "icu", table)
        if not path.exists():
            note = f"Skipped: linked ICU event table icu.{table} was unavailable."
            context.skipped_candidates.append({**candidate_identity(candidate), "reason": note})
            output.append((empty_counts_row(candidate, context, note), timing_row(candidate, pd.DataFrame(), "", note)))
            continue
        matched = matched_by_itemid.get(itemid, pd.DataFrame()).copy()
        if candidate.proxy_domain == "code_status_dnr_dni":
            if "value" not in matched.columns:
                matched = matched.iloc[0:0].copy()
            else:
                matched = matched[positive_code_status_mask(matched["value"])].copy()
        timestamp_field = timestamp_by_table.get(table, "")
        cohort_records = link_to_cohort(matched, cohort)
        note = f"Counted from icu.{table} itemid {itemid}."
        if candidate.proxy_domain == "code_status_dnr_dni":
            note += " Full-code/resuscitate-only values were not counted as limitation-positive."
        output.append(
            (
                counts_row(candidate, context, matched, cohort_records, bool(timestamp_field), note),
                timing_row(candidate, cohort_records, timestamp_field, note),
            )
        )
    return output


def count_poe_candidate(
    candidate: Candidate,
    cohort: pd.DataFrame,
    context: RunContext,
) -> tuple[dict[str, object], dict[str, object]]:
    poe = read_table(
        context.mimic_root,
        "hosp",
        "poe",
        usecols=["poe_id", "subject_id", "hadm_id", "ordertime", "order_type", "order_subtype", "order_status"],
    )
    detail = read_table(
        context.mimic_root,
        "hosp",
        "poe_detail",
        usecols=["poe_id", "subject_id", "field_name", "field_value"],
    )
    context.tables_accessed.update({"hosp.poe", "hosp.poe_detail"})
    if poe is None or detail is None:
        note = "Skipped: hosp.poe and/or hosp.poe_detail unavailable in this MIMIC root."
        context.skipped_candidates.append({**candidate_identity(candidate), "reason": note})
        return empty_counts_row(candidate, context, note), timing_row(candidate, pd.DataFrame(), "ordertime", note)
    if candidate.proxy_domain == "code_status_dnr_dni":
        detail_hits = detail[
            detail["field_name"].fillna("").astype(str).str.contains("code status", case=False, na=False)
            & positive_code_status_mask(detail["field_value"])
        ].copy()
        if candidate.source_table == "poe_detail" and "|" in candidate.raw_label:
            target_value = normalize_text(candidate.raw_label.split("|", 1)[1])
            detail_hits = detail_hits[
                detail_hits["field_value"].fillna("").astype(str).map(normalize_text).eq(target_value)
            ].copy()
        matched = detail_hits.merge(
            poe[["poe_id", "hadm_id", "ordertime"]],
            on="poe_id",
            how="left",
            suffixes=("", "_poe"),
        )
        if "hadm_id_poe" in matched.columns:
            matched["hadm_id"] = matched["hadm_id"].fillna(matched["hadm_id_poe"])
    elif candidate.proxy_domain == "palliative_care":
        matched = poe[
            poe["order_subtype"].fillna("").astype(str).str.contains("palliative", case=False, na=False)
        ].copy()
        if "|" in candidate.raw_label:
            parts = [part.strip() for part in candidate.raw_label.split("|")]
            if len(parts) >= 2:
                target_subtype = normalize_text(parts[1])
                matched = matched[
                    matched["order_subtype"].fillna("").astype(str).map(normalize_text).eq(target_subtype)
                ].copy()
    else:
        matched = poe.iloc[0:0].copy()
    if "ordertime" in matched.columns:
        matched["ordertime"] = pd.to_datetime(matched["ordertime"], errors="coerce")
    cohort_records = link_to_cohort(matched, cohort)
    note = "Counted from POE structured orders/details."
    return (
        counts_row(candidate, context, matched, cohort_records, "ordertime" in matched.columns, note),
        timing_row(candidate, cohort_records, "ordertime", note),
    )


def count_candidate(
    candidate: Candidate,
    cohort: pd.DataFrame,
    context: RunContext,
    chunksize: int,
) -> tuple[dict[str, object], dict[str, object]]:
    if candidate.source_table == "admissions":
        return count_admissions_candidate(candidate, cohort, context)
    if candidate.source_table in {"d_icd_diagnoses", "d_icd_procedures"}:
        return count_icd_candidate(candidate, cohort, context)
    if candidate.source_table == "d_items":
        return count_d_items_candidate(candidate, cohort, context, chunksize)
    if candidate.source_table in {"poe", "poe_detail"}:
        return count_poe_candidate(candidate, cohort, context)
    note = f"Skipped: unsupported source table {candidate.source_table}."
    context.skipped_candidates.append({**candidate_identity(candidate), "reason": note})
    return empty_counts_row(candidate, context, note), timing_row(candidate, pd.DataFrame(), "", note)


def strength_max(values: Iterable[str]) -> str:
    order = {"reject": 0, "weak": 1, "moderate": 2, "strong": 3}
    clean = [value for value in values if value in order]
    if not clean:
        return ""
    return max(clean, key=lambda value: order[value])


def timing_rank(values: Iterable[str]) -> str:
    joined = " ".join(str(value) for value in values)
    if "stay_timed_structured_proxy" in joined:
        return "full_data_timing_check"
    if "post_event_or_discharge_context" in joined:
        return "stay_level_descriptive_only"
    if "post_event_or_unclear" in joined:
        return "post_event_or_unclear"
    return "not_usable"


def domain_summary(counts: pd.DataFrame, timing: pd.DataFrame, context: RunContext) -> pd.DataFrame:
    rows = []
    for domain in DOMAIN_ORDER:
        subset = counts[counts["proxy_domain"].eq(domain)].copy()
        if subset.empty:
            rows.append(
                {
                    "proxy_domain": domain,
                    "proxy_strength_max": "",
                    "n_sources": 0,
                    "n_proxy_positive_stays_any_source": 0,
                    "n_proxy_positive_fatal_stays_any_source": 0,
                    "share_proxy_positive_stays_any_source": share(0, context.denominator_stays),
                    "share_proxy_positive_fatal_stays_any_source": share(0, context.denominator_fatal_stays),
                    "best_timing_usability": "not_usable",
                    "recommended_use_domain": "",
                    "main_limitation_domain": "No approved full-data candidate rows counted.",
                    "notes": "",
                }
            )
            continue
        positive_union: set[int] = set()
        fatal_union: set[int] = set()
        for stays in subset.get("_positive_stay_ids", []):
            if isinstance(stays, set):
                positive_union.update(stays)
        for stays in subset.get("_positive_fatal_stay_ids", []):
            if isinstance(stays, set):
                fatal_union.update(stays)
        any_positive = len(positive_union)
        any_fatal = len(fatal_union)
        t_subset = timing[timing["proxy_domain"].eq(domain)] if not timing.empty else pd.DataFrame()
        rows.append(
            {
                "proxy_domain": domain,
                "proxy_strength_max": strength_max(subset["proxy_strength"].dropna().astype(str)),
                "n_sources": int(subset.shape[0]),
                "n_proxy_positive_stays_any_source": any_positive,
                "n_proxy_positive_fatal_stays_any_source": any_fatal,
                "share_proxy_positive_stays_any_source": share(any_positive, context.denominator_stays),
                "share_proxy_positive_fatal_stays_any_source": share(any_fatal, context.denominator_fatal_stays),
                "best_timing_usability": timing_rank(t_subset.get("timing_interpretation", [])),
                "recommended_use_domain": ";".join(sorted(set(subset["recommended_use"].dropna().astype(str)))),
                "main_limitation_domain": "; ".join(sorted(set(subset["main_limitation"].dropna().astype(str))))[:500],
                "notes": "Any-source row-level flags were computed internally only; no stay-level rows exported.",
            }
        )
    return pd.DataFrame(rows, columns=DOMAIN_COLUMNS)


def write_note(
    path: Path,
    *,
    counts: pd.DataFrame,
    timing: pd.DataFrame,
    domain: pd.DataFrame,
    candidates: list[Candidate],
    deferred: list[dict[str, str]],
    context: RunContext,
    outputs: list[Path],
    inventory_path: Path,
) -> None:
    counted = counts[~counts["notes"].astype(str).str.startswith("Skipped:")].copy()
    included_sources = counted[["proxy_domain", "source_schema", "source_table"]].drop_duplicates()
    source_lines = "\n".join(
        f"- `{row.proxy_domain}`: `{row.source_schema}.{row.source_table}`"
        for row in included_sources.itertuples(index=False)
    ) or "- No approved candidates had an available source table."
    deferred_lines = "\n".join(
        f"- {item.get('raw_label', '')}: {item.get('reason', '')}" for item in deferred
    ) or "- None."
    skipped_lines = "\n".join(
        f"- {item.get('raw_label', '')}: {item.get('reason', '')}"
        for item in context.skipped_candidates
    ) or "- None."
    prevalence_lines = "\n".join(
        f"- `{row.proxy_domain}`: {row.n_proxy_positive_stays_any_source} stays "
        f"({row.share_proxy_positive_stays_any_source}); fatal stays {row.n_proxy_positive_fatal_stays_any_source} "
        f"({row.share_proxy_positive_fatal_stays_any_source})."
        for row in domain.itertuples(index=False)
        if int(row.n_sources) > 0
    ) or "- No approved domains had countable sources."
    timing_lines = "\n".join(
        f"- `{row.proxy_domain}` `{row.raw_label}`: {row.timing_interpretation}"
        for row in timing.itertuples(index=False)
    ) or "- No candidate had usable timestamp summaries."
    files_lines = "\n".join(f"- `{path}`" for path in outputs)
    text = f"""# MIMIC Treatment-Limitation Proxy Full-Data Counts Note

## Scope

This is aggregate full-data counting for reviewed 5.2a structured proxy candidates. It does not export row-level patient data, analyze low-predicted fatal cases, add proxies to risk models, change the Chapter 1 cohort definition, exclude proxy-positive patients, use notes/NLP, or write a final feasibility verdict.

## Inputs

- Candidate inventory: `{inventory_path}`
- Chapter 1 cohort artifact: `{context.cohort_path}`
- MIMIC root: `{context.mimic_root}`
- Retained Chapter 1 MIMIC stays used as denominator: {context.denominator_stays}
- Fatal retained stays by established `icu_mortality`: {context.denominator_fatal_stays}

## Approved candidates counted

{source_lines}

## Excluded or deferred candidates

Deferred by review rules or missing source availability:

{deferred_lines}

Skipped during counting:

{skipped_lines}

## Prevalence summary

{prevalence_lines}

Code-status counts are limitation-positive only. Full-code/resuscitate-only values were kept separate from treatment-limitation positivity and were not counted as `code_status_dnr_dni` proxy-positive stays.

## Timing summary

{timing_lines}

Timing summaries use only available structured timestamps and established cohort anchors (`intime`, `outtime`, and `deathtime` where present). Unsupported timing anchors are not inferred.

## Data-boundary statement

No row-level patient or stay-level proxy-positive data were exported. The output CSV and JSON files contain aggregate counts and aggregate timing summaries only.

## Files produced

{files_lines}
"""
    path.write_text(text)


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def write_manifest(
    path: Path,
    *,
    context: RunContext,
    inventory_path: Path,
    outputs: list[Path],
) -> None:
    payload = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "script_paths": [
            "scripts/chapter1_mimic/count_mimic_treatment_limitation_proxy_prevalence.py"
        ],
        "input_inventory_path": str(inventory_path),
        "chapter1_mimic_cohort_file_used": str(context.cohort_path),
        "mimic_root": str(context.mimic_root),
        "mimic_tables_accessed": sorted(context.tables_accessed),
        "output_files": [str(path) for path in outputs],
        "patient_level_data_accessed_on_cluster": True,
        "outputs_are_aggregate_local_safe": True,
        "row_level_patient_data_exported": False,
        "no_low_predicted_fatal_case_analysis_attempted": True,
        "no_notes_or_nlp_used": True,
        "no_cohort_or_risk_model_changes": True,
        "missing_timing_anchors_or_limitations": context.missing_timing_anchors,
        "skipped_candidates": context.skipped_candidates,
        "statement": "Outputs contain aggregate/local-safe counts and timing summaries only.",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count aggregate Chapter 1 MIMIC structured treatment-limitation proxy candidates."
    )
    parser.add_argument("--mimic-root", type=Path, default=DEFAULT_MIMIC_ROOT)
    parser.add_argument("--cohort-path", type=Path, default=None)
    parser.add_argument("--inventory-path", type=Path, default=DEFAULT_INVENTORY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mimic_root = args.mimic_root.resolve()
    inventory_path = args.inventory_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not inventory_path.exists():
        raise FileNotFoundError(f"Missing 5.2a inventory: {inventory_path}")

    cohort_path = resolve_cohort_path(mimic_root, args.cohort_path)
    cohort = load_cohort(cohort_path)
    context = RunContext(
        denominator_stays=int(cohort["stay_id"].nunique()),
        denominator_fatal_stays=int(cohort.drop_duplicates("stay_id")["icu_mortality"].eq(1).sum()),
        denominator_nonfatal_stays=int(cohort.drop_duplicates("stay_id")["icu_mortality"].eq(0).sum()),
        cohort_path=cohort_path,
        mimic_root=mimic_root,
        tables_accessed=set(),
        skipped_candidates=[],
        missing_timing_anchors=[],
    )
    for anchor in ["intime", "outtime", "deathtime"]:
        if anchor not in cohort.columns or cohort[anchor].notna().sum() == 0:
            context.missing_timing_anchors.append(f"cohort anchor {anchor} unavailable or entirely missing")

    candidates, deferred = reviewed_candidates(inventory_path)
    count_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    d_item_candidates = [candidate for candidate in candidates if candidate.source_table == "d_items"]
    other_candidates = [candidate for candidate in candidates if candidate.source_table != "d_items"]
    for count, timing in count_d_items_candidates(d_item_candidates, cohort, context, args.chunksize):
        count_rows.append(count)
        timing_rows.append(timing)
    for candidate in other_candidates:
        count, timing = count_candidate(candidate, cohort, context, args.chunksize)
        count_rows.append(count)
        timing_rows.append(timing)

    counts_df = pd.DataFrame(count_rows)
    timing_df = pd.DataFrame(timing_rows, columns=TIMING_COLUMNS)
    domain_df = domain_summary(counts_df, timing_df, context)
    if counts_df["source_table"].astype(str).eq("d_icd_diagnoses").any():
        context.missing_timing_anchors.append(
            "ICD diagnosis proxy rows have no event timestamp in diagnoses_icd; timing summaries are unavailable for those source rows."
        )
    if any("poe" in item.get("reason", "").lower() for item in context.skipped_candidates):
        context.missing_timing_anchors.append(
            "hosp.poe and/or hosp.poe_detail were unavailable in the selected full MIMIC root, so POE timing could not be counted."
        )
    if counts_df["source_table"].astype(str).eq("admissions").any():
        context.missing_timing_anchors.append(
            "Admissions discharge-context proxies use dischtime only; this is a hospital discharge timestamp, not an ICU treatment-limitation timestamp."
        )

    counts_path = output_dir / DEFAULT_COUNTS_NAME
    timing_path = output_dir / DEFAULT_TIMING_NAME
    domain_path = output_dir / DEFAULT_DOMAIN_NAME
    note_path = output_dir / DEFAULT_NOTE_NAME
    manifest_path = output_dir / DEFAULT_MANIFEST_NAME
    outputs = [counts_path, timing_path, domain_path, note_path, manifest_path]

    counts_df[COUNTS_COLUMNS].to_csv(counts_path, index=False)
    timing_df.to_csv(timing_path, index=False)
    domain_df.to_csv(domain_path, index=False)
    write_note(
        note_path,
        counts=counts_df,
        timing=timing_df,
        domain=domain_df,
        candidates=candidates,
        deferred=deferred,
        context=context,
        outputs=outputs,
        inventory_path=inventory_path,
    )
    write_manifest(manifest_path, context=context, inventory_path=inventory_path, outputs=outputs)

    print(f"Wrote {counts_path}")
    print(f"Wrote {timing_path}")
    print(f"Wrote {domain_path}")
    print(f"Wrote {note_path}")
    print(f"Wrote {manifest_path}")
    print(f"Retained cohort stays: {context.denominator_stays}")
    print(f"Candidate rows counted or skipped: {len(counts_df)}")


if __name__ == "__main__":
    main()
