#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MIMIC_ROOT = REPO_ROOT / "mimic-iv-demo" / "data"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "analysis_artifacts" / "chapter1_mimic_treatment_limitation_proxies"
)

INVENTORY_COLUMNS = [
    "proxy_domain",
    "source_schema",
    "source_table",
    "source_field",
    "itemid_or_code",
    "raw_label",
    "normalized_label",
    "matched_terms",
    "candidate_source_type",
    "proxy_strength_preliminary",
    "has_timestamp_field",
    "timestamp_fields_available",
    "linkage_level",
    "has_stay_id",
    "has_hadm_id",
    "has_subject_id",
    "timing_usability_preliminary",
    "recommended_next_step",
    "decision_preliminary",
    "main_limitation",
    "notes",
]

SEARCH_TERMS = [
    "code status",
    "DNR",
    "DNI",
    "do not resuscitate",
    "do not intubate",
    "no CPR",
    "CPR",
    "full code",
    "comfort",
    "comfort care",
    "comfort measures",
    "comfort measures only",
    "CMO",
    "withdrawal",
    "withdraw",
    "withholding",
    "withhold",
    "life support",
    "terminal extubation",
    "no escalation",
    "limitation of care",
    "palliative",
    "hospice",
    "brain death",
    "organ donor",
    "organ donation",
    "goals of care",
    "family meeting",
    "against medical advice",
    "left AMA",
    "AMA",
    "eloped",
    "left without being seen",
    "expired",
]

PATIENT_LEVEL_TABLES = {
    "admissions",
    "poe",
    "poe_detail",
    "diagnoses_icd",
    "procedures_icd",
    "services",
}

TIMESTAMP_FIELDS = {
    "admittime",
    "chartdate",
    "charttime",
    "deathtime",
    "dischtime",
    "endtime",
    "ordertime",
    "starttime",
    "storetime",
    "transfertime",
}


@dataclass(frozen=True)
class TableStatus:
    schema: str
    table: str
    path: str
    available: bool
    columns: tuple[str, ...]
    rows_scanned: bool
    note: str


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def matched_terms(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    padded = f" {normalized} "
    matches: list[str] = []
    for term in SEARCH_TERMS:
        term_norm = normalize_text(term)
        if not term_norm:
            continue
        if term_norm in {"dnr", "dni", "cmo", "ama", "cpr"}:
            if re.search(rf"\b{re.escape(term_norm)}\b", normalized):
                matches.append(term)
        elif f" {term_norm} " in padded or term_norm in normalized:
            matches.append(term)
    return matches


def table_path(mimic_root: Path, schema: str, table: str) -> Path:
    return mimic_root / schema / f"{table}.csv.gz"


def read_columns(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    return tuple(pd.read_csv(path, nrows=0).columns)


def is_demo_root(path: Path) -> bool:
    return any("demo" in part.lower() for part in path.resolve().parts)


def can_scan_rows(table: str, mimic_root: Path, allow_patient_level_rows: bool) -> bool:
    if table not in PATIENT_LEVEL_TABLES:
        return True
    return is_demo_root(mimic_root) or allow_patient_level_rows


def read_table(
    mimic_root: Path,
    schema: str,
    table: str,
    *,
    usecols: Iterable[str] | None = None,
    allow_patient_level_rows: bool = False,
) -> tuple[pd.DataFrame | None, TableStatus]:
    path = table_path(mimic_root, schema, table)
    columns = read_columns(path)
    if not path.exists():
        return None, TableStatus(schema, table, str(path), False, (), False, "missing")

    if not can_scan_rows(table, mimic_root, allow_patient_level_rows):
        return None, TableStatus(
            schema,
            table,
            str(path),
            True,
            columns,
            False,
            "header only; patient-level row scan not allowed for non-demo root",
        )

    selected = None
    if usecols is not None:
        requested = set(usecols)
        selected = [column for column in columns if column in requested]
    if selected == []:
        selected = None

    df = pd.read_csv(path, usecols=selected, low_memory=False)
    return df, TableStatus(schema, table, str(path), True, columns, True, "rows scanned")


def linked_event_columns(mimic_root: Path, linksto: str) -> tuple[str, ...]:
    link = normalize_text(linksto).replace(" ", "")
    if not link:
        return ()
    schema = "icu"
    path = table_path(mimic_root, schema, link)
    return read_columns(path)


def source_linkage(columns: Iterable[str]) -> tuple[str, bool, str, bool, bool, bool]:
    column_set = set(columns)
    timestamps = sorted(TIMESTAMP_FIELDS & column_set)
    has_stay = "stay_id" in column_set
    has_hadm = "hadm_id" in column_set
    has_subject = "subject_id" in column_set
    levels = []
    if has_stay:
        levels.append("stay_id")
    if has_hadm:
        levels.append("hadm_id")
    if has_subject:
        levels.append("subject_id")
    linkage = "+".join(levels) if levels else "no direct patient linkage in source"
    return linkage, bool(timestamps), ";".join(timestamps), has_stay, has_hadm, has_subject


def classify_candidate(
    *,
    raw_label: str,
    source_table: str,
    source_field: str,
    candidate_source_type: str,
    has_timestamp: bool,
) -> dict[str, str]:
    normalized = normalize_text(raw_label)
    terms = set(matched_terms(raw_label))
    notes: list[str] = []

    def has_any(*needles: str) -> bool:
        return any(normalize_text(needle) in normalized for needle in needles)

    if candidate_source_type == "admission_outcome_field" or has_any("expired"):
        return {
            "proxy_domain": "reject_false_positive",
            "proxy_strength_preliminary": "reject",
            "timing_usability_preliminary": "not_usable",
            "recommended_next_step": "exclude",
            "decision_preliminary": "exclude",
            "main_limitation": "Outcome/death-discharge marker, not a treatment-limitation proxy.",
            "notes": "Retained as a rejected false positive or outcome/timing support field.",
        }

    if terms <= {"CPR"} and "CPR" in terms:
        return {
            "proxy_domain": "reject_false_positive",
            "proxy_strength_preliminary": "reject",
            "timing_usability_preliminary": "not_usable",
            "recommended_next_step": "exclude",
            "decision_preliminary": "exclude",
            "main_limitation": "CPR alone can denote resuscitation procedure/context, not treatment limitation.",
            "notes": "Requires explicit code-status/no-CPR/DNR/DNI context to be a proxy candidate.",
        }

    withdrawal_false_positive_patterns = [
        "adjustment reaction with withdrawal",
        "alcohol dependence with withdrawal",
        "alcohol withdrawal",
        "cannabis dependence with withdrawal",
        "cannabis use",
        "cocaine dependence with withdrawal",
        "coma scale",
        "drug withdrawal",
        "flexion withdrawal",
        "neonatal withdrawal",
        "nicotine dependence",
        "opioid dependence with withdrawal",
        "opioid use",
        "other psychoactive substance",
        "other stimulant",
        "sedative hypnotic",
        "substance withdrawal",
        "therapeutic use of drugs",
        "tobacco product",
        "withdrawal symptoms",
    ]
    if has_any("withdrawal", "withdraw") and any(
        pattern in normalized for pattern in withdrawal_false_positive_patterns
    ):
        return {
            "proxy_domain": "reject_false_positive",
            "proxy_strength_preliminary": "reject",
            "timing_usability_preliminary": "not_usable",
            "recommended_next_step": "exclude",
            "decision_preliminary": "exclude",
            "main_limitation": "Withdrawal keyword denotes substance withdrawal, neurologic withdrawal response, or another non-treatment-limitation concept.",
            "notes": "Retained as a rejected false positive from broad keyword matching.",
        }

    if has_any("against medical advice", "left ama", " ama ", "eloped", "left without being seen"):
        return {
            "proxy_domain": "ama_or_nonstandard_discharge",
            "proxy_strength_preliminary": "weak",
            "timing_usability_preliminary": "post_event_or_unclear",
            "recommended_next_step": "full_data_prevalence_count",
            "decision_preliminary": "include_descriptive_only_for_5_2b",
            "main_limitation": "Discharge/care-process context only; not treatment-limitation or end-of-life proxy.",
            "notes": "Keep separate from treatment-limitation/end-of-life proxy domains.",
        }

    if has_any("goals of care", "family meeting"):
        return {
            "proxy_domain": "ambiguous_goals_of_care",
            "proxy_strength_preliminary": "weak",
            "timing_usability_preliminary": (
                "requires_full_data_check" if has_timestamp else "post_event_or_unclear"
            ),
            "recommended_next_step": "manual_review_before_counts",
            "decision_preliminary": "needs_review",
            "main_limitation": "Goals-of-care/family-meeting language is nonspecific without value context.",
            "notes": "Human review needed before any prevalence counting.",
        }

    if has_any("brain death", "organ donor", "organ donation"):
        return {
            "proxy_domain": "brain_death_or_organ_donation",
            "proxy_strength_preliminary": "moderate",
            "timing_usability_preliminary": (
                "requires_full_data_check" if has_timestamp else "post_event_or_unclear"
            ),
            "recommended_next_step": "full_data_timing_check",
            "decision_preliminary": "include_for_5_2b",
            "main_limitation": "End-of-life context proxy; not direct evidence of treatment limitation.",
            "notes": "Use as a separate interpretive threat/sensitivity-support domain.",
        }

    if has_any("hospice"):
        timing = "post_event_or_unclear" if source_table == "admissions" else "requires_full_data_check"
        return {
            "proxy_domain": "hospice",
            "proxy_strength_preliminary": "weak" if source_table == "admissions" else "moderate",
            "timing_usability_preliminary": timing,
            "recommended_next_step": "full_data_prevalence_count",
            "decision_preliminary": "include_descriptive_only_for_5_2b",
            "main_limitation": "Hospice is discharge/end-of-life context, not direct ICU treatment limitation.",
            "notes": "Do not interpret as DNR/DNI, withdrawal, or withholding.",
        }

    if has_any("therapeutic or palliative substances"):
        return {
            "proxy_domain": "reject_false_positive",
            "proxy_strength_preliminary": "reject",
            "timing_usability_preliminary": "not_usable",
            "recommended_next_step": "exclude",
            "decision_preliminary": "exclude",
            "main_limitation": "Palliative keyword refers to medication/substance purpose in a procedure label, not palliative-care service or goals-of-care context.",
            "notes": "Retained as a rejected ICD procedure false positive.",
        }

    if has_any("palliative"):
        return {
            "proxy_domain": "palliative_care",
            "proxy_strength_preliminary": "moderate",
            "timing_usability_preliminary": (
                "requires_full_data_check" if has_timestamp else "post_event_or_unclear"
            ),
            "recommended_next_step": "full_data_timing_check",
            "decision_preliminary": "include_descriptive_only_for_5_2b",
            "main_limitation": "Palliative care marker is not equivalent to treatment limitation.",
            "notes": "Keep distinct from DNR/DNI/withdrawal/withholding domains.",
        }

    if has_any("comfort") and not has_any(
        "comfort measures only", "comfort measures", "comfort care", "cmo"
    ):
        return {
            "proxy_domain": "reject_false_positive",
            "proxy_strength_preliminary": "reject",
            "timing_usability_preliminary": "not_usable",
            "recommended_next_step": "exclude",
            "decision_preliminary": "exclude",
            "main_limitation": "Comfort keyword lacks comfort-care/comfort-measures/CMO context.",
            "notes": "Retained as a rejected broad comfort-keyword false positive.",
        }

    if has_any("comfort measures only", "comfort measures", "comfort care", "cmo"):
        return {
            "proxy_domain": "comfort_measures_only",
            "proxy_strength_preliminary": "strong" if has_timestamp else "moderate",
            "timing_usability_preliminary": (
                "prediction_time_usable" if has_timestamp else "requires_full_data_check"
            ),
            "recommended_next_step": "full_data_timing_check",
            "decision_preliminary": "include_for_5_2b",
            "main_limitation": "Structured label discovery only; full-data timing and value semantics not checked.",
            "notes": "Direct comfort/CMO proxy candidate if event timing and value semantics hold.",
        }

    withdrawal_direct_context = has_any(
        "withdraw life support",
        "withdrawal of life support",
        "withdrawing life support",
        "terminal extubation",
        "no escalation",
        "limitation of care",
        "withholding",
        "withhold",
        "withdraw care",
        "withdraw support",
    )
    if has_any("withdrawal", "withdraw") and not withdrawal_direct_context:
        return {
            "proxy_domain": "reject_false_positive",
            "proxy_strength_preliminary": "reject",
            "timing_usability_preliminary": "not_usable",
            "recommended_next_step": "exclude",
            "decision_preliminary": "exclude",
            "main_limitation": "Withdrawal keyword is not explicitly about withholding, no escalation, terminal extubation, or life-support withdrawal.",
            "notes": "Retained as a rejected broad withdrawal-keyword hit pending human review if needed.",
        }

    if has_any(
        "withdrawal",
        "withdraw",
        "withholding",
        "withhold",
        "life support",
        "terminal extubation",
        "no escalation",
        "limitation of care",
    ):
        return {
            "proxy_domain": "withdrawal_or_withholding",
            "proxy_strength_preliminary": "strong" if has_timestamp else "moderate",
            "timing_usability_preliminary": (
                "prediction_time_usable" if has_timestamp else "requires_full_data_check"
            ),
            "recommended_next_step": "full_data_timing_check",
            "decision_preliminary": "include_for_5_2b",
            "main_limitation": "Structured label discovery only; full-data timing and value semantics not checked.",
            "notes": "Direct withdrawal/withholding proxy candidate if event timing and value semantics hold.",
        }

    if has_any(
        "code status",
        "dnr",
        "dni",
        "do not resuscitate",
        "do not intubate",
        "no cpr",
        "full code",
    ):
        if has_any("full code"):
            return {
                "proxy_domain": "code_status_dnr_dni",
                "proxy_strength_preliminary": "weak",
                "timing_usability_preliminary": (
                    "requires_full_data_check" if has_timestamp else "post_event_or_unclear"
                ),
                "recommended_next_step": "manual_review_before_counts",
                "decision_preliminary": "needs_review",
                "main_limitation": "Full-code value identifies a code-status source but is not treatment-limitation positive.",
                "notes": "Retain for value-set review so 5.2b can separate code-status source availability from DNR/DNI proxy positivity.",
            }
        return {
            "proxy_domain": "code_status_dnr_dni",
            "proxy_strength_preliminary": "strong" if has_timestamp else "moderate",
            "timing_usability_preliminary": (
                "prediction_time_usable" if has_timestamp else "requires_full_data_check"
            ),
            "recommended_next_step": "full_data_timing_check",
            "decision_preliminary": "include_for_5_2b",
            "main_limitation": "Structured label discovery only; absence of this marker cannot imply absence of limitation.",
            "notes": " ".join(notes) or "Direct code-status proxy candidate if event timing and value semantics hold.",
        }

    return {
        "proxy_domain": "reject_false_positive",
        "proxy_strength_preliminary": "reject",
        "timing_usability_preliminary": "not_usable",
        "recommended_next_step": "exclude",
        "decision_preliminary": "exclude",
        "main_limitation": "Keyword match did not map to a supported proxy domain.",
        "notes": "Rejected during preliminary schema scan.",
    }


def inventory_row(
    *,
    source_schema: str,
    source_table: str,
    source_field: str,
    itemid_or_code: object = "",
    raw_label: object,
    candidate_source_type: str,
    columns_for_linkage: Iterable[str],
    linkage_override: str | None = None,
    notes_suffix: str = "",
) -> dict[str, object]:
    raw = "" if pd.isna(raw_label) else str(raw_label)
    linkage, has_timestamp, timestamp_fields, has_stay, has_hadm, has_subject = source_linkage(
        columns_for_linkage
    )
    if linkage_override:
        linkage = linkage_override
    classification = classify_candidate(
        raw_label=raw,
        source_table=source_table,
        source_field=source_field,
        candidate_source_type=candidate_source_type,
        has_timestamp=has_timestamp,
    )
    if notes_suffix:
        classification["notes"] = f"{classification['notes']} {notes_suffix}".strip()
    return {
        "proxy_domain": classification["proxy_domain"],
        "source_schema": source_schema,
        "source_table": source_table,
        "source_field": source_field,
        "itemid_or_code": "" if pd.isna(itemid_or_code) else str(itemid_or_code),
        "raw_label": raw,
        "normalized_label": normalize_text(raw),
        "matched_terms": ";".join(matched_terms(raw)),
        "candidate_source_type": candidate_source_type,
        "proxy_strength_preliminary": classification["proxy_strength_preliminary"],
        "has_timestamp_field": str(has_timestamp).lower(),
        "timestamp_fields_available": timestamp_fields,
        "linkage_level": linkage,
        "has_stay_id": str(has_stay).lower(),
        "has_hadm_id": str(has_hadm).lower(),
        "has_subject_id": str(has_subject).lower(),
        "timing_usability_preliminary": classification["timing_usability_preliminary"],
        "recommended_next_step": classification["recommended_next_step"],
        "decision_preliminary": classification["decision_preliminary"],
        "main_limitation": classification["main_limitation"],
        "notes": classification["notes"],
    }


def add_matching_unique_rows(
    inventory: list[dict[str, object]],
    df: pd.DataFrame | None,
    *,
    status: TableStatus,
    fields: list[str],
    candidate_source_type: str,
    itemid_field: str | None = None,
    columns_for_linkage: Iterable[str] | None = None,
    linkage_override: str | None = None,
    source_field_joiner: str = "+",
    label_joiner: str = " | ",
    notes_suffix: str = "",
) -> None:
    if df is None or df.empty:
        return
    available = [field for field in fields if field in df.columns]
    if not available:
        return
    subset_fields = available + ([itemid_field] if itemid_field and itemid_field in df.columns else [])
    subset = df[subset_fields].drop_duplicates()
    for _, row in subset.iterrows():
        labels = [str(row[field]) for field in available if not pd.isna(row[field])]
        raw = label_joiner.join(label.strip() for label in labels if label.strip())
        if not matched_terms(raw):
            continue
        itemid = row[itemid_field] if itemid_field and itemid_field in row else ""
        inventory.append(
            inventory_row(
                source_schema=status.schema,
                source_table=status.table,
                source_field=source_field_joiner.join(available),
                itemid_or_code=itemid,
                raw_label=raw,
                candidate_source_type=candidate_source_type,
                columns_for_linkage=columns_for_linkage or status.columns,
                linkage_override=linkage_override,
                notes_suffix=notes_suffix,
            )
        )


def scan_d_items(
    mimic_root: Path,
    inventory: list[dict[str, object]],
    statuses: list[TableStatus],
) -> None:
    df, status = read_table(
        mimic_root,
        "icu",
        "d_items",
        usecols=["itemid", "label", "abbreviation", "linksto", "category", "unitname", "param_type"],
    )
    statuses.append(status)
    if df is None or df.empty:
        return
    for _, row in df.drop_duplicates(subset=["itemid"]).iterrows():
        raw = " | ".join(
            str(row[column])
            for column in ["label", "abbreviation", "category"]
            if column in row and not pd.isna(row[column]) and str(row[column]).strip()
        )
        if not matched_terms(raw):
            continue
        event_columns = linked_event_columns(mimic_root, str(row.get("linksto", "")))
        inventory.append(
            inventory_row(
                source_schema="icu",
                source_table="d_items",
                source_field="label+abbreviation+category",
                itemid_or_code=row.get("itemid", ""),
                raw_label=raw,
                candidate_source_type="icu_item_dictionary",
                columns_for_linkage=event_columns,
                linkage_override=(
                    f"itemid links to icu.{row.get('linksto')}"
                    if str(row.get("linksto", "")).strip()
                    else None
                ),
            )
        )


def scan_event_table_headers(mimic_root: Path, statuses: list[TableStatus]) -> None:
    for table in ["chartevents", "datetimeevents", "procedureevents", "inputevents", "outputevents"]:
        path = table_path(mimic_root, "icu", table)
        statuses.append(
            TableStatus(
                "icu",
                table,
                str(path),
                path.exists(),
                read_columns(path),
                False,
                "header inspected for d_items linkage/timing fields" if path.exists() else "missing",
            )
        )


def scan_poe(
    mimic_root: Path,
    inventory: list[dict[str, object]],
    statuses: list[TableStatus],
    allow_patient_level_rows: bool,
) -> None:
    poe, poe_status = read_table(
        mimic_root,
        "hosp",
        "poe",
        usecols=["poe_id", "subject_id", "hadm_id", "ordertime", "order_type", "order_subtype", "order_status"],
        allow_patient_level_rows=allow_patient_level_rows,
    )
    statuses.append(poe_status)
    add_matching_unique_rows(
        inventory,
        poe,
        status=poe_status,
        fields=["order_type", "order_subtype", "order_status"],
        candidate_source_type="poe_order_value",
    )

    detail, detail_status = read_table(
        mimic_root,
        "hosp",
        "poe_detail",
        usecols=["poe_id", "poe_seq", "subject_id", "field_name", "field_value"],
        allow_patient_level_rows=allow_patient_level_rows,
    )
    statuses.append(detail_status)
    add_matching_unique_rows(
        inventory,
        detail,
        status=detail_status,
        fields=["field_name", "field_value"],
        candidate_source_type="poe_detail_value",
        linkage_override="subject_id direct; poe_id links to hosp.poe for hadm_id/order time",
        notes_suffix="hosp.poe_detail has no direct ordertime; timing requires poe_id link to hosp.poe.",
    )


def scan_admissions(
    mimic_root: Path,
    inventory: list[dict[str, object]],
    statuses: list[TableStatus],
    allow_patient_level_rows: bool,
) -> None:
    admissions, status = read_table(
        mimic_root,
        "hosp",
        "admissions",
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "hospital_expire_flag",
            "discharge_location",
        ],
        allow_patient_level_rows=allow_patient_level_rows,
    )
    statuses.append(status)
    columns = status.columns
    for field in ["deathtime", "hospital_expire_flag"]:
        if field in columns:
            inventory.append(
                inventory_row(
                    source_schema="hosp",
                    source_table="admissions",
                    source_field=field,
                    raw_label=field,
                    candidate_source_type="admission_outcome_field",
                    columns_for_linkage=columns,
                )
            )
    add_matching_unique_rows(
        inventory,
        admissions,
        status=status,
        fields=["discharge_location"],
        candidate_source_type="admission_discharge_value",
    )


def scan_icd(
    mimic_root: Path,
    inventory: list[dict[str, object]],
    statuses: list[TableStatus],
) -> None:
    code_table_columns = {
        "d_icd_diagnoses": read_columns(table_path(mimic_root, "hosp", "diagnoses_icd")),
        "d_icd_procedures": read_columns(table_path(mimic_root, "hosp", "procedures_icd")),
    }
    for table in ["diagnoses_icd", "procedures_icd"]:
        path = table_path(mimic_root, "hosp", table)
        statuses.append(
            TableStatus(
                "hosp",
                table,
                str(path),
                path.exists(),
                read_columns(path),
                False,
                "header inspected only; no patient-level ICD rows scanned",
            )
        )

    for dict_table in ["d_icd_diagnoses", "d_icd_procedures"]:
        df, status = read_table(
            mimic_root,
            "hosp",
            dict_table,
            usecols=["icd_code", "icd_version", "long_title"],
        )
        statuses.append(status)
        if df is None or df.empty:
            continue
        linked_columns = code_table_columns[dict_table]
        for _, row in df.drop_duplicates(subset=["icd_code", "icd_version"]).iterrows():
            raw = row.get("long_title", "")
            if not matched_terms(str(raw)):
                continue
            code = f"ICD{row.get('icd_version', '')}:{row.get('icd_code', '')}"
            inventory.append(
                inventory_row(
                    source_schema="hosp",
                    source_table=dict_table,
                    source_field="long_title",
                    itemid_or_code=code,
                    raw_label=raw,
                    candidate_source_type="icd_dictionary",
                    columns_for_linkage=linked_columns,
                    linkage_override=(
                        "icd_code links to hosp.diagnoses_icd"
                        if dict_table == "d_icd_diagnoses"
                        else "icd_code links to hosp.procedures_icd"
                    ),
                )
            )


def scan_services(
    mimic_root: Path,
    inventory: list[dict[str, object]],
    statuses: list[TableStatus],
    allow_patient_level_rows: bool,
) -> None:
    services, status = read_table(
        mimic_root,
        "hosp",
        "services",
        usecols=["subject_id", "hadm_id", "transfertime", "prev_service", "curr_service"],
        allow_patient_level_rows=allow_patient_level_rows,
    )
    statuses.append(status)
    if services is None or services.empty:
        return
    for field in ["prev_service", "curr_service"]:
        if field not in services.columns:
            continue
        values = services[[field]].drop_duplicates()
        for _, row in values.iterrows():
            raw = str(row[field])
            terms = matched_terms(raw)
            if not terms:
                continue
            if not any(term.lower() in {"palliative", "hospice"} for term in terms):
                continue
            inventory.append(
                inventory_row(
                    source_schema="hosp",
                    source_table="services",
                    source_field=field,
                    raw_label=raw,
                    candidate_source_type="service_value",
                    columns_for_linkage=status.columns,
                    notes_suffix="Only explicit palliative/hospice service markers are retained.",
                )
            )


def status_markdown(statuses: list[TableStatus]) -> str:
    lines = []
    seen: set[tuple[str, str]] = set()
    for status in statuses:
        key = (status.schema, status.table)
        if key in seen:
            continue
        seen.add(key)
        availability = "available" if status.available else "missing"
        scan = "rows scanned" if status.rows_scanned else "header/schema only"
        fields = ", ".join(status.columns) if status.columns else "not available"
        lines.append(
            f"- `{status.schema}.{status.table}`: {availability}; {scan}; columns: {fields}. {status.note}"
        )
    return "\n".join(lines)


def domain_summary_markdown(inventory: list[dict[str, object]]) -> str:
    counter = Counter(str(row["proxy_domain"]) for row in inventory)
    domains = [
        "code_status_dnr_dni",
        "comfort_measures_only",
        "withdrawal_or_withholding",
        "palliative_care",
        "hospice",
        "brain_death_or_organ_donation",
        "ama_or_nonstandard_discharge",
        "ambiguous_goals_of_care",
        "reject_false_positive",
    ]
    lines = []
    for domain in domains:
        rows = [row for row in inventory if row["proxy_domain"] == domain]
        if not rows:
            lines.append(f"- `{domain}`: no candidate inventory rows found.")
            continue
        examples = sorted({str(row["raw_label"]) for row in rows})[:5]
        lines.append(
            f"- `{domain}`: {counter[domain]} candidate inventory row(s). Examples: "
            + "; ".join(examples)
        )
    return "\n".join(lines)


def classification_markdown(inventory: list[dict[str, object]]) -> str:
    counter = Counter(str(row["proxy_strength_preliminary"]) for row in inventory)
    return "\n".join(
        f"- `{strength}`: {counter.get(strength, 0)} candidate inventory row(s)."
        for strength in ["strong", "moderate", "weak", "reject"]
    )


def recommended_markdown(inventory: list[dict[str, object]]) -> str:
    rows = [
        row
        for row in inventory
        if row["decision_preliminary"]
        in {"include_for_5_2b", "include_descriptive_only_for_5_2b"}
    ]
    if not rows:
        return "- No candidates recommended for 5.2b from this schema/demo scan."
    lines = []
    for row in rows:
        lines.append(
            "- "
            f"`{row['proxy_domain']}` from `{row['source_schema']}.{row['source_table']}` "
            f"`{row['source_field']}` `{row['itemid_or_code']}`: {row['raw_label']} "
            f"({row['decision_preliminary']}; {row['recommended_next_step']})."
        )
    return "\n".join(lines)


def warnings_markdown(inventory: list[dict[str, object]]) -> str:
    rows = [
        row
        for row in inventory
        if row["decision_preliminary"] == "needs_review"
        or row["proxy_domain"] in {"ambiguous_goals_of_care", "reject_false_positive"}
    ]
    if not rows:
        return "- No ambiguous or rejected keyword hits were found."
    lines = []
    for row in rows:
        lines.append(
            "- "
            f"`{row['proxy_domain']}` from `{row['source_schema']}.{row['source_table']}` "
            f"`{row['source_field']}`: {row['raw_label']}. "
            f"Reason: {row['main_limitation']}"
        )
    return "\n".join(lines)


def missing_markdown(statuses: list[TableStatus]) -> str:
    missing = [status for status in statuses if not status.available]
    header_only = [status for status in statuses if status.available and not status.rows_scanned]
    lines = []
    if missing:
        for status in missing:
            lines.append(f"- `{status.schema}.{status.table}`: missing at `{status.path}`.")
    if header_only:
        for status in header_only:
            lines.append(f"- `{status.schema}.{status.table}`: not row-scanned; {status.note}.")
    return "\n".join(lines) if lines else "- None."


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


def write_inventory(path: Path, inventory: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_COLUMNS)
        writer.writeheader()
        for row in inventory:
            writer.writerow({column: row.get(column, "") for column in INVENTORY_COLUMNS})


def write_note(
    path: Path,
    *,
    inventory: list[dict[str, object]],
    statuses: list[TableStatus],
    output_dir: Path,
    run_mode_note: str,
) -> None:
    text = f"""# MIMIC Treatment-Limitation / End-of-Life Proxy Schema Scan Note

## Scope

This is schema/demo/dictionary discovery only for candidate structured proxies. It makes no full-cohort prevalence claims, performs no low-predicted fatal-case analysis, does not inspect notes/NLP, and does not alter cohort definitions or primary risk-model features. All variables named here are preliminary proxy candidates or rejected keyword hits for review.

Run mode: {run_mode_note}

## Sources inspected

{status_markdown(statuses)}

## Candidate proxy domains found

{domain_summary_markdown(inventory)}

## Sources not available or not inspected

{missing_markdown(statuses)}

## Preliminary classification

{classification_markdown(inventory)}

## Recommended candidate list for full-data aggregation

{recommended_markdown(inventory)}

## Warnings for review

{warnings_markdown(inventory)}

AMA, elopement, and left-against-medical-advice markers are discharge-process or care-process context flags only. They are not treatment-limitation or end-of-life proxies. Palliative care is not equivalent to DNR/DNI, withdrawal, or withholding. Hospice discharge is discharge/end-of-life context, not direct ICU treatment limitation unless stronger structured evidence is found in a later full-data timing/value check. Absence of a structured marker must not be interpreted as absence of treatment limitation.

## Files produced

- `{output_dir / "mimic_treatment_limitation_proxy_inventory_schema_scan.csv"}`
- `{output_dir / "mimic_treatment_limitation_schema_scan_note.md"}`
- `{output_dir / "manifest_schema_scan.json"}`
"""
    path.write_text(text)


def write_manifest(
    path: Path,
    *,
    mimic_root: Path,
    output_dir: Path,
    statuses: list[TableStatus],
    allow_patient_level_rows: bool,
) -> None:
    demo = is_demo_root(mimic_root)
    row_scanned_patient_tables = sorted(
        {
            f"{status.schema}.{status.table}"
            for status in statuses
            if status.rows_scanned and status.table in PATIENT_LEVEL_TABLES
        }
    )
    payload = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "script_paths": [
            "scripts/chapter1_mimic/inventory_mimic_treatment_limitation_schema_scan.py"
        ],
        "mimic_root": str(mimic_root),
        "tables_schemas_searched": [
            {
                "schema": status.schema,
                "table": status.table,
                "available": status.available,
                "rows_scanned": status.rows_scanned,
                "columns": list(status.columns),
                "note": status.note,
            }
            for status in statuses
        ],
        "outputs_created": [
            str(output_dir / "mimic_treatment_limitation_proxy_inventory_schema_scan.csv"),
            str(output_dir / "mimic_treatment_limitation_schema_scan_note.md"),
            str(output_dir / "manifest_schema_scan.json"),
        ],
        "metadata_mode": (
            "demo_schema_and_structured_demo_values"
            if demo
            else "dictionary_metadata_and_schema_headers"
        ),
        "demo_schema_only_or_full_dictionary_metadata_used": (
            "demo/schema structured metadata used"
            if demo
            else "full/non-demo dictionary metadata and schema headers only by default"
        ),
        "patient_level_data_accessed": bool(row_scanned_patient_tables),
        "patient_level_data_access_note": (
            "Only structured MIMIC-IV demo rows were scanned for unique candidate values; no protected full-data patient rows were accessed."
            if row_scanned_patient_tables and demo
            else (
                "Patient-level rows were scanned because --allow-patient-level-rows was set."
                if row_scanned_patient_tables and allow_patient_level_rows
                else "No patient-level rows were scanned."
            )
        ),
        "patient_level_tables_row_scanned": row_scanned_patient_tables,
        "prevalence_claims_made": False,
        "statement_no_prevalence_claims": "No prevalence claims are made by this schema/demo/dictionary discovery scan.",
        "notes_nlp_used": False,
        "low_predicted_fatal_case_analysis_attempted": False,
        "cohort_or_risk_model_features_changed": False,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory structured MIMIC treatment-limitation/end-of-life proxy schema candidates."
    )
    parser.add_argument("--mimic-root", type=Path, default=DEFAULT_MIMIC_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-patient-level-rows",
        action="store_true",
        help=(
            "Allow row scans of patient-level structured tables for non-demo MIMIC roots. "
            "By default, non-demo roots are restricted to dictionary rows and schema headers."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mimic_root = args.mimic_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory: list[dict[str, object]] = []
    statuses: list[TableStatus] = []

    scan_event_table_headers(mimic_root, statuses)
    scan_poe(mimic_root, inventory, statuses, args.allow_patient_level_rows)
    scan_d_items(mimic_root, inventory, statuses)
    scan_admissions(mimic_root, inventory, statuses, args.allow_patient_level_rows)
    scan_icd(mimic_root, inventory, statuses)
    scan_services(mimic_root, inventory, statuses, args.allow_patient_level_rows)

    inventory = sorted(
        inventory,
        key=lambda row: (
            str(row["proxy_domain"]),
            str(row["source_schema"]),
            str(row["source_table"]),
            str(row["source_field"]),
            str(row["itemid_or_code"]),
            str(row["raw_label"]),
        ),
    )

    inventory_path = output_dir / "mimic_treatment_limitation_proxy_inventory_schema_scan.csv"
    note_path = output_dir / "mimic_treatment_limitation_schema_scan_note.md"
    manifest_path = output_dir / "manifest_schema_scan.json"

    write_inventory(inventory_path, inventory)
    run_mode_note = (
        "MIMIC-IV demo structured rows and dictionaries were inspected."
        if is_demo_root(mimic_root)
        else "Non-demo root: dictionary metadata and schema headers were inspected; patient-level row scans require --allow-patient-level-rows."
    )
    write_note(
        note_path,
        inventory=inventory,
        statuses=statuses,
        output_dir=output_dir,
        run_mode_note=run_mode_note,
    )
    write_manifest(
        manifest_path,
        mimic_root=mimic_root,
        output_dir=output_dir,
        statuses=statuses,
        allow_patient_level_rows=args.allow_patient_level_rows,
    )

    print(f"Wrote {inventory_path}")
    print(f"Wrote {note_path}")
    print(f"Wrote {manifest_path}")
    print(f"Candidate inventory rows: {len(inventory)}")


if __name__ == "__main__":
    main()
