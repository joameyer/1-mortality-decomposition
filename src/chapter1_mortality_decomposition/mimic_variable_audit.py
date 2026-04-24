from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "ch1_mimic_variable_audit.yaml"
OPTIONAL_EVENT_TABLES = {"ingredientevents"}


@dataclass(frozen=True)
class AuditConfig:
    mimic_root: Path
    mapping_csv: Path
    crosscheck_csv: Path
    unit_reference_csv: Path
    reports_dir: Path
    include_etco2: bool = True
    include_ph_art: bool = True
    chunksize: int = 200_000


@dataclass(frozen=True)
class Candidate:
    variable: str
    itemid: int
    table: str
    family: str
    role: str
    context_note: str
    value_column: str = "valuenum"
    unit_column: str = "valueuom"


@dataclass
class CandidateStats:
    candidate: Candidate
    label: str = ""
    abbreviation: str = ""
    category: str = ""
    unitname: str = ""
    linksto: str = ""
    param_type: str = ""
    fluid: str = ""
    total_row_count: int = 0
    non_null_numeric_count: int = 0
    blank_unit_count: int = 0
    subject_ids: set[str] = field(default_factory=set)
    hadm_ids: set[str] = field(default_factory=set)
    stay_ids: set[str] = field(default_factory=set)
    unit_counts: Counter[str] = field(default_factory=Counter)
    value_counts: Counter[str] = field(default_factory=Counter)
    numeric_values: list[float] = field(default_factory=list)
    audit_status: str = "scanned"
    skip_reason: str = ""

    def update(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        self.total_row_count += int(len(frame))
        for column, target in (
            ("subject_id", self.subject_ids),
            ("hadm_id", self.hadm_ids),
            ("stay_id", self.stay_ids),
        ):
            if column in frame.columns:
                values = frame[column].dropna().astype("string")
                target.update(str(value) for value in values if str(value) != "<NA>")

        if self.candidate.unit_column in frame.columns:
            units = frame[self.candidate.unit_column].fillna("").astype("string")
            for value, count in units.value_counts(dropna=False).items():
                unit = "" if str(value) == "<NA>" else str(value)
                self.unit_counts[unit] += int(count)
            self.blank_unit_count += int(units.fillna("").str.strip().eq("").sum())

        if "value" in frame.columns and self.candidate.family == "temperature_site":
            values = frame["value"].fillna("").astype("string")
            for value, count in values.value_counts(dropna=False).items():
                text = "" if str(value) == "<NA>" else str(value)
                self.value_counts[text] += int(count)

        if self.candidate.value_column in frame.columns:
            numeric = pd.to_numeric(frame[self.candidate.value_column], errors="coerce")
            numeric = numeric.dropna()
            self.non_null_numeric_count += int(numeric.shape[0])
            if not numeric.empty:
                self.numeric_values.extend(float(value) for value in numeric.to_numpy())

    def as_row(self) -> dict[str, object]:
        values = np.asarray(self.numeric_values, dtype=float)
        if values.size:
            p25, median, p75 = np.percentile(values, [25, 50, 75])
            minimum = float(np.min(values))
            maximum = float(np.max(values))
        else:
            minimum = p25 = median = p75 = maximum = ""
        return {
            "variable": self.candidate.variable,
            "candidate_itemid": self.candidate.itemid,
            "source_table": self.candidate.table,
            "candidate_family": self.candidate.family,
            "candidate_role": self.candidate.role,
            "label": self.label,
            "abbreviation": self.abbreviation,
            "category": self.category,
            "unitname": self.unitname,
            "linksto": self.linksto,
            "param_type": self.param_type,
            "fluid": self.fluid,
            "total_row_count": self.total_row_count,
            "distinct_icu_stay_count": len(self.stay_ids) if self.stay_ids else 0,
            "distinct_hadm_count": len(self.hadm_ids) if self.hadm_ids else 0,
            "distinct_subject_count": len(self.subject_ids) if self.subject_ids else 0,
            "non_null_numeric_count": self.non_null_numeric_count,
            "missing_blank_unit_count": self.blank_unit_count,
            "min": minimum,
            "p25": float(p25) if values.size else "",
            "median": float(median) if values.size else "",
            "p75": float(p75) if values.size else "",
            "max": maximum,
            "observed_unit_counts_json": json.dumps(dict(sorted(self.unit_counts.items()))),
            "observed_value_counts_json": json.dumps(dict(self.value_counts.most_common(20))),
            "context_note": self.candidate.context_note,
            "audit_status": self.audit_status,
            "skip_reason": self.skip_reason,
        }


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_config(path: Path) -> AuditConfig:
    raw: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(f"Unsupported config line in {path}: {line!r}")
        key, value = stripped.split(":", 1)
        raw[key.strip()] = value.strip().strip("'\"")

    required = [
        "mimic_root",
        "mapping_csv",
        "crosscheck_csv",
        "unit_reference_csv",
        "reports_dir",
    ]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Audit config {path} is missing keys: {missing}")

    return AuditConfig(
        mimic_root=_resolve_path(raw["mimic_root"]),
        mapping_csv=_resolve_path(raw["mapping_csv"]),
        crosscheck_csv=_resolve_path(raw["crosscheck_csv"]),
        unit_reference_csv=_resolve_path(raw["unit_reference_csv"]),
        reports_dir=_resolve_path(raw["reports_dir"]),
        include_etco2=_parse_bool(raw.get("include_etco2", "true")),
        include_ph_art=_parse_bool(raw.get("include_ph_art", "true")),
        chunksize=int(raw.get("chunksize", "200000")),
    )


def table_path(mimic_root: Path, table: str) -> Path:
    module = {
        "chartevents": "icu",
        "inputevents": "icu",
        "ingredientevents": "icu",
        "d_items": "icu",
        "icustays": "icu",
        "labevents": "hosp",
        "d_labitems": "hosp",
        "patients": "hosp",
    }[table]
    return mimic_root / module / f"{table}.csv.gz"


def require_inputs(config: AuditConfig) -> None:
    required_paths = [
        config.mapping_csv,
        config.crosscheck_csv,
        config.unit_reference_csv,
        table_path(config.mimic_root, "d_items"),
        table_path(config.mimic_root, "d_labitems"),
        table_path(config.mimic_root, "chartevents"),
        table_path(config.mimic_root, "labevents"),
        table_path(config.mimic_root, "inputevents"),
        table_path(config.mimic_root, "patients"),
        table_path(config.mimic_root, "icustays"),
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required audit inputs:\n" + "\n".join(missing))


def candidate_plan(include_etco2: bool, include_ph_art: bool) -> list[Candidate]:
    candidates = [
        Candidate("core_temp", 223762, "chartevents", "generic_temperature", "candidate", "Celsius body-temperature channel; does not by itself prove core-temperature site."),
        Candidate("core_temp", 223761, "chartevents", "generic_temperature", "candidate", "Fahrenheit body-temperature channel requiring Celsius conversion; does not by itself prove core-temperature site."),
        Candidate("core_temp", 224642, "chartevents", "temperature_site", "support", "Text site field that may indicate whether later site restriction is feasible.", value_column="valuenum"),
        Candidate("urea", 51006, "labevents", "bun_urea_nitrogen_lab", "primary_proxy", "Chemistry Urea Nitrogen/BUN lab source; supports analyte-conversion proxy, not native urea."),
        Candidate("urea", 52647, "labevents", "bun_urea_nitrogen_lab", "primary_proxy", "Additional Urea Nitrogen lab source; often absent in demo."),
        Candidate("urea", 225624, "chartevents", "bun_chart_mirror", "secondary_proxy", "Charted BUN mirror; supports BUN/urea proxy only."),
        Candidate("sao2", 220227, "chartevents", "arterial_labeled_chart", "preferred", "Arterial-labeled chart candidate."),
        Candidate("sao2", 50817, "labevents", "blood_gas_lab", "conditional_secondary", "Broader blood-gas oxygen saturation; arteriality/provenance must stay visible."),
        Candidate("lactate_art", 225668, "chartevents", "chart_lactate_like", "conditional_mirror", "Charted Lactic Acid candidate; same analyte but arterial/blood-gas context not proven."),
        Candidate("lactate_art", 50813, "labevents", "blood_gas_lab", "preferred", "Blood-gas lactate source; arteriality remains a provenance question."),
        Candidate("lactate_art", 52442, "labevents", "blood_gas_lab", "preferred", "Additional blood-gas lactate source; often absent in demo."),
        Candidate("vt", 224684, "chartevents", "tidal_volume_set", "preferred", "Setting-oriented VT candidate."),
        Candidate("vt", 224685, "chartevents", "tidal_volume_observed", "fallback", "Observed VT candidate; not interchangeable with set VT."),
        Candidate("vt", 224686, "chartevents", "tidal_volume_spontaneous", "fallback", "Spontaneous VT candidate; not interchangeable with set VT."),
        Candidate("vt_per_kg_ibw", 224684, "chartevents", "vt_support", "support", "VT set support for later VT/IBW derivation."),
        Candidate("vt_per_kg_ibw", 224685, "chartevents", "vt_support", "support", "VT observed support for later VT/IBW sensitivity."),
        Candidate("vt_per_kg_ibw", 224686, "chartevents", "vt_support", "support", "VT spontaneous support for later VT/IBW sensitivity."),
        Candidate("vt_per_kg_ibw", 226730, "chartevents", "height_support", "support", "Height in cm for later IBW derivation."),
        Candidate("vt_per_kg_ibw", 226707, "chartevents", "height_support", "support", "Height in inches for later IBW derivation after conversion."),
        Candidate("vt_per_kg_ibw", 226512, "chartevents", "weight_context", "support", "Admission weight context only; should not replace IBW."),
        Candidate("vt_per_kg_ibw", 226531, "chartevents", "weight_context", "support", "Admission weight in pounds context only; should not replace IBW."),
        Candidate("vt_per_kg_ibw", 224639, "chartevents", "weight_context", "support", "Daily weight context only; should not replace IBW."),
        Candidate("pf_ratio", 220224, "chartevents", "pao2_chart", "support", "Arterial-labeled chart PaO2 support for later PF derivation."),
        Candidate("pf_ratio", 50821, "labevents", "pao2_lab", "support", "Broader blood-gas pO2 support; arteriality/provenance remains visible."),
        Candidate("pf_ratio", 223835, "chartevents", "fio2_chart", "support", "FiO2 support for later PF derivation; percent/fraction handling remains explicit."),
        Candidate("bicarbonate_art", 50803, "labevents", "blood_gas_measurement", "preferred_sparse", "Calculated Bicarbonate, Whole Blood; sparse and not explicitly arterial."),
        Candidate("bicarbonate_art", 50882, "labevents", "serum_measurement", "wrong_context", "Serum chemistry bicarbonate; should not masquerade as arterial bicarbonate."),
        Candidate("bicarbonate_art", 224826, "chartevents", "serum_measurement", "wrong_context", "Charted serum bicarbonate mirror; wrong context for arterial bicarbonate."),
        Candidate("bicarbonate_art", 227443, "chartevents", "serum_measurement", "wrong_context", "Charted serum bicarbonate mirror; wrong context for arterial bicarbonate."),
        Candidate("bicarbonate_art", 226759, "chartevents", "apache_score", "wrong_context", "APACHE-derived value; not a raw measurement."),
        Candidate("bicarbonate_art", 226760, "chartevents", "apache_score", "wrong_context", "APACHE score component; not a raw measurement."),
        Candidate("bicarbonate_art", 225165, "inputevents", "treatment_input", "wrong_context", "Bicarbonate treatment/input, not a patient measurement.", value_column="amount", unit_column="amountuom"),
        Candidate("bicarbonate_art", 220995, "inputevents", "treatment_input", "wrong_context", "Sodium bicarbonate medication/input, not a patient measurement.", value_column="amount", unit_column="amountuom"),
        Candidate("bicarbonate_art", 227533, "inputevents", "treatment_input", "wrong_context", "Sodium bicarbonate amp input, not a patient measurement.", value_column="amount", unit_column="amountuom"),
        Candidate("bicarbonate_art", 221211, "inputevents", "treatment_input", "wrong_context", "Sodium bicarbonate input item not in use; not a patient measurement.", value_column="amount", unit_column="amountuom"),
        Candidate("bicarbonate_art", 220994, "ingredientevents", "ingredient", "wrong_context", "Ingredient record, not a patient measurement.", value_column="amount", unit_column="amountuom"),
    ]
    if include_etco2:
        candidates.append(Candidate("etco2", 228640, "chartevents", "etco2_chart", "optional", "EtCO2 candidate; unit metadata and sparse coverage require review."))
    if include_ph_art:
        candidates.extend(
            [
                Candidate("ph_art", 223830, "chartevents", "arterial_labeled_chart", "resolved_preferred", "Resolved direct semantic mapping; audit coverage/provenance only."),
                Candidate("ph_art", 50820, "labevents", "blood_gas_lab", "conditional_secondary", "Broader blood-gas pH candidate; only coverage/provenance check."),
            ]
        )
    return candidates


def load_mapping_context(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        return {row["asic_base_variable"]: row for row in csv.DictReader(handle)}


def load_dictionary_metadata(config: AuditConfig) -> dict[tuple[str, int], dict[str, str]]:
    def clean(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value)

    metadata: dict[tuple[str, int], dict[str, str]] = {}
    d_items = pd.read_csv(table_path(config.mimic_root, "d_items"))
    for row in d_items.to_dict("records"):
        metadata[("d_items", int(row["itemid"]))] = {
            "label": clean(row.get("label", "")),
            "abbreviation": clean(row.get("abbreviation", "")),
            "category": clean(row.get("category", "")),
            "unitname": clean(row.get("unitname", "")),
            "linksto": clean(row.get("linksto", "")),
            "param_type": clean(row.get("param_type", "")),
            "fluid": "",
        }

    d_labitems = pd.read_csv(table_path(config.mimic_root, "d_labitems"))
    for row in d_labitems.to_dict("records"):
        metadata[("d_labitems", int(row["itemid"]))] = {
            "label": clean(row.get("label", "")),
            "abbreviation": "",
            "category": clean(row.get("category", "")),
            "unitname": "",
            "linksto": "labevents",
            "param_type": "",
            "fluid": clean(row.get("fluid", "")),
        }
    return metadata


def dictionary_key(candidate: Candidate) -> tuple[str, int]:
    if candidate.table == "labevents":
        return ("d_labitems", candidate.itemid)
    return ("d_items", candidate.itemid)


def initialize_stats(
    candidates: Iterable[Candidate],
    metadata: dict[tuple[str, int], dict[str, str]],
) -> dict[tuple[str, int, str], CandidateStats]:
    stats: dict[tuple[str, int, str], CandidateStats] = {}
    for candidate in candidates:
        item_metadata = metadata.get(dictionary_key(candidate), {})
        stat = CandidateStats(candidate=candidate)
        for key, value in item_metadata.items():
            setattr(stat, key, value)
        stats[(candidate.variable, candidate.itemid, candidate.table)] = stat
    return stats


def scan_event_table(
    config: AuditConfig,
    table: str,
    table_candidates: list[Candidate],
    stats: dict[tuple[str, int, str], CandidateStats],
) -> None:
    path = table_path(config.mimic_root, table)
    itemids = sorted({candidate.itemid for candidate in table_candidates})
    if not itemids:
        return
    if not path.exists():
        if table in OPTIONAL_EVENT_TABLES:
            for candidate in table_candidates:
                stat = stats[(candidate.variable, candidate.itemid, candidate.table)]
                stat.audit_status = "skipped_optional_table_missing"
                stat.skip_reason = f"Optional source table unavailable: {path}"
            return
        raise FileNotFoundError(f"Missing required audit table: {path}")
    itemid_set = set(itemids)
    value_columns = sorted({candidate.value_column for candidate in table_candidates})
    unit_columns = sorted({candidate.unit_column for candidate in table_candidates})
    requested_columns = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "itemid",
        "value",
        *value_columns,
        *unit_columns,
    ]
    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [column for column in requested_columns if column in header]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=config.chunksize):
        filtered = chunk[chunk["itemid"].isin(itemid_set)]
        if filtered.empty:
            continue
        for candidate in table_candidates:
            candidate_rows = filtered[filtered["itemid"].eq(candidate.itemid)]
            stats[(candidate.variable, candidate.itemid, candidate.table)].update(candidate_rows)


def audit_candidates(config: AuditConfig, candidates: list[Candidate]) -> pd.DataFrame:
    metadata = load_dictionary_metadata(config)
    stats = initialize_stats(candidates, metadata)
    by_table: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_table.setdefault(candidate.table, []).append(candidate)
    for table, table_candidates in by_table.items():
        scan_event_table(config, table, table_candidates, stats)
    rows = [stat.as_row() for stat in stats.values()]
    return pd.DataFrame(rows)


def compact_rows(frame: pd.DataFrame, variables: Iterable[str]) -> pd.DataFrame:
    return frame[frame["variable"].isin(set(variables))].copy()


def read_population_tables(config: AuditConfig) -> dict[str, int]:
    icustays = pd.read_csv(
        table_path(config.mimic_root, "icustays"),
        usecols=["subject_id", "hadm_id", "stay_id"],
    )
    patients = pd.read_csv(
        table_path(config.mimic_root, "patients"),
        usecols=["subject_id", "gender"],
    )
    return {
        "total_icu_stays": int(icustays["stay_id"].nunique()),
        "total_hadms_in_icustays": int(icustays["hadm_id"].nunique()),
        "total_subjects_in_icustays": int(icustays["subject_id"].nunique()),
        "total_subjects_in_patients": int(patients["subject_id"].nunique()),
        "subjects_with_gender": int(patients.dropna(subset=["gender"])["subject_id"].nunique()),
    }


def build_temperature_audit(overview: pd.DataFrame) -> pd.DataFrame:
    frame = compact_rows(overview, ["core_temp"])
    temp_rows = frame[frame["candidate_family"].eq("generic_temperature")]
    total_temp_rows = int(temp_rows["total_row_count"].sum())
    max_share = 0.0
    dominant_itemid = ""
    if total_temp_rows:
        idx = temp_rows["total_row_count"].idxmax()
        dominant_itemid = str(int(temp_rows.loc[idx, "candidate_itemid"]))
        max_share = float(temp_rows.loc[idx, "total_row_count"]) / float(total_temp_rows)
    frame["temperature_audit_summary"] = ""
    frame.loc[:, "temperature_audit_summary"] = (
        f"Generic temperature rows={total_temp_rows}; dominant itemid={dominant_itemid}; "
        f"dominant share={max_share:.3f}; site item availability should determine whether "
        "core-temperature restriction is feasible."
    )
    return frame


def build_derived_readiness(overview: pd.DataFrame, population: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def subset(variable: str, families: Iterable[str]) -> pd.DataFrame:
        return overview[
            overview["variable"].eq(variable)
            & overview["candidate_family"].isin(set(families))
        ]

    vt_support = subset("vt_per_kg_ibw", ["vt_support"])
    height_support = subset("vt_per_kg_ibw", ["height_support"])
    weight_context = subset("vt_per_kg_ibw", ["weight_context"])
    rows.append(
        {
            "derived_variable": "vt_per_kg_ibw",
            "support_family": "vt_candidates",
            "candidate_itemids": "|".join(vt_support["candidate_itemid"].astype(str)),
            "total_rows": int(vt_support["total_row_count"].sum()),
            "distinct_icu_stays": int(vt_support["distinct_icu_stay_count"].max() if not vt_support.empty else 0),
            "distinct_hadms": int(vt_support["distinct_hadm_count"].max() if not vt_support.empty else 0),
            "readiness_note": "VT candidates exist; final derivation still depends on choosing set vs fallback/sensitivity source.",
        }
    )
    rows.append(
        {
            "derived_variable": "vt_per_kg_ibw",
            "support_family": "height_candidates",
            "candidate_itemids": "|".join(height_support["candidate_itemid"].astype(str)),
            "total_rows": int(height_support["total_row_count"].sum()),
            "distinct_icu_stays": int(height_support["distinct_icu_stay_count"].max() if not height_support.empty else 0),
            "distinct_hadms": int(height_support["distinct_hadm_count"].max() if not height_support.empty else 0),
            "readiness_note": "Height availability supports later IBW derivation; no patient-level VT/IBW values are exported here.",
        }
    )
    rows.append(
        {
            "derived_variable": "vt_per_kg_ibw",
            "support_family": "weight_context_not_ibw",
            "candidate_itemids": "|".join(weight_context["candidate_itemid"].astype(str)),
            "total_rows": int(weight_context["total_row_count"].sum()),
            "distinct_icu_stays": int(weight_context["distinct_icu_stay_count"].max() if not weight_context.empty else 0),
            "distinct_hadms": int(weight_context["distinct_hadm_count"].max() if not weight_context.empty else 0),
            "readiness_note": "Weight exists for context/QC but should not replace IBW.",
        }
    )
    rows.append(
        {
            "derived_variable": "vt_per_kg_ibw",
            "support_family": "sex_gender",
            "candidate_itemids": "patients.gender",
            "total_rows": population["subjects_with_gender"],
            "distinct_icu_stays": population["total_icu_stays"],
            "distinct_hadms": population["total_hadms_in_icustays"],
            "readiness_note": "Aggregated sex/gender availability only; no patient-level records exported.",
        }
    )

    pao2_chart = subset("pf_ratio", ["pao2_chart"])
    pao2_lab = subset("pf_ratio", ["pao2_lab"])
    fio2 = subset("pf_ratio", ["fio2_chart"])
    chart_pao2_stays = int(pao2_chart["distinct_icu_stay_count"].max() if not pao2_chart.empty else 0)
    fio2_stays = int(fio2["distinct_icu_stay_count"].max() if not fio2.empty else 0)
    rows.append(
        {
            "derived_variable": "pf_ratio",
            "support_family": "pao2_chart_candidates",
            "candidate_itemids": "|".join(pao2_chart["candidate_itemid"].astype(str)),
            "total_rows": int(pao2_chart["total_row_count"].sum()),
            "distinct_icu_stays": chart_pao2_stays,
            "distinct_hadms": int(pao2_chart["distinct_hadm_count"].max() if not pao2_chart.empty else 0),
            "readiness_note": "Chart PaO2 support; exact PF ratio derivation and timing alignment are intentionally not performed.",
        }
    )
    rows.append(
        {
            "derived_variable": "pf_ratio",
            "support_family": "pao2_lab_candidates",
            "candidate_itemids": "|".join(pao2_lab["candidate_itemid"].astype(str)),
            "total_rows": int(pao2_lab["total_row_count"].sum()),
            "distinct_icu_stays": int(pao2_lab["distinct_icu_stay_count"].max() if not pao2_lab.empty else 0),
            "distinct_hadms": int(pao2_lab["distinct_hadm_count"].max() if not pao2_lab.empty else 0),
            "readiness_note": "Lab pO2 support has admission-level IDs only in labevents; arteriality/provenance remains visible.",
        }
    )
    rows.append(
        {
            "derived_variable": "pf_ratio",
            "support_family": "fio2_candidates",
            "candidate_itemids": "|".join(fio2["candidate_itemid"].astype(str)),
            "total_rows": int(fio2["total_row_count"].sum()),
            "distinct_icu_stays": fio2_stays,
            "distinct_hadms": int(fio2["distinct_hadm_count"].max() if not fio2.empty else 0),
            "readiness_note": "FiO2 support exists; percent/fraction handling must remain explicit.",
        }
    )
    rows.append(
        {
            "derived_variable": "pf_ratio",
            "support_family": "coarse_chart_stay_overlap_upper_bound",
            "candidate_itemids": "220224|223835",
            "total_rows": "",
            "distinct_icu_stays": min(chart_pao2_stays, fio2_stays),
            "distinct_hadms": "",
            "readiness_note": "Coarse aggregate upper bound from separate stay counts only; no patient-level or timestamp-level PF trajectory computed.",
        }
    )
    return pd.DataFrame(rows)


def enrich_with_mapping_context(
    overview: pd.DataFrame,
    mapping_context: dict[str, dict[str, str]],
) -> pd.DataFrame:
    frame = overview.copy()
    frame["current_mapping_quality"] = frame["variable"].map(
        lambda variable: mapping_context.get(variable, {}).get("mapping_quality", "")
    )
    frame["current_risk_flag"] = frame["variable"].map(
        lambda variable: mapping_context.get(variable, {}).get("risk_flag", "")
    )
    frame["current_mapping_logic"] = frame["variable"].map(
        lambda variable: mapping_context.get(variable, {}).get(
            "candidate_itemids_or_source_logic",
            "",
        )
    )
    return frame


def write_outputs(config: AuditConfig, overview: pd.DataFrame) -> None:
    reports_dir = config.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    overview_path = reports_dir / "ch1_mimic_variable_audit_overview.csv"
    overview.to_csv(overview_path, index=False)
    build_temperature_audit(overview).to_csv(
        reports_dir / "ch1_mimic_temperature_audit.csv",
        index=False,
    )
    compact_rows(overview, ["sao2", "lactate_art", "ph_art"]).to_csv(
        reports_dir / "ch1_mimic_blood_gas_audit.csv",
        index=False,
    )
    compact_rows(overview, ["vt"]).to_csv(
        reports_dir / "ch1_mimic_vt_audit.csv",
        index=False,
    )
    compact_rows(overview, ["bicarbonate_art"]).to_csv(
        reports_dir / "ch1_mimic_bicarbonate_audit.csv",
        index=False,
    )
    compact_rows(overview, ["urea"]).to_csv(
        reports_dir / "ch1_mimic_urea_audit.csv",
        index=False,
    )
    population = read_population_tables(config)
    build_derived_readiness(overview, population).to_csv(
        reports_dir / "ch1_mimic_derived_readiness_audit.csv",
        index=False,
    )
    write_note(config, overview, population)


def _top_candidate(frame: pd.DataFrame, variable: str) -> str:
    subset = frame[frame["variable"].eq(variable)]
    if subset.empty:
        return "none"
    idx = subset["total_row_count"].astype(int).idxmax()
    row = subset.loc[idx]
    return f"{int(row['candidate_itemid'])} {row['label']} ({int(row['total_row_count'])} rows)"


def _top_by_role(frame: pd.DataFrame, variable: str, roles: set[str]) -> str:
    subset = frame[frame["variable"].eq(variable) & frame["candidate_role"].isin(roles)]
    if subset.empty:
        return "none"
    idx = subset["total_row_count"].astype(int).idxmax()
    row = subset.loc[idx]
    return f"{int(row['candidate_itemid'])} {row['label']} ({int(row['total_row_count'])} rows)"


def optional_source_notes(overview: pd.DataFrame) -> list[str]:
    if "audit_status" not in overview.columns:
        return []
    skipped = overview[overview["audit_status"].eq("skipped_optional_table_missing")]
    notes: list[str] = []
    for _, row in skipped.iterrows():
        notes.append(
            f"`{row['source_table']}` unavailable for `{row['variable']}` "
            f"candidate `{int(row['candidate_itemid'])} {row['label']}`; "
            "the branch was kept in the audit output as skipped with zero counts."
        )
    return notes


def write_note(config: AuditConfig, overview: pd.DataFrame, population: dict[str, int]) -> None:
    audited_variables = [
        "core_temp",
        "urea",
        "sao2",
        "lactate_art",
        "vt",
        "vt_per_kg_ibw",
        "pf_ratio",
        "bicarbonate_art",
    ]
    if config.include_etco2:
        audited_variables.append("etco2")
    if config.include_ph_art:
        audited_variables.append("ph_art")

    unsafe = [
        "core_temp remains semantically unsafe as direct core temperature without site/provenance restriction.",
        "bicarbonate_art remains unsafe to broaden to serum, APACHE, medication, input, or ingredient bicarbonate candidates.",
        "urea remains an explicit BUN/urea analyte-conversion proxy, not a native urea match.",
        "lactate_art and sao2 need source-context caution if broader lab candidates are used.",
    ]
    full_data_dependent = [
        "Whether sparse demo candidates such as bicarbonate_art and etco2 have enough full-data support.",
        "Whether broader lab candidates are needed for sao2, lactate_art, pf_ratio inputs, or ph_art coverage.",
        "Whether temperature-site distributions permit a defensible core-temperature restriction.",
        "Whether VT set dominates enough to avoid observed/spontaneous fallback rules.",
    ]
    optional_notes = optional_source_notes(overview)

    lines = [
        "# Chapter 1 MIMIC Variable Semantic Audit",
        "",
        "## Purpose",
        "",
        "This report is an aggregated semantic-audit support artifact for Chapter 1 ASIC-to-MIMIC mapping review. It is not preprocessing, cohort construction, 8h block construction, label generation, model fitting, or feature freeze.",
        "",
        "All generated outputs are aggregated descriptive summaries only. No patient-level rows, trajectories, timestamps, or extract files are written.",
        "",
        "## Data Source",
        "",
        f"- Audited MIMIC root: `{config.mimic_root}`",
        f"- ICU stays in local source: {population['total_icu_stays']}",
        f"- Subjects in local source: {population['total_subjects_in_icustays']}",
        "",
        "The same script can be rerun on demo or full MIMIC by changing `mimic_root` in `config/ch1_mimic_variable_audit.yaml` or passing `--mimic-root`.",
        "",
        "## Optional Sources",
        "",
        "Core audit tables are required. Auxiliary tables used only for wrong-context checks are optional; if an optional table is unavailable, its candidate rows are retained with `audit_status=skipped_optional_table_missing` and zero counts.",
        "",
        "- " + "\n- ".join(optional_notes) if optional_notes else "- No optional source tables were skipped in this run.",
        "",
        "## Variables Audited",
        "",
        "- " + "\n- ".join(audited_variables),
        "",
        "`ph_art` was treated as a resolved direct semantic mapping when anchored to `223830 PH (Arterial)`. It is included only as an optional coverage/provenance check, not as an unresolved semantic problem.",
        "",
        "## Dominant Demo Sources",
        "",
        f"- `core_temp`: measurement rows are dominated by {_top_by_role(overview, 'core_temp', {'candidate'})}; `224642 Temperature Site` is support metadata, not a measurement source.",
        f"- `urea`: accepted proxy sources are dominated by {_top_by_role(overview, 'urea', {'primary_proxy', 'secondary_proxy'})}.",
        f"- `sao2`: accepted/conditional measurement sources are dominated by {_top_by_role(overview, 'sao2', {'preferred', 'conditional_secondary'})}.",
        f"- `lactate_art`: accepted/conditional measurement sources are dominated by {_top_by_role(overview, 'lactate_art', {'preferred', 'conditional_mirror'})}.",
        f"- `vt`: row counts are dominated by {_top_candidate(overview, 'vt')}, but the Chapter 1 preferred source remains `224684 Tidal Volume (set)`.",
        f"- `bicarbonate_art`: row counts are dominated by wrong-context serum bicarbonate ({_top_by_role(overview, 'bicarbonate_art', {'wrong_context'})}); the retained blood-gas candidate is {_top_by_role(overview, 'bicarbonate_art', {'preferred_sparse'})}.",
        f"- `etco2`: {_top_candidate(overview, 'etco2') if config.include_etco2 else 'not included'}",
        f"- `ph_art`: {_top_by_role(overview, 'ph_art', {'resolved_preferred'}) if config.include_ph_art else 'not included'} for resolved direct mapping; broader lab pH is coverage/provenance context only.",
        "",
        "Dominance here is based on demo row counts only and should not be treated as a full-data freeze decision.",
        "",
        "## Semantically Unsafe Or Still Cautious",
        "",
        "- " + "\n- ".join(unsafe),
        "",
        "## Likely Full-Data Dependent Decisions",
        "",
        "- " + "\n- ".join(full_data_dependent),
        "",
        "## Output Files",
        "",
        "- `reports/ch1_mimic_variable_audit_overview.csv`",
        "- `reports/ch1_mimic_temperature_audit.csv`",
        "- `reports/ch1_mimic_blood_gas_audit.csv`",
        "- `reports/ch1_mimic_vt_audit.csv`",
        "- `reports/ch1_mimic_bicarbonate_audit.csv`",
        "- `reports/ch1_mimic_urea_audit.csv`",
        "- `reports/ch1_mimic_derived_readiness_audit.csv`",
        "",
    ]
    (config.reports_dir / "ch1_mimic_variable_audit_note.md").write_text(
        "\n".join(lines)
    )


def run_audit(config: AuditConfig) -> pd.DataFrame:
    require_inputs(config)
    mapping_context = load_mapping_context(config.mapping_csv)
    candidates = candidate_plan(
        include_etco2=config.include_etco2,
        include_ph_art=config.include_ph_art,
    )
    overview = audit_candidates(config, candidates)
    overview = enrich_with_mapping_context(overview, mapping_context)
    write_outputs(config, overview)
    return overview


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate aggregated MIMIC semantic-audit summaries for Chapter 1 variables."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the flat audit config file.",
    )
    parser.add_argument(
        "--mimic-root",
        type=Path,
        default=None,
        help="Override MIMIC root path, e.g. /path/to/mimic-iv.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Override output reports directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.mimic_root is not None:
        config = AuditConfig(**{**config.__dict__, "mimic_root": _resolve_path(args.mimic_root)})
    if args.reports_dir is not None:
        config = AuditConfig(**{**config.__dict__, "reports_dir": _resolve_path(args.reports_dir)})
    overview = run_audit(config)
    print(
        f"Wrote aggregated audit outputs to {config.reports_dir} "
        f"for {overview['variable'].nunique()} variables and {len(overview)} candidate rows."
    )


if __name__ == "__main__":
    main()
