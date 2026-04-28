from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "ch1_mimic_blocks.yaml"
DEFAULT_MAPPING_PATH = REPO_ROOT / "config" / "ch1_asic_to_mimic_variable_map.csv"

BLOCK_INDEX_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "block_index",
    "block_start_h",
    "block_end_h",
    "prediction_time_h",
    "completed_block_count",
]


@dataclass(frozen=True)
class MimicBlockConfig:
    mimic_root: Path
    reports_dir: Path
    cohort_path: Path
    processed_output_root: Path
    mapping_path: Path
    block_index_csv: str
    blocked_dynamic_features_csv: str
    stay_block_counts_csv: str
    block_hours: int = 8
    chunksize: int = 200_000

    @property
    def block_index_path(self) -> Path:
        return self.processed_output_root / self.block_index_csv

    @property
    def blocked_dynamic_features_path(self) -> Path:
        return self.processed_output_root / self.blocked_dynamic_features_csv

    @property
    def stay_block_counts_path(self) -> Path:
        return self.processed_output_root / self.stay_block_counts_csv


@dataclass(frozen=True)
class Candidate:
    variable: str
    source_table: str
    itemids: tuple[int, ...]
    role: str = "candidate"


@dataclass(frozen=True)
class SourcePreference:
    variable: str
    preferred_source_table: str
    preferred_itemids: tuple[int, ...]
    mapping_note: str


CHARTEVENT_CANDIDATES: tuple[Candidate, ...] = (
    Candidate("heart_rate", "chartevents", (220045,)),
    Candidate("sbp", "chartevents", (220179, 220050)),
    Candidate("map", "chartevents", (220181, 220052)),
    Candidate("dbp", "chartevents", (220180, 220051)),
    Candidate("resp_rate", "chartevents", (220210, 224690)),
    Candidate("core_temp", "chartevents", (223762, 223761)),
    Candidate("spo2", "chartevents", (220277,)),
    Candidate("sao2", "chartevents", (220227,)),
    Candidate("fio2", "chartevents", (223835,)),
    Candidate("peep", "chartevents", (220339, 224700, 224699)),
    Candidate("vt", "chartevents", (224684,)),
    Candidate("etco2", "chartevents", (228640,)),
    Candidate("pao2", "chartevents", (220224,)),
    Candidate("paco2", "chartevents", (220235,)),
    Candidate("ph_art", "chartevents", (223830,)),
    Candidate("base_excess_art", "chartevents", (224828,)),
    Candidate("hemoglobin", "chartevents", (220228,)),
    Candidate("hematocrit", "chartevents", (220545, 226540)),
    Candidate("wbc", "chartevents", (220546,)),
    Candidate("platelets", "chartevents", (227457,)),
    Candidate("inr", "chartevents", (227467,)),
    Candidate("ptt", "chartevents", (227466,)),
    Candidate("albumin", "chartevents", (227456,)),
    Candidate("crp", "chartevents", (227444,)),
    Candidate("bilirubin_total", "chartevents", (225690,)),
    Candidate("urea", "chartevents", (225624,)),
    Candidate("creatinine", "chartevents", (220615, 229761)),
)

LABEVENT_CANDIDATES: tuple[Candidate, ...] = (
    Candidate("sao2", "labevents", (50817,)),
    Candidate("pao2", "labevents", (50821,)),
    Candidate("paco2", "labevents", (50818,)),
    Candidate("ph_art", "labevents", (50820,)),
    Candidate("base_excess_art", "labevents", (50802,)),
    Candidate("lactate_art", "labevents", (50813, 52442)),
    Candidate("hemoglobin", "labevents", (51222, 51640, 50811)),
    Candidate("hematocrit", "labevents", (51221, 51638, 51639, 52028, 50810)),
    Candidate("wbc", "labevents", (51301,)),
    Candidate("platelets", "labevents", (51265, 51704)),
    Candidate("inr", "labevents", (51237, 51675)),
    Candidate("ptt", "labevents", (51275, 52923)),
    Candidate("albumin", "labevents", (50862, 53085)),
    Candidate("crp", "labevents", (50889, 51652)),
    Candidate("bilirubin_total", "labevents", (50885, 53089)),
    Candidate("urea", "labevents", (51006, 52647)),
    Candidate("creatinine", "labevents", (50912, 52546, 52024)),
)

DERIVED_ONLY_VARIABLES = ("pf_ratio", "vt_per_kg_ibw")
HEIGHT_CM_ITEMID = 226730
HEIGHT_INCH_ITEMID = 226707
HEIGHT_SUPPORT_ITEMIDS = (HEIGHT_CM_ITEMID, HEIGHT_INCH_ITEMID)
ADULT_HEIGHT_CM_MIN = 100.0
ADULT_HEIGHT_CM_MAX = 250.0
DERIVED_VALUE_STATISTICS = ("mean", "median", "min", "max", "last")
SHARED_PRIMARY_VARIABLES = tuple(
    dict.fromkeys(
        [
            *(candidate.variable for candidate in CHARTEVENT_CANDIDATES),
            *(candidate.variable for candidate in LABEVENT_CANDIDATES),
            *DERIVED_ONLY_VARIABLES,
        ]
    )
)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _resolved_no_strict(path: Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        _resolved_no_strict(child).relative_to(_resolved_no_strict(parent))
    except ValueError:
        return False
    return True


def _is_demo_mimic_root(mimic_root: Path) -> bool:
    resolved = _resolved_no_strict(mimic_root)
    return "mimic-iv-demo" in resolved.parts


def enforce_processed_output_storage_policy(config: MimicBlockConfig) -> None:
    if _is_demo_mimic_root(config.mimic_root):
        return

    if _is_relative_to(config.processed_output_root, REPO_ROOT):
        raise ValueError(
            "Unsafe full-MIMIC processed output root inside the project repo: "
            f"{_resolved_no_strict(config.processed_output_root)}. Full-MIMIC row-level "
            "or block-level processed artifacts must be written outside the repo. "
            "Pass --processed-output-root /path/outside/repo, for example under the "
            "private MIMIC root. Aggregated reports may remain under reports/."
        )


def load_config(path: Path) -> MimicBlockConfig:
    raw: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(f"Unsupported config line in {path}: {line!r}")
        key, value = stripped.split(":", 1)
        raw[key.strip()] = value.split("#", 1)[0].strip().strip("'\"")

    required = [
        "mimic_root",
        "reports_dir",
        "block_index_csv",
        "blocked_dynamic_features_csv",
        "stay_block_counts_csv",
    ]
    if "cohort_path" not in raw and ("processed_dir" not in raw or "cohort_csv" not in raw):
        required.extend(["cohort_path"])
    if "processed_output_root" not in raw and "processed_dir" not in raw:
        required.extend(["processed_output_root"])
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"MIMIC block config {path} is missing keys: {missing}")

    cohort_path = (
        _resolve_path(raw["cohort_path"])
        if "cohort_path" in raw
        else _resolve_path(raw["processed_dir"]) / raw["cohort_csv"]
    )
    processed_output_root = _resolve_path(
        raw.get("processed_output_root", raw.get("processed_dir", "mimic-iv-demo/data/processed"))
    )

    return MimicBlockConfig(
        mimic_root=_resolve_path(raw["mimic_root"]),
        reports_dir=_resolve_path(raw["reports_dir"]),
        cohort_path=cohort_path,
        processed_output_root=processed_output_root,
        mapping_path=_resolve_path(raw.get("mapping_path", DEFAULT_MAPPING_PATH)),
        block_index_csv=raw["block_index_csv"],
        blocked_dynamic_features_csv=raw["blocked_dynamic_features_csv"],
        stay_block_counts_csv=raw["stay_block_counts_csv"],
        block_hours=int(raw.get("block_hours", "8")),
        chunksize=int(raw.get("chunksize", "200000")),
    )


def table_path(mimic_root: Path, table: str) -> Path:
    module = {"chartevents": "icu", "labevents": "hosp"}[table]
    return mimic_root / module / f"{table}.csv.gz"


def require_columns(path: Path, required_columns: Iterable[str], *, table_name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required MIMIC table for {table_name}: {path}")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(required_columns) - set(header))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def validate_inputs(config: MimicBlockConfig) -> None:
    if config.block_hours != 8:
        raise ValueError("MIMIC Chapter 1 b2 must use 8-hour blocks.")
    enforce_processed_output_storage_policy(config)
    if not config.cohort_path.exists():
        raise FileNotFoundError(f"Missing retained MIMIC stay-level cohort: {config.cohort_path}")
    if not config.mapping_path.exists():
        raise FileNotFoundError(f"Missing ASIC->MIMIC mapping table: {config.mapping_path}")
    require_columns(
        table_path(config.mimic_root, "chartevents"),
        {"subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"},
        table_name="icu.chartevents",
    )
    require_columns(
        table_path(config.mimic_root, "labevents"),
        {"subject_id", "hadm_id", "charttime", "itemid", "valuenum"},
        table_name="hosp.labevents",
    )


def _candidate_itemids_by_variable_source() -> dict[tuple[str, str], set[int]]:
    lookup: dict[tuple[str, str], set[int]] = {}
    for candidate in (*CHARTEVENT_CANDIDATES, *LABEVENT_CANDIDATES):
        lookup.setdefault((candidate.variable, candidate.source_table), set()).update(
            candidate.itemids
        )
    return lookup


def _extract_preferred_itemids(mapping_logic: object) -> tuple[int, ...]:
    first_clause = str(mapping_logic).split(";", 1)[0]
    return tuple(dict.fromkeys(int(value) for value in re.findall(r"\b\d{5,6}\b", first_clause)))


def build_source_preferences(mapping_path: Path) -> dict[str, SourcePreference]:
    mapping = pd.read_csv(mapping_path)
    required = {
        "asic_base_variable",
        "final_role",
        "freeze_decision",
        "mimic_primary_table",
        "candidate_itemids_or_source_logic",
    }
    missing = sorted(required - set(mapping.columns))
    if missing:
        raise ValueError(f"Mapping table is missing required columns for source preference: {missing}")

    candidate_lookup = _candidate_itemids_by_variable_source()
    preferences: dict[str, SourcePreference] = {}
    unresolved: list[str] = []

    shared_primary = mapping[
        mapping["final_role"].eq("shared_primary")
        & ~mapping["freeze_decision"].eq("derived_only")
    ]
    for row in shared_primary.itertuples(index=False):
        variable = str(row.asic_base_variable)
        if variable not in SHARED_PRIMARY_VARIABLES:
            continue
        preferred_source = str(row.mimic_primary_table).strip()
        if preferred_source not in {"chartevents", "labevents"}:
            unresolved.append(
                f"{variable}: unsupported preferred source {preferred_source!r}"
            )
            continue

        available_itemids = candidate_lookup.get((variable, preferred_source), set())
        preferred_itemids = _extract_preferred_itemids(row.candidate_itemids_or_source_logic)
        preferred_itemids = tuple(itemid for itemid in preferred_itemids if itemid in available_itemids)
        if not preferred_itemids:
            if len(available_itemids) == 1:
                preferred_itemids = tuple(sorted(available_itemids))
            else:
                unresolved.append(
                    f"{variable}: could not resolve preferred itemids from mapping logic "
                    f"for {preferred_source}"
                )
                continue

        preferences[variable] = SourcePreference(
            variable=variable,
            preferred_source_table=preferred_source,
            preferred_itemids=preferred_itemids,
            mapping_note=str(row.candidate_itemids_or_source_logic),
        )

    expected_variables = [
        variable
        for variable in SHARED_PRIMARY_VARIABLES
        if variable not in DERIVED_ONLY_VARIABLES
    ]
    for variable in expected_variables:
        if variable not in preferences:
            unresolved.append(f"{variable}: no source preference resolved")

    if unresolved:
        formatted = "\n".join(f"- {item}" for item in unresolved)
        raise ValueError(
            "Could not resolve preferred MIMIC source for every non-derived "
            f"shared-primary variable from {mapping_path}:\n{formatted}"
        )
    return preferences


def load_retained_cohort(path: Path) -> pd.DataFrame:
    required = {
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime",
        "icu_los_hours",
        "retained_stay_level_cohort",
    }
    cohort = pd.read_csv(path)
    missing = sorted(required - set(cohort.columns))
    if missing:
        raise ValueError(f"Retained cohort table is missing required columns: {missing}")
    cohort = cohort[cohort["retained_stay_level_cohort"].eq(1)].copy()
    cohort["intime"] = pd.to_datetime(cohort["intime"], errors="coerce")
    cohort["outtime"] = pd.to_datetime(cohort["outtime"], errors="coerce")
    cohort["icu_los_hours"] = pd.to_numeric(cohort["icu_los_hours"], errors="coerce")
    if cohort[["subject_id", "hadm_id", "stay_id"]].isna().any(axis=None):
        raise ValueError("Retained cohort contains missing subject_id, hadm_id, or stay_id.")
    if cohort[["intime", "outtime", "icu_los_hours"]].isna().any(axis=None):
        raise ValueError("Retained cohort contains missing intime, outtime, or icu_los_hours.")
    return cohort.reset_index(drop=True)


def build_stay_block_counts(cohort: pd.DataFrame, *, block_hours: int) -> pd.DataFrame:
    keep_columns = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime",
        "icu_los_hours",
        *(["gender"] if "gender" in cohort.columns else []),
    ]
    stays = cohort[keep_columns].copy()
    stays["completed_block_count"] = np.floor(stays["icu_los_hours"] / block_hours).astype("int64")
    stays.loc[stays["completed_block_count"].lt(0), "completed_block_count"] = 0
    stays["has_completed_block"] = stays["completed_block_count"].ge(1)
    stays["ends_exactly_on_8h_boundary"] = np.isclose(
        np.mod(stays["icu_los_hours"].to_numpy(dtype=float), block_hours),
        0.0,
    )
    stays["terminal_block_end_h"] = stays["completed_block_count"] * block_hours
    return stays


def _normalize_gender_for_ibw(values: pd.Series) -> pd.Series:
    first_letter = values.astype("string").str.strip().str.upper().str[0]
    return first_letter.where(first_letter.isin(["M", "F"]))


def load_ibw_support(config: MimicBlockConfig, cohort: pd.DataFrame) -> pd.DataFrame:
    base_columns = ["subject_id", "hadm_id", "stay_id"]
    support = cohort[
        [*base_columns, *(["gender"] if "gender" in cohort.columns else [])]
    ].drop_duplicates().copy()
    if "gender" not in support.columns:
        support["gender"] = pd.NA

    stay_ids = set(pd.to_numeric(support["stay_id"], errors="coerce").dropna().astype("int64"))
    height_frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        table_path(config.mimic_root, "chartevents"),
        usecols=["subject_id", "hadm_id", "stay_id", "itemid", "valuenum"],
        chunksize=config.chunksize,
    ):
        chunk = chunk[
            chunk["itemid"].isin(HEIGHT_SUPPORT_ITEMIDS) & chunk["stay_id"].isin(stay_ids)
        ].copy()
        if chunk.empty:
            continue
        values = pd.to_numeric(chunk["valuenum"], errors="coerce")
        itemids = pd.to_numeric(chunk["itemid"], errors="coerce")
        chunk["height_cm"] = values
        chunk.loc[itemids.eq(HEIGHT_INCH_ITEMID), "height_cm"] = (
            values.loc[itemids.eq(HEIGHT_INCH_ITEMID)] * 2.54
        )
        chunk = chunk[
            chunk["height_cm"].between(ADULT_HEIGHT_CM_MIN, ADULT_HEIGHT_CM_MAX)
        ].copy()
        if not chunk.empty:
            height_frames.append(chunk[[*base_columns, "height_cm"]])

    if height_frames:
        heights = pd.concat(height_frames, ignore_index=True)
        height_by_stay = (
            heights.groupby(base_columns, dropna=False)["height_cm"]
            .median()
            .reset_index()
        )
        support = support.merge(height_by_stay, on=base_columns, how="left")
    else:
        support["height_cm"] = pd.NA

    support["sex_for_ibw"] = _normalize_gender_for_ibw(support["gender"])
    support["height_in"] = pd.to_numeric(support["height_cm"], errors="coerce") / 2.54
    support["ibw_kg"] = pd.NA
    male = support["sex_for_ibw"].eq("M") & support["height_in"].notna()
    female = support["sex_for_ibw"].eq("F") & support["height_in"].notna()
    support.loc[male, "ibw_kg"] = 50.0 + 2.3 * (support.loc[male, "height_in"] - 60.0)
    support.loc[female, "ibw_kg"] = 45.5 + 2.3 * (support.loc[female, "height_in"] - 60.0)
    support["ibw_kg"] = pd.to_numeric(support["ibw_kg"], errors="coerce")
    support.loc[~support["ibw_kg"].gt(0), "ibw_kg"] = pd.NA
    return support[[*base_columns, "gender", "height_cm", "sex_for_ibw", "ibw_kg"]]


def build_block_index(stay_block_counts: pd.DataFrame, *, block_hours: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stay in stay_block_counts.itertuples(index=False):
        for block_index in range(int(stay.completed_block_count)):
            block_start_h = block_index * block_hours
            block_end_h = block_start_h + block_hours
            rows.append(
                {
                    "subject_id": stay.subject_id,
                    "hadm_id": stay.hadm_id,
                    "stay_id": stay.stay_id,
                    "block_index": block_index,
                    "block_start_h": block_start_h,
                    "block_end_h": block_end_h,
                    "prediction_time_h": block_end_h,
                    "completed_block_count": int(stay.completed_block_count),
                }
            )
    return pd.DataFrame(rows, columns=BLOCK_INDEX_COLUMNS)


def _candidate_lookup(candidates: tuple[Candidate, ...]) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for candidate in candidates:
        for itemid in candidate.itemids:
            lookup[itemid] = candidate.variable
    return lookup


def _normalize_values(events: pd.DataFrame) -> pd.Series:
    values = pd.to_numeric(events["valuenum"], errors="coerce")
    itemids = pd.to_numeric(events["itemid"], errors="coerce")
    variables = events["variable"].astype("string")

    fahrenheit_temp = variables.eq("core_temp") & itemids.eq(223761)
    values.loc[fahrenheit_temp] = (values.loc[fahrenheit_temp] - 32.0) * 5.0 / 9.0

    fio2_fraction = variables.eq("fio2") & values.gt(0) & values.le(1)
    values.loc[fio2_fraction] = values.loc[fio2_fraction] * 100.0

    values.loc[variables.eq("hemoglobin")] = values.loc[variables.eq("hemoglobin")] / 1.611
    values.loc[variables.eq("albumin")] = values.loc[variables.eq("albumin")] / 0.01
    values.loc[variables.eq("bilirubin_total")] = (
        values.loc[variables.eq("bilirubin_total")] / 0.0585
    )
    values.loc[variables.eq("urea")] = values.loc[variables.eq("urea")] / 2.8
    values.loc[variables.eq("creatinine")] = values.loc[variables.eq("creatinine")] / 0.0113
    return values


def _empty_source_count_rows() -> list[dict[str, object]]:
    rows = [
        {
            "source_table": source_table,
            "variable": pd.NA,
            "raw_candidate_row_count_loaded": 0,
            "raw_row_count_retained_after_stay_time_filtering": 0,
            "assigned_block_row_count": 0,
            "raw_rows_dropped_negative_time_h": 0,
            "raw_rows_landing_beyond_completed_grid": 0,
            "raw_rows_exactly_on_8h_boundary": 0,
            "total_assigned_observations": 0,
            "status": "scanned",
        }
        for source_table in ("chartevents", "labevents")
    ]
    for variable in DERIVED_ONLY_VARIABLES:
        rows.append(
            {
                "source_table": "derived",
                "variable": variable,
                "raw_candidate_row_count_loaded": 0,
                "raw_row_count_retained_after_stay_time_filtering": 0,
                "assigned_block_row_count": 0,
                "raw_rows_dropped_negative_time_h": 0,
                "raw_rows_landing_beyond_completed_grid": 0,
                "raw_rows_exactly_on_8h_boundary": 0,
                "total_assigned_observations": 0,
                "status": "deferred_derived_only",
            }
        )
    return rows


def _align_chartevents(
    chunk: pd.DataFrame,
    stay_lookup: pd.DataFrame,
    *,
    block_hours: int,
    source_row_start: int,
) -> pd.DataFrame:
    events = chunk.merge(
        stay_lookup[["subject_id", "hadm_id", "stay_id", "intime", "completed_block_count"]],
        on=["subject_id", "hadm_id", "stay_id"],
        how="inner",
    )
    if events.empty:
        return events
    events["charttime"] = pd.to_datetime(events["charttime"], errors="coerce")
    events["time_h"] = (events["charttime"] - events["intime"]).dt.total_seconds() / 3600.0
    events["source_row_order"] = np.arange(source_row_start, source_row_start + len(events))
    events["block_index"] = np.floor(events["time_h"] / block_hours)
    return events


def _align_labevents(
    chunk: pd.DataFrame,
    stay_lookup: pd.DataFrame,
    *,
    block_hours: int,
    source_row_start: int,
) -> pd.DataFrame:
    events = chunk.merge(
        stay_lookup[["subject_id", "hadm_id", "stay_id", "intime", "completed_block_count"]],
        on=["subject_id", "hadm_id"],
        how="inner",
    )
    if events.empty:
        return events
    events["charttime"] = pd.to_datetime(events["charttime"], errors="coerce")
    events["time_h"] = (events["charttime"] - events["intime"]).dt.total_seconds() / 3600.0
    events["source_row_order"] = np.arange(source_row_start, source_row_start + len(events))
    events["block_index"] = np.floor(events["time_h"] / block_hours)
    return events


def load_assigned_events(
    config: MimicBlockConfig,
    stay_block_counts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    source_count_rows = _empty_source_count_rows()
    source_count_index = {row["source_table"]: row for row in source_count_rows if pd.isna(row["variable"])}
    variable_count_rows: dict[tuple[str, str], dict[str, object]] = {}
    dropped = {
        "raw_rows_dropped_negative_time_h": 0,
        "raw_rows_landing_beyond_completed_grid": 0,
        "raw_rows_exactly_on_8h_boundary": 0,
        "raw_rows_skipped_unaligned_timestamp": 0,
    }
    assigned_frames: list[pd.DataFrame] = []

    stay_lookup = stay_block_counts[
        ["subject_id", "hadm_id", "stay_id", "intime", "completed_block_count"]
    ].copy()
    for key in ("subject_id", "hadm_id", "stay_id"):
        stay_lookup[key] = pd.to_numeric(stay_lookup[key], errors="raise")

    plan = {
        "chartevents": (CHARTEVENT_CANDIDATES, _align_chartevents, ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum", "valueuom"]),
        "labevents": (LABEVENT_CANDIDATES, _align_labevents, ["subject_id", "hadm_id", "charttime", "itemid", "valuenum", "valueuom"]),
    }

    source_row_counter = 0
    for source_table, (candidates, aligner, usecols) in plan.items():
        path = table_path(config.mimic_root, source_table)
        itemid_to_variable = _candidate_lookup(candidates)
        itemids = set(itemid_to_variable)
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=config.chunksize):
            chunk = chunk[chunk["itemid"].isin(itemids)].copy()
            if chunk.empty:
                continue
            chunk["variable"] = chunk["itemid"].map(itemid_to_variable)
            source_count_index[source_table]["raw_candidate_row_count_loaded"] += int(len(chunk))

            aligned = aligner(
                chunk,
                stay_lookup,
                block_hours=config.block_hours,
                source_row_start=source_row_counter,
            )
            source_row_counter += len(aligned)
            if aligned.empty:
                continue

            unaligned_timestamp = aligned["charttime"].isna() | aligned["time_h"].isna()
            negative_time = aligned["time_h"].lt(0)
            exact_boundary = (
                aligned["time_h"].notna()
                & aligned["time_h"].ge(0)
                & np.isclose(np.mod(aligned["time_h"].to_numpy(dtype=float), config.block_hours), 0.0)
            )
            beyond_grid = (
                aligned["time_h"].notna()
                & aligned["time_h"].ge(0)
                & aligned["block_index"].ge(aligned["completed_block_count"])
            )
            valid_time = ~(unaligned_timestamp | negative_time | beyond_grid)
            retained = aligned.loc[valid_time].copy()

            source_count_index[source_table][
                "raw_row_count_retained_after_stay_time_filtering"
            ] += int((~(unaligned_timestamp | negative_time)).sum())
            source_count_index[source_table]["assigned_block_row_count"] += int(len(retained))
            source_count_index[source_table]["raw_rows_dropped_negative_time_h"] += int(
                negative_time.sum()
            )
            source_count_index[source_table]["raw_rows_landing_beyond_completed_grid"] += int(
                beyond_grid.sum()
            )
            source_count_index[source_table]["raw_rows_exactly_on_8h_boundary"] += int(
                exact_boundary.sum()
            )
            dropped["raw_rows_skipped_unaligned_timestamp"] += int(unaligned_timestamp.sum())
            dropped["raw_rows_dropped_negative_time_h"] += int(negative_time.sum())
            dropped["raw_rows_landing_beyond_completed_grid"] += int(beyond_grid.sum())
            dropped["raw_rows_exactly_on_8h_boundary"] += int(exact_boundary.sum())

            if retained.empty:
                continue
            retained["block_index"] = retained["block_index"].astype("int64")
            retained["source_table"] = source_table
            retained["value"] = _normalize_values(retained)
            retained = retained[retained["value"].notna()].copy()
            if retained.empty:
                continue
            assigned_frames.append(
                retained[
                    [
                        "subject_id",
                        "hadm_id",
                        "stay_id",
                        "block_index",
                        "charttime",
                        "time_h",
                        "source_row_order",
                        "source_table",
                        "itemid",
                        "variable",
                        "value",
                    ]
                ]
            )

            variable_counts = retained.groupby("variable", dropna=False).size()
            for variable, count in variable_counts.items():
                key = (source_table, str(variable))
                row = variable_count_rows.setdefault(
                    key,
                    {
                        "source_table": source_table,
                        "variable": str(variable),
                        "raw_candidate_row_count_loaded": 0,
                        "raw_row_count_retained_after_stay_time_filtering": 0,
                        "assigned_block_row_count": 0,
                        "raw_rows_dropped_negative_time_h": 0,
                        "raw_rows_landing_beyond_completed_grid": 0,
                        "raw_rows_exactly_on_8h_boundary": 0,
                        "total_assigned_observations": 0,
                        "status": "assigned",
                    },
                )
                row["assigned_block_row_count"] += int(count)
                row["total_assigned_observations"] += int(count)

    if assigned_frames:
        assigned_events = pd.concat(assigned_frames, ignore_index=True)
    else:
        assigned_events = pd.DataFrame(
            columns=[
                "subject_id",
                "hadm_id",
                "stay_id",
                "block_index",
                "charttime",
                "time_h",
                "source_row_order",
                "source_table",
                "itemid",
                "variable",
                "value",
            ]
        )

    source_counts = pd.DataFrame([*source_count_rows, *variable_count_rows.values()])
    return assigned_events, source_counts, dropped


def apply_source_preferences(
    assigned_events: pd.DataFrame,
    source_counts: pd.DataFrame,
    preferences: dict[str, SourcePreference],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = [
        "variable",
        "preferred_source_table",
        "preferred_itemids",
        "secondary_source_table_excluded",
        "rows_retained_for_aggregation",
        "rows_excluded_by_source_preference",
        "note",
    ]
    if assigned_events.empty:
        resolution = pd.DataFrame(columns=columns)
        source_counts = source_counts.copy()
        source_counts["rows_retained_for_aggregation"] = 0
        source_counts["rows_excluded_by_source_preference"] = 0
        return assigned_events, source_counts, resolution

    assigned = assigned_events.copy()
    assigned["retain_for_aggregation"] = False
    rows: list[dict[str, object]] = []

    for variable, preference in preferences.items():
        variable_mask = assigned["variable"].eq(variable)
        variable_rows = assigned.loc[variable_mask]
        if variable_rows.empty:
            rows.append(
                {
                    "variable": variable,
                    "preferred_source_table": preference.preferred_source_table,
                    "preferred_itemids": "|".join(str(itemid) for itemid in preference.preferred_itemids),
                    "secondary_source_table_excluded": pd.NA,
                    "rows_retained_for_aggregation": 0,
                    "rows_excluded_by_source_preference": 0,
                    "note": "No accepted candidate rows were available after stay/time filtering.",
                }
            )
            continue

        preferred_mask = (
            variable_mask
            & assigned["source_table"].eq(preference.preferred_source_table)
            & assigned["itemid"].isin(preference.preferred_itemids)
        )
        preferred_count = int(preferred_mask.sum())
        if preferred_count > 0:
            retain_mask = preferred_mask
            note = "Preferred source/item candidates retained for main b2 aggregation."
        else:
            retain_mask = variable_mask
            note = (
                "Preferred source/item candidates had zero available rows; retained available "
                "secondary candidates as only available accepted source for this run."
            )

        assigned.loc[retain_mask, "retain_for_aggregation"] = True
        excluded_count = int((variable_mask & ~retain_mask).sum())
        retained_count = int(retain_mask.sum())
        secondary_sources = sorted(
            assigned.loc[variable_mask & ~retain_mask, "source_table"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        rows.append(
            {
                "variable": variable,
                "preferred_source_table": preference.preferred_source_table,
                "preferred_itemids": "|".join(str(itemid) for itemid in preference.preferred_itemids),
                "secondary_source_table_excluded": "|".join(secondary_sources) if secondary_sources else pd.NA,
                "rows_retained_for_aggregation": retained_count,
                "rows_excluded_by_source_preference": excluded_count,
                "note": note,
            }
        )

    aggregation_events = assigned[assigned["retain_for_aggregation"]].drop(
        columns=["retain_for_aggregation"]
    )
    resolution = pd.DataFrame(rows, columns=columns)

    source_counts = source_counts.copy()
    retained_counts = (
        aggregation_events.groupby(["source_table", "variable"], dropna=False)
        .size()
        .rename("rows_retained_for_aggregation")
        .reset_index()
    )
    candidate_counts = (
        assigned_events.groupby(["source_table", "variable"], dropna=False)
        .size()
        .rename("candidate_assigned_rows")
        .reset_index()
    )
    source_counts = source_counts.merge(
        retained_counts,
        on=["source_table", "variable"],
        how="left",
    ).merge(
        candidate_counts,
        on=["source_table", "variable"],
        how="left",
    )
    source_counts["rows_retained_for_aggregation"] = (
        pd.to_numeric(source_counts["rows_retained_for_aggregation"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )
    source_counts["candidate_assigned_rows"] = (
        pd.to_numeric(source_counts["candidate_assigned_rows"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )
    variable_rows = source_counts["variable"].notna()
    source_counts["rows_excluded_by_source_preference"] = 0
    source_counts.loc[variable_rows, "rows_excluded_by_source_preference"] = (
        source_counts.loc[variable_rows, "candidate_assigned_rows"]
        - source_counts.loc[variable_rows, "rows_retained_for_aggregation"]
    )
    source_counts.loc[variable_rows, "total_assigned_observations"] = source_counts.loc[
        variable_rows, "rows_retained_for_aggregation"
    ]
    source_counts = source_counts.drop(columns=["candidate_assigned_rows"])
    return aggregation_events, source_counts, resolution


def aggregate_blocks(block_index: pd.DataFrame, assigned_events: pd.DataFrame) -> pd.DataFrame:
    output = block_index.copy()
    for column in ("dynamic_row_count", "non_missing_measurements_in_block", "observed_variables_in_block"):
        output[column] = pd.Series(0, index=output.index, dtype="Int64")
    for variable in SHARED_PRIMARY_VARIABLES:
        for statistic in ("obs_count", "mean", "median", "min", "max", "last"):
            output[f"{variable}_{statistic}"] = (
                pd.Series(0, index=output.index, dtype="Int64")
                if statistic == "obs_count"
                else pd.Series(pd.NA, index=output.index, dtype="Float64")
            )

    if assigned_events.empty:
        return output

    sort_columns = ["subject_id", "hadm_id", "stay_id", "block_index", "time_h", "source_row_order"]
    assigned = assigned_events.sort_values(sort_columns, kind="stable").copy()
    group_columns = ["subject_id", "hadm_id", "stay_id", "block_index"]
    variable_group_columns = [*group_columns, "variable"]

    block_counts = assigned.groupby(group_columns, dropna=False).agg(
        dynamic_row_count=("value", "size"),
        non_missing_measurements_in_block=("value", "count"),
        observed_variables_in_block=("variable", "nunique"),
    )
    variable_stats = assigned.groupby(variable_group_columns, dropna=False)["value"].agg(
        obs_count="count",
        mean="mean",
        median="median",
        min="min",
        max="max",
        last="last",
    )
    wide = variable_stats.unstack("variable")
    wide.columns = [f"{variable}_{statistic}" for statistic, variable in wide.columns]
    aggregated = block_counts.join(wide, how="left").reset_index()
    output = output.drop(
        columns=[
            column
            for column in aggregated.columns
            if column in output.columns and column not in group_columns
        ]
    ).merge(aggregated, on=group_columns, how="left")

    for column in ("dynamic_row_count", "non_missing_measurements_in_block", "observed_variables_in_block"):
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0).astype("Int64")
    for variable in SHARED_PRIMARY_VARIABLES:
        obs_column = f"{variable}_obs_count"
        if obs_column not in output.columns:
            output[obs_column] = pd.Series(0, index=output.index, dtype="Int64")
        else:
            output[obs_column] = pd.to_numeric(output[obs_column], errors="coerce").fillna(0).astype("Int64")
        for statistic in ("mean", "median", "min", "max", "last"):
            column = f"{variable}_{statistic}"
            if column not in output.columns:
                output[column] = pd.Series(pd.NA, index=output.index, dtype="Float64")
            else:
                output[column] = pd.to_numeric(output[column], errors="coerce")

    ordered_columns = [
        *BLOCK_INDEX_COLUMNS,
        "dynamic_row_count",
        "non_missing_measurements_in_block",
        "observed_variables_in_block",
    ]
    for variable in SHARED_PRIMARY_VARIABLES:
        ordered_columns.extend(
            [
                f"{variable}_obs_count",
                f"{variable}_mean",
                f"{variable}_median",
                f"{variable}_min",
                f"{variable}_max",
                f"{variable}_last",
            ]
        )
    return output[ordered_columns]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator_numeric = pd.to_numeric(numerator, errors="coerce")
    denominator_numeric = pd.to_numeric(denominator, errors="coerce")
    result = numerator_numeric / denominator_numeric
    result = result.where(denominator_numeric.gt(0) & numerator_numeric.notna())
    return result.astype("Float64")


def materialize_derived_variables(
    blocked_dynamic_features: pd.DataFrame,
    ibw_support: pd.DataFrame,
) -> pd.DataFrame:
    derived = blocked_dynamic_features.copy()

    for statistic in DERIVED_VALUE_STATISTICS:
        pao2 = derived[f"pao2_{statistic}"]
        fio2_fraction = pd.to_numeric(derived[f"fio2_{statistic}"], errors="coerce") / 100.0
        derived[f"pf_ratio_{statistic}"] = _safe_divide(pao2, fio2_fraction)
    derived["pf_ratio_obs_count"] = (
        pd.to_numeric(derived["pf_ratio_last"], errors="coerce").notna().astype("Int64")
    )

    support_columns = ["subject_id", "hadm_id", "stay_id", "ibw_kg"]
    with_ibw = derived[["subject_id", "hadm_id", "stay_id"]].merge(
        ibw_support[support_columns],
        on=["subject_id", "hadm_id", "stay_id"],
        how="left",
    )
    ibw_kg = pd.to_numeric(with_ibw["ibw_kg"], errors="coerce")
    for statistic in DERIVED_VALUE_STATISTICS:
        derived[f"vt_per_kg_ibw_{statistic}"] = _safe_divide(
            derived[f"vt_{statistic}"],
            ibw_kg,
        )
    derived["vt_per_kg_ibw_obs_count"] = (
        pd.to_numeric(derived["vt_per_kg_ibw_last"], errors="coerce")
        .notna()
        .astype("Int64")
    )
    return derived


def update_source_counts_for_derived(
    source_counts: pd.DataFrame,
    blocked_dynamic_features: pd.DataFrame,
) -> pd.DataFrame:
    updated = source_counts.copy()
    for variable in DERIVED_ONLY_VARIABLES:
        count = int(pd.to_numeric(blocked_dynamic_features[f"{variable}_last"], errors="coerce").notna().sum())
        mask = updated["source_table"].eq("derived") & updated["variable"].eq(variable)
        if not mask.any():
            updated = pd.concat(
                [
                    updated,
                    pd.DataFrame(
                        [
                            {
                                "source_table": "derived",
                                "variable": variable,
                                "raw_candidate_row_count_loaded": 0,
                                "raw_row_count_retained_after_stay_time_filtering": 0,
                                "assigned_block_row_count": 0,
                                "raw_rows_dropped_negative_time_h": 0,
                                "raw_rows_landing_beyond_completed_grid": 0,
                                "raw_rows_exactly_on_8h_boundary": 0,
                                "total_assigned_observations": count,
                                "status": "materialized_block_level_derivation",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            continue
        for column in (
            "total_assigned_observations",
            "rows_retained_for_aggregation",
        ):
            if column in updated.columns:
                updated.loc[mask, column] = count
        if "rows_excluded_by_source_preference" in updated.columns:
            updated.loc[mask, "rows_excluded_by_source_preference"] = 0
        updated.loc[mask, "status"] = "materialized_block_level_derivation"
    return updated


def build_derived_variable_qc_summary(
    blocked_dynamic_features: pd.DataFrame,
    ibw_support: pd.DataFrame,
) -> pd.DataFrame:
    total_blocks = int(len(blocked_dynamic_features))
    block_support = blocked_dynamic_features[["subject_id", "hadm_id", "stay_id"]].merge(
        ibw_support,
        on=["subject_id", "hadm_id", "stay_id"],
        how="left",
    )

    rows = []
    for variable, support_1, support_2, support_3, note in (
        (
            "pf_ratio",
            "pao2_last",
            "fio2_last",
            pd.NA,
            "Block-level PF ratio = pao2_last / (fio2_last / 100). Other PF summary statistics use the same-statistic PaO2 and FiO2 block summaries.",
        ),
        (
            "vt_per_kg_ibw",
            "vt_last",
            "ibw_kg",
            "sex_for_ibw",
            "Block-level VT/IBW = vt_last / Devine IBW kg. Other VT/IBW summary statistics use the same-statistic VT block summary divided by stay-level IBW.",
        ),
    ):
        if variable == "pf_ratio":
            support_1_count = int(
                pd.to_numeric(blocked_dynamic_features[support_1], errors="coerce").notna().sum()
            )
            support_2_count = int(
                pd.to_numeric(blocked_dynamic_features[support_2], errors="coerce").notna().sum()
            )
            support_3_count = pd.NA
        else:
            support_1_count = int(
                pd.to_numeric(blocked_dynamic_features[support_1], errors="coerce").notna().sum()
            )
            support_2_count = int(pd.to_numeric(block_support[support_2], errors="coerce").notna().sum())
            support_3_count = int(block_support[support_3].notna().sum())
        derived_non_missing = int(
            pd.to_numeric(blocked_dynamic_features[f"{variable}_last"], errors="coerce")
            .notna()
            .sum()
        )
        rows.append(
            {
                "variable": variable,
                "support_input_1_name": support_1,
                "support_input_1_non_missing_count": support_1_count,
                "support_input_2_name": support_2,
                "support_input_2_non_missing_count": support_2_count,
                "support_input_3_name": support_3,
                "support_input_3_non_missing_count": support_3_count,
                "derived_non_missing_count": derived_non_missing,
                "derived_missing_count": total_blocks - derived_non_missing,
                "derived_non_missing_fraction": (
                    float(derived_non_missing / total_blocks) if total_blocks else pd.NA
                ),
                "model_ready_non_missing_count": pd.NA,
                "model_ready_non_missing_fraction": pd.NA,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def _summary_value_rows(name: str, values: pd.Series) -> list[dict[str, object]]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return [
            {"metric": f"{name}_p25", "value": pd.NA},
            {"metric": f"{name}_median", "value": pd.NA},
            {"metric": f"{name}_p75", "value": pd.NA},
        ]
    return [
        {"metric": f"{name}_p25", "value": float(numeric.quantile(0.25))},
        {"metric": f"{name}_median", "value": float(numeric.quantile(0.50))},
        {"metric": f"{name}_p75", "value": float(numeric.quantile(0.75))},
    ]


def build_qc_summary(
    stay_block_counts: pd.DataFrame,
    block_index: pd.DataFrame,
    blocked_dynamic_features: pd.DataFrame,
    dropped_counts: dict[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {"metric": "retained_stay_count_entering_block_construction", "value": int(len(stay_block_counts))},
        {"metric": "total_completed_blocks_emitted", "value": int(len(block_index))},
        {"metric": "stays_with_zero_completed_blocks", "value": int(stay_block_counts["completed_block_count"].eq(0).sum())},
        {"metric": "stays_with_at_least_one_completed_block", "value": int(stay_block_counts["completed_block_count"].ge(1).sum())},
        {"metric": "completed_blocks_with_zero_dynamic_rows", "value": int(blocked_dynamic_features["dynamic_row_count"].eq(0).sum())},
        {"metric": "completed_blocks_with_zero_observed_variables", "value": int(blocked_dynamic_features["observed_variables_in_block"].eq(0).sum())},
        {"metric": "raw_rows_dropped_negative_time_h", "value": int(dropped_counts["raw_rows_dropped_negative_time_h"])},
        {"metric": "raw_rows_landing_beyond_completed_grid", "value": int(dropped_counts["raw_rows_landing_beyond_completed_grid"])},
        {"metric": "raw_rows_exactly_on_8h_boundary", "value": int(dropped_counts["raw_rows_exactly_on_8h_boundary"])},
        {"metric": "raw_rows_skipped_unaligned_timestamp", "value": int(dropped_counts["raw_rows_skipped_unaligned_timestamp"])},
    ]
    rows.extend(_summary_value_rows("completed_blocks_per_stay", stay_block_counts["completed_block_count"]))
    return pd.DataFrame(rows)


def build_edge_cases(
    stay_block_counts: pd.DataFrame,
    blocked_dynamic_features: pd.DataFrame,
    dropped_counts: dict[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    zero_block_stays = stay_block_counts[stay_block_counts["completed_block_count"].eq(0)]
    for row in zero_block_stays.head(50).itertuples(index=False):
        rows.append(
            {
                "edge_case_type": "stay_with_zero_completed_blocks",
                "subject_id": row.subject_id,
                "hadm_id": row.hadm_id,
                "stay_id": row.stay_id,
                "block_index": pd.NA,
                "value": int(row.completed_block_count),
                "note": "ICU LOS is shorter than one completed 8h block.",
            }
        )

    if not blocked_dynamic_features.empty:
        empty_by_stay = (
            blocked_dynamic_features[blocked_dynamic_features["dynamic_row_count"].eq(0)]
            .groupby(["subject_id", "hadm_id", "stay_id"], dropna=False)
            .size()
            .rename("empty_completed_block_count")
            .reset_index()
            .sort_values("empty_completed_block_count", ascending=False)
        )
        for row in empty_by_stay.head(50).itertuples(index=False):
            rows.append(
                {
                    "edge_case_type": "stay_with_many_empty_completed_blocks",
                    "subject_id": row.subject_id,
                    "hadm_id": row.hadm_id,
                    "stay_id": row.stay_id,
                    "block_index": pd.NA,
                    "value": int(row.empty_completed_block_count),
                    "note": "Structurally completed blocks retained despite zero assigned dynamic rows.",
                }
            )

    for key, note in (
        ("raw_rows_exactly_on_8h_boundary", "Rows exactly on an 8h boundary are assigned to the following block."),
        ("raw_rows_skipped_unaligned_timestamp", "Rows skipped because charttime could not be aligned."),
        ("raw_rows_dropped_negative_time_h", "Rows before ICU admission were not assigned to blocks."),
        ("raw_rows_landing_beyond_completed_grid", "Rows at or beyond the completed grid were not assigned."),
    ):
        rows.append(
            {
                "edge_case_type": key,
                "subject_id": pd.NA,
                "hadm_id": pd.NA,
                "stay_id": pd.NA,
                "block_index": pd.NA,
                "value": int(dropped_counts.get(key, 0)),
                "note": note,
            }
        )

    return pd.DataFrame(
        rows,
        columns=["edge_case_type", "subject_id", "hadm_id", "stay_id", "block_index", "value", "note"],
    )


def write_note(
    config: MimicBlockConfig,
    qc_summary: pd.DataFrame,
    *,
    unresolved_notes: list[str],
) -> None:
    metrics = qc_summary.set_index("metric")["value"].to_dict()
    note = f"""# Chapter 1 MIMIC Completed 8h Block Construction Note

## Purpose

This report documents subtask 5.1.b2: structural completed-block construction for the retained MIMIC Chapter 1 stay-level cohort. It mirrors the recovered ASIC Chapter 1 block logic and is not valid-instance filtering, carry-forward, horizon-label generation, model-ready construction, or model fitting.

## Data Source

- MIMIC root: `{config.mimic_root}`
- Processed output root: `{config.processed_output_root}`
- Retained stays entering block construction: {int(metrics.get("retained_stay_count_entering_block_construction", 0))}
- Total completed 8h blocks emitted: {int(metrics.get("total_completed_blocks_emitted", 0))}
- Stays with at least one completed block: {int(metrics.get("stays_with_at_least_one_completed_block", 0))}

## Translation Implemented

- Anchor: elapsed time from retained-stay `intime`.
- Width: 8 hours only.
- Interval convention: half-open `[start, end)`, implemented by `block_index = floor(time_h / 8)`.
- Completed blocks: `floor(icu_los_hours / 8)`, using retained-stay `intime`/`outtime` duration from the 5.1.b1 cohort.
- Prediction time: `prediction_time_h = block_end_h`.
- Empty completed blocks are retained with zero counts.
- Current-block sufficiency is not applied in b2.

## Included Raw Sources

- `icu.chartevents` for frozen shared-primary charted variables.
- `hosp.labevents` for frozen shared-primary laboratory and blood-gas variables.

B2 applies preferred source/item filtering before aggregation using `config/ch1_asic_to_mimic_variable_map.csv`. Broader secondary sources remain available in source-resolution QC for later sensitivity or provenance review, but are not pooled into the main block summaries when the preferred source is available.

The stable ordering used for `last` is `subject_id`, `hadm_id`, `stay_id`, `block_index`, `time_h`, then source row order from chunked loading after source preference filtering.

## Derived-Only Shared-Primary Variables

- `pf_ratio` is materialized as `pao2 / (fio2 / 100)` from block-level preferred PaO2 and FiO2 summaries.
- `vt_per_kg_ibw` is materialized as preferred block-level VT divided by Devine IBW from retained-stay gender and height itemids `226730`/`226707`.
- Missing support inputs leave the derived value missing; actual body weight is not substituted.

## Deferred Beyond b2

- current-block core-vital sufficiency filtering
- carry-forward / LOCF
- final missingness handling
- valid prediction-instance filtering
- horizon labels
- model-ready construction
- model fitting
- secondary-source sensitivity choices

## QC Highlights

- Completed blocks with zero dynamic rows: {int(metrics.get("completed_blocks_with_zero_dynamic_rows", 0))}
- Completed blocks with zero observed variables: {int(metrics.get("completed_blocks_with_zero_observed_variables", 0))}
- Raw rows dropped for negative `time_h`: {int(metrics.get("raw_rows_dropped_negative_time_h", 0))}
- Raw rows landing beyond the completed block grid: {int(metrics.get("raw_rows_landing_beyond_completed_grid", 0))}
- Raw rows exactly on an 8h boundary: {int(metrics.get("raw_rows_exactly_on_8h_boundary", 0))}

## Translation Limitations

"""
    if unresolved_notes:
        note += "\n".join(f"- {item}" for item in unresolved_notes) + "\n"
    else:
        note += "- No unresolved source-integration issues were detected for raw chart/lab event block assignment.\n"
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    (config.reports_dir / "ch1_mimic_block_note.md").write_text(note)


def run_mimic_blocks(config: MimicBlockConfig) -> dict[str, Path]:
    validate_inputs(config)
    config.processed_output_root.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    cohort = load_retained_cohort(config.cohort_path)
    stay_block_counts = build_stay_block_counts(cohort, block_hours=config.block_hours)
    block_index = build_block_index(stay_block_counts, block_hours=config.block_hours)
    source_preferences = build_source_preferences(config.mapping_path)
    assigned_events, source_counts, dropped_counts = load_assigned_events(config, stay_block_counts)
    ibw_support = load_ibw_support(config, cohort)
    aggregation_events, source_counts, source_resolution = apply_source_preferences(
        assigned_events,
        source_counts,
        source_preferences,
    )
    blocked_dynamic_features = aggregate_blocks(block_index, aggregation_events)
    blocked_dynamic_features = materialize_derived_variables(
        blocked_dynamic_features,
        ibw_support,
    )
    source_counts = update_source_counts_for_derived(source_counts, blocked_dynamic_features)
    derived_qc_summary = build_derived_variable_qc_summary(blocked_dynamic_features, ibw_support)
    qc_summary = build_qc_summary(
        stay_block_counts,
        block_index,
        blocked_dynamic_features,
        dropped_counts,
    )
    edge_cases = build_edge_cases(stay_block_counts, blocked_dynamic_features, dropped_counts)
    unresolved_notes = [
        "`pf_ratio` and `vt_per_kg_ibw` are materialized as block-level derived variables from frozen preferred source summaries; timestamp-level paired derivation and sensitivity variants remain out of scope.",
        "Preferred source/item filtering is applied for main b2 aggregation; secondary sources remain documented in source-resolution QC for later sensitivity or provenance review.",
    ]

    stay_block_counts.to_csv(config.stay_block_counts_path, index=False)
    block_index.to_csv(config.block_index_path, index=False)
    blocked_dynamic_features.to_csv(config.blocked_dynamic_features_path, index=False)
    qc_summary.to_csv(config.reports_dir / "ch1_mimic_block_qc_summary.csv", index=False)
    source_counts.to_csv(config.reports_dir / "ch1_mimic_block_source_counts.csv", index=False)
    source_resolution.to_csv(
        config.reports_dir / "ch1_mimic_block_source_resolution_summary.csv",
        index=False,
    )
    derived_qc_summary.to_csv(
        config.reports_dir / "ch1_mimic_derived_variable_qc_summary.csv",
        index=False,
    )
    edge_cases.to_csv(config.reports_dir / "ch1_mimic_block_edge_cases.csv", index=False)
    write_note(config, qc_summary, unresolved_notes=unresolved_notes)

    return {
        "stay_block_counts": config.stay_block_counts_path,
        "block_index": config.block_index_path,
        "blocked_dynamic_features": config.blocked_dynamic_features_path,
        "qc_summary": config.reports_dir / "ch1_mimic_block_qc_summary.csv",
        "source_counts": config.reports_dir / "ch1_mimic_block_source_counts.csv",
        "source_resolution": config.reports_dir / "ch1_mimic_block_source_resolution_summary.csv",
        "derived_variable_qc": config.reports_dir / "ch1_mimic_derived_variable_qc_summary.csv",
        "edge_cases": config.reports_dir / "ch1_mimic_block_edge_cases.csv",
        "note": config.reports_dir / "ch1_mimic_block_note.md",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mimic-root", type=Path, default=None)
    parser.add_argument("--cohort-path", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--processed-output-root", type=Path, default=None)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Deprecated alias for --processed-output-root.",
    )
    parser.add_argument("--chunksize", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    if args.mimic_root is not None:
        config = replace(config, mimic_root=_resolve_path(args.mimic_root))
    if args.reports_dir is not None:
        config = replace(config, reports_dir=_resolve_path(args.reports_dir))
    processed_output_root = args.processed_output_root or args.processed_dir
    if processed_output_root is not None:
        config = replace(config, processed_output_root=_resolve_path(processed_output_root))
    if args.cohort_path is not None:
        config = replace(config, cohort_path=_resolve_path(args.cohort_path))
    if args.chunksize is not None:
        config = replace(config, chunksize=int(args.chunksize))

    paths = run_mimic_blocks(config)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
