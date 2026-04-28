from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from chapter1_mortality_decomposition.artifacts import Chapter1InputTables
from chapter1_mortality_decomposition.config import default_chapter1_config, updated_chapter1_config
from chapter1_mortality_decomposition.mimic_blocks import (
    DEFAULT_MAPPING_PATH,
    REPO_ROOT,
    _is_demo_mimic_root,
    _is_relative_to,
    _resolve_path,
    apply_source_preferences,
    build_source_preferences,
    load_assigned_events,
    table_path,
)
from chapter1_mortality_decomposition.pipeline import (
    build_chapter1_dataset,
    write_chapter1_dataset,
)
from chapter1_mortality_decomposition.utils import ensure_directory


DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "ch1_mimic_preprocessing_adapter.yaml"
HOSPITAL_ID = "MIMIC-IV"
INVASIVE_VENT_ITEMID = 225792


@dataclass(frozen=True)
class MimicPreprocessingAdapterConfig:
    mimic_root: Path
    reports_dir: Path
    cohort_path: Path
    mimic_block_root: Path
    adapter_output_root: Path
    preprocessing_output_root: Path
    mapping_path: Path
    feature_set_config_path: Path
    block_index_csv: str
    blocked_dynamic_features_csv: str
    stay_block_counts_csv: str
    run_preprocessing_core: bool = True
    output_format: str = "csv"
    chunksize: int = 200_000

    @property
    def mimic_block_index_path(self) -> Path:
        return self.mimic_block_root / self.block_index_csv

    @property
    def mimic_blocked_dynamic_features_path(self) -> Path:
        return self.mimic_block_root / self.blocked_dynamic_features_csv

    @property
    def mimic_stay_block_counts_path(self) -> Path:
        return self.mimic_block_root / self.stay_block_counts_csv


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def load_config(path: Path) -> MimicPreprocessingAdapterConfig:
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
        "cohort_path",
        "mimic_block_root",
        "adapter_output_root",
        "preprocessing_output_root",
        "block_index_csv",
        "blocked_dynamic_features_csv",
        "stay_block_counts_csv",
    ]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"MIMIC preprocessing adapter config {path} is missing keys: {missing}")

    return MimicPreprocessingAdapterConfig(
        mimic_root=_resolve_path(raw["mimic_root"]),
        reports_dir=_resolve_path(raw["reports_dir"]),
        cohort_path=_resolve_path(raw["cohort_path"]),
        mimic_block_root=_resolve_path(raw["mimic_block_root"]),
        adapter_output_root=_resolve_path(raw["adapter_output_root"]),
        preprocessing_output_root=_resolve_path(raw["preprocessing_output_root"]),
        mapping_path=_resolve_path(raw.get("mapping_path", DEFAULT_MAPPING_PATH)),
        feature_set_config_path=_resolve_path(
            raw.get("feature_set_config_path", "config/ch1_feature_sets.json")
        ),
        block_index_csv=raw["block_index_csv"],
        blocked_dynamic_features_csv=raw["blocked_dynamic_features_csv"],
        stay_block_counts_csv=raw["stay_block_counts_csv"],
        run_preprocessing_core=_parse_bool(raw.get("run_preprocessing_core", "true")),
        output_format=raw.get("output_format", "csv"),
        chunksize=int(raw.get("chunksize", "200000")),
    )


def enforce_storage_policy(config: MimicPreprocessingAdapterConfig) -> None:
    if _is_demo_mimic_root(config.mimic_root):
        return
    unsafe_outputs = [
        ("adapter_output_root", config.adapter_output_root),
        ("preprocessing_output_root", config.preprocessing_output_root),
    ]
    unsafe_inputs = [
        ("cohort_path", config.cohort_path),
        ("mimic_block_root", config.mimic_block_root),
    ]
    violations = [
        f"{label}={path}"
        for label, path in (*unsafe_outputs, *unsafe_inputs)
        if _is_relative_to(path, REPO_ROOT)
    ]
    if violations:
        raise ValueError(
            "Unsafe full-MIMIC adapter path inside the project repo. Full-MIMIC row-level "
            "inputs and outputs for b3 must live outside the repo. Violations: "
            + "; ".join(violations)
        )


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _add_contract_ids(df: pd.DataFrame) -> pd.DataFrame:
    adapted = df.copy()
    if "stay_id" not in adapted.columns:
        raise ValueError("MIMIC table is missing stay_id for contract ID adaptation.")
    adapted["stay_id_global"] = adapted["stay_id"].astype("string")
    adapted["hospital_id"] = HOSPITAL_ID
    return adapted


def load_retained_cohort(path: Path) -> pd.DataFrame:
    cohort = _read_csv(path)
    required = {
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime",
        "icu_los_hours",
        "icu_mortality",
    }
    missing = sorted(required - set(cohort.columns))
    if missing:
        raise ValueError(f"Retained MIMIC cohort is missing required columns: {missing}")
    if "retained_stay_level_cohort" in cohort.columns:
        cohort = cohort[cohort["retained_stay_level_cohort"].eq(1)].copy()
    cohort = _add_contract_ids(cohort)
    cohort["intime"] = pd.to_datetime(cohort["intime"], errors="coerce")
    cohort["outtime"] = pd.to_datetime(cohort["outtime"], errors="coerce")
    cohort["icu_los_hours"] = pd.to_numeric(cohort["icu_los_hours"], errors="coerce")
    if cohort[["intime", "outtime", "icu_los_hours"]].isna().any(axis=None):
        raise ValueError("Retained MIMIC cohort contains missing ICU timing fields.")
    return cohort.reset_index(drop=True)


def load_icd_codes(mimic_root: Path, cohort: pd.DataFrame, *, chunksize: int) -> tuple[pd.Series, str]:
    path = mimic_root / "hosp" / "diagnoses_icd.csv.gz"
    if not path.exists():
        return pd.Series("MISSING_ICD_CODES", index=cohort.index, dtype="string"), "placeholder_missing_diagnoses_icd"

    hadm_ids = set(pd.to_numeric(cohort["hadm_id"], errors="coerce").dropna().astype("int64"))
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=["hadm_id", "seq_num", "icd_code", "icd_version"],
        chunksize=chunksize,
    ):
        chunk = chunk[chunk["hadm_id"].isin(hadm_ids)].copy()
        if not chunk.empty:
            frames.append(chunk)
    if not frames:
        return pd.Series("", index=cohort.index, dtype="string"), "diagnoses_icd_available_no_matches"

    diagnoses = pd.concat(frames, ignore_index=True)
    diagnoses["seq_num"] = pd.to_numeric(diagnoses["seq_num"], errors="coerce")
    diagnoses = diagnoses.sort_values(["hadm_id", "seq_num"], kind="stable")
    diagnoses["code_token"] = (
        diagnoses["icd_version"].astype("string").fillna("")
        + ":"
        + diagnoses["icd_code"].astype("string").fillna("")
    )
    grouped = diagnoses.groupby("hadm_id", dropna=False)["code_token"].agg(
        lambda values: ";".join(value for value in values.astype(str) if value and value != ":")
    )
    mapped = cohort["hadm_id"].map(grouped).fillna("").astype("string")
    return mapped, "diagnoses_icd_joined"


def build_static_harmonized(
    cohort: pd.DataFrame,
    config: MimicPreprocessingAdapterConfig,
) -> tuple[pd.DataFrame, str]:
    icd_codes, icd_status = load_icd_codes(config.mimic_root, cohort, chunksize=config.chunksize)
    static = cohort[["stay_id_global", "hospital_id"]].copy()
    static["icu_readmit"] = 0
    static["icu_mortality"] = pd.to_numeric(cohort["icu_mortality"], errors="coerce").astype("Int64")
    static["icd10_codes"] = icd_codes
    return static, icd_status


def build_stay_block_counts(cohort: pd.DataFrame, mimic_stay_block_counts: pd.DataFrame) -> pd.DataFrame:
    adapted_counts = _add_contract_ids(mimic_stay_block_counts)
    keep_extra = [
        column
        for column in [
            "completed_block_count",
            "has_completed_block",
            "ends_exactly_on_8h_boundary",
            "terminal_block_end_h",
        ]
        if column in adapted_counts.columns
    ]
    counts = adapted_counts[
        [
            "stay_id_global",
            "hospital_id",
            "intime",
            "outtime",
            "icu_los_hours",
            *keep_extra,
        ]
    ].copy()
    counts = counts.rename(
        columns={
            "intime": "icu_admission_time",
            "outtime": "icu_end_time_proxy",
            "icu_los_hours": "icu_end_time_proxy_hours",
        }
    )
    return counts


def build_block_index(mimic_block_index: pd.DataFrame) -> pd.DataFrame:
    adapted = _add_contract_ids(mimic_block_index)
    return adapted[
        [
            "stay_id_global",
            "hospital_id",
            "block_index",
            "block_start_h",
            "block_end_h",
            "prediction_time_h",
        ]
    ].copy()


def build_blocked_dynamic_features(mimic_blocked: pd.DataFrame) -> pd.DataFrame:
    adapted = _add_contract_ids(mimic_blocked)
    id_columns = [
        "stay_id_global",
        "hospital_id",
        "block_index",
        "block_start_h",
        "block_end_h",
        "prediction_time_h",
    ]
    feature_columns = [
        column
        for column in adapted.columns
        if column not in {"subject_id", "hadm_id", "stay_id", "completed_block_count"}
        and column not in id_columns
    ]
    return adapted[[*id_columns, *feature_columns]].copy()


def build_mech_vent_stay_level_qc(cohort: pd.DataFrame) -> pd.DataFrame:
    qc = cohort[["stay_id_global", "hospital_id"]].copy()
    if "passed_vent_los_qc" in cohort.columns:
        qc["mech_vent_ge_24h_qc"] = cohort["passed_vent_los_qc"].astype(bool)
    elif "passed_invasive_vent_ge_24h" in cohort.columns:
        qc["mech_vent_ge_24h_qc"] = cohort["passed_invasive_vent_ge_24h"].astype(bool)
    else:
        qc["mech_vent_ge_24h_qc"] = True
    return qc


def _format_timedelta_hours(hours: pd.Series) -> pd.Series:
    return pd.to_timedelta(hours, unit="h").astype("string")


def build_mech_vent_episode_level(
    config: MimicPreprocessingAdapterConfig,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    path = config.mimic_root / "icu" / "procedureevents.csv.gz"
    stay_lookup = cohort[["stay_id", "stay_id_global", "hospital_id", "intime"]].copy()
    stay_ids = set(pd.to_numeric(stay_lookup["stay_id"], errors="coerce").dropna().astype("int64"))
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=["stay_id", "starttime", "endtime", "itemid"],
        chunksize=config.chunksize,
    ):
        chunk = chunk[
            chunk["itemid"].eq(INVASIVE_VENT_ITEMID) & chunk["stay_id"].isin(stay_ids)
        ].copy()
        if chunk.empty:
            continue
        frames.append(chunk)
    if not frames:
        return pd.DataFrame(
            columns=["stay_id_global", "hospital_id", "episode_start_time", "episode_end_time"]
        )

    episodes = pd.concat(frames, ignore_index=True)
    episodes = episodes.merge(stay_lookup, on="stay_id", how="inner")
    episodes["starttime"] = pd.to_datetime(episodes["starttime"], errors="coerce")
    episodes["endtime"] = pd.to_datetime(episodes["endtime"], errors="coerce")
    episodes = episodes.dropna(subset=["starttime", "endtime", "intime"]).copy()
    episodes["episode_start_h"] = (episodes["starttime"] - episodes["intime"]).dt.total_seconds() / 3600.0
    episodes["episode_end_h"] = (episodes["endtime"] - episodes["intime"]).dt.total_seconds() / 3600.0
    episodes = episodes[episodes["episode_end_h"].gt(episodes["episode_start_h"])].copy()
    episodes["episode_start_time"] = _format_timedelta_hours(episodes["episode_start_h"])
    episodes["episode_end_time"] = _format_timedelta_hours(episodes["episode_end_h"])
    return episodes[
        ["stay_id_global", "hospital_id", "episode_start_time", "episode_end_time"]
    ].reset_index(drop=True)


@dataclass(frozen=True)
class _EventLoadConfig:
    mimic_root: Path
    mapping_path: Path
    block_hours: int
    chunksize: int


def build_dynamic_harmonized(
    config: MimicPreprocessingAdapterConfig,
    mimic_stay_block_counts: pd.DataFrame,
) -> pd.DataFrame:
    stay_counts = mimic_stay_block_counts.copy()
    if "intime" in stay_counts.columns:
        stay_counts["intime"] = pd.to_datetime(stay_counts["intime"], errors="coerce")
    if "outtime" in stay_counts.columns:
        stay_counts["outtime"] = pd.to_datetime(stay_counts["outtime"], errors="coerce")
    event_config = _EventLoadConfig(
        mimic_root=config.mimic_root,
        mapping_path=config.mapping_path,
        block_hours=8,
        chunksize=config.chunksize,
    )
    assigned_events, source_counts, _ = load_assigned_events(event_config, stay_counts)
    preferences = build_source_preferences(config.mapping_path)
    aggregation_events, _, _ = apply_source_preferences(
        assigned_events,
        source_counts,
        preferences,
    )
    if aggregation_events.empty:
        return pd.DataFrame(columns=["stay_id_global", "hospital_id", "minutes_since_admit"])

    events = _add_contract_ids(aggregation_events)
    events["minutes_since_admit"] = events["time_h"] * 60.0
    events = events.sort_values(
        ["stay_id_global", "minutes_since_admit", "source_row_order"],
        kind="stable",
    ).reset_index(drop=True)
    row_index = pd.RangeIndex(len(events), name="adapter_event_row")
    dynamic = events[["stay_id_global", "hospital_id", "minutes_since_admit"]].copy()
    for variable, variable_values in events.groupby("variable", sort=False)["value"]:
        dynamic[str(variable)] = pd.NA
        dynamic.loc[variable_values.index, str(variable)] = variable_values
    dynamic.index = row_index
    return dynamic.reset_index(drop=True)


def write_standardized_inputs(
    config: MimicPreprocessingAdapterConfig,
    inputs: Chapter1InputTables,
) -> dict[str, Path]:
    root = config.adapter_output_root
    paths = {
        "static_harmonized": root / "static" / "harmonized.csv",
        "dynamic_harmonized": root / "dynamic" / "harmonized.csv",
        "block_index": root / "blocked" / "asic_8h_block_index.csv",
        "blocked_dynamic_features": root / "blocked" / "asic_8h_blocked_dynamic_features.csv",
        "stay_block_counts": root / "blocked" / "asic_8h_stay_block_counts.csv",
        "mech_vent_stay_level_qc": root / "qc" / "mech_vent_ge_24h_stay_level.csv",
        "mech_vent_episode_level": root / "qc" / "mech_vent_ge_24h_episode_level.csv",
    }
    table_lookup = {
        "static_harmonized": inputs.static_harmonized,
        "dynamic_harmonized": inputs.dynamic_harmonized,
        "block_index": inputs.block_index,
        "blocked_dynamic_features": inputs.blocked_dynamic_features,
        "stay_block_counts": inputs.stay_block_counts,
        "mech_vent_stay_level_qc": inputs.mech_vent_stay_level_qc,
        "mech_vent_episode_level": inputs.mech_vent_episode_level,
    }
    for name, path in paths.items():
        ensure_directory(path.parent)
        table_lookup[name].to_csv(path, index=False)
    return paths


def build_contract_check(inputs: Chapter1InputTables) -> pd.DataFrame:
    checks = {
        "static_harmonized": (inputs.static_harmonized, {"stay_id_global", "hospital_id", "icu_readmit", "icu_mortality", "icd10_codes"}),
        "dynamic_harmonized": (inputs.dynamic_harmonized, {"stay_id_global", "hospital_id", "minutes_since_admit"}),
        "block_index": (inputs.block_index, {"stay_id_global", "hospital_id", "block_index", "block_start_h", "block_end_h", "prediction_time_h"}),
        "blocked_dynamic_features": (inputs.blocked_dynamic_features, {"stay_id_global", "hospital_id", "block_index", "block_start_h", "block_end_h", "prediction_time_h"}),
        "stay_block_counts": (inputs.stay_block_counts, {"stay_id_global", "hospital_id", "icu_admission_time", "icu_end_time_proxy", "icu_end_time_proxy_hours"}),
        "mech_vent_stay_level_qc": (inputs.mech_vent_stay_level_qc, {"stay_id_global", "hospital_id", "mech_vent_ge_24h_qc"}),
        "mech_vent_episode_level": (inputs.mech_vent_episode_level, {"stay_id_global", "hospital_id", "episode_start_time", "episode_end_time"}),
    }
    rows = []
    for component, (df, required) in checks.items():
        missing = sorted(required - set(df.columns))
        rows.append(
            {
                "component": component,
                "created": True,
                "row_count": int(df.shape[0]),
                "required_columns_present": not missing,
                "status": "pass" if not missing else "fail",
                "missing_columns": "|".join(missing),
                "notes": "Required contract columns present." if not missing else "Missing required columns.",
            }
        )
    return pd.DataFrame(rows)


def build_qc_summary(
    inputs: Chapter1InputTables,
    *,
    preprocessing_ran_successfully: bool,
    preprocessing_error: str | None,
) -> pd.DataFrame:
    rows = [
        {"metric": "retained_stay_count_in_adapter_outputs", "value": int(inputs.static_harmonized["stay_id_global"].nunique())},
        {"metric": "adapted_block_index_rows", "value": int(inputs.block_index.shape[0])},
        {"metric": "adapted_blocked_dynamic_features_rows", "value": int(inputs.blocked_dynamic_features.shape[0])},
        {"metric": "adapted_dynamic_harmonized_rows", "value": int(inputs.dynamic_harmonized.shape[0])},
        {"metric": "mech_vent_stay_level_qc_stays", "value": int(inputs.mech_vent_stay_level_qc["stay_id_global"].nunique())},
        {"metric": "mech_vent_episode_level_rows", "value": int(inputs.mech_vent_episode_level.shape[0])},
        {"metric": "reused_asic_preprocessing_entrypoint_ran_successfully", "value": bool(preprocessing_ran_successfully)},
    ]
    if preprocessing_error:
        rows.append({"metric": "preprocessing_error", "value": preprocessing_error})
    return pd.DataFrame(rows)


def update_derived_variable_qc_with_model_ready(
    config: MimicPreprocessingAdapterConfig,
) -> None:
    qc_path = config.reports_dir / "ch1_mimic_derived_variable_qc_summary.csv"
    model_ready_path = (
        config.preprocessing_output_root
        / "model_ready"
        / f"chapter1_primary_model_ready_dataset.{config.output_format}"
    )
    if not qc_path.exists() or not model_ready_path.exists():
        return

    if config.output_format == "csv":
        model_ready = pd.read_csv(
            model_ready_path,
            usecols=lambda column: column in {"pf_ratio_last", "vt_per_kg_ibw_last"},
        )
    else:
        model_ready = pd.read_parquet(
            model_ready_path,
            columns=["pf_ratio_last", "vt_per_kg_ibw_last"],
        )

    total_rows = int(model_ready.shape[0])
    counts = {
        "pf_ratio": int(model_ready["pf_ratio_last"].notna().sum())
        if "pf_ratio_last" in model_ready.columns
        else pd.NA,
        "vt_per_kg_ibw": int(model_ready["vt_per_kg_ibw_last"].notna().sum())
        if "vt_per_kg_ibw_last" in model_ready.columns
        else pd.NA,
    }
    fractions = {
        variable: (count / total_rows if total_rows and not pd.isna(count) else pd.NA)
        for variable, count in counts.items()
    }

    qc = pd.read_csv(qc_path)
    qc["model_ready_non_missing_count"] = qc["variable"].map(counts).astype("Int64")
    qc["model_ready_non_missing_fraction"] = qc["variable"].map(fractions)
    qc.to_csv(qc_path, index=False)


def write_adapter_note(
    config: MimicPreprocessingAdapterConfig,
    *,
    preprocessing_ran_successfully: bool,
    preprocessing_error: str | None,
    icd_status: str,
) -> None:
    status = "successfully" if preprocessing_ran_successfully else "not successfully"
    lines = [
        "# Chapter 1 MIMIC Preprocessing Adapter Note",
        "",
        "## Purpose",
        "",
        "This report documents subtask 5.1.b3: a thin MIMIC-to-ASIC input-contract adapter for reusing the existing frozen ASIC Chapter 1 preprocessing core.",
        "",
        "## Storage",
        "",
        f"- MIMIC root: `{config.mimic_root}`",
        f"- Adapter standardized input root: `{config.adapter_output_root}`",
        f"- Reused preprocessing output root: `{config.preprocessing_output_root}`",
        "- Full-MIMIC row-level adapter and preprocessing outputs must remain outside the repo.",
        "",
        "## Adapter Decisions",
        "",
        f"- `stay_id_global = string(stay_id)`",
        f"- `hospital_id = {HOSPITAL_ID}`",
        "- `icu_readmit = 0` for the retained first-stay MIMIC cohort.",
        f"- `icd10_codes` status: `{icd_status}`",
        "- `dynamic_harmonized` is built from preferred source/item MIMIC event rows, with `minutes_since_admit = time_h * 60`.",
        "- `mech_vent_episode_level` uses invasive ventilation procedureevents itemid `225792` only, expressed as timedeltas since ICU admission.",
        "",
        "## Reuse Result",
        "",
        f"The existing ASIC preprocessing core was {status} run on the adapted MIMIC inputs.",
    ]
    if preprocessing_error:
        lines.extend(["", "Error:", "", f"`{preprocessing_error}`"])
    lines.extend(
        [
            "",
            "## Deferred Beyond b3",
            "",
            "- model fitting",
            "- any redesign of LOCF, valid-instance, observation-process, or model-ready logic",
            "- review of proxy horizon label semantics for final external-validation reporting",
        ]
    )
    (config.reports_dir / "ch1_mimic_preprocessing_adapter_note.md").write_text("\n".join(lines))


def run_adapter(config: MimicPreprocessingAdapterConfig) -> tuple[Chapter1InputTables, bool, str | None]:
    validate_inputs(config)
    ensure_directory(config.reports_dir)
    ensure_directory(config.adapter_output_root)
    ensure_directory(config.preprocessing_output_root)

    cohort = load_retained_cohort(config.cohort_path)
    mimic_stay_block_counts = _read_csv(config.mimic_stay_block_counts_path)
    static_harmonized, icd_status = build_static_harmonized(cohort, config)
    inputs = Chapter1InputTables(
        static_harmonized=static_harmonized,
        dynamic_harmonized=build_dynamic_harmonized(config, mimic_stay_block_counts),
        block_index=build_block_index(_read_csv(config.mimic_block_index_path)),
        blocked_dynamic_features=build_blocked_dynamic_features(
            _read_csv(config.mimic_blocked_dynamic_features_path)
        ),
        stay_block_counts=build_stay_block_counts(cohort, mimic_stay_block_counts),
        mech_vent_stay_level_qc=build_mech_vent_stay_level_qc(cohort),
        mech_vent_episode_level=build_mech_vent_episode_level(config, cohort),
    )
    write_standardized_inputs(config, inputs)

    preprocessing_ran_successfully = False
    preprocessing_error: str | None = None
    if config.run_preprocessing_core:
        try:
            chapter1_config = updated_chapter1_config(
                default_chapter1_config(),
                feature_set_config_path=config.feature_set_config_path,
            )
            dataset = build_chapter1_dataset(inputs, config=chapter1_config)
            write_chapter1_dataset(
                dataset,
                output_dir=config.preprocessing_output_root,
                output_format=config.output_format,
            )
            preprocessing_ran_successfully = True
            update_derived_variable_qc_with_model_ready(config)
        except Exception as exc:  # pragma: no cover - surfaced in report
            preprocessing_error = f"{type(exc).__name__}: {exc}"

    contract_check = build_contract_check(inputs)
    qc_summary = build_qc_summary(
        inputs,
        preprocessing_ran_successfully=preprocessing_ran_successfully,
        preprocessing_error=preprocessing_error,
    )
    contract_check.to_csv(config.reports_dir / "ch1_mimic_adapter_contract_check.csv", index=False)
    qc_summary.to_csv(config.reports_dir / "ch1_mimic_adapter_qc_summary.csv", index=False)
    write_adapter_note(
        config,
        preprocessing_ran_successfully=preprocessing_ran_successfully,
        preprocessing_error=preprocessing_error,
        icd_status=icd_status,
    )
    return inputs, preprocessing_ran_successfully, preprocessing_error


def validate_inputs(config: MimicPreprocessingAdapterConfig) -> None:
    enforce_storage_policy(config)
    required_paths = [
        config.cohort_path,
        config.mimic_block_index_path,
        config.mimic_blocked_dynamic_features_path,
        config.mimic_stay_block_counts_path,
        config.mapping_path,
        table_path(config.mimic_root, "chartevents"),
        table_path(config.mimic_root, "labevents"),
        config.mimic_root / "icu" / "procedureevents.csv.gz",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required MIMIC adapter inputs: " + ", ".join(missing))
    if config.output_format not in {"csv", "parquet"}:
        raise ValueError("output_format must be csv or parquet.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mimic-root", type=Path, default=None)
    parser.add_argument("--cohort-path", type=Path, default=None)
    parser.add_argument("--mimic-block-root", type=Path, default=None)
    parser.add_argument("--adapter-output-root", type=Path, default=None)
    parser.add_argument("--preprocessing-output-root", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--chunksize", type=int, default=None)
    parser.add_argument("--skip-preprocessing-core", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    replacements = {}
    for field_name, arg_value in (
        ("mimic_root", args.mimic_root),
        ("cohort_path", args.cohort_path),
        ("mimic_block_root", args.mimic_block_root),
        ("adapter_output_root", args.adapter_output_root),
        ("preprocessing_output_root", args.preprocessing_output_root),
        ("reports_dir", args.reports_dir),
    ):
        if arg_value is not None:
            replacements[field_name] = _resolve_path(arg_value)
    if args.chunksize is not None:
        replacements["chunksize"] = int(args.chunksize)
    if args.skip_preprocessing_core:
        replacements["run_preprocessing_core"] = False
    if replacements:
        config = replace(config, **replacements)

    inputs, success, error = run_adapter(config)
    print(f"Adapter static rows: {inputs.static_harmonized.shape[0]}")
    print(f"Adapter dynamic rows: {inputs.dynamic_harmonized.shape[0]}")
    print(f"Adapter block rows: {inputs.block_index.shape[0]}")
    print(f"ASIC preprocessing core success: {success}")
    if error:
        print(f"ASIC preprocessing core error: {error}")


if __name__ == "__main__":
    main()
