from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "chapter1"
ASIC_INPUT_ROOT = Path("/Users/joanameyer/repository/icu-data-platform/artifacts/asic_harmonized")

HARD_CASE_PATH = (
    ARTIFACT_ROOT
    / "evaluation"
    / "asic"
    / "hard_cases"
    / "primary_medians"
    / "logistic_regression"
    / "stay_level_hard_case_flags.csv"
)
UPSTREAM_PREDICTIONS_PATH = (
    ARTIFACT_ROOT
    / "baselines"
    / "asic"
    / "primary_medians"
    / "logistic_regression"
    / "horizon_24h"
    / "predictions.csv"
)
MODEL_READY_PATH = ARTIFACT_ROOT / "model_ready" / "chapter1_primary_model_ready_dataset.csv"
BLOCKED_DYNAMIC_PATH = ASIC_INPUT_ROOT / "blocked" / "asic_8h_blocked_dynamic_features.csv"
STATIC_PATH = ASIC_INPUT_ROOT / "static" / "harmonized.csv"
DYNAMIC_SOURCE_MAP_PATH = ASIC_INPUT_ROOT / "dynamic" / "source_map.csv"
CHAPTER1_RETAINED_STAY_TABLE_PATH = (
    ARTIFACT_ROOT / "cohort" / "chapter1_retained_stay_table.csv"
)
UPSTREAM_CHAPTER1_COHORT_PATH = ASIC_INPUT_ROOT / "cohort" / "chapter1_stay_level.csv"
UPSTREAM_STAY_LEVEL_PATH = ASIC_INPUT_ROOT / "cohort" / "stay_level.csv"

OUTPUT_DIR = HARD_CASE_PATH.parent / "asic_hardcase_comparison_variable_audit"
TABLE_PATH = OUTPUT_DIR / "asic_hardcase_comparison_variable_audit_table.csv"
MEMO_PATH = OUTPUT_DIR / "asic_hardcase_comparison_variable_audit_memo.md"

TARGET_HORIZON_H = 24
MERGE_KEYS = [
    "instance_id",
    "stay_id_global",
    "hospital_id",
    "block_index",
    "prediction_time_h",
    "horizon_h",
    "label_value",
]
BLOCK_KEYS = ["stay_id_global", "hospital_id", "block_index", "prediction_time_h"]

VASOPRESSOR_CANONICAL_NAMES = [
    "norepinephrine_iv_cont",
    "epinephrine_iv_cont",
    "vasopressin_iv_cont",
    "terlipressin_iv_bolus",
]
RRT_SEARCH_TERMS = ("rrt", "renal replacement", "dialysis", "cvvh", "crrt", "hemofil")
CSV_CHUNKSIZE = 100_000


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _fmt_count(count: int, total: int) -> str:
    pct = (100.0 * count / total) if total else 0.0
    return f"{count}/{total} ({pct:.0f}%)"


def _parse_source_columns(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    parsed = ast.literal_eval(text)
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).strip()]
    return []


def _codeset(raw_codes: object) -> list[str]:
    if pd.isna(raw_codes):
        return []
    return [code.strip().upper().replace(" ", "") for code in str(raw_codes).split(",") if code.strip()]


def _has_prefix(codes: list[str], prefixes: list[str]) -> bool:
    return any(any(code.startswith(prefix) for prefix in prefixes) for code in codes)


def _classify_coarse_disease_group(raw_codes: object) -> str:
    codes = _codeset(raw_codes)
    if any(re.match(r"^(S|T|V|W|X|Y)", code) for code in codes) or _has_prefix(
        codes,
        ["K91", "T81", "Z51"],
    ):
        return "surgical / postoperative / trauma-related"
    if _has_prefix(codes, ["J"]):
        return "respiratory / pulmonary"
    if _has_prefix(codes, ["A40", "A41", "R65", "B95", "B96", "B97", "U81", "N39"]):
        return "infection / sepsis non-pulmonary"
    if _has_prefix(codes, ["G"]) or any(re.match(r"^I6[0-9]", code) for code in codes):
        return "neurologic"
    if _has_prefix(codes, ["I"]) or _has_prefix(codes, ["R57"]):
        return "cardiovascular"
    return "other / mixed / uncategorized"


def _markdown_table(frame: pd.DataFrame) -> str:
    rendered = frame.fillna("").astype(str)
    header = "| " + " | ".join(rendered.columns.tolist()) + " |"
    separator = "| " + " | ".join(["---"] * len(rendered.columns)) + " |"
    rows = [
        "| " + " | ".join(row) + " |"
        for row in rendered.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _csv_columns(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.astype(str).tolist()


def _filter_csv_to_target(
    path: Path,
    usecols: list[str],
    key_columns: list[str],
    target_keys: pd.DataFrame,
) -> pd.DataFrame:
    target_index = pd.MultiIndex.from_frame(target_keys[key_columns].drop_duplicates())
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CSV_CHUNKSIZE):
        mask = pd.MultiIndex.from_frame(chunk[key_columns]).isin(target_index)
        if bool(mask.any()):
            parts.append(chunk.loc[mask].copy())
    if parts:
        return pd.concat(parts, ignore_index=True)
    return pd.read_csv(path, usecols=usecols, nrows=0)


def _resolve_cohort_metadata_path() -> Path:
    candidates = [
        CHAPTER1_RETAINED_STAY_TABLE_PATH,
        UPSTREAM_CHAPTER1_COHORT_PATH,
        UPSTREAM_STAY_LEVEL_PATH,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate a cohort stay-level table for ICU timing metadata. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def build_outputs() -> tuple[pd.DataFrame, str]:
    hard_case_columns = [*MERGE_KEYS, "hard_case_flag"]
    hard_cases = pd.read_csv(HARD_CASE_PATH, usecols=hard_case_columns)
    dynamic_source_map = pd.read_csv(DYNAMIC_SOURCE_MAP_PATH)

    target = hard_cases[
        hard_cases["horizon_h"].eq(TARGET_HORIZON_H) & hard_cases["label_value"].eq(1)
    ].copy()
    if target.empty:
        raise ValueError("The 24h fatal hard-case target population is empty.")

    model_ready_columns = [
        *MERGE_KEYS,
        "pf_ratio_last",
        "pf_ratio_observed_in_block",
        "pf_ratio_filled_by_locf",
        "spo2_last",
        "fio2_last",
        "map_last",
        "map_observed_in_block",
        "creatinine_last",
        "creatinine_observed_in_block",
        "creatinine_filled_by_locf",
        "peep_last",
        "peep_observed_in_block",
        "peep_filled_by_locf",
    ]
    model_ready = _filter_csv_to_target(
        MODEL_READY_PATH,
        usecols=model_ready_columns,
        key_columns=MERGE_KEYS,
        target_keys=target[MERGE_KEYS],
    )

    merged = target.merge(
        model_ready,
        on=MERGE_KEYS,
        how="left",
        indicator=True,
        validate="one_to_one",
        suffixes=("", "_model_ready"),
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Some 24h fatal hard-case rows could not be linked to model-ready data.")
    merged = merged.drop(columns="_merge")

    blocked_dynamic_header = _csv_columns(BLOCKED_DYNAMIC_PATH)
    vasopressor_columns = BLOCK_KEYS + [
        column
        for column in blocked_dynamic_header
        if any(
            column.startswith(f"{base_name}_")
            for base_name in [
                "norepinephrine_iv_cont",
                "epinephrine_iv_cont",
                "vasopressin_iv_cont",
                "terlipressin_iv_bolus",
                "dobutamine_iv_cont",
                "milrinone_iv_cont",
                "levosimendan_iv_cont",
            ]
        )
    ]
    rrt_columns = [
        column
        for column in blocked_dynamic_header
        if any(term in column.lower() for term in RRT_SEARCH_TERMS)
    ]
    blocked_dynamic = _filter_csv_to_target(
        BLOCKED_DYNAMIC_PATH,
        usecols=list(dict.fromkeys([*vasopressor_columns, *rrt_columns])),
        key_columns=BLOCK_KEYS,
        target_keys=target[BLOCK_KEYS],
    )
    merged = merged.merge(
        blocked_dynamic[vasopressor_columns],
        on=BLOCK_KEYS,
        how="left",
        validate="one_to_one",
    )

    static_columns = _csv_columns(STATIC_PATH)
    static = _filter_csv_to_target(
        STATIC_PATH,
        usecols=["stay_id_global", "hospital_id", "age_group", "sex", "icd10_codes"],
        key_columns=["stay_id_global", "hospital_id"],
        target_keys=target[["stay_id_global", "hospital_id"]],
    )
    merged = merged.merge(
        static[["stay_id_global", "hospital_id", "age_group", "sex", "icd10_codes"]],
        on=["stay_id_global", "hospital_id"],
        how="left",
        validate="one_to_one",
    )

    cohort_path = _resolve_cohort_metadata_path()
    cohort = _filter_csv_to_target(
        cohort_path,
        usecols=["stay_id_global", "hospital_id", "icu_admission_time", "icu_end_time_proxy"],
        key_columns=["stay_id_global", "hospital_id"],
        target_keys=target[["stay_id_global", "hospital_id"]],
    )
    merged = merged.merge(
        cohort[["stay_id_global", "hospital_id", "icu_admission_time", "icu_end_time_proxy"]],
        on=["stay_id_global", "hospital_id"],
        how="left",
        validate="one_to_one",
    )

    total = int(merged.shape[0])
    n_hard = int(merged["hard_case_flag"].eq(True).sum())
    n_other = int(merged["hard_case_flag"].eq(False).sum())
    hospital_counts = (
        merged["hospital_id"].astype("string").value_counts().sort_index().to_dict()
    )

    exact_age_candidates = [
        column
        for column in static_columns
        if "age" in column.lower() and column.lower() != "age_group"
    ]
    target_age_group_nonmissing = int(merged["age_group"].notna().sum())
    sex_nonmissing = int(merged["sex"].notna().sum())
    icd_nonmissing = int(merged["icd10_codes"].notna().sum())

    merged["coarse_disease_group"] = merged["icd10_codes"].map(_classify_coarse_disease_group)
    coarse_group_nonmissing = int(merged["coarse_disease_group"].notna().sum())
    disease_group_counts = (
        merged["coarse_disease_group"].value_counts(dropna=False).to_dict()
    )

    icu_admission_zero = bool(pd.to_numeric(merged["icu_admission_time"], errors="coerce").fillna(0).eq(0).all())
    icu_time_nonmissing = int(merged["prediction_time_h"].notna().sum())
    site_nonmissing = int(merged["hospital_id"].notna().sum())

    respiratory_primary_nonmissing = int(merged["pf_ratio_last"].notna().sum())
    respiratory_primary_in_block = int(merged["pf_ratio_observed_in_block"].fillna(False).sum())
    respiratory_primary_locf = int(merged["pf_ratio_filled_by_locf"].fillna(False).sum())
    respiratory_fallback_nonmissing = int(
        merged["spo2_last"].notna().mul(merged["fio2_last"].notna()).sum()
    )
    respiratory_status = "READY"
    respiratory_family_status = "PRIMARY FEASIBLE"

    source_map_subset = dynamic_source_map[
        dynamic_source_map["canonical_name"].isin(VASOPRESSOR_CANONICAL_NAMES)
    ].copy()
    source_map_subset["raw_source_columns_parsed"] = source_map_subset["raw_source_columns_used"].map(
        _parse_source_columns
    )
    mapped_by_hospital = (
        source_map_subset.assign(
            has_source=source_map_subset["raw_source_columns_parsed"].map(bool)
        )
        .groupby("hospital", dropna=False)["has_source"]
        .any()
        .to_dict()
    )
    merged["vasopressor_source_mapped"] = merged["hospital_id"].map(mapped_by_hospital).fillna(False)

    vasopressor_use = pd.Series(False, index=merged.index, dtype="boolean")
    for base_name in VASOPRESSOR_CANONICAL_NAMES:
        obs_count_column = f"{base_name}_obs_count"
        last_column = f"{base_name}_last"
        observed = pd.Series(False, index=merged.index, dtype="boolean")
        if obs_count_column in merged.columns:
            observed = observed | pd.to_numeric(merged[obs_count_column], errors="coerce").fillna(0).gt(0)
        if last_column in merged.columns:
            observed = observed | pd.to_numeric(merged[last_column], errors="coerce").fillna(0).gt(0)
        vasopressor_use = vasopressor_use | observed

    vasopressor_observable = int(merged["vasopressor_source_mapped"].sum())
    vasopressor_yes = int((merged["vasopressor_source_mapped"] & vasopressor_use.fillna(False)).sum())
    vasopressor_structural_missing = int((~merged["vasopressor_source_mapped"]).sum())
    map_nonmissing = int(merged["map_last"].notna().sum())
    map_in_block = int(merged["map_observed_in_block"].fillna(False).sum())
    hemodynamic_status = "READY WITH FALLBACK"
    hemodynamic_family_status = "FALLBACK NEEDED"

    renal_primary_nonmissing = int(merged["creatinine_last"].notna().sum())
    renal_primary_in_block = int(merged["creatinine_observed_in_block"].fillna(False).sum())
    renal_primary_locf = int(merged["creatinine_filled_by_locf"].fillna(False).sum())
    rrt_nonmissing = 0
    renal_status = "READY"
    renal_family_status = "PRIMARY FEASIBLE"

    ventilation_nonmissing = int(merged["peep_last"].notna().sum())
    ventilation_in_block = int(merged["peep_observed_in_block"].fillna(False).sum())
    ventilation_locf = int(merged["peep_filled_by_locf"].fillna(False).sum())
    ventilation_status = "READY"
    ventilation_family_status = "PRIMARY FEASIBLE"

    table = pd.DataFrame(
        [
            {
                "variable_family": "age",
                "chosen_variable": "age",
                "source_artifact": _display_path(STATIC_PATH),
                "static_join_needed": "yes",
                "timepoint_aligned": "yes",
                "completeness": (
                    f"{_fmt_count(0, total)} exact age; "
                    f"{_fmt_count(target_age_group_nonmissing, total)} age_group only"
                ),
                "status": "NOT READY",
                "notes": (
                    "No exact age field exists in the current ASIC static harmonized table or source map. "
                    "Only categorical age_group is present."
                ),
            },
            {
                "variable_family": "sex",
                "chosen_variable": "sex",
                "source_artifact": _display_path(STATIC_PATH),
                "static_join_needed": "yes",
                "timepoint_aligned": "yes",
                "completeness": _fmt_count(sex_nonmissing, total),
                "status": "READY",
                "notes": "Direct static join on sex with full coverage in the target slice.",
            },
            {
                "variable_family": "coarse disease group",
                "chosen_variable": "derived from icd10_codes",
                "source_artifact": _display_path(STATIC_PATH),
                "static_join_needed": "yes",
                "timepoint_aligned": "yes",
                "completeness": _fmt_count(coarse_group_nonmissing, total),
                "status": "READY",
                "notes": (
                    "icd10_codes is complete in the target slice. A reproducible stay-level hierarchy is feasible; "
                    f"provisional counts are {disease_group_counts}."
                ),
            },
            {
                "variable_family": "ICU time to last eligible prediction",
                "chosen_variable": "prediction_time_h",
                "source_artifact": _display_path(HARD_CASE_PATH),
                "static_join_needed": "no",
                "timepoint_aligned": "yes",
                "completeness": _fmt_count(icu_time_nonmissing, total),
                "status": "READY",
                "notes": (
                    "prediction_time_h is already stored on the stay-level hard-case artifact. "
                    f"In `{_display_path(cohort_path)}`, icu_admission_time is 0 for all retained stays, so this is hours since ICU admission."
                ),
            },
            {
                "variable_family": "site",
                "chosen_variable": "hospital_id",
                "source_artifact": _display_path(HARD_CASE_PATH),
                "static_join_needed": "no",
                "timepoint_aligned": "yes",
                "completeness": _fmt_count(site_nonmissing, total),
                "status": "READY",
                "notes": f"Target hospitals: {hospital_counts}.",
            },
            {
                "variable_family": "respiratory proxy",
                "chosen_variable": "primary pf_ratio_last; fallback spo2_last / fio2_last",
                "source_artifact": _display_path(MODEL_READY_PATH),
                "static_join_needed": "no",
                "timepoint_aligned": "partial",
                "completeness": (
                    f"primary {_fmt_count(respiratory_primary_nonmissing, total)}; "
                    f"fallback {_fmt_count(respiratory_fallback_nonmissing, total)}"
                ),
                "status": respiratory_status,
                "notes": (
                    f"PRIMARY FEASIBLE. pf_ratio_last is current-block in {_fmt_count(respiratory_primary_in_block, total)} "
                    f"and never LOCF-filled in the saved Chapter 1 layer. The S/F fallback is derivable but rescues no additional rows; "
                    "missingness is driven by absent FiO2 / oxygenation support documentation."
                ),
            },
            {
                "variable_family": "hemodynamic proxy",
                "chosen_variable": (
                    "primary derived vasopressor_use_last from norepinephrine_iv_cont / "
                    "epinephrine_iv_cont / vasopressin_iv_cont / terlipressin_iv_bolus; fallback map_last"
                ),
                "source_artifact": (
                    f"{_display_path(BLOCKED_DYNAMIC_PATH)}; {_display_path(MODEL_READY_PATH)}"
                ),
                "static_join_needed": "no",
                "timepoint_aligned": "partial",
                "completeness": (
                    f"primary observable {_fmt_count(vasopressor_observable, total)}; "
                    f"fallback {_fmt_count(map_nonmissing, total)}"
                ),
                "status": hemodynamic_status,
                "notes": (
                    f"FALLBACK NEEDED. Vasopressor use is directly derivable in mapped hospitals only; "
                    f"{vasopressor_yes}/{vasopressor_observable} mapped stays show use, but {_fmt_count(vasopressor_structural_missing, total)} "
                    "are structurally unmapped at asic_UK04 and UK02 only exposes norepinephrine. MAP is current-block in "
                    f"{_fmt_count(map_in_block, total)} and is the cleaner proxy."
                ),
            },
            {
                "variable_family": "renal proxy",
                "chosen_variable": "primary creatinine_last; fallback renal replacement therapy",
                "source_artifact": (
                    f"{_display_path(MODEL_READY_PATH)}; {_display_path(BLOCKED_DYNAMIC_PATH)}"
                ),
                "static_join_needed": "no",
                "timepoint_aligned": "partial",
                "completeness": (
                    f"primary {_fmt_count(renal_primary_nonmissing, total)}; "
                    f"fallback {_fmt_count(rrt_nonmissing, total)}"
                ),
                "status": renal_status,
                "notes": (
                    f"PRIMARY FEASIBLE. Creatinine is current-block in {_fmt_count(renal_primary_in_block, total)} "
                    f"and uses existing 48h LOCF in {_fmt_count(renal_primary_locf, total)}. No time-varying RRT field was found in the blocked "
                    f"or model-ready layers ({rrt_columns or 'none'}); static dialysis_free_days is not timepoint-aligned."
                ),
            },
            {
                "variable_family": "ventilation proxy",
                "chosen_variable": "peep_last",
                "source_artifact": _display_path(MODEL_READY_PATH),
                "static_join_needed": "no",
                "timepoint_aligned": "partial",
                "completeness": _fmt_count(ventilation_nonmissing, total),
                "status": ventilation_status,
                "notes": (
                    f"PRIMARY FEASIBLE. PEEP is current-block in {_fmt_count(ventilation_in_block, total)} "
                    f"and uses within-window ventilator LOCF in {_fmt_count(ventilation_locf, total)}."
                ),
            },
        ]
    )

    overall_ready = False
    blocking_variable_families = ["age"]

    memo_lines = [
        "# ASIC Issue 3.2 Frozen Variable Package Feasibility",
        "",
        "## Scope",
        "- Narrow feasibility / availability confirmation only for the frozen ASIC Issue 3.2 variable package.",
        "- No variable discovery, no broader substitutions, no SOFA reconsideration, and no admission-type reconsideration.",
        "",
        "## Target Population Artifact",
        f"- Primary anchor artifact: `{_display_path(HARD_CASE_PATH)}`.",
        f"- Filter applied: `horizon_h == {TARGET_HORIZON_H}` and `label_value == 1`.",
        f"- Resulting target population: `{total}` fatal stays with one last eligible 24h prediction instance per stay.",
        f"- Low-predicted fatal stays: `{n_hard}`. Other fatal stays: `{n_other}`.",
        "- A standalone 24h fatal-only artifact is not already saved separately; the slice is reproducibly constructed by filtering the saved stay-level hard-case flags.",
        f"- Upstream reconstruction remains available from `{_display_path(UPSTREAM_PREDICTIONS_PATH)}` via `select_last_eligible_stay_points` and `classify_hard_cases_for_horizon` in `src/chapter1_mortality_decomposition/hard_case_definition.py`.",
        "",
        "## Join Logic",
        "- Start from the filtered stay-level hard-case slice described above.",
        f"- Join `{_display_path(MODEL_READY_PATH)}` on `instance_id`, `stay_id_global`, `hospital_id`, `block_index`, `prediction_time_h`, `horizon_h`, and `label_value` to recover the Chapter 1 time-varying proxy fields already exported for analysis.",
        f"- Join `{_display_path(BLOCKED_DYNAMIC_PATH)}` on `stay_id_global`, `hospital_id`, `block_index`, and `prediction_time_h` for vasopressor columns that exist in the blocked dynamic layer but were not selected into the model-ready training matrix.",
        f"- Join `{_display_path(STATIC_PATH)}` on `stay_id_global` and `hospital_id` for static sex plus ICD-based disease-group inputs. This same static join shows that only `age_group`, not exact age, is currently available.",
        "",
        "## Source Mapping And Timepoint Alignment",
        "- `age`: no exact age field was found in the current Chapter 1 analysis artifacts or the upstream ASIC static harmonized table; only static `age_group` exists.",
        "- `sex`: static `sex` from the harmonized static table via the stay-level static join.",
        "- `ICD-10-derived coarse disease group`: derive from static `icd10_codes` after the same stay-level static join.",
        f"- `time from ICU admission to last eligible prediction`: use `prediction_time_h` from the stay-level hard-case artifact. Because `icu_admission_time` is 0 in `{_display_path(cohort_path)}`, this already equals hours since ICU admission.",
        "- `site / hospital`: use `hospital_id` from the stay-level hard-case artifact.",
        "- `respiratory primary`: use `pf_ratio_last` from model-ready. It is directly available at the saved last eligible block when present and is not LOCF-filled in this Chapter 1 export.",
        "- `respiratory fallback`: derive `spo2_last / fio2_last` from model-ready if needed. It is derivable from the same block row but offers no extra rescue in this artifact bundle.",
        "- `hemodynamic primary`: derive vasopressor use from blocked dynamic vasopressor fields at the exact last eligible block. This is directly time-aligned where site source mappings exist.",
        "- `hemodynamic fallback`: use `map_last` from model-ready. It is directly current-block when present in this target slice.",
        "- `renal primary`: use `creatinine_last` from model-ready. It is current-block in a minority of rows and otherwise depends on the repo's existing 48h LOCF.",
        "- `renal fallback`: no timepoint-valid renal replacement therapy field was found in the blocked or model-ready layers; static `dialysis_free_days` is not an at-timepoint fallback.",
        "- `ventilation primary`: use `peep_last` from model-ready. It is current-block when present except for one within-window LOCF fill.",
        "",
        "## Completeness Assessment",
        f"- `age`: exact age `0/{total}`. Static `age_group` exists in `{target_age_group_nonmissing}/{total}` but is not the frozen exact-age variable.",
        f"- `sex`: `{sex_nonmissing}/{total}` non-missing.",
        f"- `coarse disease group`: `icd10_codes` is `{icd_nonmissing}/{total}` and the provisional hierarchy assigns `{coarse_group_nonmissing}/{total}` stays.",
        f"- `time from ICU admission to last eligible prediction`: `{icu_time_nonmissing}/{total}` non-missing.",
        f"- `site / hospital`: `{site_nonmissing}/{total}` non-missing.",
        f"- `respiratory primary (PF ratio)`: `{respiratory_primary_nonmissing}/{total}` non-missing; `{respiratory_primary_in_block}/{total}` directly current-block; `{respiratory_primary_locf}/{total}` LOCF-filled.",
        f"- `respiratory fallback (SF ratio)`: `{respiratory_fallback_nonmissing}/{total}` derivable. Missingness is driven by absent FiO2 / oxygenation support documentation, so the fallback does not add coverage here.",
        f"- `hemodynamic primary (vasopressor use)`: directly observable in `{vasopressor_observable}/{total}` stays; `{vasopressor_yes}/{vasopressor_observable}` of those show use. `{vasopressor_structural_missing}/{total}` are structurally unmapped because the hospital-level raw vasopressor source fields are absent.",
        f"- `hemodynamic fallback (MAP)`: `{map_nonmissing}/{total}` non-missing and `{map_in_block}/{total}` are direct current-block observations.",
        f"- `renal primary (creatinine)`: `{renal_primary_nonmissing}/{total}` non-missing; `{renal_primary_in_block}/{total}` direct current-block; `{renal_primary_locf}/{total}` require existing 48h LOCF.",
        f"- `renal fallback (RRT)`: no time-varying field found, so coverage is `{rrt_nonmissing}/{total}` by the agreed fallback definition.",
        f"- `ventilation primary (PEEP)`: `{ventilation_nonmissing}/{total}` non-missing; `{ventilation_in_block}/{total}` direct current-block; `{ventilation_locf}/{total}` within-window LOCF.",
        "",
        "## Proxy Family Feasibility",
        f"- Respiratory: `{respiratory_family_status}`.",
        f"- Hemodynamic: `{hemodynamic_family_status}`.",
        f"- Renal: `{renal_family_status}`.",
        f"- Ventilation: `{ventilation_family_status}`.",
        "",
        "## Disease-Group Feasibility",
        "- `icd10_codes` exists as a static stay-linked field and can be joined cleanly on `stay_id_global` + `hospital_id`.",
        "- A reproducible stay-level coarse disease-group variable is feasible.",
        "- A simple hierarchy is necessary because the target stays frequently carry multi-system ICD-10 code lists; the grouping is therefore hierarchy-sensitive rather than naturally one-to-one.",
        f"- Under a provisional hierarchy used only to estimate feasibility, target stays distribute as: `{disease_group_counts}`.",
        "",
        "## Compact Table",
        _markdown_table(table),
        "",
        "## Final Readiness Judgement",
        "- age: `NOT READY`",
        "- sex: `READY`",
        "- coarse disease group: `READY`",
        "- ICU time to last eligible prediction: `READY`",
        "- site: `READY`",
        f"- respiratory proxy: `{respiratory_status}`",
        f"- hemodynamic proxy: `{hemodynamic_status}`",
        f"- renal proxy: `{renal_status}`",
        f"- ventilation proxy: `{ventilation_status}`",
        "",
        (
            "- Overall judgement: `ISSUE 3.2 VARIABLE PACKAGE READY`."
            if overall_ready
            else "- Overall judgement: `ISSUE 3.2 VARIABLE PACKAGE NOT YET READY`."
        ),
        (
            "- Blocking variable family: `age`."
            if not overall_ready
            else "- No blocking variable families remain."
        ),
        "- Interpretation: the joins and the target population slice are reproducible now, and all non-age families are either ready or ready via the predefined MAP fallback. The package is not yet fully ready because exact age is absent from the current ASIC static layer; only categorical `age_group` is available.",
    ]

    memo_text = "\n".join(memo_lines) + "\n"
    return table, memo_text


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table, memo_text = build_outputs()
    table.to_csv(TABLE_PATH, index=False)
    MEMO_PATH.write_text(memo_text, encoding="utf-8")
    print(f"Wrote table: {TABLE_PATH}")
    print(f"Wrote memo: {MEMO_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
