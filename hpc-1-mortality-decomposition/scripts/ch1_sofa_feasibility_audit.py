from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "chapter1"

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
FEATURE_DEFINITION_PATH = (
    ARTIFACT_ROOT / "feature_sets" / "chapter1_feature_set_definition.csv"
)
LOCF_SUMMARY_PATH = ARTIFACT_ROOT / "carry_forward" / "chapter1_primary_locf_feature_summary.csv"
OUTPUT_DIR = HARD_CASE_PATH.parent / "asic_sofa_feasibility_audit"
TABLE_PATH = OUTPUT_DIR / "sofa_component_feasibility_table.csv"
MEMO_PATH = OUTPUT_DIR / "sofa_feasibility_memo.md"

MERGE_KEYS = [
    "instance_id",
    "stay_id_global",
    "hospital_id",
    "block_index",
    "prediction_time_h",
    "horizon_h",
    "label_value",
]


def _relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _fmt_count(count: int, total: int) -> str:
    pct = (100.0 * count / total) if total else 0.0
    return f"{count}/{total} ({pct:.0f}%)"


def _bool_count(series: pd.Series) -> int:
    return int(series.fillna(False).astype("boolean").sum())


def _notna_count(series: pd.Series) -> int:
    return int(series.notna().sum())


def _group_count_string(
    frame: pd.DataFrame,
    *,
    selector: pd.Series,
) -> str:
    hard = int(selector[frame["hard_case_flag"].eq(True)].sum())
    other = int(selector[frame["hard_case_flag"].eq(False)].sum())
    return f"{other}/{int(frame['hard_case_flag'].eq(False).sum())} other fatal, {hard}/{int(frame['hard_case_flag'].eq(True).sum())} low-predicted fatal"


def _find_matches(columns: list[str], base_variables: list[str], terms: tuple[str, ...]) -> list[str]:
    matches: list[str] = []
    for value in columns + base_variables:
        lowered = value.lower()
        if any(term in lowered for term in terms):
            matches.append(value)
    return sorted(set(matches))


def _match_summary(matches: list[str]) -> str:
    return ", ".join(matches) if matches else "none"


def build_audit_outputs() -> tuple[pd.DataFrame, str]:
    hard_cases = pd.read_csv(HARD_CASE_PATH)
    model_ready = pd.read_csv(MODEL_READY_PATH)
    feature_def = pd.read_csv(FEATURE_DEFINITION_PATH)
    locf_summary = pd.read_csv(LOCF_SUMMARY_PATH)

    target = hard_cases[
        hard_cases["horizon_h"].eq(24) & hard_cases["label_value"].eq(1)
    ].copy()
    if target.empty:
        raise ValueError("The 24h fatal stay-level hard-case population is empty.")

    merged = target.merge(
        model_ready,
        on=MERGE_KEYS,
        how="left",
        indicator=True,
        validate="one_to_one",
        suffixes=("", "_model_ready"),
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Some 24h fatal stay-level rows could not be linked to model-ready data.")
    merged = merged.drop(columns="_merge")

    total = int(merged.shape[0])
    n_hard = int(merged["hard_case_flag"].eq(True).sum())
    n_other = int(merged["hard_case_flag"].eq(False).sum())

    base_variables = (
        feature_def["base_variable"].dropna().astype("string").drop_duplicates().tolist()
    )
    model_ready_columns = model_ready.columns.astype(str).tolist()
    gcs_matches = _find_matches(model_ready_columns, base_variables, ("gcs", "glasgow"))
    vasopressor_matches = _find_matches(
        model_ready_columns,
        base_variables,
        ("vaso", "dopamine", "dobut", "norepi", "epine"),
    )
    urine_matches = _find_matches(model_ready_columns, base_variables, ("urine",))

    locf_windows = {
        row.base_variable: row.locf_window_hours
        for row in locf_summary.itertuples(index=False)
        if pd.notna(row.locf_window_hours)
    }

    resp_after = merged["pf_ratio_last"].notna() & merged["fio2_ventilation_window_active"].fillna(False)
    resp_in_block = (
        merged["pf_ratio_observed_in_block"].fillna(False)
        & merged["fio2_ventilation_window_active"].fillna(False)
    )
    coag_after = merged["platelets_last"].notna()
    coag_in_block = merged["platelets_observed_in_block"].fillna(False)
    liver_after = merged["bilirubin_total_last"].notna()
    liver_in_block = merged["bilirubin_total_observed_in_block"].fillna(False)
    cv_after = merged["map_last"].notna()
    cv_in_block = merged["map_observed_in_block"].fillna(False)
    renal_after = merged["creatinine_last"].notna()
    renal_in_block = merged["creatinine_observed_in_block"].fillna(False)

    available_after_cols = [resp_after, coag_after, liver_after, cv_after, renal_after]
    available_in_block_cols = [resp_in_block, coag_in_block, liver_in_block, cv_in_block, renal_in_block]
    complete_after = pd.concat(available_after_cols, axis=1).all(axis=1)
    complete_in_block = pd.concat(available_in_block_cols, axis=1).all(axis=1)
    pf_ratio_available = merged["pf_ratio_last"].notna()
    pao2_available = merged["pao2_last"].notna()
    fio2_available = merged["fio2_last"].notna()
    spo2_available = merged["spo2_last"].notna()

    table = pd.DataFrame(
        [
            {
                "component": "Respiratory",
                "required_inputs": "PaO2/FiO2 ratio plus mechanical ventilation or respiratory-support status",
                "available_variables": "; ".join(
                    [
                        "pf_ratio_last",
                        "pf_ratio_observed_in_block",
                        "pf_ratio_filled_by_locf",
                        "pao2_last",
                        "fio2_last",
                        "fio2_ventilation_window_active",
                        "spo2_last",
                        "sao2_last",
                    ]
                ),
                "source_artifacts": "; ".join(
                    [
                        _relative(MODEL_READY_PATH),
                        _relative(FEATURE_DEFINITION_PATH),
                        _relative(LOCF_SUMMARY_PATH),
                    ]
                ),
                "timepoint_aligned": "partial",
                "completeness_in_target_population": (
                    f"{_fmt_count(int(resp_after.sum()), total)} scorable with existing PF ratio + "
                    f"ventilation flag; {_fmt_count(int(resp_in_block.sum()), total)} observed in current block"
                ),
                "feasible": "partial",
                "notes": (
                    "PF ratio exists in the current analysis layer and is already derived. "
                    f"No additional LOCF rescue was observed in the target rows. "
                    f"Ventilation status is only indirectly available via fio2_ventilation_window_active. "
                    f"SpO2 is present in {_fmt_count(_notna_count(merged['spo2_last']), total)} rows, but an S/F rescue rule "
                    "would be a nonstandard proxy and is not defensible for Issue 3.2."
                ),
            },
            {
                "component": "Coagulation",
                "required_inputs": "Platelet count",
                "available_variables": "; ".join(
                    [
                        "platelets_last",
                        "platelets_observed_in_block",
                        "platelets_filled_by_locf",
                        "platelets_missing_after_locf",
                    ]
                ),
                "source_artifacts": "; ".join(
                    [
                        _relative(MODEL_READY_PATH),
                        _relative(FEATURE_DEFINITION_PATH),
                        _relative(LOCF_SUMMARY_PATH),
                    ]
                ),
                "timepoint_aligned": "partial",
                "completeness_in_target_population": (
                    f"{_fmt_count(int(coag_after.sum()), total)} after existing LOCF; "
                    f"{_fmt_count(int(coag_in_block.sum()), total)} observed in current block"
                ),
                "feasible": "partial",
                "notes": (
                    f"Standard platelet mapping is straightforward when values exist, but "
                    f"{_fmt_count(_bool_count(merged['platelets_filled_by_locf']), total)} of target rows rely on "
                    f"{int(locf_windows['platelets'])}h LOCF rather than a current-block observation."
                ),
            },
            {
                "component": "Liver",
                "required_inputs": "Total bilirubin",
                "available_variables": "; ".join(
                    [
                        "bilirubin_total_last",
                        "bilirubin_total_observed_in_block",
                        "bilirubin_total_filled_by_locf",
                        "bilirubin_total_missing_after_locf",
                    ]
                ),
                "source_artifacts": "; ".join(
                    [
                        _relative(MODEL_READY_PATH),
                        _relative(FEATURE_DEFINITION_PATH),
                        _relative(LOCF_SUMMARY_PATH),
                    ]
                ),
                "timepoint_aligned": "partial",
                "completeness_in_target_population": (
                    f"{_fmt_count(int(liver_after.sum()), total)} after existing LOCF; "
                    f"{_fmt_count(int(liver_in_block.sum()), total)} observed in current block"
                ),
                "feasible": "partial",
                "notes": (
                    f"Bilirubin is sparse and depends on {int(locf_windows['bilirubin_total'])}h LOCF in "
                    f"{_fmt_count(_bool_count(merged['bilirubin_total_filled_by_locf']), total)} rows. "
                    "Missingness is strongly structured by hard-case status."
                ),
            },
            {
                "component": "Cardiovascular",
                "required_inputs": "MAP and vasopressor information sufficient for SOFA-style scoring",
                "available_variables": "; ".join(
                    [
                        "map_last",
                        "map_observed_in_block",
                        "map_filled_by_locf",
                        "map_missing_after_locf",
                    ]
                ),
                "source_artifacts": "; ".join(
                    [
                        _relative(MODEL_READY_PATH),
                        _relative(FEATURE_DEFINITION_PATH),
                    ]
                ),
                "timepoint_aligned": "partial",
                "completeness_in_target_population": (
                    f"MAP {_fmt_count(int(cv_after.sum()), total)} after existing LOCF; "
                    f"vasopressors {_fmt_count(0, total)}"
                ),
                "feasible": "no",
                "notes": (
                    "MAP is mostly available and largely current-block, but no vasopressor columns were found in "
                    "the Chapter 1 feature dictionary or model-ready table. Standard SOFA cardiovascular scoring "
                    "is therefore not defensible."
                ),
            },
            {
                "component": "CNS",
                "required_inputs": "GCS",
                "available_variables": "None found",
                "source_artifacts": "; ".join(
                    [
                        _relative(MODEL_READY_PATH),
                        _relative(FEATURE_DEFINITION_PATH),
                        "repo-wide search in src/chapter1_mortality_decomposition",
                    ]
                ),
                "timepoint_aligned": "no",
                "completeness_in_target_population": _fmt_count(0, total),
                "feasible": "no",
                "notes": "No GCS or Glasgow-derived variable is represented in the current Chapter 1 analysis layer.",
            },
            {
                "component": "Renal",
                "required_inputs": "Creatinine and/or urine output",
                "available_variables": "; ".join(
                    [
                        "creatinine_last",
                        "creatinine_observed_in_block",
                        "creatinine_filled_by_locf",
                        "creatinine_missing_after_locf",
                    ]
                ),
                "source_artifacts": "; ".join(
                    [
                        _relative(MODEL_READY_PATH),
                        _relative(FEATURE_DEFINITION_PATH),
                        _relative(LOCF_SUMMARY_PATH),
                    ]
                ),
                "timepoint_aligned": "partial",
                "completeness_in_target_population": (
                    f"Creatinine {_fmt_count(int(renal_after.sum()), total)} after existing LOCF; "
                    f"{_fmt_count(int(renal_in_block.sum()), total)} observed in current block; "
                    f"urine output {_fmt_count(0, total)}"
                ),
                "feasible": "partial",
                "notes": (
                    f"Creatinine is available, but {_fmt_count(_bool_count(merged['creatinine_filled_by_locf']), total)} "
                    f"rows depend on {int(locf_windows['creatinine'])}h LOCF and no urine-output field is available "
                    "to rescue missing or discordant renal status."
                ),
            },
        ]
    )

    memo = "\n".join(
        [
            "# SOFA Feasibility Audit for ASIC Chapter 1 / Sprint 3 / Issue 3.2",
            "",
            "## Target Population Used",
            f"- Exact artifact: `{_relative(HARD_CASE_PATH)}`.",
            "- Filter applied: `horizon_h == 24` and `label_value == 1`.",
            f"- Resulting stay-level fatal comparison population: `{total}` stays.",
            f"- Low-predicted fatal stays (`hard_case_flag == True`): `{n_hard}`.",
            f"- Other fatal stays (`hard_case_flag == False`): `{n_other}`.",
            f"- This artifact already stores one last eligible prediction instance per stay and horizon; upstream reconstruction is available from `{_relative(UPSTREAM_PREDICTIONS_PATH)}` via `select_last_eligible_stay_points` and `classify_hard_cases_for_horizon` in `src/chapter1_mortality_decomposition/hard_case_definition.py`.",
            "",
            "## Component-by-Component Inventory",
            f"- Respiratory: `pf_ratio_last` plus `fio2_ventilation_window_active` make a partial standard mapping possible in `{_fmt_count(int(resp_after.sum()), total)}` fatal stays. `pao2_last` is present in `{_fmt_count(int(pao2_available.sum()), total)}`, `fio2_last` is present in `{_fmt_count(int(fio2_available.sum()), total)}`, and the derived PF ratio is available in `{_fmt_count(int(pf_ratio_available.sum()), total)}`. `spo2_last` is present in `{_fmt_count(int(spo2_available.sum()), total)}`, but an S/F rescue mapping would be nonstandard and should not be introduced for Issue 3.2.",
            f"- Coagulation: `platelets_last` is available in `{_fmt_count(int(coag_after.sum()), total)}`. The raw mapping is standard, but only `{_fmt_count(int(coag_in_block.sum()), total)}` are observed in the current 8h block and `{_fmt_count(_bool_count(merged['platelets_filled_by_locf']), total)}` depend on 24h LOCF.",
            f"- Liver: `bilirubin_total_last` is available in `{_fmt_count(int(liver_after.sum()), total)}`. Only `{_fmt_count(int(liver_in_block.sum()), total)}` are observed in the current block; `{_fmt_count(_bool_count(merged['bilirubin_total_filled_by_locf']), total)}` rely on 48h LOCF and `{_fmt_count(_bool_count(merged['bilirubin_total_missing_after_locf']), total)}` remain missing.",
            f"- Cardiovascular: `map_last` is available in `{_fmt_count(int(cv_after.sum()), total)}` and is current-block in `{_fmt_count(int(cv_in_block.sum()), total)}`. No vasopressor variables were found in the feature dictionary or the model-ready layer, so standard SOFA cardiovascular scoring cannot be completed.",
            f"- CNS: no GCS field was found in the feature dictionary or model-ready dataset (`{_match_summary(gcs_matches)}`), so the CNS component is absent.",
            f"- Renal: `creatinine_last` is available in `{_fmt_count(int(renal_after.sum()), total)}`. Only `{_fmt_count(int(renal_in_block.sum()), total)}` are current-block observations, `{_fmt_count(_bool_count(merged['creatinine_filled_by_locf']), total)}` depend on 48h LOCF, and no urine-output field was found (`{_match_summary(urine_matches)}`).",
            "",
            "## Timepoint Alignment Assessment",
            "- The Chapter 1 analysis layer is block-based. Candidate values are 8h-block summaries ending at `prediction_time_h`, not instantaneous bedside measurements.",
            f"- Current-block availability across the partially represented SOFA organs is: respiratory `{_fmt_count(int(resp_in_block.sum()), total)}`, coagulation `{_fmt_count(int(coag_in_block.sum()), total)}`, liver `{_fmt_count(int(liver_in_block.sum()), total)}`, cardiovascular MAP `{_fmt_count(int(cv_in_block.sum()), total)}`, renal creatinine `{_fmt_count(int(renal_in_block.sum()), total)}`.",
            f"- Allowing only the repo's existing bounded LOCF windows raises availability to: respiratory `{_fmt_count(int(resp_after.sum()), total)}`, coagulation `{_fmt_count(int(coag_after.sum()), total)}`, liver `{_fmt_count(int(liver_after.sum()), total)}`, cardiovascular MAP `{_fmt_count(int(cv_after.sum()), total)}`, renal creatinine `{_fmt_count(int(renal_after.sum()), total)}`.",
            f"- Complete-case coverage across those five partially represented organs is `{_fmt_count(int(complete_in_block.sum()), total)}` using current-block observations only and `{_fmt_count(int(complete_after.sum()), total)}` after LOCF.",
            "- Respiratory support status is only indirectly aligned through `fio2_ventilation_window_active`, which flags overlap between the current block and a documented mechanical-ventilation episode.",
            "- Coagulation, liver, and renal coverage are materially dependent on carry-forward from earlier blocks. That makes any score a mixture of current physiology and stale laboratory values rather than a clean same-timepoint severity snapshot.",
            "",
            "## Missingness and Completeness Assessment",
            f"- Respiratory scorable rows are structured by subgroup: {_group_count_string(merged, selector=resp_after.astype(int))}.",
            f"- Bilirubin availability is strongly structured: {_group_count_string(merged, selector=liver_after.astype(int))}.",
            f"- Creatinine availability is also structured: {_group_count_string(merged, selector=renal_after.astype(int))}.",
            f"- Available-organ complete-case coverage after LOCF is also selective: {_group_count_string(merged, selector=complete_after.astype(int))}.",
            "- These patterns do not look plausibly missing completely at random. They reflect measurement intensity and case mix, which would bias any complete-case SOFA comparison.",
            "- Component defaulting or imputation would be methodologically unacceptable here. Treating unmeasured GCS, bilirubin, urine output, or vasopressors as normal would mechanically bias the low-predicted fatal comparison.",
            "",
            "## Feasibility Classification",
            "- Final classification: `NOT FEASIBLE`.",
            f"- Rationale: standard SOFA is blocked by absent CNS, absent vasopressors, and absent urine output. Even a reduced pseudo-SOFA would still depend on nonstandard omissions plus heavy 24-48h carry-forward and would have only `{_fmt_count(int(complete_after.sum()), total)}` complete cases across the available organs after LOCF.",
            "",
            "## Recommendation",
            "- Recommendation: `C. Do not use SOFA in Issue 3.2; proceed with direct organ-support/dysfunction proxies only.`",
            "- A transparent descriptive table of direct proxies is cleaner and more reproducible than introducing a pseudo-SOFA that is missing entire standard domains.",
            "",
            "## Implementation Sketch",
            "- Not provided, because feasibility is not clearly positive.",
        ]
    )

    if gcs_matches:
        raise ValueError(f"Unexpected GCS-like fields found: {gcs_matches}")
    if vasopressor_matches:
        raise ValueError(f"Unexpected vasopressor-like fields found: {vasopressor_matches}")
    if urine_matches:
        raise ValueError(f"Unexpected urine-output-like fields found: {urine_matches}")

    return table, memo


def main() -> None:
    table, memo = build_audit_outputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_PATH, index=False)
    MEMO_PATH.write_text(memo + "\n")

    print(f"Wrote {TABLE_PATH}")
    print(f"Wrote {MEMO_PATH}")


if __name__ == "__main__":
    main()
