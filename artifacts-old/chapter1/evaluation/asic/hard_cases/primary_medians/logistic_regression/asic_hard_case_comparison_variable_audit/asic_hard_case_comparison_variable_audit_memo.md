# ASIC Issue 3.2 Frozen Variable Package Feasibility

## Scope
- Narrow feasibility / availability confirmation only for the frozen ASIC Issue 3.2 variable package.
- No variable discovery, no broader substitutions, no SOFA reconsideration, and no admission-type reconsideration.

## Target Population Artifact
- Primary anchor artifact: `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/stay_level_hard_case_flags.csv`.
- Filter applied: `horizon_h == 24` and `label_value == 1`.
- Resulting target population: `10` fatal stays with one last eligible 24h prediction instance per stay.
- Low-predicted fatal stays: `4`. Other fatal stays: `6`.
- A standalone 24h fatal-only artifact is not already saved separately; the slice is reproducibly constructed by filtering the saved stay-level hard-case flags.
- Upstream reconstruction remains available from `artifacts/chapter1/baselines/asic/primary_medians/logistic_regression/horizon_24h/predictions.csv` via `select_last_eligible_stay_points` and `classify_hard_cases_for_horizon` in `src/chapter1_mortality_decomposition/hard_case_definition.py`.

## Join Logic
- Start from the filtered stay-level hard-case slice described above.
- Join `artifacts/chapter1/model_ready/chapter1_primary_model_ready_dataset.csv` on `instance_id`, `stay_id_global`, `hospital_id`, `block_index`, `prediction_time_h`, `horizon_h`, and `label_value` to recover the Chapter 1 time-varying proxy fields already exported for analysis.
- Join `/Users/joanameyer/repository/icu-data-platform/artifacts/asic_harmonized/blocked/asic_8h_blocked_dynamic_features.csv` on `stay_id_global`, `hospital_id`, `block_index`, and `prediction_time_h` for vasopressor columns that exist in the blocked dynamic layer but were not selected into the model-ready training matrix.
- Join `/Users/joanameyer/repository/icu-data-platform/artifacts/asic_harmonized/static/harmonized.csv` on `stay_id_global` and `hospital_id` for static sex plus ICD-based disease-group inputs. This same static join shows that only `age_group`, not exact age, is currently available.

## Source Mapping And Timepoint Alignment
- `age`: no exact age field was found in the current Chapter 1 analysis artifacts or the upstream ASIC static harmonized table; only static `age_group` exists.
- `sex`: static `sex` from the harmonized static table via the stay-level static join.
- `ICD-10-derived coarse disease group`: derive from static `icd10_codes` after the same stay-level static join.
- `time from ICU admission to last eligible prediction`: use `prediction_time_h` from the stay-level hard-case artifact. Because `icu_admission_time` is 0 in `artifacts/chapter1/cohort/chapter1_retained_stay_table.csv`, this already equals hours since ICU admission.
- `site / hospital`: use `hospital_id` from the stay-level hard-case artifact.
- `respiratory primary`: use `pf_ratio_last` from model-ready. It is directly available at the saved last eligible block when present and is not LOCF-filled in this Chapter 1 export.
- `respiratory fallback`: derive `spo2_last / fio2_last` from model-ready if needed. It is derivable from the same block row but offers no extra rescue in this artifact bundle.
- `hemodynamic primary`: derive vasopressor use from blocked dynamic vasopressor fields at the exact last eligible block. This is directly time-aligned where site source mappings exist.
- `hemodynamic fallback`: use `map_last` from model-ready. It is directly current-block when present in this target slice.
- `renal primary`: use `creatinine_last` from model-ready. It is current-block in a minority of rows and otherwise depends on the repo's existing 48h LOCF.
- `renal fallback`: no timepoint-valid renal replacement therapy field was found in the blocked or model-ready layers; static `dialysis_free_days` is not an at-timepoint fallback.
- `ventilation primary`: use `peep_last` from model-ready. It is current-block when present except for one within-window LOCF fill.

## Completeness Assessment
- `age`: exact age `0/10`. Static `age_group` exists in `10/10` but is not the frozen exact-age variable.
- `sex`: `10/10` non-missing.
- `coarse disease group`: `icd10_codes` is `10/10` and the provisional hierarchy assigns `10/10` stays.
- `time from ICU admission to last eligible prediction`: `10/10` non-missing.
- `site / hospital`: `10/10` non-missing.
- `respiratory primary (PF ratio)`: `7/10` non-missing; `7/10` directly current-block; `0/10` LOCF-filled.
- `respiratory fallback (SF ratio)`: `7/10` derivable. Missingness is driven by absent FiO2 / oxygenation support documentation, so the fallback does not add coverage here.
- `hemodynamic primary (vasopressor use)`: directly observable in `8/10` stays; `3/8` of those show use. `2/10` are structurally unmapped because the hospital-level raw vasopressor source fields are absent.
- `hemodynamic fallback (MAP)`: `9/10` non-missing and `9/10` are direct current-block observations.
- `renal primary (creatinine)`: `8/10` non-missing; `3/10` direct current-block; `5/10` require existing 48h LOCF.
- `renal fallback (RRT)`: no time-varying field found, so coverage is `0/10` by the agreed fallback definition.
- `ventilation primary (PEEP)`: `7/10` non-missing; `6/10` direct current-block; `1/10` within-window LOCF.

## Proxy Family Feasibility
- Respiratory: `PRIMARY FEASIBLE`.
- Hemodynamic: `FALLBACK NEEDED`.
- Renal: `PRIMARY FEASIBLE`.
- Ventilation: `PRIMARY FEASIBLE`.

## Disease-Group Feasibility
- `icd10_codes` exists as a static stay-linked field and can be joined cleanly on `stay_id_global` + `hospital_id`.
- A reproducible stay-level coarse disease-group variable is feasible.
- A simple hierarchy is necessary because the target stays frequently carry multi-system ICD-10 code lists; the grouping is therefore hierarchy-sensitive rather than naturally one-to-one.
- Under a provisional hierarchy used only to estimate feasibility, target stays distribute as: `{'surgical / postoperative / trauma-related': 5, 'respiratory / pulmonary': 3, 'infection / sepsis non-pulmonary': 1, 'cardiovascular': 1}`.

## Compact Table
| variable_family | chosen_variable | source_artifact | static_join_needed | timepoint_aligned | completeness | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| age | age | /Users/joanameyer/repository/icu-data-platform/artifacts/asic_harmonized/static/harmonized.csv | yes | yes | 0/10 (0%) exact age; 10/10 (100%) age_group only | NOT READY | No exact age field exists in the current ASIC static harmonized table or source map. Only categorical age_group is present. |
| sex | sex | /Users/joanameyer/repository/icu-data-platform/artifacts/asic_harmonized/static/harmonized.csv | yes | yes | 10/10 (100%) | READY | Direct static join on sex with full coverage in the target slice. |
| coarse disease group | derived from icd10_codes | /Users/joanameyer/repository/icu-data-platform/artifacts/asic_harmonized/static/harmonized.csv | yes | yes | 10/10 (100%) | READY | icd10_codes is complete in the target slice. A reproducible stay-level hierarchy is feasible; provisional counts are {'surgical / postoperative / trauma-related': 5, 'respiratory / pulmonary': 3, 'infection / sepsis non-pulmonary': 1, 'cardiovascular': 1}. |
| ICU time to last eligible prediction | prediction_time_h | artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/stay_level_hard_case_flags.csv | no | yes | 10/10 (100%) | READY | prediction_time_h is already stored on the stay-level hard-case artifact. In `artifacts/chapter1/cohort/chapter1_retained_stay_table.csv`, icu_admission_time is 0 for all retained stays, so this is hours since ICU admission. |
| site | hospital_id | artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/stay_level_hard_case_flags.csv | no | yes | 10/10 (100%) | READY | Target hospitals: {'asic_UK02': 4, 'asic_UK04': 2, 'asic_UK07': 3, 'asic_UK08': 1}. |
| respiratory proxy | primary pf_ratio_last; fallback spo2_last / fio2_last | artifacts/chapter1/model_ready/chapter1_primary_model_ready_dataset.csv | no | partial | primary 7/10 (70%); fallback 7/10 (70%) | READY | PRIMARY FEASIBLE. pf_ratio_last is current-block in 7/10 (70%) and never LOCF-filled in the saved Chapter 1 layer. The S/F fallback is derivable but rescues no additional rows; missingness is driven by absent FiO2 / oxygenation support documentation. |
| hemodynamic proxy | primary derived vasopressor_use_last from norepinephrine_iv_cont / epinephrine_iv_cont / vasopressin_iv_cont / terlipressin_iv_bolus; fallback map_last | /Users/joanameyer/repository/icu-data-platform/artifacts/asic_harmonized/blocked/asic_8h_blocked_dynamic_features.csv; artifacts/chapter1/model_ready/chapter1_primary_model_ready_dataset.csv | no | partial | primary observable 8/10 (80%); fallback 9/10 (90%) | READY WITH FALLBACK | FALLBACK NEEDED. Vasopressor use is directly derivable in mapped hospitals only; 3/8 mapped stays show use, but 2/10 (20%) are structurally unmapped at asic_UK04 and UK02 only exposes norepinephrine. MAP is current-block in 9/10 (90%) and is the cleaner proxy. |
| renal proxy | primary creatinine_last; fallback renal replacement therapy | artifacts/chapter1/model_ready/chapter1_primary_model_ready_dataset.csv; /Users/joanameyer/repository/icu-data-platform/artifacts/asic_harmonized/blocked/asic_8h_blocked_dynamic_features.csv | no | partial | primary 8/10 (80%); fallback 0/10 (0%) | READY | PRIMARY FEASIBLE. Creatinine is current-block in 3/10 (30%) and uses existing 48h LOCF in 5/10 (50%). No time-varying RRT field was found in the blocked or model-ready layers (none); static dialysis_free_days is not timepoint-aligned. |
| ventilation proxy | peep_last | artifacts/chapter1/model_ready/chapter1_primary_model_ready_dataset.csv | no | partial | 7/10 (70%) | READY | PRIMARY FEASIBLE. PEEP is current-block in 6/10 (60%) and uses within-window ventilator LOCF in 1/10 (10%). |

## Final Readiness Judgement
- age: `NOT READY`
- sex: `READY`
- coarse disease group: `READY`
- ICU time to last eligible prediction: `READY`
- site: `READY`
- respiratory proxy: `READY`
- hemodynamic proxy: `READY WITH FALLBACK`
- renal proxy: `READY`
- ventilation proxy: `READY`

- Overall judgement: `ISSUE 3.2 VARIABLE PACKAGE NOT YET READY`.
- Blocking variable family: `age`.
- Interpretation: the joins and the target population slice are reproducible now, and all non-age families are either ready or ready via the predefined MAP fallback. The package is not yet fully ready because exact age is absent from the current ASIC static layer; only categorical `age_group` is available.
