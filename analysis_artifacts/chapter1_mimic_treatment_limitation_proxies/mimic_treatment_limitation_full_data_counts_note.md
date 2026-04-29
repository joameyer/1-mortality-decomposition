# MIMIC Treatment-Limitation Proxy Full-Data Counts Note

## Scope

This is aggregate full-data counting for reviewed 5.2a structured proxy candidates. It does not export row-level patient data, analyze low-predicted fatal cases, add proxies to risk models, change the Chapter 1 cohort definition, exclude proxy-positive patients, use notes/NLP, or write a final feasibility verdict.

## Inputs

- Candidate inventory: `/Users/joanameyer/repository/1-mortality-decomposition/analysis_artifacts/chapter1_mimic_treatment_limitation_proxies/mimic_treatment_limitation_proxy_inventory_schema_scan.csv`
- Chapter 1 cohort artifact: `/Users/joanameyer/data/mimic-iv/mimic-iv-3.1/1-mortality-decomposition/processed/ch1_mimic_stay_level_cohort.csv`
- MIMIC root: `/Users/joanameyer/data/mimic-iv/mimic-iv-3.1`
- Retained Chapter 1 MIMIC stays used as denominator: 10648
- Fatal retained stays by established `icu_mortality`: 2210

## Approved candidates counted

- `brain_death_or_organ_donation`: `icu.d_items`
- `code_status_dnr_dni`: `icu.d_items`
- `palliative_care`: `icu.d_items`
- `brain_death_or_organ_donation`: `hosp.d_icd_diagnoses`
- `code_status_dnr_dni`: `hosp.d_icd_diagnoses`
- `hospice`: `hosp.admissions`
- `palliative_care`: `hosp.d_icd_diagnoses`
- `ama_or_nonstandard_discharge`: `hosp.admissions`

## Excluded or deferred candidates

Deferred by review rules or missing source availability:

- Family Meeting | Family Meeting | Restraint/Support Systems: 5.2a candidate remained needs_review and was not approved by the 5.2b review instructions
- Family meeting held | Family meeting held | 7-Communication: 5.2a candidate remained needs_review and was not approved by the 5.2b review instructions
- Family meeting attempted, unable | Family meeting attempted, unable | 7-Communication: 5.2a candidate remained needs_review and was not approved by the 5.2b review instructions
- Code status | Full code  (attempt resuscitation): 5.2a candidate remained needs_review and was not approved by the 5.2b review instructions
- Code status | Resuscitate (Full code): 5.2a candidate remained needs_review and was not approved by the 5.2b review instructions
- 64 inventory rows with decision_preliminary=exclude: Rejected false positives or outcome fields were not counted
- General Care | Code status | Inactive: POE order source row lacks limitation-positive value; POE detail values are counted separately when available

Skipped during counting:

- Code status | DNAR (DO NOT attempt resuscitation for cardiac arrest): Skipped: hosp.poe and/or hosp.poe_detail unavailable in this MIMIC root.
- Code status | Do not resuscitate (DNR/DNI): Skipped: hosp.poe and/or hosp.poe_detail unavailable in this MIMIC root.
- Consults | Palliative Care | Inactive: Skipped: hosp.poe and/or hosp.poe_detail unavailable in this MIMIC root.
- Consults | Palliative Care/Ethics Support | Inactive: Skipped: hosp.poe and/or hosp.poe_detail unavailable in this MIMIC root.

## Prevalence summary

- `code_status_dnr_dni`: 2638 stays (0.247746); fatal stays 1386 (0.627149).
- `palliative_care`: 1806 stays (0.169609); fatal stays 1129 (0.51086).
- `hospice`: 231 stays (0.021694); fatal stays 1 (0.000452).
- `brain_death_or_organ_donation`: 146 stays (0.013711); fatal stays 128 (0.057919).
- `ama_or_nonstandard_discharge`: 0 stays (0.0); fatal stays 0 (0.0).

Code-status counts are limitation-positive only. Full-code/resuscitate-only values were kept separate from treatment-limitation positivity and were not counted as `code_status_dnr_dni` proxy-positive stays.

## Timing summary

- `brain_death_or_organ_donation` `Brain Death | Brain Death | 3-Significant Events`: stay_timed_structured_proxy; horizon-specific prediction-time check still required
- `code_status_dnr_dni` `Code Status | Code Status | General`: stay_timed_structured_proxy; horizon-specific prediction-time check still required
- `code_status_dnr_dni` `Code Status. | Code Status. | MD Progress Note`: stay_timed_structured_proxy; horizon-specific prediction-time check still required
- `code_status_dnr_dni` `Code Status (Intubation) | Code Status (Intubation) | Intubation`: timing_not_available
- `palliative_care` `Palliative Care NCP - Expected outcomes | Palliative Care NCP - Expected outcomes | Care Plans`: stay_timed_structured_proxy; horizon-specific prediction-time check still required
- `palliative_care` `Palliative Care NCP - Interventions | Palliative Care NCP - Interventions | Care Plans`: stay_timed_structured_proxy; horizon-specific prediction-time check still required
- `palliative_care` `Palliative Care NCP - Plan revised | Palliative Care NCP - Plan revised | Care Plans`: stay_timed_structured_proxy; horizon-specific prediction-time check still required
- `palliative_care` `Palliative Care NCP - Problem resolved | Palliative Care NCP - Problem resolved | Care Plans`: stay_timed_structured_proxy; horizon-specific prediction-time check still required
- `brain_death_or_organ_donation` `Brain death`: timing_not_available
- `brain_death_or_organ_donation` `Brain death`: timing_not_available
- `code_status_dnr_dni` `Do not resuscitate`: timing_not_available
- `code_status_dnr_dni` `Do not resuscitate status`: timing_not_available
- `code_status_dnr_dni` `Code status | DNAR (DO NOT attempt resuscitation for cardiac arrest)`: timing_not_available
- `code_status_dnr_dni` `Code status | Do not resuscitate (DNR/DNI)`: timing_not_available
- `hospice` `HOSPICE`: post_event_or_discharge_context
- `palliative_care` `Encounter for palliative care`: timing_not_available
- `palliative_care` `Encounter for palliative care`: timing_not_available
- `palliative_care` `Consults | Palliative Care | Inactive`: timing_not_available
- `palliative_care` `Consults | Palliative Care/Ethics Support | Inactive`: timing_not_available
- `ama_or_nonstandard_discharge` `AMA / against medical advice / elopement discharge values`: timing_not_available

Timing summaries use only available structured timestamps and established cohort anchors (`intime`, `outtime`, and `deathtime` where present). Unsupported timing anchors are not inferred.

## Data-boundary statement

No row-level patient or stay-level proxy-positive data were exported. The output CSV and JSON files contain aggregate counts and aggregate timing summaries only.

## Files produced

- `/Users/joanameyer/repository/1-mortality-decomposition/analysis_artifacts/chapter1_mimic_treatment_limitation_proxies/mimic_treatment_limitation_proxy_full_data_counts.csv`
- `/Users/joanameyer/repository/1-mortality-decomposition/analysis_artifacts/chapter1_mimic_treatment_limitation_proxies/mimic_treatment_limitation_proxy_timing_summary.csv`
- `/Users/joanameyer/repository/1-mortality-decomposition/analysis_artifacts/chapter1_mimic_treatment_limitation_proxies/mimic_treatment_limitation_proxy_domain_summary.csv`
- `/Users/joanameyer/repository/1-mortality-decomposition/analysis_artifacts/chapter1_mimic_treatment_limitation_proxies/mimic_treatment_limitation_full_data_counts_note.md`
- `/Users/joanameyer/repository/1-mortality-decomposition/analysis_artifacts/chapter1_mimic_treatment_limitation_proxies/manifest_full_data_counts.json`
