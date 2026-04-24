# Chapter 1 MIMIC Stay-Level Cohort Extraction Note

## Purpose

This report documents the 5.1.b1 stay-level MIMIC operationalization of the frozen Chapter 1 cohort. It is not 8h block construction, horizon-label generation, valid prediction-instance filtering, or model fitting.

## Data Source

- MIMIC root: `/Users/joanameyer/data/mimic-iv/mimic-iv-3.1`
- ICU stays considered: 94458
- Retained stay-level cohort rows: 10648
- Retained in-ICU mortality count: 2210

## Implemented Gates

- Adult inclusion: `age_at_icu_intime = anchor_age + (year(icustays.intime) - anchor_year)`, retain age >= 18.
- First ICU stay: `ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY intime ASC, stay_id ASC)`, retain `stay_rank = 1`.
- Mechanical ventilation: sum invasive ventilation procedure time for itemid `225792`; retain `total_invasive_hours >= 24`.
- Ventilation QC guard: retain `total_invasive_hours <= icu_los_hours + 4`.
- ICU mortality: `admissions.deathtime` non-null and `deathtime <= icustays.outtime`.
- ICU entry/exit: `icustays.intime` and `icustays.outtime` are authoritative.
- Transfer/discharge handling: each `stay_id` remains distinct; multiple ICU stays are not merged.

Non-invasive ventilation itemid `225794` is summarized only for QC/context and is not counted toward the >=24h gate.

No LOS >=48h, trauma, AMA, hospice, or discharge-location exclusions are applied in 5.1.b1.

## Discharge Location Clarification

`admissions.discharge_location` is a hospital-discharge disposition field, not an ICU-discharge outcome field. Therefore `discharge_location = DIED` is not the primary Chapter 1 ICU mortality definition. The ICU mortality definition remains `deathtime` non-null and `deathtime <= icustays.outtime`. Differences between discharge-location counts and ICU-mortality counts are expected because these fields describe different event levels.

## Ventilation Timing QC Addendum

Ventilation-vs-LOS edge cases are limited in the retained cohort: 1 of 10648 retained stays have ICU LOS <24h; 164 have summed invasive ventilation time greater than ICU LOS; 6 exceed ICU LOS by more than 2h; and 0 exceed ICU LOS by more than 4h. The current +4h QC guard therefore appears sufficient for this stay-level gate. Short-LOS retained stays and small positive timing differences should be carried forward as documented procedure-timing limitations rather than triggering a cohort rule change.

## Deferred Requirement

Valid prediction-instance eligibility is required later for final Chapter 1 preprocessing, but is intentionally not enforced in 5.1.b1 because it depends on block construction and horizon-specific label availability.

## Outputs

- `reports/ch1_mimic_cohort_flow.csv`
- `reports/ch1_mimic_cohort_qc_summary.csv`
- `reports/ch1_mimic_transfer_discharge_summary.csv`
- `reports/ch1_mimic_ventilation_qc_addendum.csv`
- `/Users/joanameyer/repository/1-mortality-decomposition/data/processed/ch1_mimic_stay_level_cohort.csv`
