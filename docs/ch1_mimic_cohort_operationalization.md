# Chapter 1 MIMIC Cohort Operationalization

## Purpose

This document defines the 5.1.b1 stay-level MIMIC operationalization of the frozen Chapter 1 cohort. It implements cohort extraction and verification up to the retained ICU-stay table only.

This is not 8h block construction, horizon-label generation, valid prediction-instance filtering, or model fitting.

## Data Sources

Required MIMIC tables:

- `icu/icustays`
- `hosp/patients`
- `hosp/admissions`
- `icu/procedureevents`

ICU entry and exit are taken directly from `icustays.intime` and `icustays.outtime`; these timestamps are authoritative for this task.

## Adult Inclusion

Adults are defined as ICU stays with age at ICU admission at least 18 years.

Operational formula:

```text
age_at_icu_intime = patients.anchor_age + (year(icustays.intime) - patients.anchor_year)
```

Retain:

```text
age_at_icu_intime >= 18
```

## First ICU Stay

The cohort retains the first chronologic ICU stay per subject.

Operational SQL rule:

```sql
ROW_NUMBER() OVER (
    PARTITION BY subject_id
    ORDER BY intime ASC, stay_id ASC
) AS stay_rank
```

Retain:

```text
stay_rank = 1
```

Each `stay_id` remains a distinct ICU stay. Multiple ICU stays within the same hospitalization are not merged.

## Mechanical Ventilation >=24h

The primary MIMIC mechanical-ventilation gate uses invasive ventilation procedure time only.

Source:

- `icu.procedureevents`
- itemid `225792` = Invasive Ventilation

Operational formula:

```text
total_invasive_hours = sum(date_diff('minute', starttime, endtime)) / 60.0
```

implemented with SQLite `julianday` hour differences.

Retain:

```text
total_invasive_hours >= 24
```

QC guard:

```text
total_invasive_hours <= icu_los_hours + 4
```

Non-invasive ventilation itemid `225794` is not counted toward the primary >=24h gate. It is summarized only as QC/context.

## In-ICU Mortality

The primary stay-level outcome is ICU mortality.

Operational rule:

```text
icu_mortality = 1
if admissions.deathtime is non-null
and admissions.deathtime <= icustays.outtime
else 0
```

Hospital death after ICU discharge is not ICU mortality. `hospital_expire_flag` and death after ICU outtime are preserved only for QC/context.

## Discharge Location Clarification

`admissions.discharge_location` is a hospital-discharge disposition field, not an ICU-discharge outcome field. Therefore `discharge_location = DIED` is not the primary Chapter 1 ICU mortality definition.

The Chapter 1 ICU mortality definition remains:

```text
deathtime is non-null and deathtime <= icustays.outtime
```

Any discrepancy between discharge-location counts and ICU-mortality counts is expected because the fields describe different events and levels of care.

## Transfer And Discharge Handling

The extraction treats each `stay_id` as a distinct ICU stay.

Rules:

- Do not merge multiple ICU stays within the same hospitalization.
- First ICU stay means first chronologic `stay_id` per subject, not first hospitalization.
- Hospital death after ICU discharge remains outside the primary ICU mortality endpoint.
- Discharge location and later ICU stay within the same hospitalization are summarized for verification only; they are not exclusion rules.

No LOS >=48h, trauma, AMA, hospice, or discharge-location exclusions are applied in 5.1.b1.

## Deferred Valid Prediction-Instance Requirement

Valid prediction-instance eligibility is required later for final Chapter 1 preprocessing, but it is intentionally deferred beyond 5.1.b1.

It must not be enforced until after temporal block construction and horizon-specific label availability are known.

## Outputs

Extraction code:

- `src/chapter1_mortality_decomposition/mimic_cohort.py`
- `scripts/run_mimic_cohort.py`
- `config/ch1_mimic_cohort.yaml`

Aggregated verification reports:

- `reports/ch1_mimic_cohort_flow.csv`
- `reports/ch1_mimic_cohort_qc_summary.csv`
- `reports/ch1_mimic_transfer_discharge_summary.csv`
- `reports/ch1_mimic_ventilation_qc_addendum.csv`
- `reports/ch1_mimic_cohort_note.md`

The retained stay-level cohort is written under the configured processed directory:

- demo mode: `mimic-iv-demo/data/processed/ch1_mimic_stay_level_cohort.csv`
- full MIMIC: `.../mimic-iv-3.1/1-mortality-decomposition/processed/ch1_mimic_stay_level_cohort.csv`

This repo-local path is allowed only for MIMIC demo-derived outputs. Full-MIMIC retained stay-level cohort output is row-level private data and must be written to an external/private processed directory outside the project repo, for example under the private MIMIC root. The b1 runner fails before writing if a full-MIMIC run points `processed_dir` inside the repo. Safe aggregated reports may remain under `reports/`.

This retained table is an interim extraction artifact for later preprocessing, not a block, label, or model-ready dataset.
