# Chapter 1 MIMIC Preprocessing Adapter

## Purpose

This note documents subtask 5.1.b3: the thin MIMIC-to-ASIC input-contract adapter used to reuse the existing frozen ASIC Chapter 1 preprocessing core on MIMIC artifacts. This is not a separate MIMIC preprocessing pipeline and does not redesign carry-forward, LOCF, missingness handling, valid-instance construction, observation-process features, model-ready construction, or model fitting.

## Reuse Strategy

The adapter constructs the existing `Chapter1InputTables` contract and then calls the existing ASIC preprocessing entrypoint:

```text
build_chapter1_dataset(inputs: Chapter1InputTables, config=...)
write_chapter1_dataset(...)
```

The standardized adapter inputs are also written using the ASIC input layout so they can be inspected or loaded through the existing file contract:

- `static/harmonized.csv`
- `dynamic/harmonized.csv`
- `blocked/asic_8h_block_index.csv`
- `blocked/asic_8h_blocked_dynamic_features.csv`
- `blocked/asic_8h_stay_block_counts.csv`
- `qc/mech_vent_ge_24h_stay_level.csv`
- `qc/mech_vent_ge_24h_episode_level.csv`

## Identifier Mapping

MIMIC native identifiers are translated without changing the retained b1/b2 cohort or block definitions:

- `stay_id_global = string(stay_id)`
- `hospital_id = "MIMIC-IV"`

The MIMIC-native `subject_id`, `hadm_id`, and `stay_id` remain upstream provenance fields in the b1/b2 artifacts, but the reused ASIC preprocessing core receives the standardized identifiers above.

## Adapted Tables

`static_harmonized` is built from the retained MIMIC stay-level cohort. The adapter sets `icu_readmit = 0` because 5.1.b1 retains first chronologic ICU stays only. `icu_mortality` is copied from the stay-level ICU mortality endpoint. `icd10_codes` is populated from `hosp/diagnoses_icd.csv.gz` when available, using semicolon-separated `icd_version:icd_code` tokens by `hadm_id`; if that table is unavailable, the adapter writes a documented placeholder rather than omitting the required column.

`stay_block_counts` is adapted from the MIMIC b2 stay block-count artifact. `intime` becomes `icu_admission_time`, `outtime` becomes `icu_end_time_proxy`, and `icu_los_hours` becomes `icu_end_time_proxy_hours`.

`block_index` and `blocked_dynamic_features` are adapted from the MIMIC completed-block outputs. The block anchor, 8-hour width, half-open intervals, completed-block rule, and empty-block retention are unchanged from b2. The existing `{variable}_{obs_count,mean,median,min,max,last}` columns are preserved.

`mech_vent_stay_level_qc` is built from the retained invasive-ventilation gate/QC status in the b1 cohort and exposed as `mech_vent_ge_24h_qc`.

`mech_vent_episode_level` is rebuilt from `icu/procedureevents.csv.gz` using invasive ventilation itemid `225792` only. Episode start and end times are expressed as timedelta strings relative to ICU admission, matching the existing ASIC parser expectations.

`dynamic_harmonized` is built as a raw-history table for retained stays from the preferred-source MIMIC event rows already used in b2. It uses `minutes_since_admit = time_h * 60` and writes sparse variable columns for the source-resolved Chapter 1 variables so the existing dynamic-presence and observation-process logic can run unchanged.

## Storage Policy

Demo-derived adapter and preprocessing outputs may be written to repo-local configured paths. Full-MIMIC row-level adapter inputs and reused-preprocessing outputs must be written outside the project repo. The adapter enforces this by rejecting full-MIMIC runs where any of these paths resolve under the repo:

- retained cohort input
- MIMIC b2 block input root
- adapter standardized input root
- reused preprocessing output root

Safe aggregated reports remain under `reports/`.

## Reused Unchanged

The following existing ASIC preprocessing components are reused through `build_chapter1_dataset`:

- cohort construction from `static_harmonized`
- valid-instance construction
- proxy horizon label construction as implemented by the existing core
- observation-process features
- carry-forward and LOCF logic
- model-ready table construction
- artifact writing through `write_chapter1_dataset`

## Validation

The adapter was run on MIMIC demo-derived b1/b2 artifacts under `/tmp/ch1_mimic_adapter_demo`. It created all seven required `Chapter1InputTables` components, all required columns passed the contract check, and the existing ASIC preprocessing core ran successfully on the adapted MIMIC inputs.

The repo-local validation reports are:

- `reports/ch1_mimic_adapter_contract_check.csv`
- `reports/ch1_mimic_adapter_qc_summary.csv`
- `reports/ch1_mimic_preprocessing_adapter_note.md`

## Remaining Caveats

No contract mismatch remained in the demo validation. The adapter keeps the `icd10_codes` behavior explicit: codes are joined from MIMIC diagnoses when available, otherwise a placeholder status is reported. Because the reused ASIC core also produces proxy horizon labels and model-ready outputs, final scientific use of those outputs on full MIMIC still requires the planned downstream review of MIMIC label semantics and valid-instance counts. This task does not fit models.
