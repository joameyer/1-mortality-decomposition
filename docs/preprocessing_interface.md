# Chapter 1 Preprocessing Interface

## Purpose

This document separates:

- what the standalone Chapter 1 repo currently requires as input
- what it currently enforces itself
- what is scientifically intended but still unresolved at the interface level

## Current Required Upstream Artifacts

The code currently requires only the following standardized ASIC artifacts:

- `static/harmonized.{csv|parquet}`
- `dynamic/harmonized.{csv|parquet}`
- `blocked/asic_8h_block_index.{csv|parquet}`
- `blocked/asic_8h_blocked_dynamic_features.{csv|parquet}`
- `blocked/asic_8h_stay_block_counts.{csv|parquet}`
- `qc/mech_vent_ge_24h_stay_level.{csv|parquet}`
- `qc/mech_vent_ge_24h_episode_level.{csv|parquet}`

These inputs are assumed to be:

- adult-only upstream
- harmonized
- semantically cleaned upstream
- generically blocked upstream
- accompanied by upstream stay-level mechanical-ventilation QC for `>=24h`
- accompanied by upstream ventilation-supported episode windows for ventilator carry-forward restriction

## What Is Enforced In This Repo

The current standalone Chapter 1 code enforces:

- Chapter 1 site-level exclusion based on ICU mortality availability and core-vital coverage
- stay-level exclusion of any stay with missing `mech_vent_ge_24h_qc`
- stay-level exclusion of any stay with `mech_vent_ge_24h_qc == False`
- stay-level exclusion for site ineligibility
- stay-level exclusion for missing dynamic data
- stay-level exclusion for missing readmission and readmission-flagged stays
- exclusion of stays with missing or unusable ICU mortality labels
- valid-instance generation from generic 8-hour blocks using block structure, ICU-end proxy, and 3-of-4 core-group coverage within each block
- proxy within-horizon label generation using `icu_mortality` and `icu_end_time_proxy_hours`
- bounded preprocessing-time LOCF for configured feature families
- ventilator-variable LOCF restricted to blocks overlapping upstream ventilation-supported episodes
- explicit missingness indicators in the model-ready export
- canonical stay-level train/validation/test split assignment from the retained stay cohort, with hospital-preserving within-hospital splitting and stay-level outcome stratification as far as feasible
- feature-set-specific model-ready exports driven by `config/ch1_feature_sets.json`

## Trusted Upstream Assumptions

The frozen Chapter 1 analysis specification defines the scientific cohort as:

- adults only
- mechanically ventilated ICU stays only
- ventilation duration `>= 24h`

The current standalone preprocessing interface now treats these as follows:

- adult-only is trusted upstream and not rederived here
- mechanically ventilated `>=24h` is explicitly provided by upstream QC and enforced here

## Remaining Caveats

What remains unresolved is narrower now:

- the repo does not itself verify age; it assumes the upstream adult contract is already satisfied
- ASIC still lacks patient identifiers, so any downstream splitting remains effectively stay-level rather than patient-level
- horizon mortality labels remain proxy labels because true ICU discharge and death timestamps are unavailable in standardized ASIC artifacts
- no final imputation is performed during preprocessing export generation; that remains a downstream modeling-stage responsibility fit on the training split only
- exact `70/15/15` split proportions and exact within-hospital outcome balance may be impossible in very small hospital-specific strata, so the repo writes split-balance summaries and verification artifacts alongside the split assignments

## Working Interpretation

Treat the contract as:

- guaranteed upstream: adult-only harmonized and generically blocked ASIC artifacts plus stay-level `mech_vent_ge_24h_qc`
- enforced here: Chapter 1 site/stay/instance logic, bounded LOCF preprocessing, proxy label generation, and stay-level split assignment
- caveat: labels are proxy within-horizon labels and splitting remains stay-level

## Carry-Forward Contract

The Chapter 1 preprocessing export now applies this missingness policy:

- bounded LOCF only for prespecified feature families and windows
- no semantic substitutions such as deriving `spont_resp_rate` from `resp_rate`
- no global median imputation or other final imputation in preprocessing exports
- ventilator-variable LOCF only within upstream ventilation-supported episode windows
- explicit indicator columns marking whether each base feature was observed in-block, filled by LOCF, or still missing after LOCF

Any final imputation remains a downstream modeling-stage step and must be fit on the training split only.

## Split Contract

The primary Chapter 1 split design is:

- source cohort: the canonical retained Chapter 1 stay cohort artifact
- split unit: `stay_id_global`
- target proportions: `70%` train, `15%` validation, `15%` test
- hospital handling: split within each retained hospital and then pool across hospitals
- stratification target: stay-level `icu_mortality`, as far as feasible within each hospital
- reproducibility control: `split_random_seed` from `config/ch1_run_config.json`

All valid prediction instances and all labelable/model-ready rows inherit their split from the stay-level assignment table.
