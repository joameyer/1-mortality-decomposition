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

These inputs are assumed to be:

- adult-only upstream
- harmonized
- semantically cleaned upstream
- generically blocked upstream
- accompanied by upstream stay-level mechanical-ventilation QC for `>=24h`

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

## Working Interpretation

Treat the contract as:

- guaranteed upstream: adult-only harmonized and generically blocked ASIC artifacts plus stay-level `mech_vent_ge_24h_qc`
- enforced here: Chapter 1 site/stay/instance logic listed above
- caveat: labels are proxy within-horizon labels and splitting remains stay-level
