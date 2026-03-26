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

These inputs are assumed to be:

- harmonized
- semantically cleaned upstream
- generically blocked upstream

## What Is Enforced In This Repo

The current standalone Chapter 1 code enforces:

- Chapter 1 site-level exclusion based on ICU mortality availability and core-vital coverage
- stay-level exclusion for site ineligibility
- stay-level exclusion for missing dynamic data
- stay-level exclusion for missing readmission and readmission-flagged stays
- exclusion of stays with missing ICU mortality labels
- valid-instance generation from generic 8-hour blocks
- proxy within-horizon label generation using `icu_mortality` and `icu_end_time_proxy_hours`

## Scientifically Intended But Not Yet Guaranteed By The Current Interface

The frozen Chapter 1 analysis specification defines the scientific cohort as:

- adults only
- mechanically ventilated ICU stays only
- ventilation duration `>= 24h`

The current standalone preprocessing interface does not yet require standardized columns that make those restrictions enforceable here.

Therefore:

- this repo does not currently prove that the adult-only restriction has already been applied upstream
- this repo does not currently prove that the ventilation-duration `>= 24h` restriction has already been applied upstream
- this remains an unresolved interface issue that must be settled before continued Chapter 1 preprocessing work

## Working Interpretation

Until the interface is finalized, treat the contract as:

- guaranteed upstream: harmonized and generically blocked ASIC artifacts
- enforced here: Chapter 1 site/stay/instance logic listed above
- unresolved: whether adult-only and ventilated-`>=24h` cohort restrictions are guaranteed upstream or still need repo-local enforcement once standardized columns exist
