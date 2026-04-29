# MIMIC Observation-Process Variable Derivation Rules

## Scope

This freezes the MIMIC observation-process variable set for Chapter 1 external validation and later hard-case sensitivity. The variables are interpretive observation-process checks, not primary mortality-model features.

## Design decision

MIMIC uses the same frozen ASIC observation-process concept set. The frozen set contains exactly nine variables: four current-block group indicators, one current-block completeness summary, and four group recency variables.

## Source data requirement

Variables must be derived from raw harmonized timestamped dynamic observations before blocking, LOCF, imputation, or model-ready feature construction. In the current MIMIC pipeline this is the adapter `dynamic/harmonized.csv` raw-history table built from preferred-source MIMIC event rows, with `minutes_since_admit = time_h * 60`.

## Group mapping

- HR group: `heart_rate`
- BP group: any of `sbp`, `dbp`, `map`
- Respiratory group: `resp_rate`
- Oxygenation group: any of `spo2`, `sao2`

The MIMIC harmonized names match the ASIC concept names used by `src/chapter1_mortality_decomposition/observation_process.py`. No unrelated variables are substituted.

## Variable definitions

`obs_hr_grp_block`, `obs_bp_grp_block`, `obs_resp_grp_block`, and `obs_oxy_grp_block` are 1 when at least one non-missing raw harmonized measurement from the corresponding group occurs within the current completed 8h block, otherwise 0.

`n_core_grps_obs_block` is the row-wise sum of the four block observation indicators and ranges from 0 to 4.

`tsl_hr_grp_h`, `tsl_bp_grp_h`, `tsl_resp_grp_h`, and `tsl_oxy_grp_h` are the hours from prediction time, equivalently block end, to the most recent prior non-missing raw harmonized measurement for that group. Measurements in the current block count, so current-block observations produce values from 0 to less than 8 hours under the half-open block convention.

## Missingness rules

- Block indicators are 0/1.
- `n_core_grps_obs_block` ranges 0-4.
- `tsl_*` remains missing if the group has never been observed before prediction time.
- No artificial large-value fill is used for `tsl_*`.

## Excluded variables

The frozen set excludes:

- broad per-feature observation counts,
- all-feature missingness fractions,
- longest-gap variables,
- cumulative density measures,
- rhythm irregularity metrics,
- lab-specific missingness inventories.

## Intended use

These variables are intended for later observation-process sensitivity and hard-case characterization only. They are not for primary risk-model training, primary cohort construction, prediction-label changes, valid-instance rule changes, or model evaluation.
