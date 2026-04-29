# MIMIC Observation-Process Implementation Status

## What was frozen

The frozen MIMIC observation-process variable set contains exactly:

- `obs_hr_grp_block`
- `obs_bp_grp_block`
- `obs_resp_grp_block`
- `obs_oxy_grp_block`
- `n_core_grps_obs_block`
- `tsl_hr_grp_h`
- `tsl_bp_grp_h`
- `tsl_resp_grp_h`
- `tsl_oxy_grp_h`

## What was verified

The required MIMIC harmonized raw-history variables were found:

| Harmonized variable | Group | Source/timestamp/value fields | Availability | Direct ASIC group mapping |
|---|---|---|---|---|
| `heart_rate` | HR | `icu.chartevents`; `charttime`; `valuenum`; adapter `minutes_since_admit` | Non-missing raw values available | Yes |
| `sbp` | BP | `icu.chartevents`; `charttime`; `valuenum`; adapter `minutes_since_admit` | Non-missing raw values available | Yes |
| `dbp` | BP | `icu.chartevents`; `charttime`; `valuenum`; adapter `minutes_since_admit` | Non-missing raw values available | Yes |
| `map` | BP | `icu.chartevents`; `charttime`; `valuenum`; adapter `minutes_since_admit` | Non-missing raw values available | Yes |
| `resp_rate` | Respiratory | `icu.chartevents`; `charttime`; `valuenum`; adapter `minutes_since_admit` | Non-missing raw values available | Yes |
| `spo2` | Oxygenation | `icu.chartevents`; `charttime`; `valuenum`; adapter `minutes_since_admit` | Non-missing raw values available | Yes |
| `sao2` | Oxygenation | preferred `icu.chartevents` with secondary `hosp.labevents` documented; `charttime`; `valuenum`; adapter `minutes_since_admit` | Non-missing raw values available | Yes |

The detailed source-variable check is in `mimic_observation_process_source_variable_check.csv`. Existing full-MIMIC b2 source-resolution reports show retained preferred-source rows for all seven required harmonized variables. The demo adapter raw-history table also contains all seven columns.

## What remains for implementation

Derivation code already exists in `src/chapter1_mortality_decomposition/observation_process.py` and is invoked by the Chapter 1 preprocessing pipeline after the MIMIC adapter constructs raw-history `dynamic/harmonized.csv`. The existing code derives only the frozen nine observation-process columns at the completed 8h block level, preserves missing `tsl_*` for never-observed groups, and writes observation-process outputs separately from the primary model-ready tables.

No new standalone producer script was added in this issue. A separate script is not required for the freeze because the current repo convention routes MIMIC through the reused Chapter 1 preprocessing pipeline. If a later producer issue wants a direct command-line wrapper, it should call the existing `build_chapter1_observation_process_features` function and use the adapter `dynamic/harmonized.csv` plus `instances/chapter1_valid_instances.csv` or the unique completed-block index as inputs.

## Not done in this issue

- no hard-case analysis,
- no model evaluation,
- no primary feature-set changes,
- no broad missingness feature expansion.
