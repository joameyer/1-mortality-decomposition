# Chapter 1 Preprocessing Reuse Check for MIMIC

## Purpose

This note checks whether the existing frozen ASIC Chapter 1 preprocessing logic can be reused for the newly constructed MIMIC 8-hour block artifacts. It is an input-contract and compatibility check only. It does not implement MIMIC preprocessing, build labels, fit models, or fork the ASIC preprocessing logic.

## Files and Functions Inspected

Active code and config inspected:

- `src/chapter1_mortality_decomposition/cli.py`
- `src/chapter1_mortality_decomposition/artifacts.py`
- `src/chapter1_mortality_decomposition/pipeline.py`
- `src/chapter1_mortality_decomposition/cohort.py`
- `src/chapter1_mortality_decomposition/instances.py`
- `src/chapter1_mortality_decomposition/labels.py`
- `src/chapter1_mortality_decomposition/observation_process.py`
- `src/chapter1_mortality_decomposition/carry_forward.py`
- `src/chapter1_mortality_decomposition/model_ready.py`
- `src/chapter1_mortality_decomposition/config.py`
- `src/chapter1_mortality_decomposition/mimic_blocks.py`
- `docs/ch1_asic_block_logic_recovery.md`
- `docs/ch1_mimic_block_construction.md`
- `config/ch1_mimic_blocks.yaml`

The private full-MIMIC processed block artifacts outside the repo were not read for this check. The comparison uses repo-local code, notes, configs, and known writer schemas.

## Identified ASIC Preprocessing Entrypoints

The standalone entrypoint is:

```text
python -m chapter1_mortality_decomposition --input-dir <standardized_input_dir> --output-dir <output_dir>
```

Implementation path:

```text
chapter1_mortality_decomposition.cli.main
  -> pipeline.build_and_write_chapter1_dataset
  -> artifacts.load_chapter1_inputs
  -> pipeline.build_chapter1_dataset
```

The in-process reusable entrypoint is:

```python
build_chapter1_dataset(inputs: Chapter1InputTables, config: Chapter1Config | None = None)
```

This is the core logic to reuse after an adapter constructs a `Chapter1InputTables` object or equivalent standardized input directory.

## ASIC Input Contract

`load_chapter1_inputs` expects a standardized ASIC-style input directory with:

- `static/harmonized.csv`
- `dynamic/harmonized.csv`
- `blocked/asic_8h_block_index.csv`
- `blocked/asic_8h_blocked_dynamic_features.csv`
- `blocked/asic_8h_stay_block_counts.csv`
- `qc/mech_vent_ge_24h_stay_level.csv`
- `qc/mech_vent_ge_24h_episode_level.csv`

The loaded object is `Chapter1InputTables` with:

- `static_harmonized`
- `dynamic_harmonized`
- `block_index`
- `blocked_dynamic_features`
- `stay_block_counts`
- `mech_vent_stay_level_qc`
- `mech_vent_episode_level`

### Required Identifiers

The ASIC preprocessing code consistently expects:

- `stay_id_global`
- `hospital_id`

These are required by cohort construction, valid-instance construction, observation-process features, carry-forward, splits, and model-ready construction.

### Static Contract

`build_chapter1_cohort` requires `static_harmonized` columns:

- `stay_id_global`
- `hospital_id`
- `icu_readmit`
- `icu_mortality`
- `icd10_codes`

The code renames `icu_readmit` to `readmission`, assumes adult age was handled upstream, and excludes readmission/missing-readmission records.

### Stay-Block Contract

`stay_block_counts` requires:

- `stay_id_global`
- `hospital_id`
- `icu_admission_time`
- `icu_end_time_proxy`
- `icu_end_time_proxy_hours`

Extra block-count fields may be present and are tolerated.

### Blocked Feature Contract

`block_index` and `blocked_dynamic_features` require:

- `stay_id_global`
- `hospital_id`
- `block_index`
- `block_start_h`
- `block_end_h`
- `prediction_time_h`

`blocked_dynamic_features` must include per-base-variable features named:

- `{base_variable}_obs_count`
- `{base_variable}_mean`
- `{base_variable}_median`
- `{base_variable}_min`
- `{base_variable}_max`
- `{base_variable}_last`

The current MIMIC b2 aggregation statistics match this naming and statistic contract after ID adaptation.

### Dynamic Raw-History Contract

`dynamic_harmonized` is still required after blocked construction. It is used for:

- site-level dynamic data presence in `cohort.py`
- site-level core-vital group coverage in `cohort.py`
- observation-process features in `observation_process.py`

Required columns include:

- `stay_id_global`
- `hospital_id`
- `minutes_since_admit`
- at least one variable column for each observation-process group:
  - cardiac rate: `heart_rate`
  - blood pressure: one or more of `map`, `sbp`, `dbp`
  - respiratory: `resp_rate`
  - oxygenation: one or more of `spo2`, `sao2`

The observation-process code uses raw history with `minutes_since_admit < prediction_time_h * 60`.

### Mechanical Ventilation QC Contract

`mech_vent_stay_level_qc` requires:

- `stay_id_global`
- `hospital_id`
- `mech_vent_ge_24h_qc`

`mech_vent_episode_level` is required for LOCF ventilation-window logic and must include:

- `stay_id_global`
- `hospital_id`
- `episode_start_time`
- `episode_end_time`

`episode_start_time` and `episode_end_time` are parsed as timedeltas and converted to hours since ICU admission.

## MIMIC Current State

The MIMIC 5.1.b1 and 5.1.b2 artifacts currently provide:

- retained stay-level cohort table with MIMIC-native identifiers:
  - `subject_id`
  - `hadm_id`
  - `stay_id`
  - `intime`
  - `outtime`
  - `icu_los_hours`
  - `icu_mortality`
  - ventilation duration/QC fields
- structural 8-hour block grid with:
  - `subject_id`
  - `hadm_id`
  - `stay_id`
  - `block_index`
  - `block_start_h`
  - `block_end_h`
  - `prediction_time_h`
  - `completed_block_count`
- blocked dynamic summaries with ASIC-compatible feature statistic suffixes, but MIMIC-native identifiers
- stay block counts with MIMIC-native identifiers plus `intime`, `outtime`, `icu_los_hours`, and completed-block metadata
- block source-resolution reports confirming preferred-source aggregation

The current b2 outputs intentionally do not provide an ASIC-style `dynamic/harmonized.csv` raw-history table or `qc/mech_vent_ge_24h_episode_level.csv`.

## Structured Comparison

The existing preprocessing logic cannot consume the MIMIC b2 outputs unchanged because the file layout and identifier columns do not match `load_chapter1_inputs`.

The core blocked feature content is close: the block grid and per-variable statistic naming are compatible after renaming IDs. The missing pieces are adapter-level standardized inputs around those blocks, not a reason to fork preprocessing logic.

The necessary differences are translational:

- map MIMIC identifiers to ASIC contract identifiers
- write or construct the standardized input directory/object expected by `Chapter1InputTables`
- convert the retained MIMIC cohort to `static_harmonized`, `stay_block_counts`, and `mech_vent_stay_level_qc`
- construct a raw MIMIC `dynamic_harmonized` table for observation-process features and site/dynamic presence checks
- construct invasive-ventilation episode rows in the ASIC timedeltas format for carry-forward ventilation-window logic

## Final Verdict

`thin_adapter_needed`

The existing ASIC preprocessing code should remain the main preprocessing implementation. A thin MIMIC adapter is needed to satisfy the same `Chapter1InputTables` contract. A separate MIMIC preprocessing pipeline is not justified by the current compatibility check.

## Minimum Adapter Steps

1. Define a stable MIMIC `stay_id_global`, for example the string form of `stay_id`, and set a single `hospital_id`, for example `MIMIC-IV`.

2. Adapt `static_harmonized` from the retained stay-level cohort:
   - `stay_id_global`
   - `hospital_id`
   - `icu_readmit = 0` for the retained first-stay cohort
   - `icu_mortality`
   - `icd10_codes`, populated from MIMIC diagnoses later if needed, or a documented placeholder if disease-stratified downstream analyses are not run in this pass

3. Adapt `stay_block_counts`:
   - `stay_id_global`
   - `hospital_id`
   - `icu_admission_time = intime`
   - `icu_end_time_proxy = outtime`
   - `icu_end_time_proxy_hours = icu_los_hours`
   - keep completed-block metadata as extra tolerated fields

4. Adapt `block_index` and `blocked_dynamic_features`:
   - rename/add `stay_id_global`
   - add `hospital_id`
   - preserve `block_index`, `block_start_h`, `block_end_h`, `prediction_time_h`
   - preserve `{base_variable}_{statistic}` feature columns

5. Adapt `mech_vent_stay_level_qc` from the retained cohort:
   - `stay_id_global`
   - `hospital_id`
   - `mech_vent_ge_24h_qc`, equivalent to the retained invasive-ventilation gate/QC result

6. Build `mech_vent_episode_level` from MIMIC invasive ventilation procedure episodes:
   - `stay_id_global`
   - `hospital_id`
   - `episode_start_time` as timedelta since ICU admission
   - `episode_end_time` as timedelta since ICU admission

7. Build `dynamic_harmonized` raw history for retained stays:
   - `stay_id_global`
   - `hospital_id`
   - `minutes_since_admit`
   - raw or source-resolved variable columns needed for observation-process groups, at minimum `heart_rate`, blood pressure variables, `resp_rate`, and oxygenation variables
   - preferably all frozen shared-primary variables if the same feature inventory is wanted for site/dynamic checks

8. Either write these as a standardized input directory with the exact ASIC filenames or instantiate `Chapter1InputTables` directly and call `build_chapter1_dataset`.

## Runtime Checks Still Needed

After the adapter exists, run a controlled demo/safe test to verify:

- all required files/columns load through `load_chapter1_inputs` or `Chapter1InputTables`
- retained MIMIC cohort count remains aligned with 5.1.b1
- completed block count remains aligned with 5.1.b2
- valid-instance counts are generated without changing block construction
- observation-process and LOCF ventilation-window checks pass
- model-ready outputs are written only to a storage-safe location for full MIMIC

## Boundary Note

The full `build_chapter1_dataset` entrypoint also builds proxy horizon labels and model-ready tables. This reuse check does not run those steps. If issue 5.1.b3 is limited to preprocessing before labels, the same internal functions can still be reused selectively, but the adapter contract remains the same.
