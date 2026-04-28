# Chapter 1 MIMIC Preprocessing Adapter Note

## Purpose

This report documents subtask 5.1.b3: a thin MIMIC-to-ASIC input-contract adapter for reusing the existing frozen ASIC Chapter 1 preprocessing core.

## Storage

- MIMIC root: `/Users/joanameyer/data/mimic-iv/mimic-iv-3.1`
- Adapter standardized input root: `/Users/joanameyer/data/mimic-iv/mimic-iv-3.1/1-mortality-decomposition/asic_contract_inputs`
- Reused preprocessing output root: `/Users/joanameyer/data/mimic-iv/mimic-iv-3.1/1-mortality-decomposition/preprocessing_outputs`
- Full-MIMIC row-level adapter and preprocessing outputs must remain outside the repo.

## Adapter Decisions

- `stay_id_global = string(stay_id)`
- `hospital_id = MIMIC-IV`
- `icu_readmit = 0` for the retained first-stay MIMIC cohort.
- `icd10_codes` status: `diagnoses_icd_joined`
- `dynamic_harmonized` is built from preferred source/item MIMIC event rows, with `minutes_since_admit = time_h * 60`.
- `mech_vent_episode_level` uses invasive ventilation procedureevents itemid `225792` only, expressed as timedeltas since ICU admission.

## Reuse Result

The existing ASIC preprocessing core was successfully run on the adapted full-MIMIC inputs. This note was restored from the known full-run storage locations after a later demo run overwrote the repo-local markdown report; the expensive b3 preprocessing step was not rerun for this restoration.

## Deferred Beyond b3

- model fitting
- any redesign of LOCF, valid-instance, observation-process, or model-ready logic
- review of proxy horizon label semantics for final external-validation reporting
