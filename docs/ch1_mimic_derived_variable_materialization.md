# Chapter 1 MIMIC Derived Variable Materialization

## Purpose

This note documents the narrow materialization of the already-frozen derived-only shared-primary MIMIC variables `pf_ratio` and `vt_per_kg_ibw`.

This is not a feature-selection revision, cohort revision, block redesign, label change, or model-fitting step. The goal is to make the frozen derived-only variables available to the existing MIMIC-to-ASIC adapter and reused ASIC preprocessing core.

## Pipeline Insertion Point

The derivations are applied in `src/chapter1_mortality_decomposition/mimic_blocks.py` after preferred-source raw event aggregation into completed 8h blocks and before `ch1_mimic_blocked_dynamic_features.csv` is written.

This keeps the downstream b3 adapter and ASIC preprocessing core unchanged: they continue to consume the existing `{variable}_{obs_count,mean,median,min,max,last}` blocked feature schema.

## `pf_ratio`

Formula:

```text
pf_ratio = pao2 / fio2_fraction
fio2_fraction = fio2_percent / 100
```

Sources:

- `pao2`: frozen preferred PaO2 block summaries from the preferred MIMIC source.
- `fio2`: frozen preferred FiO2 block summaries, stored on the percent scale after existing FiO2 normalization.

Each block-level statistic is derived from the same statistic of the support variables. For example:

```text
pf_ratio_last = pao2_last / (fio2_last / 100)
pf_ratio_mean = pao2_mean / (fio2_mean / 100)
```

If either PaO2 or FiO2 is missing for a block, the derived PF ratio for that block remains missing.

## `vt_per_kg_ibw`

Formula:

```text
vt_per_kg_ibw = vt_mL / IBW_kg
```

Sources:

- `vt`: frozen preferred `vt` block summaries from `224684 Tidal Volume (set)`.
- height: `226730 Height (cm)` and `226707 Height` from `icu.chartevents`.
- sex/gender: retained MIMIC cohort `gender`.

Height handling:

- `226730` is interpreted as centimeters.
- `226707` is interpreted as inches and converted to centimeters with `height_cm = height_in * 2.54`.
- adult-height support is restricted to 100-250 cm before deriving IBW.
- multiple height records per stay are collapsed to the stay-level median height.

IBW formula:

```text
male:   IBW_kg = 50.0 + 2.3 * (height_in - 60)
female: IBW_kg = 45.5 + 2.3 * (height_in - 60)
```

If height, sex/gender, IBW, or VT is missing, `vt_per_kg_ibw` remains missing. Actual body weight is not substituted.

## Block-Level Derivation Convention

The derivations are block-level translations from the already-frozen preferred source summaries. They are not timestamp-paired raw-event derivations and do not introduce sensitivity variants.

For each derived variable, `{variable}_obs_count` is set to 1 when `{variable}_last` is available in the block and 0 otherwise. The raw block row counters remain raw-observation counters and are not inflated by derived variables.

## Out Of Scope

- cohort changes
- structural block-grid changes
- source-preference redesign
- timestamp-paired derivation variants
- valid prediction-instance or label logic changes
- model fitting
