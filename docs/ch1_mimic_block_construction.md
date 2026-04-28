# Chapter 1 MIMIC Completed 8h Block Construction

## Purpose

This note documents subtask 5.1.b2: MIMIC completed 8-hour block construction for the retained Chapter 1 stay-level cohort. The implementation mirrors the recovered frozen ASIC blocked-data logic and is a translation layer only. It does not build valid prediction instances, carry-forward features, horizon labels, model-ready tables, or models.

## Authoritative Logic Mirrored

The source of truth is the recovered ASIC block logic in:

- `docs/ch1_asic_block_logic_recovery.md`
- `config/ch1_asic_block_logic_recovery.yaml`

The MIMIC implementation follows the ASIC staging:

1. create structural completed blocks
2. assign raw measurements to completed blocks
3. aggregate current-block values
4. defer sufficiency filtering, carry-forward, labels, and model-ready construction

## MIMIC Translation

### Elapsed-Time Anchor

MIMIC elapsed time is anchored at ICU entry using the retained stay-level cohort field `intime`, which was produced in 5.1.b1 from `icustays.intime`.

For every raw event row:

```text
time_h = (event_time - intime) in hours
```

Rows with missing event timestamps or `time_h < 0` are not assigned to blocks.

### Block Width

The block width is fixed at 8 hours. The runner rejects other widths for this b2 artifact.

### Interval Convention

Intervals are half-open:

```text
[block_start_h, block_end_h)
```

Assignment uses:

```text
block_index = floor(time_h / 8)
```

Therefore an event exactly at `8h` is assigned to block 1, not block 0. Events assigned beyond the structurally completed block grid are dropped from the b2 block aggregation.

### Completed Block Count

For each retained stay:

```text
completed_block_count = floor(icu_los_hours / 8)
```

where `icu_los_hours` comes from the retained 5.1.b1 stay-level cohort using `intime` and `outtime`.

The structural block grid emits block indices:

```text
0 ... completed_block_count - 1
```

### Prediction Time

Prediction time is the block end:

```text
prediction_time_h = block_end_h
```

### Empty Blocks

Structurally completed blocks are retained even if no raw dynamic rows are assigned. Such blocks have zero count columns and missing summary statistics.

### Current-Block Sufficiency

No current-block observation sufficiency rule is applied in b2. The ASIC-compatible core-vital group sufficiency rule is deferred to valid-instance construction.

## Raw MIMIC Sources Included

The b2 implementation currently assigns accepted frozen shared-primary raw candidates from:

- `icu.chartevents`
- `hosp.labevents`

The retained cohort table provides the stay grid and ICU entry/exit times:

- demo mode: `mimic-iv-demo/data/processed/ch1_mimic_stay_level_cohort.csv`
- full MIMIC: `.../mimic-iv-3.1/1-mortality-decomposition/processed/ch1_mimic_stay_level_cohort.csv`

Demoted `bicarbonate_art` is not part of the shared-primary blocked feature output. Frozen derived-only shared-primary variables are materialized after preferred-source block aggregation:

- `pf_ratio = pao2 / (fio2 / 100)`, using block-level PaO2 and FiO2 summaries from the frozen preferred sources.
- `vt_per_kg_ibw = vt_mL / IBW_kg`, using block-level preferred VT summaries and stay-level IBW derived from height and sex/gender support.

The derivations are block-level translations, not timestamp-paired raw-event derivations. The derived `obs_count` is set to 1 when the derived block `last` value is available and 0 otherwise.

## Source Preference

B2 applies preferred source/item filtering before main block aggregation. The preferences are read from `config/ch1_asic_to_mimic_variable_map.csv`:

- the preferred source family comes from `mimic_primary_table`
- preferred itemids are parsed from the first preferred/primary clause of `candidate_itemids_or_source_logic`
- secondary chart/lab mirror rows are excluded from the main block summaries when the preferred source is available
- if a preferred source has zero available rows for a variable in a given run, the implementation may retain the available secondary accepted source as the only available source and records that fallback in QC

This keeps mirror-source duplication out of `{variable}_obs_count`, `{variable}_mean`, `{variable}_median`, `{variable}_min`, `{variable}_max`, and `{variable}_last`. Broader secondary sources remain documented for later sensitivity or provenance review.

## Storage Safety Policy

Demo-derived processed block artifacts may be written under the repo-local `mimic-iv-demo/data/processed/` path.

Full-MIMIC-derived row-level or block-level processed artifacts must be written outside the project repo. The block runner enforces this: if `mimic_root` is not the local demo root and the processed output root resolves inside this repo, the run fails before scanning MIMIC event tables. Use `--processed-output-root` to point full-MIMIC block outputs to a private external location, for example under the local MIMIC root.

Safe aggregated QC reports may remain under `reports/` in this repo.

## Aggregation

For each structural completed block, the output includes:

- `dynamic_row_count`
- `non_missing_measurements_in_block`
- `observed_variables_in_block`

For each shared-primary base variable, the output includes:

- `{variable}_obs_count`
- `{variable}_mean`
- `{variable}_median`
- `{variable}_min`
- `{variable}_max`
- `{variable}_last`

The `last` statistic uses stable ordering by:

```text
subject_id, hadm_id, stay_id, block_index, time_h, source_row_order
```

where `source_row_order` is assigned during chunked source loading. The ordering is applied after preferred source/item filtering.

## Derived-Only Materialization

`pf_ratio` and `vt_per_kg_ibw` are frozen as shared-primary `derived_only` variables and are now materialized in b2 so the existing b3 ASIC preprocessing core can consume them without a MIMIC-specific preprocessing fork.

For `pf_ratio`, FiO2 values are stored in percent in the blocked FiO2 summaries. The derivation converts FiO2 to a fraction by dividing by 100 before the ratio. Each statistic is derived from the same statistic of the support inputs, for example `pf_ratio_last = pao2_last / (fio2_last / 100)`.

For `vt_per_kg_ibw`, height support is loaded from `226730 Height (cm)` and `226707 Height` in inches, converted to cm, restricted to 100-250 cm, and collapsed to a stay-level median height. IBW uses the Devine adult formula:

```text
male:   IBW_kg = 50.0 + 2.3 * (height_in - 60)
female: IBW_kg = 45.5 + 2.3 * (height_in - 60)
```

If height, sex/gender, or VT is unavailable, `vt_per_kg_ibw` remains missing. Actual body weight is not substituted.

## Outputs

Processed block artifacts:

- demo mode: `mimic-iv-demo/data/processed/ch1_mimic_stay_block_counts.csv`
- demo mode: `mimic-iv-demo/data/processed/ch1_mimic_block_index.csv`
- demo mode: `mimic-iv-demo/data/processed/ch1_mimic_blocked_dynamic_features.csv`
- full MIMIC: the same filenames under an external `--processed-output-root` outside this repo

QC/report artifacts:

- `reports/ch1_mimic_block_qc_summary.csv`
- `reports/ch1_mimic_block_source_counts.csv`
- `reports/ch1_mimic_block_source_resolution_summary.csv`
- `reports/ch1_mimic_derived_variable_qc_summary.csv`
- `reports/ch1_mimic_block_edge_cases.csv`
- `reports/ch1_mimic_block_note.md`

## Deferred Beyond b2

The following remain intentionally deferred:

- current-block core-vital sufficiency enforcement
- valid prediction-instance filtering
- carry-forward / LOCF
- final missingness handling
- horizon labels
- model-ready construction
- model fitting
- secondary-source sensitivity choices after preferred-source main aggregation

These are not block-construction rules and should not be folded into b2.
