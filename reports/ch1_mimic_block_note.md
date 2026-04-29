# Chapter 1 MIMIC Completed 8h Block Construction Note

## Purpose

This report documents subtask 5.1.b2: structural completed-block construction for the retained MIMIC Chapter 1 stay-level cohort. It mirrors the recovered ASIC Chapter 1 block logic and is not valid-instance filtering, carry-forward, horizon-label generation, model-ready construction, or model fitting.

## Data Source

- MIMIC root: `/Users/joanameyer/data/mimic-iv/mimic-iv-3.1`
- Processed output root: `/Users/joanameyer/data/mimic-iv/mimic-iv-3.1/1-mortality-decomposition/processed`
- Retained stays entering block construction: 10648
- Total completed 8h blocks emitted: 285406
- Stays with at least one completed block: 10648

## Translation Implemented

- Anchor: elapsed time from retained-stay `intime`.
- Width: 8 hours only.
- Interval convention: half-open `[start, end)`, implemented by `block_index = floor(time_h / 8)`.
- Completed blocks: `floor(icu_los_hours / 8)`, using retained-stay `intime`/`outtime` duration from the 5.1.b1 cohort.
- Prediction time: `prediction_time_h = block_end_h`.
- Empty completed blocks are retained with zero counts.
- Current-block sufficiency is not applied in b2.

## Included Raw Sources

- `icu.chartevents` for frozen shared-primary charted variables.
- `hosp.labevents` for frozen shared-primary laboratory and blood-gas variables.

B2 applies preferred source/item filtering before aggregation using `config/ch1_asic_to_mimic_variable_map.csv`. Broader secondary sources remain available in source-resolution QC for later sensitivity or provenance review, but are not pooled into the main block summaries when the preferred source is available.

The stable ordering used for `last` is `subject_id`, `hadm_id`, `stay_id`, `block_index`, `time_h`, then source row order from chunked loading after source preference filtering.

## Derived-Only Shared-Primary Variables

- `pf_ratio` is materialized as `pao2 / (fio2 / 100)` from block-level preferred PaO2 and FiO2 summaries.
- `vt_per_kg_ibw` is materialized as preferred block-level VT divided by Devine IBW from retained-stay gender and height itemids `226730`/`226707`.
- Missing support inputs leave the derived value missing; actual body weight is not substituted.

## Deferred Beyond b2

- current-block core-vital sufficiency filtering
- carry-forward / LOCF
- final missingness handling
- valid prediction-instance filtering
- horizon labels
- model-ready construction
- model fitting
- secondary-source sensitivity choices

## QC Highlights

- Completed blocks with zero dynamic rows: 497
- Completed blocks with zero observed variables: 497
- Raw rows dropped for negative `time_h`: 197482
- Raw rows landing beyond the completed block grid: 803526
- Raw rows exactly on an 8h boundary: 25209

## Translation Limitations

- `pf_ratio` and `vt_per_kg_ibw` are materialized as block-level derived variables from frozen preferred source summaries; timestamp-level paired derivation and sensitivity variants remain out of scope.
- Preferred source/item filtering is applied for main b2 aggregation; secondary sources remain documented in source-resolution QC for later sensitivity or provenance review.
