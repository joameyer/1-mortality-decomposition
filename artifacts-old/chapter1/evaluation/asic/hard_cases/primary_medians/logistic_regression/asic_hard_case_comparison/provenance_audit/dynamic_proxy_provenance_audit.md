# ASIC Issue 3.2 Dynamic-Proxy Provenance Audit

## Question

This memo audits the current ASIC Issue 3.2 hard-case comparison implementation to determine whether the four selected dynamic proxy variables are being pulled from the Chapter 1 model-ready analysis layer or from unfilled blocked/raw data.

Audited variables:

- `pf_ratio_last`
- `map_last`
- `creatinine_last`
- `peep_last`

## Current Issue 3.2 Assembly Path

Current implementation entrypoint:

- `run_asic_hard_case_comparison()` -> `build_stay_level_comparison_dataset()` in `src/chapter1_mortality_decomposition/asic_hard_case_comparison.py`

Current stay-level assembly path:

1. Start from saved stay-level hard-case artifact:
   - `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/stay_level_hard_case_flags.csv`
2. Filter to the current target population:
   - `horizon_h == 24`
   - `label_value == 1`
3. Join the Chapter 1 primary model-ready dataset on:
   - `instance_id`
   - `stay_id_global`
   - `hospital_id`
   - `block_index`
   - `prediction_time_h`
   - `horizon_h`
   - `label_value`
4. Pull the four dynamic comparison variables from that joined model-ready dataset:
   - `pf_ratio_last`
   - `map_last`
   - `creatinine_last`
   - `peep_last`
5. Join ASIC static data only for `age_group`, `sex`, and `icd10_codes` / `disease_group`.

Bottom line for the Issue 3.2 stage itself:

- The current comparison does **not** read these four variables from blocked dynamic tables or raw dynamic tables at the Issue 3.2 assembly stage.
- It reads them from `artifacts/chapter1/model_ready/chapter1_primary_model_ready_dataset.csv`.

## Upstream Derivation Path

The model-ready dataset is built in `src/chapter1_mortality_decomposition/model_ready.py` by merging `carry_forward.feature_frame` into the usable-label instance table.

`carry_forward.feature_frame` is built in `src/chapter1_mortality_decomposition/carry_forward.py` from the standardized blocked ASIC feature table loaded at:

- `input_dir/blocked/asic_8h_blocked_dynamic_features.csv`
- in the current repo this filename is declared in `src/chapter1_mortality_decomposition/artifacts.py`

Important implication:

- For all four audited variables, the Issue 3.2 comparison is using the **same Chapter 1 model-ready analysis layer** that underlies the baseline model pipeline.
- Variable-level bounded LOCF behavior is **not identical** across the four variables.

## Variable Audit

### `pf_ratio_last`

- Current source artifact: `artifacts/chapter1/model_ready/chapter1_primary_model_ready_dataset.csv`
- Current source column: `pf_ratio_last`
- Issue 3.2 join path: exact left join on `instance_id`, `stay_id_global`, `hospital_id`, `block_index`, `prediction_time_h`, `horizon_h`, and `label_value`
- Upstream path: blocked dynamic `pf_ratio_last` -> `carry_forward.feature_frame` -> model-ready dataset -> Issue 3.2 comparison dataset
- Bounded LOCF included in source path: `MIXED`

Why `MIXED` rather than `YES`:

- The current Issue 3.2 code uses the model-ready analysis-layer export, not blocked/raw data.
- But `pf_ratio` is **not configured** for bounded LOCF in `carry_forward.py`.
- Evidence from `artifacts/chapter1/carry_forward/chapter1_primary_locf_feature_summary.csv`:
  - `feature_family = no_bounded_locf_configured`
  - `locf_window_hours = NA`
  - `locf_filled_instances = 0 / 1328`
- Evidence from the current 24h fatal Issue 3.2 slice:
  - `pf_ratio_filled_by_locf = 0 / 10`

Interpretation:

- `pf_ratio_last` is taken from an already exported model-ready field with **no additional Issue 3.2 filling**.
- In practice it is an analysis-layer field that remains observed-or-missing.

### `map_last`

- Current source artifact: `artifacts/chapter1/model_ready/chapter1_primary_model_ready_dataset.csv`
- Current source column: `map_last`
- Issue 3.2 join path: exact left join on the same seven-key stay/instance merge
- Upstream path: blocked dynamic `map_last` -> `carry_forward.feature_frame` -> model-ready dataset -> Issue 3.2 comparison dataset
- Bounded LOCF included in source path: `MIXED`

Why `MIXED`:

- `map` is configured in `carry_forward.py` as `fast_bedside_physiology` with a `4h` bounded LOCF window.
- So the source representation is LOCF-capable and is part of the same analysis layer used by the baseline pipeline.
- But the saved primary model-ready export shows **no realized LOCF fills** for `map`.
- Evidence from `artifacts/chapter1/carry_forward/chapter1_primary_locf_feature_summary.csv`:
  - `locf_window_hours = 4`
  - `locf_filled_instances = 0 / 1328`
- Evidence from the current 24h fatal Issue 3.2 slice:
  - `map_filled_by_locf = 0 / 10`
  - `map_observed_in_block = 9 / 10`

Interpretation:

- `map_last` is coming from a carry-forward-enabled model-ready feature.
- In the current primary export and in the current Issue 3.2 slice, it behaves as a **current-block value whenever present**.

### `creatinine_last`

- Current source artifact: `artifacts/chapter1/model_ready/chapter1_primary_model_ready_dataset.csv`
- Current source column: `creatinine_last`
- Issue 3.2 join path: exact left join on the same seven-key stay/instance merge
- Upstream path: blocked dynamic `creatinine_last` -> `carry_forward.feature_frame` -> model-ready dataset -> Issue 3.2 comparison dataset
- Bounded LOCF included in source path: `YES`

Evidence:

- `creatinine` is configured in `carry_forward.py` as `slower_labs` with a `48h` bounded LOCF window.
- Evidence from `artifacts/chapter1/carry_forward/chapter1_primary_locf_feature_summary.csv`:
  - `locf_window_hours = 48`
  - `locf_filled_instances = 858 / 1328`
- Evidence from the current 24h fatal Issue 3.2 slice:
  - `creatinine_filled_by_locf = 5 / 10`
  - `creatinine_observed_in_block = 3 / 10`

Interpretation:

- The current Issue 3.2 implementation is indeed using the existing `48h` carry-forward-enabled Chapter 1 analysis-layer representation for `creatinine_last`.

### `peep_last`

- Current source artifact: `artifacts/chapter1/model_ready/chapter1_primary_model_ready_dataset.csv`
- Current source column: `peep_last`
- Issue 3.2 join path: exact left join on the same seven-key stay/instance merge
- Upstream path: blocked dynamic `peep_last` -> `carry_forward.feature_frame` -> model-ready dataset -> Issue 3.2 comparison dataset
- Bounded LOCF included in source path: `YES`

Evidence:

- `peep` is configured in `carry_forward.py` as a `ventilator_variables` feature with a `24h` bounded LOCF window.
- Ventilator LOCF is restricted to ventilation-supported windows.
- Evidence from `artifacts/chapter1/carry_forward/chapter1_primary_locf_feature_summary.csv`:
  - `locf_window_hours = 24`
  - `locf_filled_instances = 19 / 1328`
- Evidence from `artifacts/chapter1/carry_forward/chapter1_primary_ventilator_locf_summary.csv`:
  - `locf_fills_inside_ventilation_window = 19`
  - `locf_fills_outside_ventilation_window = 0`
- Evidence from the current 24h fatal Issue 3.2 slice:
  - `peep_filled_by_locf = 1 / 10`
  - the filled row remains `peep_ventilation_window_active = True`

Interpretation:

- The current Issue 3.2 implementation is using the upstream within-window ventilator LOCF representation for `peep_last` where applicable.

## Compact Verdict

| variable | current source artifact | current source column | bounded LOCF included? | short verdict |
| --- | --- | --- | --- | --- |
| `pf_ratio_last` | `artifacts/chapter1/model_ready/chapter1_primary_model_ready_dataset.csv` | `pf_ratio_last` | MIXED | model-ready analysis-layer source, but no bounded LOCF is configured for this variable |
| `map_last` | `artifacts/chapter1/model_ready/chapter1_primary_model_ready_dataset.csv` | `map_last` | MIXED | model-ready analysis-layer source with 4h LOCF capability, but zero realized fills in the current primary export |
| `creatinine_last` | `artifacts/chapter1/model_ready/chapter1_primary_model_ready_dataset.csv` | `creatinine_last` | YES | model-ready analysis-layer source with active 48h bounded LOCF |
| `peep_last` | `artifacts/chapter1/model_ready/chapter1_primary_model_ready_dataset.csv` | `peep_last` | YES | model-ready analysis-layer source with active within-window ventilator LOCF |

## Judgment Against Intended Design

Judgment: **consistent with the intended design principle**.

Reason:

- The current Issue 3.2 implementation is using the same Chapter 1 **model-ready analysis-layer representation** that the baseline model pipeline saw.
- It is **not** bypassing that layer and re-reading the four dynamic proxies directly from blocked/raw data during Issue 3.2 assembly.
- The only nuance is variable-specific carry-forward behavior:
  - `creatinine_last` and `peep_last` do incorporate bounded LOCF in the current export
  - `map_last` is routed through the carry-forward-enabled layer but has zero realized fills
  - `pf_ratio_last` is routed through the same model-ready layer but has no bounded LOCF configuration

## Final Recommendation

**KEEP CURRENT IMPLEMENTATION**

Reason for recommendation:

- The provenance is already aligned with the intended Chapter 1 design principle: Issue 3.2 uses the baseline model-ready analysis layer rather than raw/unfilled tables.
- No provenance-driven switch to blocked/raw data is indicated.
- If a later scientific decision is made that `pf_ratio_last` should itself receive bounded LOCF, that would be a separate representation-design change rather than a provenance fix.
