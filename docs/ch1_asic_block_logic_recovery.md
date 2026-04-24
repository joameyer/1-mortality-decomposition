# Chapter 1 ASIC Block-Construction Logic Recovery

## Purpose

This note recovers the authoritative ASIC Chapter 1 blocked-data logic for later MIMIC 5.1.b2 mirroring. It is a recovery artifact only: it does not redesign the block scheme, implement MIMIC blocking, build labels, or fit models.

## Sources Inspected

The active source of truth is the current implementation, not older prose. The main files/functions inspected were:

- `src/chapter1_mortality_decomposition/temporal_blocks.py`
  - `build_asic_temporal_block_artifacts`
  - `_build_stay_block_counts`
  - `_build_block_index`
  - `_build_blocked_dynamic_features`
- `src/chapter1_mortality_decomposition/artifacts.py`
  - `load_chapter1_inputs`, which expects frozen `asic_8h_*` blocked artifacts.
- `src/chapter1_mortality_decomposition/instances.py`
  - `build_chapter1_valid_instances`
- `src/chapter1_mortality_decomposition/labels.py`
  - `build_chapter1_proxy_horizon_labels`
- `src/chapter1_mortality_decomposition/config.py`
  - `default_chapter1_config`, `chapter1_group_definitions`
- `src/chapter1_mortality_decomposition/pipeline.py`
  - `build_chapter1_dataset`
- `config/ch1_run_config.json`
- `tests/test_preprocessing.py` and `tests/test_temporal_preview.py`

The current repo consumes upstream ASIC standardized inputs. In particular, the exact upstream derivation of `dynamic_harmonized.time` and `reference_stay_block_counts.icu_end_time_proxy_hours` is outside this repo; this repo treats those fields as authoritative inputs.

## Recovered Block Logic

### 1. Block Anchor

Blocks are anchored at elapsed time from ICU admission. In the block builder, `dynamic_harmonized["time"]` is converted with `pd.to_timedelta(...).dt.total_seconds() / 3600.0` into `time_h`. The reference stay table also carries `icu_admission_time`, `icu_end_time_proxy`, and `icu_end_time_proxy_hours`.

Operationally, block 0 starts at `0h` after ICU admission. For MIMIC mirroring, this implies anchoring elapsed time to ICU entry (`icustays.intime`) unless a later implementation conflict is found.

Boundary: the current repo does not recompute ASIC admission-relative time from raw timestamps; it receives `dynamic_harmonized.time` and `icu_end_time_proxy_hours` from the upstream ASIC harmonization artifacts.

### 2. Block Width

The frozen Chapter 1 blocked ASIC artifacts are 8-hour artifacts. `load_chapter1_inputs` explicitly expects:

- `blocked/asic_8h_block_index`
- `blocked/asic_8h_blocked_dynamic_features`
- `blocked/asic_8h_stay_block_counts`

The temporal block builder is parameterized by `block_hours`, but the Chapter 1 frozen path uses the 8-hour version. The default model horizons are separate: `(8, 16, 24, 48, 72)` hours.

### 3. Block Interval Convention

Dynamic rows are assigned with:

```python
block_index = time_h // block_hours
```

for rows with non-missing `time_h >= 0`, followed by an inner merge to the structurally completed block index.

This makes intervals effectively half-open:

- block 0: `[0, 8)`
- block 1: `[8, 16)`
- block k: `[8k, 8(k+1))`

Exact boundary values are assigned to the next block because of floor division. A row at exactly `8h` belongs to block 1, not block 0. A row exactly at the terminal completed-block end is assigned to the next block index and is dropped if that next block was not structurally created.

### 4. Completed-Block Rule

Completed block counts are computed per stay from `icu_end_time_proxy_hours`:

```python
completed_block_count = icu_end_time_proxy_hours // block_hours
```

for non-missing, non-negative proxy hours. Stays with fewer than one full block have `has_completed_block = False`.

The block index contains block indices:

```text
0, 1, ..., completed_block_count - 1
```

with:

```text
block_start_h = block_index * block_hours
block_end_h = block_start_h + block_hours
terminal_block_end_h = completed_block_count * block_hours
```

If `icu_end_time_proxy_hours` is exactly on an 8-hour boundary, the preceding block ending at that boundary is included, and `ends_exactly_on_8h_boundary` / `ends_exactly_on_block_boundary` is flagged in `stay_block_counts`.

### 5. Prediction Time Definition

The prediction timestamp is the end of the completed block:

```text
prediction_time_h = block_end_h
```

For 8-hour artifacts, block 0 predicts at `8h`, block 1 at `16h`, and so on.

### 6. Alive / Still-In-ICU Rule at Prediction Time

At block construction, the structural rule is completion before or at the ICU end proxy, through the completed-block count above.

At valid-instance construction, `build_chapter1_valid_instances` rechecks:

```python
block_end_h <= icu_end_time_proxy_hours
```

as `block_end_not_after_icu_end_proxy`. This is a non-strict check.

Strict prediction-before-end behavior is deferred to label construction. In `build_chapter1_proxy_horizon_labels`, positive labels require:

```python
event_time_proxy_h > prediction_time_h
event_time_proxy_h <= future_window_end_h
```

and rows where `event_time_proxy_h <= prediction_time_h` receive the unlabeled reason `prediction_time_not_strictly_before_proxy_end`.

Implication: block construction and valid-instance filtering allow a block ending exactly at the ICU end proxy, but horizon labelability later requires the prediction time to be strictly before the proxy end.

### 7. Current-Block Data Sufficiency Rule

There is no minimum observed-data sufficiency rule at the block-construction stage. All structurally completed blocks are emitted. Blocks with no assigned dynamic rows remain present in `blocked_dynamic_features` with count columns set to `0` and summary statistics missing.

The current-block sufficiency rule is downstream in `build_chapter1_valid_instances`. It requires at least `min_required_core_groups = 3` observed core vital groups in the current block. The groups are:

- `cardiac_rate`: `heart_rate`
- `blood_pressure`: `map`, `sbp`, `dbp`
- `respiratory`: `resp_rate`
- `oxygenation`: `spo2`, `sao2`

Group observation is based primarily on `*_obs_count > 0` in the current block. If obs-count columns are absent, the implementation falls back to non-missing feature columns for that group.

### 8. Per-Block Feature Aggregation

Feature columns are every dynamic-harmonized column except:

- `hospital_id`
- `stay_id_global`
- `stay_id_local`
- `time`
- `time_h`
- `minutes_since_admit`
- `source_row_order`

Values are numerically coerced before aggregation. Per completed block, the implementation emits:

- `dynamic_row_count`
- `non_missing_measurements_in_block`
- `observed_variables_in_block`
- for each feature column:
  - `{variable}_obs_count`
  - `{variable}_mean`
  - `{variable}_median`
  - `{variable}_min`
  - `{variable}_max`
  - `{variable}_last`

The `last` statistic is computed after stable sorting by:

```text
hospital_id, stay_id_global, block_index, time_h, source_row_order
```

so ties preserve upstream row order.

### 9. Enforced at Block Construction vs Deferred

Enforced during block construction:

- required input columns exist for `dynamic_harmonized` and `reference_stay_block_counts`
- one row per `stay_id_global` / `hospital_id` in the reference stay table
- positive integer block width
- elapsed dynamic times are non-missing and `>= 0` before assignment
- structural completed blocks based on `floor(icu_end_time_proxy_hours / block_hours)`
- block index with `block_start_h`, `block_end_h`, and `prediction_time_h`
- current-block aggregation of raw dynamic values into count and summary features
- empty structurally completed blocks are retained with zero count columns

Deferred downstream:

- Chapter 1 retained cohort rules and site eligibility
- valid prediction-instance filtering
- current-block core-vital group sufficiency
- strict prediction-before-proxy-end labelability
- horizon-specific label availability
- bounded carry-forward / LOCF
- observation-process features
- feature-set selection and model-ready table construction
- train/validation/test split assignment
- any model fitting

### 10. Source-of-Truth Artifacts

The real source of truth for block construction is `src/chapter1_mortality_decomposition/temporal_blocks.py`.

The frozen 8-hour artifact expectation is encoded in `src/chapter1_mortality_decomposition/artifacts.py`, which loads `asic_8h_*` blocked artifacts for Chapter 1 preprocessing.

The source of truth for valid-instance sufficiency after block construction is `src/chapter1_mortality_decomposition/instances.py`.

The source of truth for strict horizon labelability is `src/chapter1_mortality_decomposition/labels.py`.

The active run configuration is `config/ch1_run_config.json`, with horizons `[8, 16, 24, 48, 72]` and `min_required_core_groups = 3`.

## Implications for MIMIC 5.1.b2

MIMIC should mirror the ASIC block scheme as a translation, not a redesign:

- anchor elapsed time at ICU entry
- create 8-hour completed blocks
- use block intervals equivalent to `[start, end)` for measurement assignment
- define prediction time as block end
- create blocks only through the completed-block count implied by ICU end
- keep structurally completed empty blocks rather than dropping them during aggregation
- compute the same per-variable statistics: `obs_count`, `mean`, `median`, `min`, `max`, `last`
- keep current-block data sufficiency out of block construction and apply it later as valid-instance filtering
- keep horizon labels separate from block construction

MIMIC-specific translation will need to replace ASIC upstream proxy inputs with MIMIC-native fields, especially ICU entry time, ICU exit time, and mortality timing. That translation should preserve the staging above: block construction first, valid-instance filtering second, horizon labelability third.
