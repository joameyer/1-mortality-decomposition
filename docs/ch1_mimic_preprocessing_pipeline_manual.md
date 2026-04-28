# Chapter 1 MIMIC Preprocessing Pipeline Manual

## Purpose

This manual describes how to run the current Chapter 1 MIMIC preprocessing preparation pipeline. It covers the operational steps from stay-level cohort extraction through completed 8-hour block construction, the thin MIMIC-to-ASIC preprocessing adapter, and conservative proxy horizon-label verification/export.

This is an execution manual, not a new scientific design. The pipeline reuses the frozen Chapter 1 ASIC preprocessing core after MIMIC inputs are adapted to the existing `Chapter1InputTables` contract.

## Pipeline Stages

The current MIMIC preprocessing path has four ordered stages:

1. `5.1.b1`: stay-level cohort extraction
2. `5.1.b2`: completed 8-hour block construction
3. `5.1.b3`: MIMIC-to-ASIC preprocessing adapter and ASIC core reuse
4. `5.1.b4`: conservative proxy horizon-label verification and export

Do not run b2 before b1, do not run b3 before b2, and do not run b4 before b3. B3 expects the retained cohort and the b2 completed-block artifacts. B4 expects the reused ASIC preprocessing outputs produced by b3.

## Storage Policy

Full-MIMIC row-level and block-level artifacts must be written outside this repo. The scripts enforce this for full-MIMIC runs.

Safe aggregated reports may remain under:

```text
reports/
```

Demo-derived row-level artifacts may be written under repo-local `mimic-iv-demo/data/processed/`, but full-MIMIC artifacts must use an external/private root. Recommended full-MIMIC root:

```bash
FULL_ROOT=/Users/joanameyer/data/mimic-iv/mimic-iv-3.1
CH1_ROOT=$FULL_ROOT/1-mortality-decomposition
```

The external row-level/block-level output locations used below are:

```text
$CH1_ROOT/processed
$CH1_ROOT/asic_contract_inputs
$CH1_ROOT/preprocessing_outputs
$CH1_ROOT/horizon_targets
```

## Inputs

Required local MIMIC tables for the pipeline include:

- `icu/icustays.csv.gz`
- `hosp/patients.csv.gz`
- `hosp/admissions.csv.gz`
- `icu/procedureevents.csv.gz`
- `icu/chartevents.csv.gz`
- `hosp/labevents.csv.gz`
- `hosp/diagnoses_icd.csv.gz` when available for `icd10_codes`

Required repo artifacts include:

- `config/ch1_mimic_cohort.yaml`
- `config/ch1_mimic_blocks.yaml`
- `config/ch1_mimic_preprocessing_adapter.yaml`
- `config/ch1_mimic_horizon_labels.yaml`
- `config/ch1_asic_to_mimic_variable_map.csv`
- `config/ch1_feature_sets.json`
- `docs/ch1_mimic_feature_set_freeze.md`
- `docs/ch1_asic_block_logic_recovery.md`
- `docs/ch1_mimic_block_construction.md`
- `docs/ch1_preprocessing_reuse_check.md`
- `docs/ch1_mimic_horizon_label_generation.md`

## Full-MIMIC Run

Set shell variables:

```bash
FULL_ROOT=/Users/joanameyer/data/mimic-iv/mimic-iv-3.1
CH1_ROOT=$FULL_ROOT/1-mortality-decomposition
```

### 1. Run b1 Stay-Level Cohort

```bash
python scripts/run_mimic_cohort.py \
  --mimic-root $FULL_ROOT \
  --processed-dir $CH1_ROOT/processed \
  --reports-dir reports
```

Main row-level output:

```text
$CH1_ROOT/processed/ch1_mimic_stay_level_cohort.csv
```

Safe aggregated reports updated in repo:

- `reports/ch1_mimic_cohort_flow.csv`
- `reports/ch1_mimic_cohort_qc_summary.csv`
- `reports/ch1_mimic_transfer_discharge_summary.csv`
- `reports/ch1_mimic_ventilation_qc_addendum.csv`
- `reports/ch1_mimic_cohort_note.md`

### 2. Run b2 Completed 8-Hour Blocks

```bash
python scripts/run_mimic_blocks.py \
  --mimic-root $FULL_ROOT \
  --cohort-path $CH1_ROOT/processed/ch1_mimic_stay_level_cohort.csv \
  --processed-output-root $CH1_ROOT/processed \
  --reports-dir reports
```

Main row/block-level outputs:

```text
$CH1_ROOT/processed/ch1_mimic_stay_block_counts.csv
$CH1_ROOT/processed/ch1_mimic_block_index.csv
$CH1_ROOT/processed/ch1_mimic_blocked_dynamic_features.csv
```

Safe aggregated reports updated in repo:

- `reports/ch1_mimic_block_qc_summary.csv`
- `reports/ch1_mimic_block_source_counts.csv`
- `reports/ch1_mimic_block_source_resolution_summary.csv`
- `reports/ch1_mimic_block_edge_cases.csv`
- `reports/ch1_mimic_block_note.md`

### 3. Run b3 MIMIC Preprocessing Adapter

```bash
python scripts/run_mimic_preprocessing_adapter.py \
  --mimic-root $FULL_ROOT \
  --cohort-path $CH1_ROOT/processed/ch1_mimic_stay_level_cohort.csv \
  --mimic-block-root $CH1_ROOT/processed \
  --adapter-output-root $CH1_ROOT/asic_contract_inputs \
  --preprocessing-output-root $CH1_ROOT/preprocessing_outputs \
  --reports-dir reports
```

Adapter standardized input outputs:

```text
$CH1_ROOT/asic_contract_inputs/static/harmonized.csv
$CH1_ROOT/asic_contract_inputs/dynamic/harmonized.csv
$CH1_ROOT/asic_contract_inputs/blocked/asic_8h_block_index.csv
$CH1_ROOT/asic_contract_inputs/blocked/asic_8h_blocked_dynamic_features.csv
$CH1_ROOT/asic_contract_inputs/blocked/asic_8h_stay_block_counts.csv
$CH1_ROOT/asic_contract_inputs/qc/mech_vent_ge_24h_stay_level.csv
$CH1_ROOT/asic_contract_inputs/qc/mech_vent_ge_24h_episode_level.csv
```

Reused ASIC preprocessing outputs:

```text
$CH1_ROOT/preprocessing_outputs
```

Safe reports updated in repo:

- `reports/ch1_mimic_adapter_contract_check.csv`
- `reports/ch1_mimic_adapter_qc_summary.csv`
- `reports/ch1_mimic_preprocessing_adapter_note.md`

### 4. Run b4 Conservative Horizon Labels

```bash
python scripts/run_mimic_horizon_labels.py \
  --mimic-root $FULL_ROOT \
  --preprocessing-output-root $CH1_ROOT/preprocessing_outputs \
  --target-output-root $CH1_ROOT/horizon_targets \
  --reports-dir reports
```

Full-MIMIC row-level target tables:

```text
$CH1_ROOT/horizon_targets/ch1_mimic_proxy_horizon_targets_8h.csv
$CH1_ROOT/horizon_targets/ch1_mimic_proxy_horizon_targets_16h.csv
$CH1_ROOT/horizon_targets/ch1_mimic_proxy_horizon_targets_24h.csv
$CH1_ROOT/horizon_targets/ch1_mimic_proxy_horizon_targets_48h.csv
```

Safe reports updated in repo:

- `reports/ch1_mimic_horizon_label_summary.csv`
- `reports/ch1_mimic_horizon_unlabeled_reasons.csv`
- `reports/ch1_mimic_horizon_label_note.md`

## Demo Run

For demo-only testing, repo-local processed outputs are allowed because the demo data are not private full MIMIC. The repo-local demo layout is:

```text
mimic-iv-demo/
  data/
    icu/
    hosp/
    processed/
    asic_contract_inputs/
    preprocessing_outputs/
    horizon_targets/
  reports/
```

Set shell variables:

```bash
DEMO_ROOT=mimic-iv-demo/data
DEMO_OUT=mimic-iv-demo
```

Use `$DEMO_OUT/reports` for demo reports. Do not point demo runs at repo-level `reports/` if you want to preserve the full-MIMIC reports and notes.

Run b1:

```bash
python scripts/run_mimic_cohort.py \
  --mimic-root $DEMO_ROOT \
  --processed-dir $DEMO_ROOT/processed \
  --reports-dir $DEMO_OUT/reports
```

Run b2:

```bash
python scripts/run_mimic_blocks.py \
  --mimic-root $DEMO_ROOT \
  --cohort-path $DEMO_ROOT/processed/ch1_mimic_stay_level_cohort.csv \
  --processed-output-root $DEMO_ROOT/processed \
  --reports-dir $DEMO_OUT/reports
```

Run b3:

```bash
python scripts/run_mimic_preprocessing_adapter.py \
  --mimic-root $DEMO_ROOT \
  --cohort-path $DEMO_ROOT/processed/ch1_mimic_stay_level_cohort.csv \
  --mimic-block-root $DEMO_ROOT/processed \
  --adapter-output-root $DEMO_ROOT/asic_contract_inputs \
  --preprocessing-output-root $DEMO_ROOT/preprocessing_outputs \
  --reports-dir $DEMO_OUT/reports
```

Run b4:

```bash
python scripts/run_mimic_horizon_labels.py \
  --mimic-root $DEMO_ROOT \
  --preprocessing-output-root $DEMO_ROOT/preprocessing_outputs \
  --target-output-root $DEMO_ROOT/horizon_targets \
  --reports-dir $DEMO_OUT/reports
```

Demo b4 target tables are written to:

```text
$DEMO_ROOT/horizon_targets/ch1_mimic_proxy_horizon_targets_8h.csv
$DEMO_ROOT/horizon_targets/ch1_mimic_proxy_horizon_targets_16h.csv
$DEMO_ROOT/horizon_targets/ch1_mimic_proxy_horizon_targets_24h.csv
$DEMO_ROOT/horizon_targets/ch1_mimic_proxy_horizon_targets_48h.csv
```

This demo sequence runs the same storage-safe path as full MIMIC but uses the local demo source tables. Demo reports and notes are written to `$DEMO_OUT/reports`, so they do not overwrite the full-MIMIC aggregated reports under repo-local `reports/`.

## What Each Stage Does

### B1 Cohort Extraction

B1 creates the retained stay-level MIMIC cohort. It applies:

- adult inclusion using `anchor_age`, `anchor_year`, and ICU `intime`
- first chronologic ICU stay per subject
- invasive ventilation duration >=24h using procedureevents itemid `225792`
- ventilation-vs-ICU-LOS QC guard
- ICU mortality using `deathtime <= icustays.outtime`

B1 does not create blocks, labels, valid prediction instances, or model-ready features.

### B2 Block Construction

B2 mirrors the recovered ASIC completed-block logic:

- anchor elapsed time at ICU `intime`
- use 8-hour half-open intervals
- emit completed blocks only using `floor(icu_los_hours / 8)`
- retain empty completed blocks
- aggregate preferred-source dynamic observations with `obs_count`, `mean`, `median`, `min`, `max`, and `last`

B2 does not apply current-block sufficiency, carry-forward, labels, or model-ready construction.

### B3 Adapter And ASIC Core Reuse

B3 constructs the existing ASIC preprocessing input contract:

- `static_harmonized`
- `dynamic_harmonized`
- `block_index`
- `blocked_dynamic_features`
- `stay_block_counts`
- `mech_vent_stay_level_qc`
- `mech_vent_episode_level`

It maps:

- `stay_id_global = string(stay_id)`
- `hospital_id = "MIMIC-IV"`

It then calls the existing ASIC preprocessing core rather than forking MIMIC-specific preprocessing logic.

### B4 Conservative Horizon Labels

B4 verifies and exports the frozen conservative proxy horizon labels for:

- `8h`
- `16h`
- `24h`
- `48h`

B4 reuses the label output produced by the ASIC preprocessing core in b3. It does not create standard event-time mortality labels.

The frozen target semantics are:

- positive: ICU mortality stay and proxy endpoint in `(prediction_time, prediction_time + H]`
- negative: non-ICU-mortality stay with observation through at least `prediction_time + H`
- unlabeled: all other rows

This means eventual non-survivors outside the current horizon remain unlabeled, not negative, and early-discharged survivors remain unlabeled, not negative.

## Validation Checks

After a full run, inspect:

```bash
sed -n '1,120p' reports/ch1_mimic_cohort_note.md
sed -n '1,120p' reports/ch1_mimic_block_note.md
sed -n '1,120p' reports/ch1_mimic_preprocessing_adapter_note.md
sed -n '1,120p' reports/ch1_mimic_horizon_label_note.md
```

Check adapter contract status:

```bash
sed -n '1,120p' reports/ch1_mimic_adapter_contract_check.csv
```

Every row in `reports/ch1_mimic_adapter_contract_check.csv` should have:

```text
status = pass
required_columns_present = True
```

Check adapter run success:

```bash
sed -n '1,120p' reports/ch1_mimic_adapter_qc_summary.csv
```

The metric `reused_asic_preprocessing_entrypoint_ran_successfully` should be `True`.

Check b4 label verification:

```bash
sed -n '1,120p' reports/ch1_mimic_horizon_label_summary.csv
sed -n '1,120p' reports/ch1_mimic_horizon_unlabeled_reasons.csv
```

Every horizon in `reports/ch1_mimic_horizon_label_summary.csv` should have:

```text
status = pass
positive_semantic_violations = 0
negative_semantic_violations = 0
unlabeled_semantic_violations = 0
eventual_non_survivor_outside_horizon_labeled_negative = 0
early_discharged_survivor_labeled_negative = 0
```

## Common Failure Modes

If a full-MIMIC run points row-level outputs inside the repo, the script should fail with an unsafe-output-path error. Fix by passing an external path such as:

```text
--processed-dir $CH1_ROOT/processed
--processed-output-root $CH1_ROOT/processed
--adapter-output-root $CH1_ROOT/asic_contract_inputs
--preprocessing-output-root $CH1_ROOT/preprocessing_outputs
--target-output-root $CH1_ROOT/horizon_targets
```

If b2 cannot find the cohort, rerun b1 or confirm:

```text
$CH1_ROOT/processed/ch1_mimic_stay_level_cohort.csv
```

If b3 cannot find block files, rerun b2 or confirm:

```text
$CH1_ROOT/processed/ch1_mimic_block_index.csv
$CH1_ROOT/processed/ch1_mimic_blocked_dynamic_features.csv
$CH1_ROOT/processed/ch1_mimic_stay_block_counts.csv
```

If `icd10_codes` cannot be joined, b3 documents the placeholder status in `reports/ch1_mimic_preprocessing_adapter_note.md`.

If b4 cannot find label files, rerun b3 or confirm:

```text
$CH1_ROOT/preprocessing_outputs/labels/chapter1_proxy_horizon_labels.csv
$CH1_ROOT/preprocessing_outputs/labels/chapter1_usable_proxy_horizon_labels.csv
$CH1_ROOT/preprocessing_outputs/cohort/chapter1_retained_stay_table.csv
```

If b4 fails semantic verification, stop and inspect the reported violation columns in `reports/ch1_mimic_horizon_label_summary.csv`. Do not patch this by broadening the negative class or switching to true event-time mortality labels.

## Cleanup

Full-MIMIC row-level or block-level artifacts should not remain under repo-local `mimic-iv-demo/data/processed/`. If such files were produced before the storage guards were added, remove them manually.

Old unsafe pattern from the previous repo-local full-MIMIC layout:

```bash
rm data/processed/ch1_mimic_stay_level_cohort.csv
```

Do not remove demo files if they are intentionally being used for local testing.

## Deferred Beyond This Pipeline

This manual does not cover:

- model fitting
- external-validation result reporting
- post-hoc sensitivity analyses
- broad mapping redesign
- standard event-time mortality-label sensitivity analyses

B4 verifies and exports the frozen conservative proxy horizon targets. Scientific interpretation of model performance and any later sensitivity analysis against MIMIC-native event-time labels remain downstream of this operational run.
