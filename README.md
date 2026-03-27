# Chapter 1 Mortality Decomposition

This repository is the standalone Chapter 1 preprocessing layer for the mortality decomposition analysis.

It starts from standardized upstream ASIC artifacts only:

- `static/harmonized.{csv|parquet}`
- `dynamic/harmonized.{csv|parquet}`
- `blocked/asic_8h_block_index.{csv|parquet}`
- `blocked/asic_8h_blocked_dynamic_features.{csv|parquet}`
- `blocked/asic_8h_stay_block_counts.{csv|parquet}`
- `qc/mech_vent_ge_24h_stay_level.{csv|parquet}`
- `qc/mech_vent_ge_24h_episode_level.{csv|parquet}`

It does not depend on raw ASIC source tables or on `icu-data-platform` internals.

## Ownership Boundary

This repo owns Chapter 1-specific analytic preprocessing:

- site-level exclusion logic
- stay-level exclusion logic
- readmission-based first-stay proxy handling
- valid-instance generation
- Chapter 1 feature-set configuration via `config/ch1_feature_sets.json`
- proxy within-horizon in-ICU mortality label generation
- bounded preprocessing-time carry-forward and missingness handling
- model-ready dataset construction
- Chapter 1 QC and readiness summaries

These remain upstream:

- extraction from raw source systems
- harmonization of static and dynamic ASIC tables
- semantic cleaning
- generic stay/block construction
- generic shared QC

## Input Contract Status

Scientifically intended Chapter 1 cohort assumptions from the frozen analysis spec:

- adults only
- mechanically ventilated ICU stays only
- ventilation duration `>= 24h`

What is explicitly assumed upstream by the current code:

- adult age `>= 18` has already been applied upstream as part of the standardized ASIC cohort contract
- harmonized static and dynamic ASIC artifacts already exist
- generic 8-hour blocked ASIC artifacts already exist
- upstream QC provides `qc/mech_vent_ge_24h_stay_level.{csv|parquet}` with `mech_vent_ge_24h_qc`
- upstream QC provides `qc/mech_vent_ge_24h_episode_level.{csv|parquet}` for ventilation-supported episode windows

What is enforced in this repo today:

- site-level Chapter 1 exclusion rules
- stay-level exclusion of any stay with missing or failed `mech_vent_ge_24h_qc`
- stay-level dynamic-data / readmission / missing-or-unusable-label exclusions
- valid-instance construction from generic blocks using block structure, ICU-end proxy, and 3-of-4 core-group coverage within each block
- proxy within-horizon labelability and model-ready export construction
- bounded preprocessing-time LOCF on configured feature families, with missingness indicators and ventilator-variable LOCF restricted to upstream ventilation-supported windows

What remains unresolved:

- adult age `>=18` is trusted upstream, not rederived here
- ASIC still lacks patient identifiers, so downstream splitting remains effectively stay-level rather than patient-level

See [`docs/preprocessing_interface.md`](/Users/joanameyer/repository/1-mortality-decomposition/docs/preprocessing_interface.md) for the explicit contract split.

## Label Status

ASIC Chapter 1 uses explicit proxy within-horizon in-ICU mortality labels because true ICU discharge timestamps and true death timestamps are unavailable in the standardized artifacts.

- Supported horizons: `8h`, `16h`, `24h`, `48h`, `72h`
- Positive label: `icu_mortality == 1` and `icu_end_time_proxy_hours` is in `(t, t + H]`
- Negative label: `icu_mortality == 0` and `icu_end_time_proxy_hours >= t + H`
- All other cases remain unlabeled and are not coerced to negative
- These are proxy horizon labels, not true event-timed mortality labels

See [`docs/label_logic_audit.md`](/Users/joanameyer/repository/1-mortality-decomposition/docs/label_logic_audit.md) for the audit note.

## Package Layout

The canonical implementation lives in [`src/chapter1_mortality_decomposition`](/Users/joanameyer/repository/1-mortality-decomposition/src/chapter1_mortality_decomposition):

- [`config/ch1_feature_sets.json`](/Users/joanameyer/repository/1-mortality-decomposition/config/ch1_feature_sets.json): version-controlled Chapter 1 feature-set source of truth
- [`cohort.py`](/Users/joanameyer/repository/1-mortality-decomposition/src/chapter1_mortality_decomposition/cohort.py): site and stay exclusions
- [`instances.py`](/Users/joanameyer/repository/1-mortality-decomposition/src/chapter1_mortality_decomposition/instances.py): valid prediction instances
- [`labels.py`](/Users/joanameyer/repository/1-mortality-decomposition/src/chapter1_mortality_decomposition/labels.py): proxy within-horizon label logic
- [`splits.py`](/Users/joanameyer/repository/1-mortality-decomposition/src/chapter1_mortality_decomposition/splits.py): canonical stay-level train/validation/test split assignment and balance summaries
- [`model_ready.py`](/Users/joanameyer/repository/1-mortality-decomposition/src/chapter1_mortality_decomposition/model_ready.py): model-ready assembly and readiness summaries
- [`pipeline.py`](/Users/joanameyer/repository/1-mortality-decomposition/src/chapter1_mortality_decomposition/pipeline.py): end-to-end orchestration
- [`cli.py`](/Users/joanameyer/repository/1-mortality-decomposition/src/chapter1_mortality_decomposition/cli.py): runnable entrypoint

## Feature Sets

The Chapter 1 feature sets are defined once in [`config/ch1_feature_sets.json`](/Users/joanameyer/repository/1-mortality-decomposition/config/ch1_feature_sets.json):

- `primary_features`
- `extended_additional_features`

The code derives:

- `primary`
- `extended = primary + extended_additional`

The same config artifact is used for:

- feature-schema validation
- feature-set-specific valid-instance construction
- model-ready matrix creation
- feature availability and missingness summaries

The runtime input and output paths for local execution are defined in [`config/ch1_run_config.json`](/Users/joanameyer/repository/1-mortality-decomposition/config/ch1_run_config.json). Both the notebook runbook and the CLI can read this shared run config.

## Carry-Forward Policy

Chapter 1 preprocessing exports are bounded-LOCF and missingness-aware.

- bounded LOCF is applied only to prespecified feature families
- `obs_count` columns remain raw and are not carried forward
- value-summary columns remain missing when no valid carry-forward source exists
- ventilator variables are only carried forward inside upstream ventilation-supported episode windows
- missingness indicator columns are added so downstream modeling can distinguish observed values, LOCF-filled values, and values still missing after preprocessing
- final imputation is intentionally deferred to downstream modeling and must be fit on the training split only

No global median imputation or other final imputation is applied in the exported Chapter 1 model-ready tables.

## Split Strategy

Chapter 1 now includes a canonical stay-level split path driven by the retained stay cohort artifact.

- split source: the canonical retained Chapter 1 stay cohort
- split unit: `stay_id_global`
- primary design: `70%` train, `15%` validation, `15%` test
- hospital balance: stays are split within each retained hospital, then pooled
- outcome handling: stay-level `icu_mortality` is used for within-hospital stratification as far as feasible
- reproducibility: the split seed is versioned in [`config/ch1_run_config.json`](/Users/joanameyer/repository/1-mortality-decomposition/config/ch1_run_config.json) as `split_random_seed`

Because ASIC lacks patient identifiers, this remains operationally a stay-level split after readmission-based proxy first-stay filtering. Small hospital-specific strata may prevent exact `70/15/15` and exact within-hospital outcome balance, so the repo writes explicit split-balance summaries rather than assuming perfect stratification.

## Quick Start

Install the package in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Open the notebook runbook if you want a step-by-step orchestration and QC view:

```bash
jupyter notebook notebooks/ch1_preprocessing_runbook.ipynb
```

The notebook reads [`config/ch1_run_config.json`](/Users/joanameyer/repository/1-mortality-decomposition/config/ch1_run_config.json), which currently points to the standardized ASIC artifact root at `/Users/joanameyer/repository/icu-data-platform/artifacts/asic_harmonized`.

Run the preprocessing CLI:

```bash
chapter1-preprocess --input-dir path/to/standardized_asic --output-dir artifacts/chapter1
```

Or reuse the shared run config directly:

```bash
chapter1-preprocess --run-config config/ch1_run_config.json
```

Or run it as a module from the repo root:

```bash
PYTHONPATH=src python -m chapter1_mortality_decomposition \
  --input-dir path/to/standardized_asic \
  --output-dir artifacts/chapter1
```

## Outputs

The pipeline writes seven output groups under the chosen output directory:

- `cohort/`: site eligibility, stay exclusions, retained stays, cohort notes, canonical cohort summary, and verification summary
- `feature_sets/`: combined feature-set definition table and validation summary
- `instances/`: canonical candidate instances, valid instances, and exclusion summaries
- `labels/`: canonical proxy horizon label tables, horizon summaries, unlabeled-reason summaries, and notes
- `carry_forward/`: feature-level LOCF summaries, ventilator-window QC, missingness before/after LOCF summaries, and carry-forward verification tables
- `model_ready/`: feature-set-specific model-ready datasets, readiness summaries, and missingness summaries
- `splits/`: stay-level split assignments, split verification, and split-balance summaries for both the retained stay cohort and the feature-set-specific model-ready tables

## Tests

The synthetic end-to-end preprocessing tests live in [`tests/test_preprocessing.py`](/Users/joanameyer/repository/1-mortality-decomposition/tests/test_preprocessing.py).

Run them with:

```bash
python -m unittest discover -s tests -v
```

## Known Follow-Up

- Because ASIC has no patient identifiers, the split strategy cannot be patient-level.
- Split balance is only approximate within very small hospital/outcome strata, so downstream modeling should read the written split summaries rather than assuming exact `70/15/15` in every subgroup.
