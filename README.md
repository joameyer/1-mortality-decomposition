# Chapter 1 Mortality Decomposition

This repository is the standalone Chapter 1 preprocessing layer for the mortality decomposition analysis.

It starts from standardized upstream ASIC artifacts only:

- `static/harmonized.{csv|parquet}`
- `dynamic/harmonized.{csv|parquet}`
- `blocked/asic_8h_block_index.{csv|parquet}`
- `blocked/asic_8h_blocked_dynamic_features.{csv|parquet}`
- `blocked/asic_8h_stay_block_counts.{csv|parquet}`

It does not depend on raw ASIC source tables or on `icu-data-platform` internals.

## Ownership Boundary

This repo owns Chapter 1-specific analytic preprocessing:

- site-level exclusion logic
- stay-level exclusion logic
- readmission-based first-stay proxy handling
- valid-instance generation
- Chapter 1 feature-set configuration
- proxy within-horizon in-ICU mortality label generation
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

- harmonized static and dynamic ASIC artifacts already exist
- generic 8-hour blocked ASIC artifacts already exist

What is enforced in this repo today:

- site-level Chapter 1 exclusion rules
- stay-level dynamic-data / readmission / missing-label exclusions
- valid-instance construction from generic blocks

What remains unresolved:

- the current standardized input contract used by this repo does not yet require adult-age fields or ventilation-duration fields
- therefore adult-only and ventilation-duration `>= 24h` restrictions are not currently enforced in code here
- until the interface is finalized, those restrictions should be treated as scientific cohort intent, not as guaranteed upstream preprocessing

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

- [`cohort.py`](/Users/joanameyer/repository/1-mortality-decomposition/src/chapter1_mortality_decomposition/cohort.py): site and stay exclusions
- [`instances.py`](/Users/joanameyer/repository/1-mortality-decomposition/src/chapter1_mortality_decomposition/instances.py): valid prediction instances
- [`labels.py`](/Users/joanameyer/repository/1-mortality-decomposition/src/chapter1_mortality_decomposition/labels.py): proxy within-horizon label logic
- [`model_ready.py`](/Users/joanameyer/repository/1-mortality-decomposition/src/chapter1_mortality_decomposition/model_ready.py): model-ready assembly and readiness summaries
- [`pipeline.py`](/Users/joanameyer/repository/1-mortality-decomposition/src/chapter1_mortality_decomposition/pipeline.py): end-to-end orchestration
- [`cli.py`](/Users/joanameyer/repository/1-mortality-decomposition/src/chapter1_mortality_decomposition/cli.py): runnable entrypoint

## Quick Start

Install the package in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Run the preprocessing CLI:

```bash
chapter1-preprocess --input-dir path/to/standardized_asic --output-dir artifacts/chapter1
```

Or run it as a module from the repo root:

```bash
PYTHONPATH=src python -m chapter1_mortality_decomposition \
  --input-dir path/to/standardized_asic \
  --output-dir artifacts/chapter1
```

## Outputs

The pipeline writes four output groups under the chosen output directory:

- `cohort/`: site eligibility, stay exclusions, retained stays, and cohort notes
- `instances/`: candidate instances, valid instances, and exclusion summaries
- `labels/`: proxy horizon label tables, horizon summaries, unlabeled-reason summaries, and notes
- `model_ready/`: selected feature set, model-ready table, and readiness summaries

## Tests

The synthetic end-to-end preprocessing tests live in [`tests/test_preprocessing.py`](/Users/joanameyer/repository/1-mortality-decomposition/tests/test_preprocessing.py).

Run them with:

```bash
python -m unittest discover -s tests -v
```

## Known Follow-Up

- Adult-age and mechanical-ventilation-duration cohort restrictions still need a finalized standardized column contract if they are to be enforced here rather than guaranteed upstream.
