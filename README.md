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
- horizon label generation
- model-ready dataset construction
- Chapter 1 QC and readiness summaries

These remain upstream:

- extraction from raw source systems
- harmonization of static and dynamic ASIC tables
- semantic cleaning
- generic stay/block construction
- generic shared QC

## Current Label Semantics

The standalone pipeline now generates horizon-specific ICU mortality labels using `icu_end_time_proxy_hours` as the standardized event-time surrogate.

- Positive label: the stay ends in ICU death and the ICU end-time proxy falls within the configured horizon.
- Negative label: no ICU death occurs within that horizon window under the same proxy.
- Caveat: exact death timestamps are not available in the standardized artifacts, so these are proxy-based within-horizon labels rather than exact event-time labels.

## Package Layout

The canonical implementation lives in [`src/chapter1_mortality_decomposition`](/Users/joanameyer/repository/1-mortality-decomposition/src/chapter1_mortality_decomposition):

- [`cohort.py`](/Users/joanameyer/repository/1-mortality-decomposition/src/chapter1_mortality_decomposition/cohort.py): site and stay exclusions
- [`instances.py`](/Users/joanameyer/repository/1-mortality-decomposition/src/chapter1_mortality_decomposition/instances.py): valid prediction instances
- [`labels.py`](/Users/joanameyer/repository/1-mortality-decomposition/src/chapter1_mortality_decomposition/labels.py): horizon label generation
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
- `labels/`: horizon labels, label summaries, and label notes
- `model_ready/`: selected feature set, model-ready table, and readiness summaries

## Tests

The synthetic end-to-end preprocessing tests live in [`tests/test_preprocessing.py`](/Users/joanameyer/repository/1-mortality-decomposition/tests/test_preprocessing.py).

Run them with:

```bash
python -m unittest discover -s tests -v
```

## Known Follow-Up

- Adult-age and mechanical-ventilation-duration cohort restrictions are still documented as follow-up because the standardized artifact column contract for them is not yet fixed in this standalone repo.
- The broader `docs/` folder still contains planning material copied from the upstream project context; the canonical preprocessing contract is the code and this README.
