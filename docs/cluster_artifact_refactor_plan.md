# Cluster-First Artifact Refactor Plan

## Purpose

Refactor the Chapter 1 repo so that:

- protected patient-level computation happens on the cluster only
- only allowed derived artifacts are exported back to the local repo
- local notebooks, reports, and downstream analysis default to authoritative cluster exports
- synthetic local data remains available for development, tests, and smoke checks only

This plan is designed for the current repo layout:

- local dev outputs: `artifacts/chapter1/`
- authoritative true-result exports: `cluster-results/chapter1_true_results/`
- cluster upload bundle: `hpc-1-mortality-decomposition/`

## Target State

### Data boundary

Cluster-only inputs:

- standardized ASIC static tables
- standardized ASIC dynamic tables
- blocked monitoring data
- model-ready row-level datasets
- any row-level data derived directly from blocked monitoring or restricted static attributes

Allowed local exports:

- model predictions
- evaluation metrics
- figures
- manifests
- run summaries
- aggregated comparison tables
- other approved derived artifacts that do not expose restricted patient/static data

### Repo roles

Local repo:

- source of truth for code
- synthetic development and tests
- local review of exported true results
- reporting, presentation building, and artifact interpretation

HPC bundle:

- execution bundle for protected-data stages
- producer of authoritative true artifacts
- exporter of approved result bundles

`cluster-results/`:

- local mirror of approved true-result exports
- authoritative local input for scientific interpretation
- preferred input root for review notebooks and report scripts

## Artifact Tiers

Use two explicit artifact tiers and stop treating them as interchangeable.

### Tier 1: Synthetic development artifacts

Path:

- `artifacts/chapter1/`

Purpose:

- unit tests
- smoke tests
- development debugging
- schema validation
- notebook prototyping only

Status:

- never scientific authority
- should be labeled clearly as synthetic/dev-only

### Tier 2: Authoritative cluster exports

Path:

- `cluster-results/chapter1_true_results/`

Purpose:

- scientific interpretation
- reports
- figures
- presentations
- downstream local artifact-driven analysis

Status:

- authoritative local result bundle
- should be the default input for review workflows whenever available

## Module Classification

### Cluster-only producers

These depend on protected upstream inputs or restricted row-level intermediates and should remain cluster-side.

- `cohort.py`
- `instances.py`
- `labels.py`
- `carry_forward.py`
- `model_ready.py`
- `splits.py`
- `observation_process.py`
- `pipeline.py`
- `cli.py`
- `temporal_blocks.py`
- `baseline_logistic.py`
- `baseline_xgboost.py`
- `temporal_preview.py`

Notes:

- `baseline_logistic.py` and `baseline_xgboost.py` still depend on `artifacts/chapter1/model_ready/...` and feature-set outputs, so they remain cluster-only producers.
- `temporal_preview.py` currently reruns preprocessing and baseline stages, so it also remains cluster-only.

### Local-safe artifact consumers

These already work from saved predictions or saved downstream artifacts and can run locally against exported true results.

- `baseline_evaluation.py`
- `xgboost_recalibration.py`
- `hard_case_definition.py`
- `hard_case_agreement.py`
- `asic_horizon_dependence_foundation.py`
- `asic_horizon_hard_case_stability.py`
- `asic_horizon_dependence_final.py`
- `ch1_asic_descriptive_viability.py`
- `scripts/build_ch1_methods_results_presentation.py`

Notes:

- These should be refactored to accept an explicit artifact root and prefer `cluster-results/chapter1_true_results/` for local scientific work.

### Mixed / needs redesign

These currently still reach back into restricted sources and need a new export contract before they can move local.

- `asic_hard_case_comparison.py`
- `scripts/ch1_asic_hard_case_comparison_variable_audit.py`
- `scripts/ch1_sofa_feasibility_audit.py`
- notebooks that still read model-ready, blocked, dynamic, or static row-level data

Why mixed:

- `asic_hard_case_comparison.py` still joins hard-case flags to `artifacts/chapter1/model_ready/...` and to harmonized static inputs for variables such as age, sex, and ICD-10-derived groupings.
- those inputs are exactly the kinds of restricted local data that should not be mirrored outside the cluster

Refactor implication:

- keep the row-level comparison build on the cluster
- export only approved aggregate outputs for local review
- do not try to make this fully local unless data-governance rules explicitly allow the required derived row-level table

## Notebook Classification

### Keep cluster-side or cluster-first

- `notebooks/ch1_preprocessing_runbook.ipynb`
- `notebooks/ch1_cohort_characterization.ipynb`
- `notebooks/ch1_observation_process_visualization.ipynb`
- `notebooks/ch1_baseline_model_readiness_check.ipynb`
- `notebooks/ch1_risk_trajectory_shapes.ipynb`
- `notebooks/ch1_asic_hard_case_comparison.ipynb`
- `notebooks/ch1_asic_temporal_aggregation_preview_16h.ipynb`

Reason:

- these likely depend on restricted row-level data or on re-running protected-data steps

### Convert to local review notebooks

- `notebooks/ch1_asic_baseline_evaluation_review.ipynb`
- `notebooks/ch1_xgboost_recalibration_review.ipynb`
- `notebooks/ch1_asic_hard_case_review.ipynb`
- `notebooks/ch1_asic_descriptive_viability_review.ipynb`

Reason:

- these are naturally artifact-review workflows and should prefer `cluster-results/`

## Required Export Contracts

Define approved export bundles per stage.

### Export bundle A: preprocessing summaries

Destination:

- `cluster-results/chapter1_true_results/cohort/`
- `cluster-results/chapter1_true_results/splits/`
- `cluster-results/chapter1_true_results/model_ready/`
- `cluster-results/chapter1_true_results/carry_forward/`
- `cluster-results/chapter1_true_results/observation_process/`

Contents:

- cohort summaries
- site eligibility tables
- stay exclusion summaries
- split summaries
- feature availability summaries
- model-readiness summaries
- carry-forward summaries
- observation-process QC summaries

Rule:

- aggregate only
- no blocked row-level tables
- no model-ready row-level dataset

### Export bundle B: baseline predictions and evaluation artifacts

Destination:

- `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/`

Contents:

- `combined_metrics.csv`
- `reporting_split_summary.csv`
- `combined_risk_binned_summary.csv`
- evaluation figures
- run manifests
- optionally saved predictions if permitted

Decision point:

- if prediction-level exports are allowed, export them because they unlock more local downstream work
- if prediction-level exports are not allowed, keep only summary artifacts local and leave more downstream stages cluster-side

### Export bundle C: hard-case artifacts

Destination:

- `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/...`

Contents:

- saved hard-case flags
- hard-case summary tables
- agreement summary tables
- approved aggregate comparison tables
- figures
- manifests

Rule:

- row-level comparison data that still contains restricted static-derived information should remain cluster-only
- export only aggregate outputs unless governance explicitly allows more

### Export bundle D: temporal-preview artifacts

Destination:

- `cluster-results/chapter1_true_results/temporal_preview/...`

Contents:

- comparison metrics
- comparison note
- comparison figures
- run manifest

## Code Refactor Workstreams

### Workstream 1: make artifact roots explicit

Goal:

- stop hard-coding assumptions that `artifacts/chapter1/` is always the right source for scientific review

Changes:

- extend [artifacts.py](/Users/joanameyer/repository/1-mortality-decomposition/src/chapter1_mortality_decomposition/artifacts.py:1) with result-root helpers
- add a small path-resolution layer for:
  - synthetic dev root
  - cluster-results root
  - cluster-run artifact root
- teach downstream artifact consumers to accept explicit input roots instead of assuming local synthetic artifacts

Recommended new concept:

- `analysis_profile = synthetic_local | cluster_export`

Recommended behavior:

- local review commands prefer `cluster-results/chapter1_true_results/` if present
- fall back to `artifacts/chapter1/` only for development or tests

### Workstream 2: separate producer jobs from review jobs

Goal:

- make it obvious which commands are cluster execution jobs and which are local artifact-review jobs

Changes:

- rename documentation around cluster-only stages as `produce_*`
- rename local review documentation around artifact-driven stages as `review_*` or `analyze_*`
- keep current module names if needed, but clarify role in README/docs

Example split:

- cluster producer jobs:
  - preprocess
  - train baselines
  - build temporal preview
  - build hard-case comparison aggregates
- local review jobs:
  - evaluate exported predictions
  - compare recalibration variants
  - review hard-case outputs
  - build deck/report assets

### Workstream 3: export-safe derived tables for mixed analyses

Goal:

- reduce how much downstream work must stay on the cluster

Changes:

- for each mixed analysis, define the smallest safe exported table that still supports local review

Priority case:

- `asic_hard_case_comparison.py`

Recommended split:

- cluster-side step builds the row-level merge and the final aggregate outputs
- local side reads only:
  - `comparison_table.csv`
  - `standardized_difference_details.csv`
  - `summary.md`
  - figure files
  - run manifest

Do not export:

- row-level stay table with age, sex, ICD-10-derived categories, or other restricted static content unless explicitly approved

### Workstream 4: convert notebooks to review-first mode

Goal:

- stop showing synthetic outputs as if they were meaningful local results

Changes:

- add a top cell in review notebooks that resolves the artifact root
- display whether the notebook is reading:
  - `cluster-results/chapter1_true_results`
  - or `artifacts/chapter1`
- hard fail or show a warning banner when only synthetic data is available

Recommended notebook banner text:

- `Authoritative mode: cluster export`
- `Development mode: synthetic local artifacts only`

### Workstream 5: shrink the HPC bundle

Goal:

- reduce duplication and keep the upload bundle focused on protected-data execution

Changes:

- keep only cluster-relevant code, configs, job wrappers, and any notebooks genuinely needed on cluster
- remove local-only review notebooks, local reports, and presentation-generation scripts from the bundle unless explicitly needed on the cluster

Likely removable from HPC bundle later:

- report-building scripts that consume `cluster-results/`
- local presentation docs
- local review-only notebooks

## Phased Implementation Plan

## Phase 1: Establish the boundary

Target outcome:

- no ambiguity about which artifact tree is authoritative

Tasks:

- add this plan to docs
- update `README.md` with a clear synthetic-vs-cluster-results distinction
- add a short `cluster-results/README.md` describing what may be mirrored locally
- add path-resolution helpers in `artifacts.py` or a new dedicated module
- document the producer/consumer split in `hpc-1-mortality-decomposition/README.md`

Acceptance criteria:

- a new contributor can tell in under five minutes which outputs are synthetic and which are authoritative

## Phase 2: Make downstream analysis artifact-root aware

Target outcome:

- local downstream analysis can run against exported true artifacts without touching restricted data

Tasks:

- refactor local-safe artifact consumers to accept explicit input roots
- default local review workflows to `cluster-results/chapter1_true_results/`
- keep tests on synthetic fixtures

Priority modules:

- `baseline_evaluation.py`
- `xgboost_recalibration.py`
- `hard_case_definition.py`
- `hard_case_agreement.py`
- `asic_horizon_dependence_foundation.py`
- `asic_horizon_hard_case_stability.py`
- `asic_horizon_dependence_final.py`
- `ch1_asic_descriptive_viability.py`

Acceptance criteria:

- these modules can be run locally against exported true artifacts without any dependency on protected upstream tables

## Phase 3: Convert review notebooks and reports

Target outcome:

- local notebooks become trustworthy review interfaces for true exported results

Tasks:

- convert review notebooks to explicit artifact-root loading
- add warning banners when running in synthetic mode
- update report scripts to only consume `cluster-results/` for scientific outputs

Acceptance criteria:

- opening a review notebook locally no longer implies that synthetic outputs are scientifically meaningful

## Phase 4: Redesign mixed analyses around safe exports

Target outcome:

- only the minimal irreducible parts remain cluster-only

Tasks:

- split `asic_hard_case_comparison.py` into:
  - cluster-side build/export step
  - local review step
- do the same for variable audit and SOFA feasibility audit if needed
- define approved export schemas for any new aggregate tables

Acceptance criteria:

- local scientific interpretation relies on exported aggregate artifacts, not on restricted row-level joins

## Phase 5: Shrink and stabilize the HPC bundle

Target outcome:

- the HPC bundle contains only what is needed for protected-data execution and export production

Tasks:

- remove review-only payload from `hpc-1-mortality-decomposition/`
- keep job wrappers, cluster configs, and producer code
- add one documented export-sync step from cluster artifacts to local `cluster-results/`

Acceptance criteria:

- code duplication between local repo and HPC bundle is reduced
- cluster upload size and maintenance burden drop materially

## Recommended File-Level Changes

### Add or extend path-resolution helpers

Preferred options:

- extend `src/chapter1_mortality_decomposition/artifacts.py`
- or add `src/chapter1_mortality_decomposition/result_roots.py`

Responsibilities:

- resolve synthetic root
- resolve cluster-results root
- validate expected subtrees
- expose helpers for review modules

### Add local result-sync documentation

Add:

- `cluster-results/README.md`

Contents:

- what may be copied from cluster
- what must never be copied
- expected folder layout
- how local review workflows consume this tree

### Add cluster export script

Add in HPC bundle:

- a script that copies approved outputs from cluster `artifacts/chapter1/` into an export staging tree that mirrors `cluster-results/chapter1_true_results/`

Purpose:

- make exporting deliberate and repeatable
- avoid ad hoc manual copying

## Risks And Guardrails

### Risk: local notebooks silently fall back to synthetic outputs

Guardrail:

- show explicit mode banners
- require an override flag to permit synthetic fallback in review notebooks

### Risk: over-exporting restricted data

Guardrail:

- define approved export schemas per stage
- default to aggregates, manifests, and figures
- do not export row-level tables unless explicitly approved

### Risk: artifact contracts drift between local and cluster code

Guardrail:

- version export manifests
- add lightweight schema checks for exported result bundles
- keep small synthetic tests for artifact consumers

### Risk: mixed analyses remain stuck on the cluster

Guardrail:

- redesign them around aggregate exports rather than trying to mirror forbidden row-level inputs locally

## First Three Concrete Tasks

1. Add explicit artifact-root helpers and update docs so `cluster-results/` is the authoritative local result root.
2. Refactor the local-safe artifact consumers to accept a true-result root and default to `cluster-results/chapter1_true_results/`.
3. Convert the review notebooks and reporting scripts to show clear `cluster export` versus `synthetic dev` mode.

## Recommended Next Refactor Order

1. `artifacts.py` / path-resolution layer
2. `baseline_evaluation.py`
3. `xgboost_recalibration.py`
4. `hard_case_definition.py`
5. `hard_case_agreement.py`
6. `asic_horizon_dependence_foundation.py`
7. `asic_horizon_hard_case_stability.py`
8. `asic_horizon_dependence_final.py`
9. review notebooks
10. `asic_hard_case_comparison.py` split into cluster-build and local-review roles

## Definition Of Done

This refactor is done when:

- local scientific review no longer depends on synthetic outputs by default
- cluster-only steps are clearly separated from local artifact-review steps
- `cluster-results/chapter1_true_results/` is the authoritative local source for true outputs
- restricted row-level data is never mirrored locally
- the HPC bundle is smaller and focused on protected-data execution
