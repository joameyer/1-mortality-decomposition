# Chapter 1 MIMIC Preprocessing QC

## Purpose

This note documents the issue 5.1.c1 preprocessing-verification/QC pipeline for MIMIC Chapter 1 artifacts. The goal is to verify that the artifacts produced by b1-b5 are internally coherent and ready for downstream external-validation work at the schema/count/readiness level.

This is not a redesign of cohort, block, preprocessing, or label logic. It does not fit models and does not compare MIMIC distributions or performance against ASIC.

## QC Domains Covered

The QC runner covers:

- cohort flow counts
- ventilation inclusion logic
- first-stay handling
- ICU timing and completed-block counts
- valid prediction-instance counts
- per-horizon conservative proxy-label counts
- block counts per stay
- feature coverage and model-ready missingness proxies
- mapping/freeze quality proportions

## Upstream Stages Audited

The QC reads artifacts produced by:

- b1 stay-level cohort extraction
- b2 completed 8h block construction
- b3 MIMIC-to-ASIC adapter and reused ASIC preprocessing core
- b4 conservative proxy horizon-label generation
- b5 artifact-contract check

The default demo configuration reads from:

- `mimic-iv-demo/data/processed`
- `mimic-iv-demo/data/preprocessing_outputs`
- `mimic-iv-demo/data/horizon_targets`
- `mimic-iv-demo/reports`

and writes aggregated QC reports to:

- `reports/`

## Successful Completion Criteria

For 5.1.c1, successful QC completion means:

- the QC script runs on MIMIC demo outputs without needing private full-MIMIC data
- all required aggregated reports are produced
- cohort flow, block counts, valid-instance counts, horizon label counts, and mapping/freeze summaries are populated from existing artifacts
- missing optional inputs are reported as partial/not-checkable rather than silently invented
- the same script can be pointed at full-MIMIC private output roots via CLI/config overrides

## Outputs

The QC runner writes:

- `reports/ch1_mimic_preprocessing_qc_summary.csv`
- `reports/ch1_mimic_cohort_flow_verification.csv`
- `reports/ch1_mimic_valid_instance_summary.csv`
- `reports/ch1_mimic_horizon_event_summary.csv`
- `reports/ch1_mimic_block_distribution_summary.csv`
- `reports/ch1_mimic_feature_missingness_summary.csv`
- `reports/ch1_mimic_mapping_quality_summary.csv`
- `reports/ch1_mimic_preprocessing_qc_note.md`

## Running On Demo Data

```bash
python scripts/run_mimic_preprocessing_qc.py
```

## Running On Full MIMIC

Full-MIMIC row-level inputs must remain outside the repo. Point the QC runner to the private full-output roots and keep only aggregated QC reports in `reports/`:

```bash
FULL_ROOT=/Users/joanameyer/data/mimic-iv/mimic-iv-3.1
CH1_ROOT=$FULL_ROOT/1-mortality-decomposition

python scripts/run_mimic_preprocessing_qc.py \
  --mimic-processed-root $CH1_ROOT/processed \
  --preprocessing-output-root $CH1_ROOT/preprocessing_outputs \
  --horizon-target-root $CH1_ROOT/horizon_targets \
  --upstream-reports-dir reports \
  --output-reports-dir reports
```

## Outside Scope

This QC does not cover:

- model fitting
- external-validation performance
- calibration or discrimination
- substantive demo-vs-full distribution comparison
- scientific interpretation of event rates or feature distributions
- redesign of feature mapping, cohort logic, block logic, preprocessing logic, or label logic
