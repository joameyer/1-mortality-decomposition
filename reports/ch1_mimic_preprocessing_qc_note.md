# Chapter 1 MIMIC Preprocessing QC Note

## Purpose

This report documents issue 5.1.c1: an aggregated QC/readiness audit over the MIMIC Chapter 1 preprocessing artifacts produced by b1-b5.

This is a QC/readiness audit only. It does not rebuild the cohort, blocks, preprocessing, labels, or models, and it does not compare distributions or performance.

## Input Artifacts Read

- MIMIC processed root: `/Users/joanameyer/data/mimic-iv/mimic-iv-3.1/1-mortality-decomposition/processed`
- Reused preprocessing output root: `/Users/joanameyer/data/mimic-iv/mimic-iv-3.1/1-mortality-decomposition/preprocessing_outputs`
- Horizon target root: `/Users/joanameyer/data/mimic-iv/mimic-iv-3.1/1-mortality-decomposition/horizon_targets`
- Upstream report root: `/Users/joanameyer/repository/1-mortality-decomposition/reports`
- Feature freeze CSV: `/Users/joanameyer/repository/1-mortality-decomposition/config/ch1_mimic_feature_freeze.csv`

## Verified Domains

- cohort flow counts
- ventilation inclusion gate and vent-vs-LOS QC counts via b1 flow/QC artifacts
- first-stay handling via b1 flow artifacts
- ICU timing/block counts via b2 block QC and stay block counts
- valid-instance counts via reused ASIC preprocessing outputs
- per-horizon conservative proxy-label counts via b4 outputs
- block-count distribution summaries
- feature coverage and missingness using blocked features and model-ready exports
- mapping/freeze quality proportions

## Partial Checks

- Feature missingness uses each variable's `{variable}_last` model-ready column as the representative preprocessed non-missingness proxy where exact per-base-variable post-LOCF counts are not separately exported.

## Full-MIMIC Use

Run the same QC script with path overrides pointing to the private full-MIMIC processed, preprocessing, horizon-target, and upstream-report roots. The script writes aggregated reports only; full-MIMIC row-level inputs should remain outside the repo.

Example:

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

Full-MIMIC scientific interpretation still depends on the user running this QC on the private full outputs and then reviewing downstream external-validation results.