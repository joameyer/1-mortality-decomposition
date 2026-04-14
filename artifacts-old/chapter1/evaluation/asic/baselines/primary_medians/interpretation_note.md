# Chapter 1 ASIC Baseline Evaluation: First-Pass Interpretation

Models evaluated: Logistic Regression, XGBoost.
Horizons evaluated: 8h, 16h, 24h, 48h, 72h.
Reporting split usage across model-horizon pairs: validation=10.

At least one horizon had to fall back from the frozen test split to another existing split because the local sample test partition was not binary-evaluable.
Calibration curves are descriptive enough to inspect risk ordering on the currently selected holdout split, but they should still be treated as preliminary until the full ASIC test set is run.
No single hospital obviously dominates the primary-horizon evaluation rows in the current sample, although several site-level metrics remain sparse.

Primary 24h summary:
- Logistic Regression 24h used the `validation` split (AUROC 0.741, AUPRC 0.116, calibration slope 0.630).
- XGBoost 24h used the `validation` split (AUROC 0.514, AUPRC 0.032, calibration slope 0.016).

Caveats:
- This run is still sample-limited, especially on the frozen test split where local smoke-test artifacts can contain no events.
- Discrimination, calibration, and site-level plots should therefore be interpreted as pipeline-validation outputs first, and scientific evidence second.
- Later hard-case analysis should wait for the full ASIC evaluation run on a binary-evaluable holdout split.