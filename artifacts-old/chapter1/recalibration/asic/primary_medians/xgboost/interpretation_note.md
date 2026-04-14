# Chapter 1 ASIC XGBoost Recalibration Review

Recalibration methods fit on `validation` only and applied unchanged to `test`.
Reference comparisons include Logistic Regression and raw XGBoost from the saved baseline prediction artifacts.

The currently available test artifacts are not binary-evaluable for any horizon.
On this local artifact bundle, the test split has no events, so AUROC, AUPRC, calibration slope, and calibration intercept are unavailable on test.
Brier score and reliability-style plots were still written, but full probability interpretation should wait for a binary-evaluable frozen test artifact set.

- 8h test: logistic Brier 0.0010, raw XGBoost Brier 0.0009, Platt Brier 0.0001, isotonic Brier 0.0002.
- 16h test: logistic Brier 0.0009, raw XGBoost Brier 0.0051, Platt Brier 0.0003, isotonic Brier 0.0005.
- 24h test: logistic Brier 0.0022, raw XGBoost Brier 0.0081, Platt Brier 0.0006, isotonic Brier 0.0009.
- 48h test: logistic Brier 0.0074, raw XGBoost Brier 0.0688, Platt Brier 0.0031, isotonic Brier 0.0041.
- 72h test: logistic Brier 0.0150, raw XGBoost Brier 0.1046, Platt Brier 0.0141, isotonic Brier 0.0184.

Preliminary interpretation:
- Recalibration can improve XGBoost probability outputs without changing the underlying ranking model, but the current local bundle is too sparse on test to make a strong chapter-level claim.
- Logistic regression should remain the primary Chapter 1 anchor unless the full evaluable ASIC test bundle shows recalibrated XGBoost delivers stable probability improvement without losing the expected discrimination advantage.