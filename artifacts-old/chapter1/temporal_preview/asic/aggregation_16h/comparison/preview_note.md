# ASIC Temporal Aggregation Preview: 8h vs 16h

Across the comparable non-train holdout pairs, the 16h preview looks broadly similar rather than obviously unstable. On its own, it does not suggest that the Chapter 1 signal is an artifact of the 8h aggregation choice.
Calibration did not show an obvious across-the-board collapse.
The 24h mortality-vs-risk comparison should be read cautiously because at least one curve is sparse or visibly reshaped.

Comparable holdout pairs used for the compact stability summary: Logistic Regression 16h, Logistic Regression 24h, Logistic Regression 48h, Logistic Regression 72h, XGBoost 16h, XGBoost 24h, XGBoost 48h, XGBoost 72h.
Maximum absolute AUROC delta across comparable holdout pairs: 0.086.
Maximum absolute AUPRC delta across comparable holdout pairs: 0.048.
Maximum absolute Brier-score delta across comparable holdout pairs: 0.018.

Primary 24h comparison:
- Logistic Regression 24h: AUROC 0.741 -> 0.695, AUPRC 0.116 -> 0.114, slope 0.630 -> 0.545.
- XGBoost 24h: AUROC 0.514 -> 0.462, AUPRC 0.032 -> 0.035, slope 0.016 -> -0.071.

24h mortality-vs-risk structure:
- Logistic Regression 24h upper-half event share: 0.833 at 8h vs 0.500 at 16h.
- XGBoost 24h upper-half event share: 0.500 at 8h vs 0.500 at 16h.

Caveats:
- Only one alternative aggregation was tested, and it changes the prediction-time grid to completed 16h blocks.
- The preview reuses the frozen stay-level split assignments, but reporting may still fall back from test to validation when a split is not binary-evaluable.
- This is not a formal temporal-sensitivity analysis and should not be used to choose an optimal aggregation.
- Split selection was not directly comparable for: Logistic Regression 8h, XGBoost 8h.