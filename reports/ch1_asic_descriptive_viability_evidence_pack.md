# Sprint 3 ASIC Viability Evidence Pack

## Purpose
This document compresses the existing ASIC Sprint 3 hard-case and horizon-dependence artifacts into a short review pack for Issue 3.4. It is intentionally bounded to located artifacts and does not rerun the analyses. The discovered notes explicitly mark the local numbers as synthetic stand-in outputs, so this pack is for workflow and argument structure rather than scientific inference.

## Evidence located
- `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/run_manifest.json`
- `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/horizon_hard_case_summary.csv`
- `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/comparison_table.csv`
- `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/summary.md`
- `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison_variable_audit/asic_hard_case_comparison_variable_audit_memo.md`
- `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/agreement/logistic_regression_vs_xgboost_platt/horizon_hard_case_agreement_summary.csv`
- `artifacts/chapter1/evaluation/asic/horizon_dependence/foundation/horizon_summary.csv`
- `artifacts/chapter1/evaluation/asic/horizon_dependence/overlap/pairwise_overlap.csv`
- `artifacts/chapter1/evaluation/asic/horizon_dependence/final/horizon_interpretation_memo.md`
- `artifacts/chapter1/evaluation/asic/horizon_dependence/final/mortality_risk_horizon_comparison.png`

## Hard-case definition summary
- Frozen rule located: `asic_logistic_last_eligible_nonfatal_q75_v1`.
- `24h` fatal comparison slice located: `4` low-predicted fatal vs `6` other fatal stays (total `10`).
- Hard-case count / threshold source: `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/horizon_hard_case_summary.csv`.
- Horizon-specific q75 thresholds and hard-case shares were extracted:

| horizon_label | nonfatal_last_n | fatal_last_n | nonfatal_q75_threshold | hard_case_n | hard_case_share_among_fatal |
| --- | --- | --- | --- | --- | --- |
| 8h | 25 | 10 | 0.008 | 4 | 0.400 |
| 16h | 25 | 10 | 0.028 | 4 | 0.400 |
| 24h | 25 | 10 | 0.041 | 4 | 0.400 |
| 48h | 24 | 10 | 0.110 | 5 | 0.500 |
| 72h | 22 | 10 | 0.164 | 6 | 0.600 |

## ASIC hard-case comparison summary
- Low-predicted fatal stays were more common among `asic_UK07` fatal stays (3/4, 75%) than among other fatal stays (0/6, 0%).
- Low-predicted fatal stays were enriched in `respiratory / pulmonary` disease-group assignments (3/4, 75% vs 17%).
- Among the frozen timing and physiologic proxies, PF ratio was higher, creatinine was lower, and PEEP was higher among low-predicted fatal stays.
- Main comparison table: `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/comparison_table.csv`.
- Main comparison figure: `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/effect_size_figure.png`.

## Horizon dependence summary
- Hard-case share: `8h` to `24h` stays at `0.40` to `0.40`, then is higher at `48h` `0.50` and `72h` `0.60`. For the narrative anchor and main contrast, `24h` is `0.40` and `48h` is `0.50`.
- Hard-case membership: overlap is substantial but incomplete. Mean pairwise Jaccard is `0.690`; `24h` vs `48h` has Jaccard `0.500`, with directional overlap `24h -> 48h` `0.750` and the reverse `0.600`.
- Mortality-vs-risk shape: the five binned panels still show mortality increasing with risk, but the pooled-bin profile shifts enough between 24h and 48h to count as a material descriptive change in this local run. The weighted 24h vs 48h shape distance is `0.256`.
- Overall label: `change form`. On the local synthetic run, the balance between share, membership overlap, and the binned mortality-vs-risk panels is not cleanly captured by a simple persistence story, so change form is the closest label.
- Main overlap table: `artifacts/chapter1/evaluation/asic/horizon_dependence/overlap/pairwise_overlap.csv`.
- Main interpretation memo: `artifacts/chapter1/evaluation/asic/horizon_dependence/final/horizon_interpretation_memo.md`.

## Preliminary decision-relevant interpretation
- What strengthens descriptive viability:
- The hard-case rule is frozen and recoverable from a saved manifest rather than implied retrospectively.
- A concrete low-predicted-versus-other-fatal comparison package exists with tables, a figure, and a short summary note.
- Horizon dependence was materialized into summary tables, overlap tables, and a final interpretation memo rather than left implicit.
- What weakens descriptive viability:
- The located notes repeatedly say the local values are synthetic implementation-test outputs, so the current readout is not a scientific claim yet.
- The local comparison slice is very small and explicitly flagged as a bounded descriptive comparison.
- The final horizon package labels the pattern as changing form rather than a clean stable subtype.
- What strengthens decomposition:
- There is at least some recurring low-risk fatal structure across horizons, so a secondary summary device is not obviously pointless.
- The artifact set is now organized enough to ask whether decomposition adds anything beyond the descriptive hard-case story.
- What weakens decomposition:
- The descriptive story already has its own rule, comparison table, and horizon memo, so decomposition is not needed to make Chapter 1 legible.
- Cross-model hard-case agreement is limited in the saved agreement summary, which weakens confidence in a fragile summary-model layer.
- Key sensitivity pieces remain incomplete or negative, including the variable-package readiness gap and the non-feasible SOFA route.

## Main remaining risks
- All discovered hard-case and horizon notes explicitly describe the local values as synthetic implementation-test outputs, so the current readout is workflow-valid but not scientifically interpretable.
- Frozen Issue 3.2 variable-package status: ISSUE 3.2 VARIABLE PACKAGE NOT YET READY. Blocking family: age.
- SOFA feasibility audit result: NOT FEASIBLE.
- Cross-model hard-case agreement is limited at 24h: logistic vs recalibrated XGBoost Jaccard 0.20 with logistic-hard confirmation by XGBoost 0.25.
- The saved horizon package labels the pattern 'change form', which is weaker than a clean single-form persistence story.

## Missing evidence / unresolved items
- No dedicated treatment-limitation or end-of-life proxy artifact was found in the searched ASIC Sprint 3 roots.
- No pre-existing ASIC viability memo artifact was found before this workflow; the decision state still had to be reconstructed from comparison and horizon notes.
- Only local synthetic stand-in outputs were located here. The same review must be rerun on full ASIC HPC artifacts before treating the memo as a scientific decision.
