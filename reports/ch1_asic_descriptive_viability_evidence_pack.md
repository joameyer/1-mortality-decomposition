# Sprint 3 ASIC Viability Evidence Pack

## Purpose
This document compresses the existing ASIC Sprint 3 hard-case and horizon-dependence artifacts into a short review pack for Issue 3.4. It is intentionally bounded to located artifacts and does not rerun the analyses. The discovered notes explicitly mark the local numbers as synthetic stand-in outputs, so this pack is for workflow and argument structure rather than scientific inference.

## Review mode
- Active result tier: `cluster export`.
- Local hard-case-comparison review is aggregate-only: use `comparison_table.csv`, `effect_size_plot_data.csv`, `summary.md`, the effect-size figure, approved early-vs-late outputs, and the paired variable-audit exports.
- Do not treat `stay_level_comparison_dataset.csv` or other restricted row-level reconstruction tables as normal local review inputs. Those remain cluster-side unless a separate export is explicitly approved.

## Evidence located
- `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/run_manifest.json`
- `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/horizon_hard_case_summary.csv`
- `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/comparison_table.csv`
- `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/summary.md`
- `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison_variable_audit/asic_hard_case_comparison_variable_audit_memo.md`
- `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/agreement/logistic_regression_vs_xgboost_platt/horizon_hard_case_agreement_summary.csv`
- `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/foundation/horizon_summary.csv`
- `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/overlap/pairwise_overlap.csv`
- `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/final/horizon_interpretation_memo.md`
- `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/final/mortality_risk_horizon_comparison.png`

## Hard-case definition summary
- Frozen rule located: `asic_logistic_last_eligible_nonfatal_q75_v1`.
- `24h` fatal comparison slice located: `346` low-predicted fatal vs `1336` other fatal stays (total `1682`).
- Hard-case count / threshold source: `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/horizon_hard_case_summary.csv`.
- Horizon-specific q75 thresholds and hard-case shares were extracted:

| horizon_label | nonfatal_last_n | fatal_last_n | nonfatal_q75_threshold | hard_case_n | hard_case_share_among_fatal |
| --- | --- | --- | --- | --- | --- |
| 8h | 4713 | 1639 | 0.004 | 351 | 0.214 |
| 16h | 4713 | 1670 | 0.009 | 342 | 0.205 |
| 24h | 4696 | 1682 | 0.015 | 346 | 0.206 |
| 48h | 4542 | 1697 | 0.032 | 352 | 0.207 |
| 72h | 4326 | 1704 | 0.053 | 364 | 0.214 |

## ASIC hard-case comparison summary
- Low-predicted fatal stays were modestly more common among `asic_UK04` fatal stays (37/346, 11%) than among other fatal stays (71/1336, 5%).
- Low-predicted fatal stays were modestly enriched in `neurologic` disease-group assignments (7/346, 2% vs 0%).
- Among the frozen timing and physiologic proxies, MAP was higher, PF ratio was modestly higher, and PEEP was modestly lower among low-predicted fatal stays.
- Main comparison table: `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/comparison_table.csv`.
- Main comparison figure: `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/effect_size_figure.png`.

## Horizon dependence summary
- Hard-case share: `8h` to `24h` stays at `0.21` to `0.21`, then is higher at `48h` `0.21` and `72h` `0.21`. For the narrative anchor and main contrast, `24h` is `0.21` and `48h` is `0.21`.
- Hard-case membership: overlap is substantial but incomplete. Mean pairwise Jaccard is `0.828`; `24h` vs `48h` has Jaccard `0.885`, with directional overlap `24h -> 48h` `0.934` and the reverse `0.944`.
- Mortality-vs-risk shape: the five binned panels keep the same pooled risk axis and remain broadly similar in overall upward mortality-with-risk structure. The weighted 24h vs 48h shape distance is `0.089`, which is small enough for a descriptive similarity read.
- Overall label: `persist`. On the local synthetic run, the low-risk fatal burden stays present across horizons, cross-horizon membership overlap is substantial, and the 24h vs 48h mortality-vs-risk panels remain broadly similar.
- Main overlap table: `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/overlap/pairwise_overlap.csv`.
- Main interpretation memo: `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/final/horizon_interpretation_memo.md`.

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
- Cross-model hard-case agreement is limited at 24h: logistic vs recalibrated XGBoost Jaccard 0.49 with logistic-hard confirmation by XGBoost 0.54.
- The saved horizon package labels the pattern 'persist', which is weaker than a clean single-form persistence story.

## Missing evidence / unresolved items
- No dedicated treatment-limitation or end-of-life proxy artifact was found in the searched ASIC Sprint 3 roots.
- No pre-existing ASIC viability memo artifact was found before this workflow; the decision state still had to be reconstructed from comparison and horizon notes.
- Only local synthetic stand-in outputs were located here. The same review must be rerun on full ASIC HPC artifacts before treating the memo as a scientific decision.
