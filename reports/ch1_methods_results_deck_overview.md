# ASIC Chapter 1 Frozen Figure And Table Plan

Scan date: 2026-04-16

## Scope

This note freezes the lean ASIC figure and table set for Phase 1 / Chapter 1 Issue 4.7. It is aligned to the revised risk-structure-first interpretation and the saved authoritative ASIC result bundle.

## Interpretation guardrails

- Use the saved `cluster-results/chapter1_true_results/` bundle as the primary empirical record.
- Use downstream local sensitivity notes only where their run manifests point back to that saved true-result bundle.
- Keep all wording conditional on the observed feature set, documentation process, temporal aggregation, model class, and unresolved treatment-limitation confounding.
- Do not present low-predicted fatal stays as biological subtypes, irreducibly stochastic deaths, or model-invariant entities.
- Treat decomposition as optional and secondary.

## Frozen Core Figures

### Figure 1. Primary 24h logistic mortality vs predicted risk

- Source: `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/logistic_regression/horizon_24h/mortality_vs_risk_plot.png`
- Why it stays: this is the main risk-structure figure and the cleanest visual motivation for the hard-case analysis.

### Figure 2. Primary 24h hard-case comparison

- Source: `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/effect_size_figure.png`
- Why it stays: this is the most compact figure for the low-predicted fatal versus other fatal contrast at the frozen 24h anchor.

### Figure 3. Horizon dependence under the frozen representation

- Source: `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/final/mortality_risk_horizon_comparison.png`
- Why it stays: this carries the core horizon read and should be captioned as persistence with changing form rather than strict invariance.

### Figure 4. Observation-process sensitivity summary

- Source: `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_observation_process_sensitivity/effect_size_figure.png`
- Why it stays: this is the cleanest figure for the documentation-process threat check and directly supports the bounded interpretation.

### Figure 5. Disease-stratified hard-case-share summary

- Source: `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_disease_stratified_predictability_structure/asic_disease_stratified_hardcase_share.png`
- Why it stays: this is the leanest way to show suggestive heterogeneity without drifting into subtype claims.

## Frozen Core Tables

### Table 1. ASIC cohort and hard-case anchor summary

- Sources:
  - `cluster-results/chapter1_true_results/cohort/chapter1_cohort_summary.csv`
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/horizon_hard_case_summary.csv`
- Content to retain:
  - retained stays, valid prediction instances, retained hospitals
  - frozen rule `asic_logistic_last_eligible_nonfatal_q75_v1`
  - 24h hard-case anchor `346/1682 (20.6%)`

### Table 2. Baseline performance and calibration across horizons

- Sources:
  - `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/combined_metrics.csv`
  - `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/reporting_split_summary.csv`
- Content to retain:
  - logistic and XGBoost test-split AUROC, AUPRC, calibration slope, Brier across `8h/16h/24h/48h/72h`
  - explicit note that all reported pairs use the binary-evaluable test split

### Table 3. Primary 24h low-predicted fatal versus other fatal comparison

- Source: `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/comparison_table.csv`
- Content to retain:
  - core timing and physiology contrasts
  - compact site and disease-group enrichment rows
  - explicit `n=346` versus `n=1336`

### Table 4. Observation-process sensitivity summary

- Source: `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_observation_process_sensitivity/comparison_table.csv`
- Content to retain:
  - all-4-core-group completeness
  - stale-core monitoring
  - frozen-proxy missingness
  - time-since-last-observation measures

### Table 5. Temporal aggregation sensitivity summary

- Sources:
  - `cluster-results/chapter1_true_results/temporal_sensitivity/asic/comparison/reporting_metric_summary.csv`
  - `cluster-results/chapter1_true_results/temporal_sensitivity/asic/comparison/hard_case_prevalence_summary.csv`
  - `cluster-results/chapter1_true_results/temporal_sensitivity/asic/comparison/logistic_24h_hard_case_pairwise_overlap.csv`
- Content to retain:
  - logistic 24h AUROC/AUPRC/slope across `8h`, `16h`, and `24h` aggregation
  - hard-case share changes under coarsening
  - pairwise Jaccard overlap across aggregations

### Table 6. Disease-stratified summary

- Source: `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_disease_stratified_predictability_structure/asic_disease_stratified_hardcase_summary.csv`
- Content to retain:
  - fatal counts, low-predicted fatal counts, hard-case share among fatal stays
  - adequacy flag and one-line interpretation per disease group

## Appendix / Backup Only

- Cross-model agreement table:
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/agreement/logistic_regression_vs_xgboost_platt/horizon_hard_case_agreement_summary.csv`
  - keep as backup because it is interpretively important but not part of the main story spine
- Site-sensitivity table:
  - `artifacts/chapter1/site_sensitivity/asic/site_hard_case_summary.csv`
  - `artifacts/chapter1/site_sensitivity/asic/uk04_observation_process_summary.csv`
  - keep as backup only because enrichment is modest and not additive enough for the core argument
- Logistic 24h reliability plot:
  - `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/logistic_regression/horizon_24h/reliability_plot.png`
  - useful as a supporting panel or appendix figure, but not essential if Table 2 already carries calibration clearly
- SOFA and variable-package feasibility notes:
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_sofa_feasibility_audit/sofa_feasibility_memo.md`
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison_variable_audit/asic_hard_case_comparison_variable_audit_memo.md`
  - keep as limitations backup only

## Drop Or Demote

- Drop decomposition-forward visuals from the main Chapter 1 set.
- Drop synthetic or stand-in horizon notes as scientific interpretation sources.
- Drop the earlier temporal preview from the core story:
  - `cluster-results/chapter1_true_results/temporal_preview/asic/aggregation_16h/comparison/preview_note.md`
  - `cluster-results/chapter1_true_results/temporal_preview/asic/aggregation_16h/comparison/logistic_regression_24h_mortality_vs_risk_8h_vs_16h.png`
- Drop redundant overlap displays from the main story:
  - `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/overlap/directional_overlap_heatmap.png`
  - `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/overlap/jaccard_heatmap.png`
  - `cluster-results/chapter1_true_results/temporal_sensitivity/asic/comparison/logistic_24h_hard_case_directional_overlap_heatmap.png`
  - `cluster-results/chapter1_true_results/temporal_sensitivity/asic/comparison/logistic_24h_hard_case_jaccard_heatmap.png`
- Drop wording that implies subtype discovery, ontological hard cases, or stochastic mortality classes.

## Final decision

- Main-story figures frozen: `5`
- Main-story tables frozen: `6`
- Site sensitivity, cross-model agreement, reliability, and feasibility notes remain appendix-only.
