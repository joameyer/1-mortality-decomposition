# ASIC Interpretation Evidence Pack

## Purpose

This pack indexes the authoritative artifacts that support the revised Chapter 1 ASIC interpretation. It is a companion inventory for documentation review, not a new analysis.

## Core empirical anchors

- Baseline calibration and discrimination:
  - `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/combined_metrics.csv`
  - `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/reporting_split_summary.csv`
  - Primary 24h logistic anchor: AUROC `0.819`, AUPRC `0.268`, calibration slope `0.974`, Brier `0.0186`.
- Frozen hard-case definition and burden:
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/horizon_hard_case_summary.csv`
  - 24h hard cases: `346/1682` fatal stays (`20.6%`).
- Primary 24h hard-case comparison:
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/comparison_table.csv`
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/effect_size_figure.png`
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/summary.md`
- Cross-model caution:
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/agreement/logistic_regression_vs_xgboost_platt/horizon_hard_case_agreement_summary.csv`
  - 24h logistic-vs-XGBoost Jaccard: `0.488`.
- Horizon dependence:
  - `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/foundation/horizon_summary.csv`
  - `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/overlap/pairwise_overlap.csv`
  - `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/overlap/persistence_distribution.csv`
  - `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/final/mortality_risk_horizon_binned_summary.csv`
  - 24h vs 48h Jaccard: `0.885`; 24h vs 72h Jaccard: `0.824`.
- Temporal aggregation sensitivity:
  - `cluster-results/chapter1_true_results/temporal_sensitivity/asic/comparison/reporting_metric_summary.csv`
  - `cluster-results/chapter1_true_results/temporal_sensitivity/asic/comparison/hard_case_prevalence_summary.csv`
  - `cluster-results/chapter1_true_results/temporal_sensitivity/asic/comparison/logistic_24h_hard_case_pairwise_overlap.csv`
  - Best summary label: partially weakened under coarsening, not aggregation-invariant.

## Downstream sensitivity notes grounded in the saved bundle

- Observation-process sensitivity:
  - `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_observation_process_sensitivity/memo.md`
  - `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_observation_process_sensitivity/comparison_table.csv`
- Disease-stratified interpretation:
  - `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_disease_stratified_predictability_structure/asic_disease_stratified_interpretation_memo.md`
  - `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_disease_stratified_predictability_structure/asic_disease_stratified_hardcase_summary.csv`
- Site sensitivity and UK04 follow-up:
  - `artifacts/chapter1/site_sensitivity/asic/site_enrichment_decision.md`
  - `artifacts/chapter1/site_sensitivity/asic/site_hard_case_summary.csv`
  - `artifacts/chapter1/site_sensitivity/asic/uk04_observation_process_interpretation.md`

These downstream notes are acceptable for interpretation because their saved run manifests point back to the authoritative `cluster-results/chapter1_true_results/` inputs.

## Feasibility and unresolved limits

- Variable-package feasibility:
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison_variable_audit/asic_hard_case_comparison_variable_audit_memo.md`
  - Main unresolved data issue: exact age absent; only `age_group` is available.
- SOFA feasibility:
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_sofa_feasibility_audit/sofa_feasibility_memo.md`
  - Final classification: `NOT FEASIBLE`.
- Treatment-limitation interpretation status:
  - No dedicated structured ASIC treatment-limitation sensitivity artifact is present in the reviewed saved bundle.
  - This remains a visible limitation rather than a closed sensitivity.

## Do Not Use As Scientific Evidence For Issue 4.7

- `cluster-results/chapter1_true_results/temporal_preview/asic/aggregation_16h/comparison/preview_note.md`
- `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/foundation/artifact_foundation_note.md`
- `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/foundation/horizon_summary.md`
- Older notes that describe the baseline or horizon bundles as synthetic, sample-limited, smoke-test, or implementation-only.

Those files remain useful for workflow history or precursor context, but they are not the scientific basis for the frozen Chapter 1 ASIC interpretation.
