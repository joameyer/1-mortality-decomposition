# Revised ASIC Interpretation Memo for Chapter 1

## Scope

This memo updates the Chapter 1 ASIC interpretation against the authoritative saved result bundle already present in this repository. It is a documentation synthesis only. No new modeling was run for this issue.

## Authoritative evidence base

- Baseline evaluation: `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/combined_metrics.csv`, `reporting_split_summary.csv`, and `run_manifest.json`.
- Frozen hard-case anchor and 24h comparison: `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/horizon_hard_case_summary.csv`, `stay_level_hard_case_flags.csv`, `asic_hard_case_comparison/comparison_table.csv`, and `asic_hard_case_comparison/summary.md`.
- Cross-model caution signal: `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/agreement/logistic_regression_vs_xgboost_platt/horizon_hard_case_agreement_summary.csv`.
- Horizon dependence: `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/foundation/horizon_summary.csv`, `overlap/pairwise_overlap.csv`, `overlap/persistence_distribution.csv`, `final/mortality_risk_horizon_binned_summary.csv`, and `final/run_manifest.json`.
- Temporal aggregation sensitivity: `cluster-results/chapter1_true_results/temporal_sensitivity/asic/comparison/reporting_metric_summary.csv`, `hard_case_prevalence_summary.csv`, `logistic_24h_hard_case_pairwise_overlap.csv`, and `temporal_aggregation_sensitivity_interpretation.md`.
- Observation-process sensitivity: `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_observation_process_sensitivity/memo.md` and `comparison_table.csv`, with sources recorded in the local run manifest and pointing back to the saved true-result bundle.
- Disease-stratified interpretation: `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_disease_stratified_predictability_structure/asic_disease_stratified_interpretation_memo.md` and `asic_disease_stratified_hardcase_summary.csv`.
- Site sensitivity and UK04 follow-up: `artifacts/chapter1/site_sensitivity/asic/site_enrichment_decision.md`, `site_hard_case_summary.csv`, `uk04_observation_process_interpretation.md`, and the companion run manifests.
- Feasibility limits relevant to interpretation: `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison_variable_audit/asic_hard_case_comparison_variable_audit_memo.md` and `asic_sofa_feasibility_audit/sofa_feasibility_memo.md`.

For scientific interpretation, the saved `cluster-results/chapter1_true_results/` bundle is the primary empirical record. Downstream local sensitivity notes are usable only where their run manifests show that they were computed directly from those saved true-result artifacts. Older synthetic, stand-in, or preview artifacts are implementation-test or precursor outputs only and should not be cited as scientific evidence.

## Revised bounded claim

In ASIC, short-horizon mortality risk under the frozen Chapter 1 representation shows a reproducible low-predicted fatal subset, indicating heterogeneous predictability structure within the observed ICU data. This pattern is sufficient to support a bounded descriptive hard-case chapter, but interpretation must remain conditional on the recorded feature set, documentation process, temporal aggregation, model class, and unresolved treatment-limitation confounding.

## Safe follow-up sentence

The ASIC results support a risk-structure-first reading in which some fatal stays appear less well captured by routinely observed short-horizon deterioration signals, but they do not support biological subtype claims, irreducible-stochasticity claims, or causal claims about monitoring, site, or treatment processes.

## Core descriptive read

- The primary 24h logistic anchor is scientifically usable in the saved cluster bundle: AUROC `0.819`, AUPRC `0.268`, calibration slope `0.974`, test rows `33,676`, and positives `752`.
- XGBoost ranks better at 24h (AUROC `0.848`, AUPRC `0.318`) but is less suitable as the Chapter 1 narrative anchor because calibration is materially less clean (slope `1.162`, intercept `-3.712`, Brier `0.135` versus logistic `0.019`).
- Under the frozen logistic last-eligible nonfatal-q75 rule, `346/1682` fatal 24h stays are low-predicted (`20.6%`).
- The same hard-case burden remains visible across all frozen horizons: `21.4%` at 8h, `20.5%` at 16h, `20.6%` at 24h, `20.7%` at 48h, and `21.4%` at 72h.
- Cross-model agreement is only moderate. At 24h, logistic and recalibrated XGBoost overlap on `188` hard cases with Jaccard `0.488`. This supports a bounded descriptive pattern, not a model-invariant entity.

## Integrated interpretation

- Observation-process differences plausibly explain some share of the hard-case pattern, but they do not collapse the descriptive core. The main observation-process separations are modest rather than overwhelming: all-4-core-group completeness `0.237`, stale monitoring `0.301`, and frozen-proxy missingness `0.304` standardized-difference units.
- Treatment-limitation confounding remains the biggest unresolved interpretation limit. No dedicated structured ASIC treatment-limitation sensitivity artifact was found in the saved bundle, and the saved feasibility notes only justify direct proxy use for organ-support and dysfunction variables, not a clean end-of-life or code-status sensitivity.
- Temporal aggregation partially weakens the pattern under coarsening. For logistic 24h, AUPRC declines from `0.268` at 8h to `0.235` at 16h and `0.217` at 24h aggregation, while the hard-case share rises from `0.206` to `0.248` and `0.254`. This is a real sensitivity, not an invariant result.
- Horizon dependence is best read as persistence with changing form, not strict horizon invariance and not latent subtype stability. Hard-case burden remains present across horizons and overlap is substantial (`24h` vs `48h` Jaccard `0.885`; `24h` vs `72h` Jaccard `0.824`), but membership is incomplete and the horizon-specific thresholds and risk-shape summaries still move.
- Disease-stratified heterogeneity is suggestive only. Surgical/postoperative/trauma-related (`20.0%`) and respiratory/pulmonary (`19.3%`) strata stay directionally aligned with the pooled pattern, while cardiovascular (`44.7%`) is descriptively elevated but based on limited fatal counts (`38`) and therefore remains cautious-only.
- Site enrichment is present but clearly modest. The saved site decision memo reports Cramer's V `0.126`, a maximum site standardized difference of `0.199`, and `asic_UK04` as the most enriched site (`37/108`, `34.3%`). The UK04 follow-up does not show strong measured observation-process differences in the direction needed for a documentation-only account, so the pattern should not be described as site-confined.

## Interpretation boundaries

- Do not describe these results as biological subtype discovery.
- Do not describe the low-predicted fatal subset as irreducibly unpredictable.
- Do not infer causal effects of monitoring intensity, hospital context, or treatment processes from these descriptive sensitivities.
- Keep all Chapter 1 language conditional on the observed feature set, the proxy label design, the documentation process, the chosen temporal aggregation, the model class, and unresolved treatment-limitation confounding.

## Baseline provenance reconciliation

- Earlier notes that describe the baseline package as sample-limited, smoke-test, or still waiting for a usable test split are now outdated for Chapter 1 interpretation.
- The saved baseline reporting summary shows that every reported model-horizon pair used the binary-evaluable test split.
- The baseline run manifest records the exact saved cluster prediction artifacts consumed for the evaluation package, so the current baseline interpretation should be tied to that saved bundle rather than to older local stand-ins.

## Chapter 1 consequence

ASIC now supports a bounded descriptive hard-case chapter on a risk-structure-first reading without needing decomposition to carry the argument. Decomposition should remain clearly secondary and easy to drop. The main remaining interpretive burden is not whether a low-predicted fatal subset exists, but how cautiously it must be described given observation-process dependence, temporal-aggregation sensitivity, model dependence, proxy labels, and unresolved treatment-limitation confounding.
