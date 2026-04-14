# Chapter 1 Methods And Results Slide Spec

Source priority:
- Primary blueprint: `reports/ch1_methods_results_deck_overview.md`
- Frozen methods grounding: `docs/chapter1_analysis_spec_frozen_v1.md`, `docs/phase1_working_reference.md`, `docs/preprocessing_interface.md`, `docs/label_logic_audit.md`
- Current authoritative analysis state: `/Users/joanameyer/repository/phd-general/_context/context_sprint4.md`
- Authoritative empirical source: `cluster-results/chapter1_true_results/...`

Deck design note:
- Format: 16:9, white background, charcoal text, restrained teal accent, muted rust only for caveat boxes.
- Figure slides should use large visuals with one compact interpretation strip below the figure.
- Methods slides should use compact tables or schematics rather than dense prose.

Main deck count:
- 14 main slides
- 7 appendix slides

## Main Deck

## Slide 1 — Goal
- Slide number: 1
- Title: Goal
- Purpose: open the talk with the minimum technical framing needed.
- Exact on-slide content:
  - Stand-alone methods/results review of the current ASIC Chapter 1 analysis
  - Focus: cohort construction, valid-instance and proxy-label rules, frozen design choices, baseline models, hard-case analysis, horizon dependence
  - Emphasis: current exported ASIC results from `cluster-results`, not thesis framing
  - Bound the interpretation to the observed feature set, charting process, and temporal aggregation
- Exact artifact(s) used:
  - `reports/ch1_methods_results_deck_overview.md`
  - `docs/chapter1_analysis_spec_frozen_v1.md`
- Interpretive statement: this is a bounded analysis of near-term in-ICU mortality risk structure under the current recorded representation, not a generic mortality-prediction presentation.
- Status: ready

## Slide 2 — Frozen Study Setup
- Slide number: 2
- Title: Frozen Study Setup
- Purpose: establish the frozen analysis contract before operational implementation or results.
- Exact on-slide content:
  - Development dataset: ASIC
  - External-validation target: MIMIC-IV, not presented here because no result bundle is currently available
  - Endpoint: in-ICU mortality
  - Primary horizon: 24h; main contrast: 48h; sensitivities: 8h, 16h, 72h
  - Unit of analysis: patient-time prediction instances at completed 8h blocks in the first ICU stay
  - Minimum model set: logistic regression and XGBoost
  - Non-claims: no biological subtypes, no irreducible stochasticity, no causal attribution
- Exact artifact(s) used:
  - `docs/chapter1_analysis_spec_frozen_v1.md`
  - `config/ch1_run_config.json`
  - `config/ch1_feature_sets.json`
  - `docs/phase1_working_reference.md`
- Interpretive statement: the primary scientific choices were frozen early enough to separate design decisions from later empirical findings.
- Status: needs export

## Slide 3 — Cohort Construction And Exclusions
- Slide number: 3
- Title: Cohort Construction And Exclusions
- Purpose: show exactly how input hospitals and stays become the retained Chapter 1 cohort.
- Exact on-slide content:
  - Inclusion criteria: adult, mechanical ventilation `>=24h`, first ICU stay, valid in-ICU mortality label, at least one valid prediction instance
  - Site retention rule: usable ICU mortality plus at least 3 of 4 core physiologic groups with dynamic coverage
  - Input hospitals/stays: 8 hospitals, 15,969 stays
  - Retained hospitals/stays: 4 hospitals, 6,446 stays
  - Excluded sites: `asic_UK00`, `asic_UK01`, `asic_UK03`, `asic_UK06`
  - Main retained-site drops: `mech_vent_ge_24h_qc == False` and readmission-based first-stay filtering
  - Retained stays by hospital: `asic_UK02` 745, `asic_UK04` 470, `asic_UK07` 2,197, `asic_UK08` 3,034
- Exact artifact(s) used:
  - `cluster-results/chapter1_true_results/cohort/chapter1_site_eligibility.csv`
  - `cluster-results/chapter1_true_results/cohort/chapter1_stay_exclusion_summary_by_hospital.csv`
  - `cluster-results/chapter1_true_results/cohort/chapter1_counts_by_hospital.csv`
  - `cluster-results/chapter1_true_results/cohort/chapter1_cohort_summary.csv`
  - `docs/preprocessing_interface.md`
  - `docs/chapter1_analysis_spec_frozen_v1.md`
- Interpretive statement: most cohort contraction is caused by explicit site and stay eligibility logic, not by later model filtering.
- Status: needs export

## Slide 4 — Time Representation, Valid-Instance Rule, And Proxy Labels
- Slide number: 4
- Title: Time Representation, Valid-Instance Rule, And Proxy Labels
- Purpose: make the row-level analysis unit and label construction explicit.
- Exact on-slide content:
  - Time axis: completed 8h blocks
  - Valid instance requires:
    - patient alive and still in ICU at prediction time
    - sufficient observed data through the end of the block
    - at least 3 of 4 core groups observed in-block
    - unambiguous horizon-specific labelability
  - Current usable-block observation pattern:
    - 248,772 unique usable 8h blocks before horizon duplication
    - 69.4% with all 4 core groups observed
    - 30.6% with exactly 3 of 4
  - Proxy label rule:
    - positive if `icu_mortality == 1` and `icu_end_time_proxy_hours` in `(t, t+H]`
    - negative if `icu_mortality == 0` and `icu_end_time_proxy_hours >= t+H`
    - otherwise unlabeled
  - 24h counts: 231,596 labelable, 4,986 positive, 226,610 negative, 78,048 unlabeled
  - Explicit limitation: ASIC lacks true death and ICU-discharge timestamps for within-horizon labels
- Exact artifact(s) used:
  - `docs/chapter1_analysis_spec_frozen_v1.md`
  - `docs/preprocessing_interface.md`
  - `docs/label_logic_audit.md`
  - `cluster-results/chapter1_true_results/cohort/chapter1_cohort_summary.csv`
  - `cluster-results/chapter1_true_results/observation_process/chapter1_observation_process_qc_summary.csv`
  - `config/ch1_run_config.json`
- Interpretive statement: the analysis row set is intentionally restrictive, and both row inclusion and labelability are shaped by current-block measurement coverage plus the proxy ICU-end-time rule.
- Status: needs export

## Slide 5 — Frozen Modeling Design Choices
- Slide number: 5
- Title: Frozen Modeling Design Choices
- Purpose: summarize the feature boundary, preprocessing policy, split strategy, models, and metrics.
- Exact on-slide content:
  - Primary feature set: 31 routine variables; extended set: +15 sparse-lab variables
  - Model-ready export:
    - 186 selected blocked dynamic columns
    - 98 LOCF/missingness indicator columns
    - no final imputation in preprocessing export
  - Missingness policy:
    - bounded LOCF only for prespecified families
    - ventilator-variable LOCF only within upstream ventilation-supported windows
    - final imputation deferred to model training
  - Split strategy:
    - target `70/15/15`
    - operational split unit: `stay_id_global`
    - within retained hospitals
    - seed `20260327`
  - Baseline models: logistic regression, XGBoost
  - Metrics: AUROC, AUPRC, calibration intercept, calibration slope, Brier score
- Exact artifact(s) used:
  - `config/ch1_feature_sets.json`
  - `config/ch1_run_config.json`
  - `docs/preprocessing_interface.md`
  - `cluster-results/chapter1_true_results/model_ready/chapter1_primary_readiness_summary.csv`
  - `cluster-results/chapter1_true_results/carry_forward/chapter1_primary_locf_feature_summary.csv`
  - `docs/chapter1_analysis_spec_frozen_v1.md`
- Interpretive statement: the preprocessing and model-design choices are deliberately conservative, exposing missingness and using calibration-aware baselines rather than a large model zoo.
- Status: needs export

## Slide 6 — Retained Cohort And Realized Split Summary
- Slide number: 6
- Title: Retained Cohort And Realized Split Summary
- Purpose: show the realized scale of the true ASIC run and the internal split balance.
- Exact on-slide content:
  - 6,446 retained stays
  - 1,548,220 valid prediction instances across 5 horizons
  - 309,644 valid instances per horizon before horizon-specific labelability filtering
  - Realized stay splits:
    - train 4,511
    - validation 968
    - test 967
  - Stay-level mortality prevalence:
    - train 26.76%
    - validation 26.76%
    - test 26.78%
  - 24h test rows: 33,676 with 752 positives
- Exact artifact(s) used:
  - `cluster-results/chapter1_true_results/cohort/chapter1_cohort_summary.csv`
  - `cluster-results/chapter1_true_results/splits/chapter1_stay_split_summary.csv`
  - `cluster-results/chapter1_true_results/splits/chapter1_primary_split_summary.csv`
  - `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/reporting_split_summary.csv`
- Interpretive statement: the current ASIC bundle is a full-scale internal run with a balanced and binary-evaluable test split, not a local smoke-test artifact.
- Status: needs export

## Slide 7 — Baseline Model Performance And Calibration
- Slide number: 7
- Title: Baseline Model Performance And Calibration
- Purpose: present the headline quantitative baseline results.
- Exact on-slide content:
  - Report on the test split for all model-horizon pairs
  - 24h test metrics:
    - logistic regression: AUROC 0.819, AUPRC 0.268, slope 0.974, Brier 0.0186
    - XGBoost: AUROC 0.848, AUPRC 0.318, slope 1.162, Brier 0.1351
  - Logistic across horizons:
    - test AUROC 0.843 -> 0.791 from 8h to 72h
    - test calibration slope stays close to 1
  - Short note that ranking and calibration point in different directions for the two baseline models
- Exact artifact(s) used:
  - `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/combined_metrics.csv`
  - `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/reporting_split_summary.csv`
- Interpretive statement: XGBoost is the stronger ranker, but logistic regression is the cleaner probability model for a calibration-first interpretation of hard cases.
- Status: needs export

## Slide 8 — Primary 24h Mortality-Vs-Risk Structure
- Slide number: 8
- Title: Primary 24h Mortality-Vs-Risk Structure
- Purpose: show the main descriptive risk-structure result at the primary horizon.
- Exact on-slide content:
  - Large 24h logistic mortality-vs-risk plot
  - Companion 24h logistic reliability plot
  - Small footer strip with the 24h test metrics
- Exact artifact(s) used:
  - `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/logistic_regression/horizon_24h/mortality_vs_risk_plot.png`
  - `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/logistic_regression/horizon_24h/reliability_plot.png`
  - `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/combined_metrics.csv`
- Interpretive statement: the primary 24h logistic model preserves coherent risk ordering and acceptable calibration, which makes a bounded hard-case analysis interpretable.
- Status: ready

## Slide 9 — Hard-Case Definition And Burden
- Slide number: 9
- Title: Hard-Case Definition And Burden
- Purpose: define the operational hard-case rule and show its size at each horizon.
- Exact on-slide content:
  - Rule name: `asic_logistic_last_eligible_nonfatal_q75_v1`
  - One last eligible stay-level point per stay and horizon
  - Horizon-specific nonfatal q75 thresholds
  - Fatal hard-case share by horizon:
    - 8h 21.42%
    - 16h 20.48%
    - 24h 20.57%
    - 48h 20.74%
    - 72h 21.36%
  - 24h fatal stays: 1,682; hard cases: 346
- Exact artifact(s) used:
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/horizon_hard_case_summary.csv`
  - `docs/phase1_working_reference.md`
- Interpretive statement: under the frozen logistic rule, the low-predicted fatal subset is a substantial minority of fatal stays rather than a rare outlier set.
- Status: needs export

## Slide 10 — 24h Hard-Case Comparison
- Slide number: 10
- Title: 24h Hard-Case Comparison
- Purpose: show how low-predicted fatal stays differ from other fatal stays at the primary horizon.
- Exact on-slide content:
  - Large effect-size figure
  - Side box with core numbers:
    - fatal 24h stay-level slice 1,682
    - low-predicted fatal 346
    - other fatal 1,336
  - Side box with the strongest differences:
    - prediction time 176h vs 232h
    - MAP 81.0 vs 58.1
    - PF ratio 277.7 vs 172.3
    - PEEP 9.3 vs 10.1
    - modest enrichment in `asic_UK04` and neurologic group
  - Small footnote: exact age unavailable; age-group only in the current package
- Exact artifact(s) used:
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/effect_size_figure.png`
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/comparison_table.csv`
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/summary.md`
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison_variable_audit/asic_hard_case_comparison_variable_audit_memo.md`
- Local-review boundary:
  - Build this slide from the approved aggregate export bundle mirrored under `cluster-results/`.
  - Do not use `stay_level_comparison_dataset.csv` or other restricted row-level comparison tables as standard local presentation inputs.
- Interpretive statement: low-predicted fatal stays look less aligned with captured short-term physiologic severity, but the comparison remains descriptive and operational rather than typological.
- Status: ready

## Slide 11 — Horizon Dependence
- Slide number: 11
- Title: Horizon Dependence
- Purpose: show whether the hard-case burden and risk structure persist across the frozen horizon grid.
- Exact on-slide content:
  - Main horizon-comparison figure
  - Compact table with overlap summary:
    - mean pairwise Jaccard 0.828
    - 24h vs 48h Jaccard 0.885
    - 24h vs 72h Jaccard 0.824
    - run-manifest label `persist`
  - One footnote that the figure is descriptive and does not redefine the hard-case rule
- Exact artifact(s) used:
  - `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/final/mortality_risk_horizon_comparison.png`
  - `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/overlap/pairwise_overlap.csv`
  - `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/final/run_manifest.json`
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/horizon_hard_case_summary.csv`
- Interpretive statement: the current ASIC outputs support a descriptive persistence read, because the burden changes little across horizons and cross-horizon membership overlap is high.
- Status: ready

## Slide 12 — Secondary Robustness Summary
- Slide number: 12
- Title: Secondary Robustness Summary
- Purpose: summarize what is currently known about model dependence, site dependence, and temporal aggregation.
- Exact on-slide content:
  - Cross-model agreement at 24h:
    - logistic hard 346
    - XGBoost-Platt hard 227
    - overlap 188
    - Jaccard 0.488
  - Site sanity check:
    - pooled 24h result not obviously single-site-driven
    - site-level metrics are much sparser than pooled metrics
  - 8h vs 16h temporal preview:
    - logistic 24h AUROC 0.819 -> 0.816; AUPRC 0.268 -> 0.235
    - XGBoost 24h AUROC 0.848 -> 0.846; AUPRC 0.318 -> 0.291
  - Explicit status note:
    - cross-model agreement is mature enough for presentation
    - temporal preview is still provisional
- Exact artifact(s) used:
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/agreement/logistic_regression_vs_xgboost_platt/horizon_hard_case_agreement_summary.csv`
  - `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/combined_primary_site_summary.csv`
  - `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/logistic_regression/primary_24h_site_overview.png`
  - `cluster-results/chapter1_true_results/temporal_preview/asic/aggregation_16h/comparison/preview_note.md`
  - `cluster-results/chapter1_true_results/temporal_preview/asic/aggregation_16h/comparison/aggregation_comparison_metrics.csv`
- Interpretive statement: robustness is mixed rather than all-or-none, with strong cross-horizon persistence, only moderate cross-model agreement, and an encouraging but incomplete temporal preview.
- Status: provisional

## Slide 13 — Current Bounded Interpretation
- Slide number: 13
- Title: Current Bounded Interpretation
- Purpose: state the current technical readout in the most defensible form.
- Exact on-slide content:
  - Current ASIC analysis supports heterogeneous predictability under the observed feature set
  - Logistic 24h risk is a usable calibration-aware anchor for hard-case definition
  - About one-fifth of fatal stays are low-predicted under the frozen logistic rule
  - The low-predicted fatal burden persists descriptively across the frozen horizons
  - Hard cases are operational, model-dependent, and measurement-bound
  - Explicit non-claims: no biological subtypes, no irreducible randomness, no causal inference
- Exact artifact(s) used:
  - `docs/chapter1_analysis_spec_frozen_v1.md`
  - `docs/phase1_working_reference.md`
  - `/Users/joanameyer/repository/phd-general/_context/context_sprint4.md`
  - `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/combined_metrics.csv`
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/summary.md`
  - `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/final/run_manifest.json`
- Interpretive statement: the most defensible claim is about heterogeneous predictability under this recorded representation, not about discovered death classes.
- Status: ready

## Slide 14 — Open Limitations And Pending Analyses
- Slide number: 14
- Title: Open Limitations And Pending Analyses
- Purpose: close with unresolved methodological and interpretive gaps.
- Exact on-slide content:
  - ASIC lacks true death and ICU-discharge timestamps for within-horizon labels
  - Row inclusion depends on valid-instance and labelability gates
  - Operational split is stay-level because patient identifiers are unavailable
  - Current 24h hard-case package lacks exact age and some proxies rely on LOCF or partial coverage
  - Missing or incomplete sensitivity work:
    - observation-process hard-case comparison
    - treatment-limitation sensitivity or explicit absence note
    - formal temporal-aggregation sensitivity
    - disease-stratified predictability analyses
  - Practical note: some older markdown interpretation memos still contain stale synthetic/preliminary wording
- Exact artifact(s) used:
  - `docs/label_logic_audit.md`
  - `docs/preprocessing_interface.md`
  - `/Users/joanameyer/repository/phd-general/_context/context_sprint4.md`
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison_variable_audit/asic_hard_case_comparison_variable_audit_memo.md`
  - `cluster-results/chapter1_true_results/observation_process/chapter1_observation_process_qc_summary.csv`
  - `cluster-results/chapter1_true_results/temporal_preview/asic/aggregation_16h/comparison/preview_note.md`
- Interpretive statement: the current ASIC package is strong enough for a bounded methods/results presentation, but not yet closed on the most important interpretation-critical sensitivities.
- Status: ready

## Appendix

## Appendix A1 — Full Metrics By Model And Horizon
- Slide number: A1
- Title: Full Metrics By Model And Horizon
- Purpose: provide the full quantitative baseline table for reference.
- Exact on-slide content:
  - Full test-split metrics for logistic regression and XGBoost across 8h, 16h, 24h, 48h, 72h
  - Columns: sample count, event count, AUROC, AUPRC, calibration intercept, calibration slope, Brier
- Exact artifact(s) used:
  - `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/combined_metrics.csv`
- Interpretive statement: the ranking-versus-calibration tradeoff is consistent across horizons, not unique to 24h.
- Status: appendix only

## Appendix A2 — Site-Stratified 24h Sanity Check
- Slide number: A2
- Title: Site-Stratified 24h Sanity Check
- Purpose: show the available site-level backup view without making it a main-deck claim.
- Exact on-slide content:
  - 24h logistic site overview figure
  - Side table with 24h site-specific AUROC, AUPRC, calibration slope, and event counts
- Exact artifact(s) used:
  - `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/logistic_regression/primary_24h_site_overview.png`
  - `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/combined_primary_site_summary.csv`
- Interpretive statement: the pooled result is not obviously driven by one hospital, but site-level calibration remains a sparse secondary analysis.
- Status: appendix only

## Appendix A3 — Cross-Model Hard-Case Agreement By Horizon
- Slide number: A3
- Title: Cross-Model Hard-Case Agreement By Horizon
- Purpose: expand the model-dependence result beyond the 24h summary.
- Exact on-slide content:
  - Agreement table across 8h, 16h, 24h, 48h, 72h
  - Show logistic hard, XGBoost-Platt hard, overlap, Jaccard, and directional overlap
- Exact artifact(s) used:
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/agreement/logistic_regression_vs_xgboost_platt/horizon_hard_case_agreement_summary.csv`
- Interpretive statement: cross-model agreement improves somewhat at longer horizons but remains well short of model invariance.
- Status: appendix only

## Appendix A4 — Variable Audit For The 24h Hard-Case Comparison
- Slide number: A4
- Title: Variable Audit For The 24h Hard-Case Comparison
- Purpose: document which comparison variables are directly available, partially available, or unavailable.
- Exact on-slide content:
  - Exact age unavailable; age-group only
  - Respiratory, hemodynamic, renal, and ventilation proxy readiness summary
  - Coverage and LOCF dependence where relevant
- Exact artifact(s) used:
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison_variable_audit/asic_hard_case_comparison_variable_audit_memo.md`
- Local-review boundary:
  - This appendix slide should also rely on the approved mirrored aggregate exports rather than on the restricted row-level reconstruction table.
- Interpretive statement: the 24h comparison variable package is usable, but it is not a complete clinical adjustment set and some proxies rely materially on carry-forward.
- Status: appendix only

## Appendix A5 — Why SOFA Was Not Used
- Slide number: A5
- Title: Why SOFA Was Not Used
- Purpose: explain the decision to use direct proxies rather than a pseudo-SOFA score.
- Exact on-slide content:
  - Missing domains: CNS, vasopressors, urine output
  - Partial coverage and heavy LOCF dependence in available domains
  - Final classification: standard SOFA not feasible
- Exact artifact(s) used:
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_sofa_feasibility_audit/sofa_feasibility_memo.md`
- Interpretive statement: a transparent proxy table is cleaner and more reproducible than an incomplete pseudo-SOFA.
- Status: appendix only

## Appendix A6 — Observation-Process Readiness
- Slide number: A6
- Title: Observation-Process Readiness
- Purpose: show that the observation-process variable set exists, while being explicit that the actual sensitivity result is still missing.
- Exact on-slide content:
  - List of derived block-level observation-process variables
  - Coverage summary for `n_core_grps_obs_block` and time-since-last-observation variables
  - Explicit note that no observation-process hard-case comparison result is currently available
- Exact artifact(s) used:
  - `cluster-results/chapter1_true_results/observation_process/chapter1_observation_process_qc_summary.csv`
  - `cluster-results/chapter1_true_results/observation_process/chapter1_observation_process_implementation_note.md`
- Interpretive statement: the instrumentation is in place, but the explanatory sensitivity analysis has not yet been run.
- Status: appendix only

## Appendix A7 — Temporal Aggregation Preview: 8h Vs 16h
- Slide number: A7
- Title: Temporal Aggregation Preview: 8h Vs 16h
- Purpose: provide the current provisional time-grid sensitivity check.
- Exact on-slide content:
  - 24h logistic 8h-vs-16h mortality-vs-risk comparison figure
  - Small table with AUROC, AUPRC, slope, and Brier deltas for both models
  - Explicit provisional label
- Exact artifact(s) used:
  - `cluster-results/chapter1_true_results/temporal_preview/asic/aggregation_16h/comparison/logistic_regression_24h_mortality_vs_risk_8h_vs_16h.png`
  - `cluster-results/chapter1_true_results/temporal_preview/asic/aggregation_16h/comparison/aggregation_comparison_metrics.csv`
  - `cluster-results/chapter1_true_results/temporal_preview/asic/aggregation_16h/comparison/preview_note.md`
- Interpretive statement: the preview shows movement but no obvious collapse, which is encouraging but not enough to freeze a temporal-sensitivity conclusion.
- Status: appendix only

## Missing Exports / Cleanups Still Needed

- Polished cohort-flow figure that combines site eligibility and stay-level exclusions into one slide-ready visual.
- Compact frozen-definitions table for Slide 2.
- Valid-instance schematic for Slide 4.
- Proxy-label schematic with per-horizon labelable / positive / negative / unlabeled counts for Slide 4.
- Compact modeling-design table for Slide 5.
- Cleaned cohort-and-split summary table for Slide 6.
- Cleaned baseline performance table for Slide 7.
- Hard-case share by horizon mini-plot or polished summary table for Slide 9.
- Clean figure title or replacement interpretation note for the horizon figure in Slide 11, because companion markdown memos still contain stale synthetic wording.
- Optional cleaned secondary-robustness summary table for Slide 12 if the deck should not rely on text-only numbers.

## Slides Blocked By Missing Polished Artifacts

- Slide 2 — Frozen Study Setup
  - Blocker: no compact exported frozen-definitions figure/table yet.
- Slide 3 — Cohort Construction And Exclusions
  - Blocker: no polished cohort-flow / exclusion diagram yet.
- Slide 4 — Time Representation, Valid-Instance Rule, And Proxy Labels
  - Blocker: no ready-made valid-instance or label schematic.
- Slide 5 — Frozen Modeling Design Choices
  - Blocker: no compact exported methods table.
- Slide 6 — Retained Cohort And Realized Split Summary
  - Blocker: needs a cleaned presentation table assembled from multiple CSV summaries.
- Slide 7 — Baseline Model Performance And Calibration
  - Blocker: needs a cleaned presentation table rather than raw CSV.
- Slide 9 — Hard-Case Definition And Burden
  - Blocker: no polished hard-case-by-horizon visual or summary table export yet.

## Top 8 Slides For A Shortened Meeting Version

- Slide 1 — Goal
- Slide 2 — Frozen Study Setup
- Slide 3 — Cohort Construction And Exclusions
- Slide 4 — Time Representation, Valid-Instance Rule, And Proxy Labels
- Slide 7 — Baseline Model Performance And Calibration
- Slide 8 — Primary 24h Mortality-Vs-Risk Structure
- Slide 10 — 24h Hard-Case Comparison
- Slide 11 — Horizon Dependence
