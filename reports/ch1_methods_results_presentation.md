# Chapter 1 Methods And Results Presentation Draft

Authoritative empirical source:
- `cluster-results/chapter1_true_results/...`

Build note:
- This draft follows `reports/ch1_methods_results_slide_spec.md` and `reports/ch1_methods_results_deck_overview.md`.
- Slides marked for exported tables or schematics should be built from the listed artifacts rather than replaced with improvised content.

Visual direction:
- White background, dark charcoal text, muted teal accent for methods tables, muted rust for caveat strips.
- Large figures, short captions, one bounded interpretation sentence on every figure slide.

## Main Deck

## Slide 1 — Goal
Title:
- Goal

Subtitle:
- Current ASIC Chapter 1 analysis: methods, frozen design choices, and exported results

On-slide bullets:
- Stand-alone methods/results review of the current ASIC analysis
- Focus: cohort construction, valid-instance and proxy-label rules, baseline models, hard-case analysis, horizon dependence
- Empirical source: `cluster-results`, treated as authoritative over older local artifacts
- Interpretation kept measurement-bound to the recorded feature set, charting process, and temporal aggregation

Figure/table placement note:
- No main figure.
- Optional small callout box on the right: `ASIC current result bundle / cluster-results authoritative`.

Figure caption:
- None.

Speaker notes:
- This talk is intentionally narrow. I am not framing it as a PhD chapter talk or a broad motivation talk; the point is to review the current ASIC methods and results package as it exists now.
- I will focus on what was frozen up front, how rows and stays are retained or dropped, how proxy labels are constructed, what the baselines show, and what the current hard-case and horizon results support.

## Slide 2 — Frozen Study Setup
Title:
- Frozen Study Setup

Subtitle:
- Frozen v1 analysis contract

On-slide bullets:
- Development dataset: ASIC
- External-validation target: MIMIC-IV, not shown here because no result bundle is available yet
- Endpoint: in-ICU mortality
- Horizons: primary 24h; main contrast 48h; sensitivities 8h, 16h, 72h
- Unit: patient-time prediction instances at completed 8h blocks in the first ICU stay
- Minimum model set: logistic regression and XGBoost
- Non-claims: no biological subtypes, no irreducible stochasticity, no causal attribution

Figure/table placement note:
- Full-width compact frozen-definitions table to be exported from:
  - `docs/chapter1_analysis_spec_frozen_v1.md`
  - `config/ch1_run_config.json`
  - `config/ch1_feature_sets.json`
- Cleanup/export note: this needs a polished methods table export.

Figure caption:
- Frozen analysis design used for the current ASIC Chapter 1 implementation.

Speaker notes:
- The main point here is that the scientific contract was defined before looking at the current results. That includes the endpoint, the horizon hierarchy, the time representation, the minimum baseline models, and the non-claims.
- I want that visible early because later results need to be interpreted relative to this frozen design rather than as post hoc choices.

## Slide 3 — Cohort Construction And Exclusions
Title:
- Cohort Construction And Exclusions

Subtitle:
- From standardized ASIC input to the retained Chapter 1 cohort

On-slide bullets:
- Inclusion: adult, mechanical ventilation `>=24h`, first ICU stay, valid in-ICU mortality label, `>=1` valid prediction instance
- Site retention requires usable ICU mortality and at least 3 of 4 core physiologic groups with dynamic coverage
- Input: 8 hospitals, 15,969 stays
- Retained: 4 hospitals, 6,446 stays
- Retained hospitals: `asic_UK02`, `asic_UK04`, `asic_UK07`, `asic_UK08`
- Main retained-site drops: failed `mech_vent_ge_24h_qc` and readmission-based first-stay filtering

Figure/table placement note:
- Main visual: cohort flow / exclusion diagram to be exported from:
  - `cluster-results/chapter1_true_results/cohort/chapter1_site_eligibility.csv`
  - `cluster-results/chapter1_true_results/cohort/chapter1_stay_exclusion_summary_by_hospital.csv`
  - `cluster-results/chapter1_true_results/cohort/chapter1_counts_by_hospital.csv`
  - `cluster-results/chapter1_true_results/cohort/chapter1_cohort_summary.csv`
- Add a small side table with retained stays by hospital:
  - `asic_UK02` 745
  - `asic_UK04` 470
  - `asic_UK07` 2,197
  - `asic_UK08` 3,034
- Cleanup/export note: polished flow figure still needed.

Figure caption:
- Site- and stay-level eligibility logic for the retained ASIC Chapter 1 cohort.

Speaker notes:
- The critical point is that most contraction happens through explicit site and stay eligibility logic, not because of downstream model filtering.
- Site exclusion is driven by ICU-mortality usability and core-vital coverage, and then within retained sites the main drops come from `mech_vent_ge_24h_qc` failure and readmission-based first-stay filtering.

## Slide 4 — Time Representation, Valid-Instance Rule, And Proxy Labels
Title:
- Time Representation, Valid-Instance Rule, And Proxy Labels

Subtitle:
- What counts as an analyzable prediction row

On-slide bullets:
- Time axis: completed 8h blocks
- Valid instance requires:
- Alive and still in ICU at prediction time
- Sufficient observed data through the end of the block
- At least 3 of 4 core groups observed in-block
- Horizon-specific labelability must be unambiguous
- Current usable-block pattern: 248,772 unique 8h blocks; 69.4% with all 4 core groups, 30.6% with exactly 3
- Proxy labels use `icu_mortality` and `icu_end_time_proxy_hours`
- 24h counts: 231,596 labelable; 4,986 positive; 226,610 negative; 78,048 unlabeled

Figure/table placement note:
- Left: methods schematic showing completed 8h block -> valid instance gate -> horizon-specific proxy label.
- Right: compact labelability table by horizon.
- Build from:
  - `docs/chapter1_analysis_spec_frozen_v1.md`
  - `docs/preprocessing_interface.md`
  - `docs/label_logic_audit.md`
  - `cluster-results/chapter1_true_results/cohort/chapter1_cohort_summary.csv`
  - `cluster-results/chapter1_true_results/observation_process/chapter1_observation_process_qc_summary.csv`
- Cleanup/export note: both the valid-instance schematic and label table still need to be exported.

Figure caption:
- Valid-instance construction and proxy within-horizon mortality labeling in the current ASIC pipeline.

Speaker notes:
- This is one of the most important slides because the row set is highly structured. Rows are not all blocked timepoints; they are filtered by current-block physiologic coverage and by whether a horizon-specific proxy label can be defined.
- The other key point is that these are explicit proxy within-horizon labels. ASIC does not provide true death or ICU-discharge timestamps, so the within-horizon rule is based on `icu_end_time_proxy_hours`, and unlabeled rows are left unlabeled rather than forced negative.

## Slide 5 — Frozen Modeling Design Choices
Title:
- Frozen Modeling Design Choices

Subtitle:
- Feature boundary, preprocessing, split, models, and metrics

On-slide bullets:
- Primary feature set: 31 routine variables; extended set adds 15 sparse-lab variables
- Model-ready export: 186 selected dynamic columns plus 98 LOCF/missingness indicator columns
- Bounded LOCF only for prespecified families
- Ventilator-variable LOCF only inside upstream ventilation-supported windows
- No global median imputation in preprocessing export
- Split: operationally stay-level within retained hospitals; target `70/15/15`; seed `20260327`
- Models: logistic regression and XGBoost
- Metrics: AUROC, AUPRC, calibration intercept, calibration slope, Brier score

Figure/table placement note:
- Full-width methods table to be exported from:
  - `config/ch1_feature_sets.json`
  - `config/ch1_run_config.json`
  - `docs/preprocessing_interface.md`
  - `cluster-results/chapter1_true_results/model_ready/chapter1_primary_readiness_summary.csv`
  - `cluster-results/chapter1_true_results/carry_forward/chapter1_primary_locf_feature_summary.csv`
- Cleanup/export note: compact methods table still needed.

Figure caption:
- Frozen modeling and preprocessing choices used in the current ASIC analysis.

Speaker notes:
- The main design principle is restraint. The feature boundary is routine ICU data, carry-forward is bounded rather than aggressive, missingness is explicitly exposed, and final imputation is deferred to the model-training stage.
- The split is operationally stay-level because ASIC lacks patient identifiers, which is important to surface explicitly before showing performance results.

## Slide 6 — Retained Cohort And Realized Split Summary
Title:
- Retained Cohort And Realized Split Summary

Subtitle:
- Scale of the true ASIC run

On-slide bullets:
- 6,446 retained stays
- 1,548,220 valid prediction instances across 5 frozen horizons
- 309,644 valid instances per horizon before horizon-specific labelability filtering
- Realized stay splits: train 4,511; validation 968; test 967
- Stay-level mortality prevalence: train 26.76%; validation 26.76%; test 26.78%
- 24h test rows: 33,676 with 752 positives

Figure/table placement note:
- Full-width compact summary table assembled from:
  - `cluster-results/chapter1_true_results/cohort/chapter1_cohort_summary.csv`
  - `cluster-results/chapter1_true_results/splits/chapter1_stay_split_summary.csv`
  - `cluster-results/chapter1_true_results/splits/chapter1_primary_split_summary.csv`
  - `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/reporting_split_summary.csv`
- Cleanup/export note: this needs a cleaned presentation table rather than raw CSV.

Figure caption:
- Retained cohort size, valid-instance scale, and realized train/validation/test balance.

Speaker notes:
- This slide is mainly to reset expectations about scale. The current `cluster-results` bundle is a full ASIC internal run, with millions of analyzable rows and a balanced test split.
- That matters because some older local markdown notes still talk like this is a sample-limited or smoke-test package, but the numerical result bundle is clearly beyond that stage.

## Slide 7 — Baseline Model Performance And Calibration
Title:
- Baseline Model Performance And Calibration

Subtitle:
- Test-split summary across the frozen horizon set

On-slide bullets:
- All currently reported model-horizon pairs use the test split
- 24h test set: 33,676 rows; 752 positives
- Logistic 24h: AUROC 0.819; AUPRC 0.268; slope 0.974; Brier 0.0186
- XGBoost 24h: AUROC 0.848; AUPRC 0.318; slope 1.162; Brier 0.1351
- Logistic test AUROC declines gradually from 0.843 at 8h to 0.791 at 72h
- Logistic calibration slopes stay close to 1 across horizons

Figure/table placement note:
- Main content should be a cleaned table from:
  - `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/combined_metrics.csv`
  - `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/reporting_split_summary.csv`
- Optional right-hand callout: `XGBoost ranks better; logistic calibrates better`.
- Cleanup/export note: cleaned performance table still needed.

Figure caption:
- Test-split baseline metrics for logistic regression and XGBoost across the frozen horizon grid.

Speaker notes:
- The key tension is between ranking and calibration. If I only cared about AUROC or AUPRC, XGBoost looks better, but for a calibration-first interpretation of hard cases, logistic regression is the cleaner anchor.
- I would keep that tradeoff explicit and avoid turning the performance slide into a model-competition slide.

## Slide 8 — Primary 24h Mortality-Vs-Risk Structure
Title:
- Primary 24h Mortality-Vs-Risk Structure

Subtitle:
- Logistic regression, test split

On-slide bullets:
- Primary 24h logistic model used as the main hard-case anchor
- Fatal cases are not confined to the extreme upper tail
- Calibration slope remains near 1 at the primary horizon

Figure/table placement note:
- Left 65%: `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/logistic_regression/horizon_24h/mortality_vs_risk_plot.png`
  - Shows observed mortality across the predicted-risk spectrum for the 24h logistic model
- Right 35%: `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/logistic_regression/horizon_24h/reliability_plot.png`
  - Shows calibration at the same horizon
- Footer strip: 24h logistic test metrics from `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/combined_metrics.csv`
- Cleanup note: minor caption cleanup only

Figure caption:
- ASIC 24h logistic mortality-vs-risk structure and reliability on the test split. Risk ordering remains coherent and calibration is close enough to support a bounded hard-case interpretation.

Speaker notes:
- This is the core descriptive result. The logistic model is not being presented as the best possible classifier; it is being used because its probability scale is interpretable enough to inspect how fatal cases are distributed across predicted risk.
- The take-home point is that risk ordering is real, but fatal cases are not restricted to only the highest predicted-risk region, which is what opens the door to the hard-case analysis.

## Slide 9 — Hard-Case Definition And Burden
Title:
- Hard-Case Definition And Burden

Subtitle:
- Frozen logistic last-eligible rule

On-slide bullets:
- Rule name: `asic_logistic_last_eligible_nonfatal_q75_v1`
- One last eligible stay-level point per stay and horizon
- Hard case = fatal stay below the horizon-specific nonfatal q75 threshold
- 24h fatal stays: 1,682
- 24h hard cases: 346
- 24h hard-case share: 20.57%
- Share remains near 21% across 8h, 16h, 24h, 48h, 72h

Figure/table placement note:
- Preferred main visual: mini horizon-share plot or polished summary table built from:
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/horizon_hard_case_summary.csv`
- If the mini-plot is not exported in time, use a clean table with thresholds and shares.
- Cleanup/export note: polished burden-by-horizon visual still needed.

Figure caption:
- Frozen hard-case definition and hard-case burden across the five frozen horizons.

Speaker notes:
- The important thing here is that the hard-case group is operationally defined and fixed, not discovered post hoc from the comparison results.
- Numerically, it is not trivial. At the primary horizon, about one-fifth of fatal stays are low-predicted under the frozen logistic rule, and that share remains strikingly stable across the horizon grid.

## Slide 10 — 24h Hard-Case Comparison
Title:
- 24h Hard-Case Comparison

Subtitle:
- Low-predicted fatal stays versus other fatal stays

On-slide bullets:
- 24h fatal stay-level slice: 1,682 stays
- Low-predicted fatal: 346
- Other fatal: 1,336
- Strongest differences:
- Earlier last eligible prediction time: 176h vs 232h
- Higher MAP: 81.0 vs 58.1
- Higher PF ratio: 277.7 vs 172.3
- Lower PEEP: 9.3 vs 10.1
- Modest enrichments only: `asic_UK04` 11% vs 5%; neurologic 2% vs 0%

Figure/table placement note:
- Main figure: `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/effect_size_figure.png`
  - Shows standardized differences for the main timing, physiologic, and categorical comparison variables
- Side table built from:
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/comparison_table.csv`
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/summary.md`
- Footnote from:
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison_variable_audit/asic_hard_case_comparison_variable_audit_memo.md`
- Local-review boundary:
  - Treat the mirrored `cluster-results/.../asic_hard_case_comparison/` package as the approved aggregate export bundle for local deck-building.
  - Do not assume access to `stay_level_comparison_dataset.csv` or other restricted row-level comparison tables when preparing the local presentation.
- Cleanup note: add `n=346` vs `n=1336` directly onto the slide

Figure caption:
- Standardized differences for the primary 24h hard-case comparison. Low-predicted fatal stays look less aligned with captured short-term physiologic severity, but the subgroup pattern remains modest and descriptive.

Speaker notes:
- I would present this as the most concrete empirical hard-case result in the package. The main signal is physiologic and timing-related rather than a dramatic disease-group split.
- Just as importantly, I would keep the claims bounded: this is not evidence of a biological subtype, and some comparison variables remain incomplete or partially LOCF-dependent.

## Slide 11 — Horizon Dependence
Title:
- Horizon Dependence

Subtitle:
- Burden and overlap across 8h, 16h, 24h, 48h, 72h

On-slide bullets:
- Hard-case share stays narrowly within 20.5% to 21.4%
- Cross-horizon overlap is substantial
- Mean pairwise Jaccard: 0.828
- 24h vs 48h Jaccard: 0.885
- 24h vs 72h Jaccard: 0.824
- Run-manifest interpretation label: `persist`

Figure/table placement note:
- Main figure: `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/final/mortality_risk_horizon_comparison.png`
  - Shows mortality-vs-risk structure across the frozen horizons
- Add small overlap box from:
  - `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/overlap/pairwise_overlap.csv`
  - `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/final/run_manifest.json`
- Cleanup note: do not quote the stale `synthetic` wording from the companion markdown memo

Figure caption:
- Mortality-vs-risk structure and cross-horizon hard-case stability across the frozen horizon grid. The current ASIC outputs support a descriptive persistence read rather than a single-horizon artifact read.

Speaker notes:
- The key point is not that every hard case is identical across horizons. It is that the burden remains stable and the overlap is far above trivial rotation, especially around the 24h and 48h narrative anchors.
- I would keep the wording descriptive here. This supports persistence of the low-predicted fatal burden under the frozen rule, not the discovery of a persistent underlying subtype.

## Slide 12 — Secondary Robustness Summary
Title:
- Secondary Robustness Summary

Subtitle:
- Model, site, and time-grid checks

On-slide bullets:
- Cross-model 24h agreement:
- Logistic hard 346; XGBoost-Platt hard 227; overlap 188; Jaccard 0.488
- Site check:
- pooled 24h result not obviously driven by one site
- site-level metrics much sparser than pooled metrics
- Temporal preview only:
- Logistic 24h AUROC 0.819 -> 0.816; AUPRC 0.268 -> 0.235
- XGBoost 24h AUROC 0.848 -> 0.846; AUPRC 0.318 -> 0.291
- Observation-process and treatment-limitation sensitivities are still missing

Figure/table placement note:
- Three-panel summary slide:
  - Top-left compact agreement table from `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/agreement/logistic_regression_vs_xgboost_platt/horizon_hard_case_agreement_summary.csv`
  - Top-right optional site sanity-check thumbnail from `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/logistic_regression/primary_24h_site_overview.png`
  - Bottom panel temporal preview summary from:
    - `cluster-results/chapter1_true_results/temporal_preview/asic/aggregation_16h/comparison/preview_note.md`
    - `cluster-results/chapter1_true_results/temporal_preview/asic/aggregation_16h/comparison/aggregation_comparison_metrics.csv`
- Cleanup note: if visual clutter is a risk, keep this slide mostly tabular.

Figure caption:
- Secondary robustness summary. Cross-model agreement is moderate, site-level evidence is sparse but not obviously contradictory, and the current 8h-vs-16h preview is encouraging but still provisional.

Speaker notes:
- I would be explicit that these robustness checks have different maturity levels. Cross-model agreement is a real result, but the temporal preview is not yet a finished sensitivity analysis, and site-level summaries are much sparser than the pooled evaluation.
- This slide is useful because it prevents the main story from looking cleaner than it really is.

## Slide 13 — Current Bounded Interpretation
Title:
- Current Bounded Interpretation

Subtitle:
- What the current ASIC package supports

On-slide bullets:
- The current ASIC analysis supports heterogeneous predictability under the observed feature set
- Logistic 24h risk is a usable calibration-aware anchor for hard-case definition
- About one-fifth of fatal stays are low-predicted under the frozen logistic rule
- The low-predicted fatal burden persists descriptively across horizons
- Hard cases are operational and model-dependent
- Explicit non-claims:
- not biological subtypes
- not irreducible stochastic mortality
- not causal categories

Figure/table placement note:
- No large figure.
- Use a two-column interpretation table:
  - left: supported statements
  - right: explicit non-claims
- Sources:
  - `docs/chapter1_analysis_spec_frozen_v1.md`
  - `docs/phase1_working_reference.md`
  - `/Users/joanameyer/repository/phd-general/_context/context_sprint4.md`

Figure caption:
- Supported interpretation and explicit non-claims for the current ASIC result state.

Speaker notes:
- This is where I would deliberately narrow the language. The package now supports a bounded claim about heterogeneous predictability under this recorded representation, with a substantial low-predicted fatal subset and descriptive persistence across horizons.
- It does not support stronger claims about biology, causality, or ontological classes of death.

## Slide 14 — Open Limitations And Pending Analyses
Title:
- Open Limitations And Pending Analyses

Subtitle:
- What is still unresolved

On-slide bullets:
- ASIC lacks true death and ICU-discharge timestamps for within-horizon labels
- Row inclusion depends on valid-instance and labelability gates
- Operational split is stay-level because patient identifiers are unavailable
- Exact age is unavailable in the current 24h comparison package
- Several comparison proxies have partial coverage or LOCF dependence
- Missing or incomplete sensitivity work:
- observation-process hard-case comparison
- treatment-limitation sensitivity or explicit absence note
- formal temporal-aggregation sensitivity
- disease-stratified predictability analyses

Figure/table placement note:
- Full-width status table with two columns:
  - structural limitations already known
  - empirical/sensitivity work still pending
- Use source notes from:
  - `docs/label_logic_audit.md`
  - `docs/preprocessing_interface.md`
  - `/Users/joanameyer/repository/phd-general/_context/context_sprint4.md`
  - `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison_variable_audit/asic_hard_case_comparison_variable_audit_memo.md`
- No figure required.

Figure caption:
- Structural limitations of the current ASIC package and the main pending sensitivity analyses.

Speaker notes:
- I would end here so the talk stays honest about what is already solid and what is not. The key open items are not cosmetic: they include observation process, treatment limitation, and formal temporal sensitivity.
- That framing makes the current package look strong enough for a technical review, while still showing that the interpretation is not fully closed.

## Appendix

## Appendix A1 — Full Metrics By Model And Horizon
Title:
- Full Metrics By Model And Horizon

Subtitle:
- Test-split reference table

On-slide bullets:
- Full logistic and XGBoost metrics across 8h, 16h, 24h, 48h, 72h
- Include sample counts, event counts, AUROC, AUPRC, calibration intercept, calibration slope, Brier

Figure/table placement note:
- Full-width table from `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/combined_metrics.csv`

Figure caption:
- Test-split baseline metrics for both baseline models across the frozen horizon set.

Speaker notes:
- This is the backup table for anyone who wants the full numerical summary rather than the 24h headline table.

## Appendix A2 — Site-Stratified 24h Sanity Check
Title:
- Site-Stratified 24h Sanity Check

Subtitle:
- Logistic regression, pooled versus site-specific summaries

On-slide bullets:
- Site-level event counts and metrics at the primary horizon
- Pooled result not obviously single-site-driven
- Site-level calibration remains much sparser than pooled calibration

Figure/table placement note:
- Main figure: `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/logistic_regression/primary_24h_site_overview.png`
- Side table: `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/combined_primary_site_summary.csv`

Figure caption:
- Site-stratified 24h logistic performance and calibration sanity check.

Speaker notes:
- This is meant as a backup view, not a main result. It is useful if questions come up about whether the pooled pattern is hiding one strongly driving site.

## Appendix A3 — Cross-Model Hard-Case Agreement By Horizon
Title:
- Cross-Model Hard-Case Agreement By Horizon

Subtitle:
- Logistic versus XGBoost-Platt

On-slide bullets:
- Agreement table for 8h, 16h, 24h, 48h, 72h
- Show hard-case counts, overlap, Jaccard, and directional overlap

Figure/table placement note:
- Full-width agreement table from `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/agreement/logistic_regression_vs_xgboost_platt/horizon_hard_case_agreement_summary.csv`

Figure caption:
- Cross-model hard-case agreement across the frozen horizon set.

Speaker notes:
- This appendix slide lets the audience see that model dependence is not unique to 24h, although the overlap remains moderate rather than near-complete throughout.

## Appendix A4 — Variable Audit For The 24h Hard-Case Comparison
Title:
- Variable Audit For The 24h Hard-Case Comparison

Subtitle:
- Availability, coverage, and LOCF dependence

On-slide bullets:
- Exact age unavailable; age-group only
- Respiratory, hemodynamic, renal, and ventilation proxies mostly usable
- Some proxies depend materially on LOCF or partial current-block coverage
- Package is usable but not a complete clinical adjustment set

Figure/table placement note:
- Compact summary table distilled from `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison_variable_audit/asic_hard_case_comparison_variable_audit_memo.md`

Figure caption:
- Feasibility and completeness audit for the current 24h hard-case comparison variable package.

Speaker notes:
- This slide is useful when questions come up about whether the comparison variables are fully direct or partly reconstructed. The answer is mixed, and this audit is where that should be documented.

## Appendix A5 — Why SOFA Was Not Used
Title:
- Why SOFA Was Not Used

Subtitle:
- Feasibility audit summary

On-slide bullets:
- Missing domains: CNS, vasopressor variables, urine output
- Available domains still have heavy partial coverage and LOCF dependence
- Complete-case coverage after LOCF remains limited
- Final classification: standard SOFA not feasible

Figure/table placement note:
- One compact summary box from `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_sofa_feasibility_audit/sofa_feasibility_memo.md`

Figure caption:
- Rationale for using direct organ-support and dysfunction proxies rather than a pseudo-SOFA score.

Speaker notes:
- This is mainly a defensive appendix slide. It prevents the obvious question of why SOFA was not included from distracting the main discussion.

## Appendix A6 — Observation-Process Readiness
Title:
- Observation-Process Readiness

Subtitle:
- Instrumentation exists; explanatory sensitivity still missing

On-slide bullets:
- Derived block-level variables for in-block coverage and time since last observation
- 248,772 usable 8h blocks before horizon duplication
- 69.4% of blocks observe all 4 core groups; 30.6% observe exactly 3
- No observation-process hard-case comparison result yet

Figure/table placement note:
- Compact QC table from:
  - `cluster-results/chapter1_true_results/observation_process/chapter1_observation_process_qc_summary.csv`
  - `cluster-results/chapter1_true_results/observation_process/chapter1_observation_process_implementation_note.md`

Figure caption:
- Observation-process variable readiness in the current ASIC Chapter 1 pipeline.

Speaker notes:
- The important distinction is between variable readiness and empirical sensitivity results. The current bundle has the former, not the latter.

## Appendix A7 — Temporal Aggregation Preview: 8h Vs 16h
Title:
- Temporal Aggregation Preview: 8h Vs 16h

Subtitle:
- Provisional sensitivity preview, not a completed analysis

On-slide bullets:
- 24h logistic and XGBoost metrics move modestly between 8h and 16h aggregation
- AUROC changes are small; AUPRC decreases are somewhat larger
- Calibration slope is broadly stable
- The preview does not justify a temporal refreeze decision by itself

Figure/table placement note:
- Main figure: `cluster-results/chapter1_true_results/temporal_preview/asic/aggregation_16h/comparison/logistic_regression_24h_mortality_vs_risk_8h_vs_16h.png`
- Side summary table from:
  - `cluster-results/chapter1_true_results/temporal_preview/asic/aggregation_16h/comparison/aggregation_comparison_metrics.csv`
  - `cluster-results/chapter1_true_results/temporal_preview/asic/aggregation_16h/comparison/preview_note.md`
- Cleanup note: mark the slide clearly as provisional

Figure caption:
- Preliminary 8h-versus-16h aggregation comparison for the current ASIC pipeline.

Speaker notes:
- This is a useful preview, but I would avoid overstating it. It shows that the pattern does not obviously collapse at 16h, but it is still only one alternative aggregation and not a completed temporal sensitivity analysis.

## Missing Exports / Cleanups Still Needed

- Polished cohort-flow figure for Slide 3
- Frozen-definitions table export for Slide 2
- Valid-instance schematic for Slide 4
- Proxy-label and labelability table export for Slide 4
- Compact modeling-design table export for Slide 5
- Clean cohort-and-split summary table export for Slide 6
- Clean baseline performance table export for Slide 7
- Hard-case burden-by-horizon visual or cleaned summary table for Slide 9
- Cleaned caption or replacement note for the horizon comparison figure in Slide 11

## Slides Blocked By Missing Polished Artifacts

- Slide 2 — Frozen Study Setup
- Slide 3 — Cohort Construction And Exclusions
- Slide 4 — Time Representation, Valid-Instance Rule, And Proxy Labels
- Slide 5 — Frozen Modeling Design Choices
- Slide 6 — Retained Cohort And Realized Split Summary
- Slide 7 — Baseline Model Performance And Calibration
- Slide 9 — Hard-Case Definition And Burden

## Top 8 Slides For A Shortened Meeting Version

- Slide 1 — Goal
- Slide 2 — Frozen Study Setup
- Slide 3 — Cohort Construction And Exclusions
- Slide 4 — Time Representation, Valid-Instance Rule, And Proxy Labels
- Slide 7 — Baseline Model Performance And Calibration
- Slide 8 — Primary 24h Mortality-Vs-Risk Structure
- Slide 10 — 24h Hard-Case Comparison
- Slide 11 — Horizon Dependence
