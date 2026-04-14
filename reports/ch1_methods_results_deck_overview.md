# Chapter 1 Methods And Results Deck Overview

Scan date: 2026-04-13

Scope note:
- This document is a stand-alone methods-and-results deck plan for the current ASIC Chapter 1 analysis.
- It prioritizes the frozen Chapter 1 specification, the current Phase 1 working reference, the current authoritative Sprint 4 context, and the true ASIC cluster-result bundle in `cluster-results/chapter1_true_results`.
- When narrative markdown memos in the cluster bundle conflict with large-count CSV summaries, figures, or run manifests, the CSVs/figures/run manifests are treated as authoritative.
- This document separates frozen methodological choices, implementation-specific operational choices, current empirical findings, and unresolved limitations.

## 1. Presentation objective

- Present the current ASIC Chapter 1 analysis as a technical methods-and-results package, not as thesis framing.
- Show the exact cohort definition, site/stay exclusion logic, valid-instance rule, proxy within-horizon label construction, and preprocessing contract.
- Summarize the current baseline-model, hard-case, and horizon-dependence results using the true ASIC result bundle.
- Make explicit which results are presentation-ready now and which interpretation-critical sensitivity analyses are still pending or only partially available.

## 2. Recommended slide structure for a methods/results deck

### Slide 1. Objective And Analysis Snapshot
- Purpose: open the deck with a minimal statement of what the analysis is and what the presentation will cover.
- Exact content to show: 3-4 bullets on objective, development dataset, primary endpoint, primary horizon, and the fact that this is a methods/results review of the current ASIC analysis state.
- Recommended figure/table/artifact: text-only slide plus a compact frozen-design box built from `docs/chapter1_analysis_spec_frozen_v1.md`.
- Interpretive statement: this is a bounded analysis of near-term in-ICU mortality risk structure under the current recorded feature set and time representation, not a generic mortality-prediction talk.
- Status: ready

### Slide 2. Frozen Analysis Design
- Purpose: lock the frozen scientific design before showing implementation or results.
- Exact content to show: development dataset, external-validation target, primary outcome, unit of analysis, primary and sensitivity horizons, minimum baseline models, mandatory metrics, and non-claims.
- Recommended figure/table/artifact: compact frozen-definitions table exported from `docs/chapter1_analysis_spec_frozen_v1.md`, `config/ch1_run_config.json`, and `config/ch1_feature_sets.json`.
- Interpretive statement: the main analysis choices were prespecified early, which helps separate methodological design from later empirical findings.
- Status: needs export

### Slide 3. Cohort Construction: Site And Stay Exclusions
- Purpose: show exactly how input ASIC hospitals and stays became the retained Chapter 1 cohort.
- Exact content to show: one cohort flow starting with 8 hospitals / 15,969 stays, site eligibility logic, retained hospitals, then stay-level exclusions within retained sites; include the retained hospitals `asic_UK02`, `asic_UK04`, `asic_UK07`, `asic_UK08`.
- Recommended figure/table/artifact: export from `cluster-results/chapter1_true_results/cohort/chapter1_site_eligibility.csv`, `cluster-results/chapter1_true_results/cohort/chapter1_stay_exclusion_summary_by_hospital.csv`, `cluster-results/chapter1_true_results/cohort/chapter1_counts_by_hospital.csv`, and `cluster-results/chapter1_true_results/cohort/chapter1_cohort_summary.csv`.
- Interpretive statement: the main sample contraction occurs at documented site/stay filters, especially site eligibility and `mech_vent_ge_24h_qc` plus readmission filtering, not at the modeling stage.
- Status: needs export

### Slide 4. Valid Prediction Instances And Major Drop Mechanisms
- Purpose: explain what constitutes a valid prediction row and where row loss is introduced after stay retention.
- Exact content to show: completed 8-hour block representation, alive/in-ICU requirement, 3-of-4 core-group rule, horizon-specific labelability requirement, and the current valid-instance counts by horizon; explicitly state that a dedicated exported instance-drop-by-reason figure is not currently available.
- Recommended figure/table/artifact: methods schematic built from `docs/chapter1_analysis_spec_frozen_v1.md`, `docs/preprocessing_interface.md`, `config/ch1_run_config.json`, and `cluster-results/chapter1_true_results/cohort/chapter1_cohort_summary.csv`; supporting observation-process counts from `cluster-results/chapter1_true_results/observation_process/chapter1_observation_process_qc_summary.csv`.
- Interpretive statement: the analyzable unit is intentionally restrictive and is strongly shaped by within-block physiologic coverage plus horizon-specific labelability.
- Status: missing artifact

### Slide 5. Proxy Within-Horizon Label Construction
- Purpose: make the label rule explicit and surface its limitations before performance results.
- Exact content to show: positive / negative / unlabeled rule using `icu_mortality` and `icu_end_time_proxy_hours`, plus horizon-specific labelable / positive / negative / unlabeled counts.
- Recommended figure/table/artifact: label-rule schematic and compact horizon-count table built from `docs/label_logic_audit.md` and `cluster-results/chapter1_true_results/cohort/chapter1_cohort_summary.csv`.
- Interpretive statement: these are explicit proxy within-horizon labels, not true event-timed mortality labels, and unlabeled rows are left unlabeled rather than coerced negative.
- Status: needs export

### Slide 6. Feature Boundary And Preprocessing Contract
- Purpose: show what entered the baseline models and how missingness was handled.
- Exact content to show: 31 primary base features, 15 extended additions, bounded LOCF policy, ventilation-window restriction for ventilator variables, missingness indicators, and the fact that no final imputation is applied in preprocessing exports.
- Recommended figure/table/artifact: compact methods table from `config/ch1_feature_sets.json`, `docs/preprocessing_interface.md`, `cluster-results/chapter1_true_results/model_ready/chapter1_primary_readiness_summary.csv`, and `cluster-results/chapter1_true_results/carry_forward/chapter1_primary_locf_feature_summary.csv`.
- Interpretive statement: preprocessing is intentionally conservative; missingness is exposed and bounded rather than aggressively erased.
- Status: needs export

### Slide 7. Split Strategy And Realized Split Balance
- Purpose: show the frozen split intent, the operational stay-level implementation, and the realized balance.
- Exact content to show: frozen `70/15/15` target, within-hospital splitting, stay-level implementation because ASIC lacks patient identifiers, realized stay counts / prevalence by split, and confirmation that all model-horizon reporting uses the test split.
- Recommended figure/table/artifact: table built from `docs/chapter1_analysis_spec_frozen_v1.md`, `docs/preprocessing_interface.md`, `cluster-results/chapter1_true_results/splits/chapter1_stay_split_summary.csv`, `cluster-results/chapter1_true_results/splits/chapter1_primary_split_summary.csv`, and `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/reporting_split_summary.csv`.
- Interpretive statement: the implementation departs from ideal patient-level splitting, but the realized test split is balanced and binary-evaluable across all current reported horizons.
- Status: needs export

### Slide 8. Baseline Models And Headline Performance
- Purpose: summarize the current baseline model set and headline quantitative results.
- Exact content to show: logistic regression and XGBoost across the five frozen horizons; include AUROC, AUPRC, calibration slope, and Brier score, with a clean emphasis on the primary 24h horizon.
- Recommended figure/table/artifact: cleaned performance table exported from `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/combined_metrics.csv`.
- Interpretive statement: XGBoost ranks better on discrimination, but logistic regression is cleaner for calibration-oriented interpretation and remains the main hard-case anchor.
- Status: needs export

### Slide 9. Primary 24h Risk Structure: Mortality Vs Predicted Risk And Calibration
- Purpose: show the core descriptive result that justifies hard-case interpretation at 24h.
- Exact content to show: the logistic-regression 24h mortality-vs-risk plot plus the logistic 24h reliability plot; optionally add one row of 24h test metrics beneath the figure.
- Recommended figure/table/artifact: `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/logistic_regression/horizon_24h/mortality_vs_risk_plot.png` and `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/logistic_regression/horizon_24h/reliability_plot.png`.
- Interpretive statement: on the primary horizon, the logistic model shows coherent risk ordering and near-unit calibration slope, which is sufficient to support a calibration-aware hard-case analysis.
- Status: ready

### Slide 10. Hard-Case Definition And Burden
- Purpose: define the hard-case rule precisely and show how large the low-predicted fatal burden is.
- Exact content to show: the saved rule `asic_logistic_last_eligible_nonfatal_q75_v1`, one last eligible stay-level point per stay, the horizon-specific nonfatal q75 thresholds, and the proportion of fatal stays classified as hard cases at each horizon.
- Recommended figure/table/artifact: compact slide table exported from `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/horizon_hard_case_summary.csv`.
- Interpretive statement: the hard-case burden is substantial rather than trivial, with about one-fifth of fatal stays below the nonfatal q75 threshold at the primary 24h horizon.
- Status: needs export

### Slide 11. Primary 24h Hard-Case Comparison
- Purpose: show what distinguishes low-predicted fatal stays from other fatal stays at the primary horizon.
- Exact content to show: the effect-size figure plus a compact companion table or callout box for timing, MAP, PF ratio, PEEP, and modest site/disease-group enrichments.
- Recommended figure/table/artifact: `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/effect_size_figure.png`, `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/comparison_table.csv`, and `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/summary.md`.
- Interpretive statement: low-predicted fatal stays look less aligned with captured short-term physiologic severity, but they do not form a dramatic categorical subgroup and should not be presented as a biological class.
- Status: ready

### Slide 12. Horizon Dependence Of Hard-Case Burden
- Purpose: show whether the low-predicted fatal pattern is specific to 24h or remains visible across the frozen horizon grid.
- Exact content to show: the horizon comparison figure, the hard-case share by horizon, and key overlap statistics such as 24h vs 48h and 24h vs 72h Jaccard overlap.
- Recommended figure/table/artifact: `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/final/mortality_risk_horizon_comparison.png`, `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/horizon_hard_case_summary.csv`, `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/overlap/pairwise_overlap.csv`, and `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/final/run_manifest.json`.
- Interpretive statement: on the current ASIC run, both hard-case burden and membership are fairly stable across horizons, which supports a persistence read under the frozen definition.
- Status: ready

### Slide 13. Secondary Robustness And Current Gaps
- Purpose: close with what is already available beyond the core story and what remains incomplete.
- Exact content to show: one compact summary box for cross-model agreement, site-stratified 24h sanity checks, the 8h vs 16h temporal preview, and a final box listing missing observation-process, treatment-limitation, and disease-stratified results.
- Recommended figure/table/artifact: `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/agreement/logistic_regression_vs_xgboost_platt/horizon_hard_case_agreement_summary.csv`, `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/combined_primary_site_summary.csv`, `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/logistic_regression/primary_24h_site_overview.png`, `cluster-results/chapter1_true_results/temporal_preview/asic/aggregation_16h/comparison/preview_note.md`, and `cluster-results/chapter1_true_results/temporal_preview/asic/aggregation_16h/comparison/aggregation_comparison_metrics.csv`.
- Interpretive statement: robustness is mixed rather than all-or-none, with strong cross-horizon persistence, only moderate cross-model agreement, a reassuring but still limited temporal preview, and several interpretation-critical sensitivity analyses still pending.
- Status: needs export

## 3. Methods blocks that must be covered

| methods block | frozen methodological choices | implementation-specific operational choices | preferred deck form | supporting file(s) |
| --- | --- | --- | --- | --- |
| Frozen analysis design | ASIC development data, MIMIC-IV external-validation target, in-ICU mortality endpoint, 24h primary horizon, 48h main contrast, 8h/16h/72h sensitivities, logistic + XGBoost minimum model set | Current deck should stay ASIC-only because no MIMIC result artifacts are present locally | Compact table | `docs/chapter1_analysis_spec_frozen_v1.md`; `config/ch1_run_config.json`; `config/ch1_feature_sets.json` |
| Cohort construction | adult ICU stays, mechanical ventilation `>=24h`, first ICU stay, valid in-ICU mortality label, at least one valid prediction instance | Adult status is trusted upstream; site-level exclusions depend on ICU mortality availability and core-vital coverage; first-stay handling is proxied by `readmission` | Flowchart plus small table | `docs/chapter1_analysis_spec_frozen_v1.md`; `docs/preprocessing_interface.md`; `cluster-results/chapter1_true_results/cohort/chapter1_site_eligibility.csv`; `cluster-results/chapter1_true_results/cohort/chapter1_stay_exclusion_summary_by_hospital.csv`; `cluster-results/chapter1_true_results/cohort/chapter1_cohort_summary.csv` |
| Site and stay drop logic | Exclude unusable mortality labels, non-first stays, no valid instances, preprocessing failures | Site exclusion is implemented before stay-level filtering; stay-level filters include `mech_vent_ge_24h_qc == False`, missing dynamic data, missing `readmission`, and readmission-flagged stays | Flowchart or waterfall | `docs/preprocessing_interface.md`; `cluster-results/chapter1_true_results/cohort/chapter1_counts_by_hospital.csv`; `cluster-results/chapter1_true_results/cohort/chapter1_stay_exclusion_summary_by_hospital.csv` |
| Valid prediction-instance construction | Completed 8-hour blocks; patient alive and in ICU at prediction time; sufficient observed data; horizon-specific label definable | Current implementation requires at least 3 of 4 core physiologic groups within the block and uses stay-level `icu_end_time_proxy_hours` to determine horizon eligibility | Schematic | `docs/chapter1_analysis_spec_frozen_v1.md`; `docs/preprocessing_interface.md`; `config/ch1_run_config.json`; `src/chapter1_mortality_decomposition/instances.py` |
| Horizon-specific label construction | Positive and negative labels defined within horizon `H`; ambiguous cases remain unlabeled | ASIC lacks true ICU discharge and death timestamps, so labels use the explicit proxy `icu_end_time_proxy_hours`; unlabeled reasons are described in the audit but a standalone downloaded unlabeled-reason table is not currently available | Schematic plus compact table | `docs/chapter1_analysis_spec_frozen_v1.md`; `docs/label_logic_audit.md`; `src/chapter1_mortality_decomposition/labels.py`; `cluster-results/chapter1_true_results/cohort/chapter1_cohort_summary.csv` |
| Proxy-label limitations | Proxy labels must be named explicitly as proxies and not upgraded into event-timed labels | Current implementation treats `icu_end_time_proxy_hours` as an event-time proxy for fatal stays and a horizon-observation proxy for survivors | Text box or caution slide | `docs/chapter1_analysis_spec_frozen_v1.md`; `docs/label_logic_audit.md`; `/Users/joanameyer/repository/phd-general/_context/context_sprint4.md` |
| Feature boundary | Use routine ICU variables only; no high-complexity bespoke feature engineering | Primary feature set has 31 base variables; extended adds 15 more; current deck should present the primary set as the main analysis and the extended set only if needed in appendix | Compact table | `config/ch1_feature_sets.json`; `cluster-results/chapter1_true_results/model_ready/chapter1_primary_readiness_summary.csv` |
| Missingness and carry-forward policy | One prespecified preprocessing policy; missingness handling must not be revised post hoc | Bounded LOCF only for prespecified families; ventilator-variable LOCF restricted to supported windows; missingness indicators appended; final imputation deferred to model training | Table | `docs/chapter1_analysis_spec_frozen_v1.md`; `docs/preprocessing_interface.md`; `cluster-results/chapter1_true_results/model_ready/chapter1_primary_readiness_summary.csv`; `cluster-results/chapter1_true_results/carry_forward/chapter1_primary_locf_feature_summary.csv`; `cluster-results/chapter1_true_results/carry_forward/chapter1_primary_ventilator_locf_summary.csv` |
| Split strategy | Frozen design intent is `70/15/15`, within-site, mortality-stratified internal split | ASIC lacks patient identifiers, so the implemented split is stay-level rather than patient-level; all prediction rows inherit their stay-level split | Table | `docs/chapter1_analysis_spec_frozen_v1.md`; `docs/preprocessing_interface.md`; `cluster-results/chapter1_true_results/splits/chapter1_stay_split_summary.csv`; `cluster-results/chapter1_true_results/splits/chapter1_primary_split_summary.csv` |
| Baseline models and evaluation framework | Minimum model set is logistic regression and XGBoost; mandatory metrics are AUROC, AUPRC, calibration intercept/slope; calibration is central | Current exported bundle also includes Brier score and per-horizon reporting-split summaries; use the test split because it is binary-evaluable for all current reported horizons | Table | `docs/chapter1_analysis_spec_frozen_v1.md`; `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/combined_metrics.csv`; `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/reporting_split_summary.csv` |
| Hard-case definition | Hard cases are defined operationally from the predicted-risk spectrum, not biologically | Current frozen rule is `asic_logistic_last_eligible_nonfatal_q75_v1`, using one last eligible stay-level point per stay and the horizon-specific nonfatal q75 threshold | Table or rule schematic | `docs/phase1_working_reference.md`; `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/horizon_hard_case_summary.csv`; `src/chapter1_mortality_decomposition/hard_case_definition.py` |

## 4. Results blocks that must be covered

| result block | main deck or appendix | exact figure/table/artifact | concise interpretive statement | maturity |
| --- | --- | --- | --- | --- |
| Cohort summary and retained sample size | Main deck | `cluster-results/chapter1_true_results/cohort/chapter1_cohort_summary.csv` | The current ASIC analysis is no longer sample-scale: it uses 15,969 input stays, 6,446 retained stays, and 1,548,220 valid prediction instances across frozen horizons. | Mature enough for presentation, but needs slide-ready export |
| Site and stay exclusion logic | Main deck | `cluster-results/chapter1_true_results/cohort/chapter1_site_eligibility.csv`; `cluster-results/chapter1_true_results/cohort/chapter1_stay_exclusion_summary_by_hospital.csv` | Cohort reduction is dominated by documented site eligibility and stay filters, not by late-stage model exclusion. | Mature enough for presentation, but no polished cohort-flow figure exists yet |
| Split balance and evaluability | Main deck | `cluster-results/chapter1_true_results/splits/chapter1_stay_split_summary.csv`; `cluster-results/chapter1_true_results/splits/chapter1_primary_split_summary.csv`; `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/reporting_split_summary.csv` | The realized test split is balanced and binary-evaluable across all current reported horizons. | Mature enough for presentation |
| Model performance and calibration | Main deck | `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/combined_metrics.csv` | XGBoost is stronger on ranking metrics, but logistic is cleaner for calibration-based interpretation. | Mature enough for presentation after table cleanup |
| 24h mortality-vs-risk structure | Main deck | `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/logistic_regression/horizon_24h/mortality_vs_risk_plot.png` | Fatal cases are not confined to the extreme upper tail, but the risk ordering remains strong enough to define a bounded low-predicted fatal subset. | Mature enough for presentation |
| 24h reliability / calibration | Main deck | `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/logistic_regression/horizon_24h/reliability_plot.png` | The primary logistic anchor has near-unit calibration slope at 24h, which supports a calibration-aware hard-case read. | Mature enough for presentation |
| Hard-case definition and burden | Main deck | `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/horizon_hard_case_summary.csv` | Under the frozen logistic last-eligible rule, 346 of 1,682 fatal 24h stays are low-predicted, or about 20.6%. | Mature enough for presentation, but best shown as an exported summary table or figure |
| 24h hard-case comparison | Main deck | `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/effect_size_figure.png`; `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/comparison_table.csv` | Low-predicted fatal stays look less aligned with captured short-term physiologic severity, but subgroup signals remain modest. Use the approved mirrored aggregate export bundle for local review rather than the restricted row-level reconstruction table. | Mature enough for presentation |
| Horizon dependence | Main deck | `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/final/mortality_risk_horizon_comparison.png`; `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/horizon_hard_case_summary.csv`; `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/overlap/pairwise_overlap.csv` | The current ASIC outputs support a persistence read: burden changes little across horizons and 24h vs 48h overlap is high. | Mature enough for presentation, but captioning should not quote the stale synthetic memo literally |
| Cross-model agreement | Appendix or last main slide if time allows | `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/agreement/logistic_regression_vs_xgboost_platt/horizon_hard_case_agreement_summary.csv` | Cross-model agreement is only moderate, so hard cases should not be presented as model-invariant entities. | Mature enough for appendix |
| Site sensitivity | Appendix | `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/logistic_regression/primary_24h_site_overview.png`; `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/combined_primary_site_summary.csv` | No single site obviously collapses the primary-horizon result, but site-level metrics remain much sparser than the pooled evaluation. | Appendix-ready, but not strong enough for a core-slide claim |
| Observation-process sensitivity | Missing from main result bundle | No linked hard-case comparison artifact currently present; only readiness/QC artifacts such as `cluster-results/chapter1_true_results/observation_process/chapter1_observation_process_qc_summary.csv` exist | The variable set exists and is QC'd, but no current result shows whether observation process explains the hard-case pattern. | Not ready for empirical presentation |
| Temporal-aggregation sensitivity | Appendix only | `cluster-results/chapter1_true_results/temporal_preview/asic/aggregation_16h/comparison/preview_note.md`; `cluster-results/chapter1_true_results/temporal_preview/asic/aggregation_16h/comparison/aggregation_comparison_metrics.csv`; `cluster-results/chapter1_true_results/temporal_preview/asic/aggregation_16h/comparison/logistic_regression_24h_mortality_vs_risk_8h_vs_16h.png` | The current 16h preview is reassuring but too narrow to count as a completed sensitivity analysis. | Appendix-only and explicitly provisional |
| Disease-stratified analyses | Missing | Only the cohort disease-group inventory is present: `cluster-results/chapter1_true_results/evaluation/asic/icd10_disease_group_validation/final_group_counts.csv` | There is no current disease-stratified predictability analysis result to present. | Not ready |

## 5. Figure-first deck recommendation

| priority | filename/path | what it shows | why it earns a main-deck slot | interpretive statement to say aloud | cleanup needed |
| --- | --- | --- | --- | --- | --- |
| 1 | `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/logistic_regression/horizon_24h/mortality_vs_risk_plot.png` | Observed mortality across the predicted-risk spectrum for the primary 24h logistic model | This is the core descriptive output that motivates hard-case analysis | "The primary 24h logistic model preserves clear risk ordering, but fatal cases are not confined to only the highest predicted-risk region." | Minor caption cleanup only |
| 2 | `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/logistic_regression/horizon_24h/reliability_plot.png` | Calibration of the primary 24h logistic model | Calibration is a gating condition for any hard-case interpretation | "At 24h the logistic anchor is close enough to calibrated that low-predicted fatal cases are not just an obvious probability-scale artifact." | Minor caption cleanup only |
| 3 | `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/effect_size_figure.png` | Effect-size summary for the 24h low-predicted fatal vs other fatal comparison | This is the best compact figure for the actual hard-case comparison | "The largest separations are physiologic and timing-related, especially MAP, PF ratio, PEEP, and time to last eligible prediction." | Add a slide caption with `n=346` vs `n=1336` |
| 4 | `cluster-results/chapter1_true_results/evaluation/asic/horizon_dependence/final/mortality_risk_horizon_comparison.png` | Mortality-vs-risk structure across the frozen horizons | It compresses the horizon story into one figure and supports the persistence read | "The low-predicted fatal burden does not disappear when the horizon changes, although the horizon story remains descriptive rather than ontological." | Caption cleanup needed because companion memo text is stale |
| 5 | `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/logistic_regression/primary_24h_site_overview.png` | Site-stratified 24h pooled sanity check | Useful backup for questions about whether one hospital drives the main pooled result | "The main pooled 24h pattern is not obviously a single-site artifact, but site-specific metrics are much sparser than the pooled analysis." | Likely appendix-ready as is |
| 6 | `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/xgboost/horizon_24h/mortality_vs_risk_plot.png` | 24h mortality-vs-risk structure for XGBoost | Useful comparator when discussing discrimination-versus-calibration tradeoff | "XGBoost ranks better, but it is not the main structural anchor because probability calibration is materially worse." | Keep out of main deck unless explicit model-comparison slide is added |
| 7 | `cluster-results/chapter1_true_results/temporal_preview/asic/aggregation_16h/comparison/logistic_regression_24h_mortality_vs_risk_8h_vs_16h.png` | Preview comparison of 8h vs 16h temporal aggregation | Useful appendix figure when asked whether the 8h representation is obviously driving the pattern | "The current preview shows movement, but no obvious collapse of the 24h risk-structure pattern." | Appendix only; needs explicit provisional label |

## 6. Appendix recommendation

- Extra performance table across all five horizons and both models.
  - Artifact: `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/combined_metrics.csv`
- Detailed frozen-definitions slide covering cohort, outcome, unit of analysis, horizons, models, metrics, and non-claims.
  - Artifacts: `docs/chapter1_analysis_spec_frozen_v1.md`; `config/ch1_run_config.json`; `config/ch1_feature_sets.json`
- Site-level 24h performance and calibration sanity checks.
  - Artifacts: `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/logistic_regression/primary_24h_site_overview.png`; `cluster-results/chapter1_true_results/evaluation/asic/baselines/primary_medians/combined_primary_site_summary.csv`
- Cross-model hard-case agreement appendix slide.
  - Artifact: `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/agreement/logistic_regression_vs_xgboost_platt/horizon_hard_case_agreement_summary.csv`
- Variable audit / proxy feasibility appendix slide for the 24h hard-case comparison package.
  - Artifact: `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison_variable_audit/asic_hard_case_comparison_variable_audit_memo.md`
- SOFA feasibility appendix slide explaining why direct proxy variables were used instead of SOFA.
  - Artifact: `cluster-results/chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_sofa_feasibility_audit/sofa_feasibility_memo.md`
- Observation-process readiness appendix slide.
  - Artifact: `cluster-results/chapter1_true_results/observation_process/chapter1_observation_process_qc_summary.csv`
- Temporal-aggregation preview appendix slide with explicit provisional label.
  - Artifacts: `cluster-results/chapter1_true_results/temporal_preview/asic/aggregation_16h/comparison/preview_note.md`; `cluster-results/chapter1_true_results/temporal_preview/asic/aggregation_16h/comparison/aggregation_comparison_metrics.csv`
- Proxy-label caveat appendix slide covering missing true death/discharge timestamps, proxy ICU-end timing, and unlabeled rows.
  - Artifacts: `docs/label_logic_audit.md`; `docs/chapter1_analysis_spec_frozen_v1.md`

## 7. Missing artifacts still needed

- A polished cohort-flow figure that combines site eligibility and stay-level exclusions into one slide-ready export.
- A compact frozen-definitions table exported specifically for presentation use.
- A valid-instance construction schematic showing the completed-block rule, 3-of-4 core-group requirement, alive/in-ICU requirement, and horizon-specific labelability gate.
- A slide-ready proxy-label figure that shows positive / negative / unlabeled logic by horizon.
- A dedicated exported table or plot of hard-case share by horizon; the raw summary exists, but there is no polished slide artifact yet.
- A cleaned performance table that puts logistic and XGBoost side by side at 24h and then across all horizons.
- A cleaned caption or replacement note for the horizon-dependence figure, because the saved companion memo still uses stale synthetic wording.
- A formal observation-process hard-case comparison artifact; readiness/QC exists, but the sensitivity result itself does not.
- A formal treatment-limitation sensitivity artifact or explicit absence note usable in the deck.
- A disease-stratified predictability analysis result package; only disease-group inventory counts are currently available.
- A standalone unlabeled-reason summary artifact in the downloaded bundle; `docs/label_logic_audit.md` references unlabeled reasons, but no dedicated downloaded table is currently present.

## 8. Keep / drop decisions

| content block | decision | reason |
| --- | --- | --- |
| Minimal objective slide | Keep | Necessary opener for a technical deck, but keep it short |
| Broad motivation / why this matters for the thesis | Drop | Not needed for supervisor and close technical colleagues |
| PhD roadmap / chapter architecture / proposal alignment | Drop | Out of scope for a stand-alone methods/results deck |
| Frozen analysis design summary | Keep | Core to this audience and needed before results |
| Detailed repo architecture | Drop | Too implementation-heavy for this deck |
| Cohort flow and exclusion logic | Keep | One of the highest-priority methods blocks |
| Valid-instance rule and row-drop logic | Keep | Critical for understanding what is actually modeled |
| Proxy-label rule and limitations | Keep | Must be surfaced explicitly because true event times are unavailable |
| Full feature list by variable name | Appendix-only | Too dense for the main deck; main deck needs boundary and counts, not every column |
| Missingness / LOCF policy summary | Keep | Central methodological choice with interpretation consequences |
| Split strategy and realized balance | Keep | Important because implementation is stay-level, not patient-level |
| Full all-horizon metrics table | Appendix-only | Useful reference, but too dense for the main narrative |
| Clean 24h headline performance table | Keep | Necessary result summary for the primary horizon |
| Logistic 24h mortality-vs-risk figure | Keep | Core descriptive result |
| Logistic 24h reliability figure | Keep | Core calibration result |
| XGBoost 24h comparator figure | Appendix-only | Useful comparator, but not the main anchor |
| Hard-case definition and burden | Keep | Core bridge from baseline risk to hard-case analysis |
| 24h hard-case comparison figure | Keep | Core result figure |
| Horizon dependence figure | Keep | Core sensitivity/result figure |
| Cross-model agreement | Appendix-only or final backup slide | Important caveat, but secondary to the main deck spine |
| Site-stratified 24h summary | Appendix-only | Useful defensively, not part of the main claim |
| Observation-process QC summary | Appendix-only | Readiness is relevant, but there is no actual sensitivity result yet |
| Temporal-aggregation preview | Appendix-only | Useful but explicitly incomplete |
| Disease-group inventory counts | Appendix-only | Contextual only; not the same as disease-stratified analysis |
| SOFA feasibility audit | Appendix-only | Good defensive appendix, not a main-deck result |
| Decomposition framing | Drop from main deck | Not needed for a methods/results presentation of the current analysis state |
| MIMIC / external-validation slides | Drop | No current MIMIC result bundle is present |

## 9. Risks of mispresentation

- Overclaiming the proxy labels as true within-horizon death timing rather than explicit ICU-end-time proxies.
- Hiding the fact that many rows are unlabeled and that labelability itself depends on horizon and ICU-end proxy timing.
- Skipping the valid-instance rule and making the row set look like all blocked timepoints rather than a filtered analyzable subset.
- Presenting the split as patient-level when the operational implementation is stay-level because ASIC lacks patient identifiers.
- Letting AUROC dominate the interpretation and underplaying calibration, even though the frozen Chapter 1 language treats calibration as central.
- Treating low-predicted fatal stays as biological subtypes, ontological classes, or irreducibly unpredictable deaths.
- Presenting the hard-case comparison as a stable subgroup estimate rather than a bounded descriptive comparison under one operational rule.
- Overstating hard-case stability by ignoring that cross-model agreement is only moderate even though cross-horizon overlap is high.
- Quoting stale narrative language from `interpretation_note.md` or `horizon_interpretation_memo.md` as if it were the authoritative current evidence state.
- Presenting observation-process or treatment-limitation confounding as already addressed when those interpretation-critical sensitivities are still pending.
- Presenting the 16h temporal preview as a completed temporal-resolution sensitivity analysis.
- Using site-level or subgroup differences as if they were mature inferential findings rather than sparse descriptive signals.
- Failing to mention that exact age is unavailable in the current 24h hard-case comparison package and that several proxy variables have nontrivial missingness or LOCF dependence.
