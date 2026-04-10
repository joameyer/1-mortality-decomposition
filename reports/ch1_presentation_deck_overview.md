# Chapter 1 Presentation Deck Overview

Scan date: 2026-04-10

Scope note:
- This overview is based only on files and artifacts present in this repository.
- It separates frozen scientific design from current local artifact status.
- Where artifacts explicitly describe outputs as synthetic, sample-limited, or workflow-validation only, that caveat is preserved here.
- Chapter 1 is treated as risk-structure-first, not decomposition-first.

## 1. Executive Summary

- This repository is the standalone Chapter 1 preprocessing and baseline-analysis layer for a mortality risk-structure study in ventilated ICU patients.
- The central Chapter 1 question is not "can we build the best mortality model?" but "do some fatal outcomes remain weakly captured by standard short-horizon risk models under the recorded ICU feature set and time resolution?"
- The frozen development dataset is ASIC; frozen external validation is MIMIC-IV, but no MIMIC-IV artifacts are present locally yet.
- The pipeline starts from standardized upstream ASIC artifacts only and owns Chapter 1-specific cohort restriction, valid-instance construction, proxy horizon-label generation, bounded LOCF preprocessing, split assignment, and baseline evaluation artifacts.
- The frozen primary horizon is 24h, with 48h as the main contrast and 8h, 16h, and 72h as sensitivities.
- Local preprocessing artifacts are complete enough to support a methodological presentation: retained cohort tables, valid instances, proxy labels, model-ready datasets, observation-process features, split summaries, baseline metrics, hard-case artifacts, and horizon-dependence summaries all exist.
- The strongest currently available local finding is descriptive: a low-predicted fatal subset can be operationally defined, characterized, and compared across horizons. This supports a Chapter 1 narrative even without decomposition.
- The most important current caveat is that several local result packages are explicitly synthetic or sample-limited. In particular, the current primary test split has zero positive stays, so reported model evaluation falls back to validation rather than a clean held-out test readout.
- The deck should therefore emphasize frozen design, preprocessing contract, calibration-first evaluation logic, the descriptive hard-case workflow, and the status of evidence, not a mature scientific claim.
- Decomposition should be presented, if at all, as optional secondary summarization. It is not needed to make the current Chapter 1 story coherent.

## 2. Data Sources And Cohorts

### Frozen scientific design

| item | frozen status | presentation note |
| --- | --- | --- |
| Development data | ASIC cohort | Frozen primary development dataset |
| External validation | MIMIC-IV | Mandatory in the frozen spec, but not yet available locally |
| Study cohort | Adults, mechanically ventilated ICU stays, ventilation >=24h, first ICU stay, valid in-ICU mortality label, at least one valid prediction instance | State this as the intended scientific cohort |
| Primary outcome | In-ICU mortality | Must be described as the primary endpoint |
| Time representation | 8-hour blocks | Pragmatic representation, not a claim about biology |

### Current repo-local input contract

- Upstream standardized inputs:
  - `static/harmonized.{csv|parquet}`
  - `dynamic/harmonized.{csv|parquet}`
  - `blocked/asic_8h_block_index.{csv|parquet}`
  - `blocked/asic_8h_blocked_dynamic_features.{csv|parquet}`
  - `blocked/asic_8h_stay_block_counts.{csv|parquet}`
  - `qc/mech_vent_ge_24h_stay_level.{csv|parquet}`
  - `qc/mech_vent_ge_24h_episode_level.{csv|parquet}`
- The repo trusts adult age as an upstream contract and enforces mechanical-ventilation >=24h via the upstream QC artifact.

### Current local artifact cohort status

| metric | local artifact value | note |
| --- | --- | --- |
| Input hospitals | 8 | Local repo artifact, not the frozen scientific five-hospital description |
| Retained hospitals | 4 | `asic_UK02`, `asic_UK04`, `asic_UK07`, `asic_UK08` |
| Input stays | 80 | Local sample-limited artifact |
| Retained stays | 35 | Local sample-limited artifact |
| Primary valid prediction instances | 1,722 | Same count across all five horizons before labelability filtering |
| 24h labelable primary rows | 1,257 | 33 positives, 1,224 negatives, 465 unlabeled |
| Primary model-ready rows | 6,080 | Across 5 horizons and 3 splits |

### Site-restriction logic visible in local artifacts

- A hospital is retained only if ICU mortality is usable and at least 3 of 4 core physiologic groups have usable dynamic data.
- Local excluded hospitals were dropped either for missing/unusable ICU mortality (`asic_UK00`, `asic_UK01`, `asic_UK03`) or insufficient core-vital coverage (`asic_UK01`, `asic_UK06`).

## 3. Frozen Definitions And Key Design Choices

### Frozen Chapter 1 framing

- Chapter 1 is about heterogeneous predictability under the observed feature set, not biological death subtypes.
- Baseline risk models are operational instruments for identifying low-predicted fatal cases.
- Calibration is a gating issue before hard-case interpretation.
- Decomposition is secondary and should be easy to drop.

### Frozen outcome and time choices

- Primary horizon: 24h.
- Main horizon contrast: 48h.
- Sensitivity horizons: 8h, 16h, 72h.
- Unit of analysis: patient-time prediction instances at completed 8-hour blocks.

### Implemented cohort and instance rules

- Adult age >=18 is trusted upstream rather than rederived here.
- Mechanical ventilation >=24h is enforced via the upstream `mech_vent_ge_24h_qc` artifact.
- First-stay handling is proxied by `readmission == 0`.
- Valid instances require block eligibility plus at least 3 of 4 core physiologic groups within the block.
- ASIC lacks patient identifiers, so operational splitting remains stay-level rather than patient-level.

### Implemented label rule

- Labels are proxy within-horizon in-ICU mortality labels, not true event-timed labels.
- Positive at horizon `H`: `icu_mortality == 1` and `icu_end_time_proxy_hours` lies in `(t, t+H]`.
- Negative at horizon `H`: `icu_mortality == 0` and `icu_end_time_proxy_hours >= t+H`.
- All other rows stay unlabeled rather than being coerced negative.

### Feature and preprocessing boundary

- Primary feature set contains 31 base features.
- Extended adds 15 more variables.
- No high-complexity bespoke feature engineering is introduced.
- Bounded LOCF is applied only to prespecified families.
- Ventilator-variable LOCF is restricted to ventilation-supported windows.
- Missingness indicators are appended.
- Final imputation is deferred to downstream model training only.

### Split design

- Target split: 70/15/15.
- Split unit: `stay_id_global`.
- Splitting is done within retained hospitals and pooled.
- Split seed is frozen as `20260327`.
- Careful phrasing needed: the frozen spec says patient-level intent, but the actual operational implementation is stay-level because ASIC lacks patient identifiers.

## 4. Current Chapter 1 Essential Findings

### Findings that are methodological and safe to present now

- The Chapter 1 preprocessing pipeline is operational end-to-end on local artifacts: cohort restriction, valid-instance generation, proxy labels, model-ready export, observation-process features, and split assignment all exist and pass verification checks.
- Label prevalence increases with horizon in the local primary feature set:
  - 8h: 11 / 1,291 labelable rows (0.9%)
  - 16h: 22 / 1,274 (1.7%)
  - 24h: 33 / 1,257 (2.6%)
  - 48h: 63 / 1,210 (5.2%)
  - 72h: 87 / 1,163 (7.5%)
- The hard-case workflow is frozen and reproducible: `asic_logistic_last_eligible_nonfatal_q75_v1`.
- ICD-10-based coarse disease grouping is feasible with a reproducible hierarchy, though it is hierarchy-sensitive because `icd10_codes` is a multi-code bag rather than a principal diagnosis field.
- Observation-process variables are implemented and QC'd, including block-level coverage of core groups and time-since-last-observation fields.

### Findings that are present but should be presented as local/provisional only

- Current split pathology is substantial: the local primary test split has 0 positive stays and 0 positive labels, so all reported baseline evaluation metrics fall back to the validation split.
- On the local validation split at 24h, logistic regression performs materially better than XGBoost:
  - Logistic regression: AUROC 0.741, AUPRC 0.116, calibration slope 0.630
  - XGBoost: AUROC 0.514, AUPRC 0.032, calibration slope 0.016
- The local 24h risk-binned summaries are directionally consistent with a risk-structure story, but they are not a full-data ASIC result.
- Under the frozen logistic hard-case rule, 4 of 10 fatal stays (40%) are low-predicted at 24h in the local last-eligible stay-level slice.
- In the local 24h fatal-stay comparison, low-predicted fatal stays are enriched in:
  - `asic_UK07` (75% vs 0% among other fatal stays)
  - `respiratory / pulmonary` disease group (75% vs 17%)
- In that same local comparison, low-predicted fatal stays have:
  - higher PF ratio
  - lower creatinine
  - higher PEEP
- Local horizon-dependence outputs suggest the low-predicted fatal burden does not disappear with horizon change:
  - hard-case share is 0.40 at 8h, 16h, 24h
  - 0.50 at 48h
  - 0.60 at 72h
- The same horizon package labels the current pattern `change form`, not clean persistence.
- Cross-model hard-case agreement is limited at 24h: logistic vs recalibrated XGBoost Jaccard is 0.20.
- A 16h temporal preview does not suggest obvious collapse of the Chapter 1 signal, but it is only a preview and not a formal sensitivity analysis.

### Findings that should not be overstated

- The hard-case comparison slice is only 10 fatal stays at 24h.
- The hard-case summary explicitly warns `n_hard_cases_lt_20` at every horizon.
- Horizon-dependence memos explicitly say the local values are synthetic implementation-test outputs.
- Baseline interpretation notes explicitly say the current evaluation is pipeline-validation first and scientific evidence second.

## 5. Important Figures/Tables/Results Already Available

| asset | what it supports | main-deck ready? | note |
| --- | --- | --- | --- |
| `artifacts/chapter1/cohort/chapter1_cohort_summary.csv` | cohort headline numbers | yes, after simple visualization | Good source for a cohort summary slide |
| `artifacts/chapter1/cohort/chapter1_site_eligibility.csv` | hospital inclusion/exclusion logic | yes | Best source for a site-restriction flow |
| `artifacts/chapter1/splits/chapter1_stay_split_summary.csv` | split design and split pathology | yes | Use to explain zero-event test problem |
| `artifacts/chapter1/labels/chapter1_primary_proxy_label_summary_by_horizon.csv` | labelable/positive/unlabeled burden by horizon | yes | Useful for label/censoring slide |
| `artifacts/chapter1/model_ready/chapter1_primary_readiness_summary.csv` | feature-set and preprocessing status | yes | Good methods summary table |
| `artifacts/chapter1/evaluation/asic/baselines/primary_medians/combined_metrics.csv` | headline discrimination/calibration metrics | yes, with caveat banner | Strong supporting table |
| `artifacts/chapter1/evaluation/asic/baselines/primary_medians/logistic_regression/horizon_24h/mortality_vs_risk_plot.png` | 24h risk-structure view | yes, but label as local/validation | Prefer logistic over XGBoost for main deck |
| `artifacts/chapter1/evaluation/asic/baselines/primary_medians/logistic_regression/horizon_24h/reliability_plot.png` | calibration-first framing | yes | Important for risk-structure-first story |
| `artifacts/chapter1/evaluation/asic/baselines/primary_medians/logistic_regression/primary_24h_site_overview.png` | site-stratified sanity check | maybe | Useful if you need site heterogeneity context |
| `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/horizon_hard_case_summary.csv` | frozen hard-case definition across horizons | yes | Core descriptive table |
| `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/comparison_table.csv` | 24h hard-case vs other fatal comparison | yes | Best current hard-case table |
| `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/effect_size_figure.png` | compact hard-case figure | yes | Strong candidate main-deck figure |
| `artifacts/chapter1/evaluation/asic/horizon_dependence/final/mortality_risk_horizon_comparison.png` | horizon-dependence visual | yes, with strong caveat | Use only if clearly labeled provisional |
| `artifacts/chapter1/evaluation/asic/horizon_dependence/overlap/pairwise_overlap.csv` | membership overlap across horizons | yes | Good secondary table |
| `artifacts/chapter1/temporal_preview/asic/aggregation_16h/comparison/logistic_regression_24h_mortality_vs_risk_8h_vs_16h.png` | temporal aggregation preview | maybe | Better in appendix unless needed for reviewer defense |
| `artifacts/chapter1/observation_process/chapter1_observation_process_qc_summary.csv` | observation-process variable readiness | appendix | Supports methods and sensitivity-readiness claims |
| `artifacts/chapter1/evaluation/asic/icd10_disease_group_validation/final_group_counts.csv` | disease-group inventory | appendix or backup | Context, not a central result |

## 6. Open Gaps Where A Figure/Table Is Still Missing

- A polished cohort flow figure for the actual Chapter 1 retained cohort is not present yet, even though the underlying tables exist.
- A scientifically usable held-out test-set performance table is missing because the current test split has zero positive stays.
- External validation figures/tables for MIMIC-IV are missing entirely.
- A direct observation-process vs hard-case comparison figure/table is missing. The variables exist, but no linked hard-case sensitivity result was found.
- A treatment-limitation / end-of-life proxy inventory figure/table is missing. The evidence pack explicitly says no dedicated artifact was found.
- A full-data ASIC hard-case comparison figure/table is missing. The current saved artifacts are local/sample-limited and repeatedly flagged as synthetic or provisional.
- A polished slide-ready label-censoring figure is missing. The tables exist, but there is no saved visual explaining positive/negative/unlabeled logic by horizon.
- A formal temporal-resolution sensitivity figure set beyond the current 8h vs 16h preview is missing.
- A robust site-stratified calibration figure with adequate event counts is missing; current site-level artifacts are sparse.
- A final covariate summary table for the frozen hard-case comparison package is incomplete because exact age is absent; only `age_group` is available.

## 7. Recommended Slide Sections For A 12-15 Slide Presentation

### Recommended 13-slide spine

1. Title and one-sentence Chapter 1 question
2. Why this chapter matters: limits of predictability, not generic mortality prediction
3. Data sources and cohort contract: ASIC development, MIMIC-IV external validation target
4. Cohort restriction and site eligibility flow
5. Prediction setup: 8h blocks, 24h primary horizon, proxy label logic
6. Feature boundary and preprocessing policy: bounded LOCF, missingness indicators, no final imputation
7. Split design and current local limitations: stay-level split, no patient IDs, zero-event test split
8. Baseline models and calibration-first evaluation logic
9. Main 24h risk-structure view: mortality vs predicted risk plus calibration
10. Hard-case definition and 24h low-predicted fatal comparison
11. Horizon dependence: what changes, what persists, and why 24h remains the narrative anchor
12. What is currently solid vs provisional: synthetic/sample-limited outputs, agreement weaknesses, missing external validation
13. Next required analyses before strong chapter claims: full ASIC rerun, MIMIC replication, observation-process and treatment-limitation sensitivity

### If you need only 12 slides

- Merge slides 2 and 3.

### If you can use 15 slides

- Add one dedicated slide on observation-process instrumentation and why it matters.
- Add one dedicated slide on appendix-level sensitivities already attempted: temporal preview, SOFA not feasible, variable-package readiness.

## 8. Appendix Material That Should Not Go In The Main Deck

- Full per-hospital risk-binned tables and site-specific reliability details.
- Early-vs-late death timing sensitivity. The note itself says it is too sparse and best treated as decorative.
- SOFA feasibility audit. It is useful defensively, but it should not distract from the main analysis.
- Dynamic-proxy provenance audit and LOCF provenance details.
- Full feature-availability tables and LOCF summaries.
- The detailed ICD-10 hierarchy memo and exception logic.
- Notebook-level readiness checks and implementation runbooks.
- Full overlap matrices and heatmaps if the audience is not asking about stability in detail.
- Any decomposition-specific exploratory material, unless later full-data runs materially strengthen it.

## 9. Risks Of Misframing Or Overclaiming

| risk | why it is risky | safer phrasing |
| --- | --- | --- |
| Presenting Chapter 1 as death-subtype discovery | Frozen docs explicitly reject biological subtype claims | "heterogeneous predictability under the observed feature set" |
| Saying low-predicted deaths are inherently unpredictable | The chapter cannot distinguish unobserved from unobservable | "poorly captured in the recorded data at this resolution" |
| Treating proxy labels as true death timing | Labels use `icu_end_time_proxy_hours`, not true death timestamps | "proxy within-horizon in-ICU mortality labels" |
| Saying the split is patient-level | ASIC lacks patient identifiers | "operationally stay-level after first-stay proxy filtering" |
| Presenting validation fallback as held-out test evidence | Current test split has zero events | "local holdout/validation readout because the current test partition is not binary-evaluable" |
| Presenting hard-case proportions as stable subgroup estimates | The fatal comparison is only 10 stays and carries small-n warnings | "bounded descriptive signal in the current local artifacts" |
| Claiming model-invariant hard cases | Logistic vs recalibrated XGBoost agreement is weak | "definition- and model-sensitive hard-case structure" |
| Claiming horizon persistence of a subtype | Current horizon memo labels the pattern `change form` | "the low-predicted fatal burden changes in form across horizons" |
| Framing decomposition as the chapter backbone | Repo memos say the descriptive story already stands without it | "optional secondary summary layer" |
| Using SOFA as if it were available | Audit classifies SOFA as not feasible | "direct organ-support/dysfunction proxies instead of SOFA" |
| Implying external validation is already done | No local MIMIC-IV artifacts are present | "external validation is frozen as required work, not current evidence" |
| Ignoring treatment-limitation and observation-process confounding | Both are central interpretive threats in the frozen docs | Explicitly state that these are open sensitivity requirements |

## 10. File-Level Inventory: Which Files/Artifacts Support Each Section

| section | key supporting files |
| --- | --- |
| 1. Executive summary | `README.md`; `docs/chapter1_analysis_spec_frozen_v1.md`; `docs/phase1_working_reference.md`; `reports/ch1_asic_descriptive_viability_memo_draft.md` |
| 2. Data sources and cohorts | `README.md`; `docs/preprocessing_interface.md`; `artifacts/chapter1/cohort/chapter1_cohort_summary.csv`; `artifacts/chapter1/cohort/chapter1_site_eligibility.csv`; `artifacts/chapter1/cohort/chapter1_counts_by_hospital.csv`; `artifacts/chapter1/cohort/chapter1_stay_exclusion_summary_by_hospital.csv`; `notebooks/ch1_cohort_characterization.ipynb` |
| 3. Frozen definitions and key design choices | `docs/chapter1_analysis_spec_frozen_v1.md`; `docs/preprocessing_interface.md`; `docs/label_logic_audit.md`; `config/ch1_feature_sets.json`; `config/ch1_run_config.json`; `src/chapter1_mortality_decomposition/cohort.py`; `src/chapter1_mortality_decomposition/instances.py`; `src/chapter1_mortality_decomposition/labels.py`; `src/chapter1_mortality_decomposition/model_ready.py`; `src/chapter1_mortality_decomposition/splits.py`; `src/chapter1_mortality_decomposition/carry_forward.py` |
| 4. Current Chapter 1 essential findings | `artifacts/chapter1/model_ready/chapter1_primary_readiness_summary.csv`; `artifacts/chapter1/labels/chapter1_primary_proxy_label_summary_by_horizon.csv`; `artifacts/chapter1/splits/chapter1_stay_split_summary.csv`; `artifacts/chapter1/evaluation/asic/baselines/primary_medians/combined_metrics.csv`; `artifacts/chapter1/evaluation/asic/baselines/primary_medians/interpretation_note.md`; `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/horizon_hard_case_summary.csv`; `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/summary.md`; `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/agreement/logistic_regression_vs_xgboost_platt/horizon_hard_case_agreement_summary.csv`; `artifacts/chapter1/evaluation/asic/horizon_dependence/final/horizon_interpretation_memo.md`; `artifacts/chapter1/temporal_preview/asic/aggregation_16h/comparison/preview_note.md` |
| 5. Important figures/tables/results already available | `artifacts/chapter1/evaluation/asic/baselines/primary_medians/logistic_regression/horizon_24h/mortality_vs_risk_plot.png`; `artifacts/chapter1/evaluation/asic/baselines/primary_medians/logistic_regression/horizon_24h/reliability_plot.png`; `artifacts/chapter1/evaluation/asic/baselines/primary_medians/logistic_regression/primary_24h_site_overview.png`; `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/effect_size_figure.png`; `artifacts/chapter1/evaluation/asic/horizon_dependence/final/mortality_risk_horizon_comparison.png`; `artifacts/chapter1/temporal_preview/asic/aggregation_16h/comparison/logistic_regression_24h_mortality_vs_risk_8h_vs_16h.png` |
| 6. Open gaps | `reports/ch1_asic_descriptive_viability_evidence_pack.md`; `reports/ch1_asic_descriptive_viability_memo_draft.md`; `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison_variable_audit/asic_hard_case_comparison_variable_audit_memo.md`; `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_sofa_feasibility_audit/sofa_feasibility_memo.md`; `docs/phase1_working_reference.md` |
| 7. Recommended slide sections | `docs/chapter1_analysis_spec_frozen_v1.md`; `docs/phase1_working_reference.md`; `artifacts/chapter1/evaluation/asic/baselines/primary_medians/interpretation_note.md`; `reports/ch1_asic_descriptive_viability_memo_draft.md` |
| 8. Appendix material | `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/early_vs_late_death_split/early_vs_late_interpretation_note.md`; `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_sofa_feasibility_audit/sofa_feasibility_memo.md`; `artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/provenance_audit/dynamic_proxy_provenance_audit.md`; `docs/ch1_asic_icd10_disease_group_mapping_memo.md`; `artifacts/chapter1/observation_process/chapter1_observation_process_implementation_note.md`; `notebooks/ch1_baseline_model_readiness_check.ipynb` |
| 9. Risks of misframing or overclaiming | `docs/chapter1_analysis_spec_frozen_v1.md`; `docs/phase1_working_reference.md`; `docs/context.md`; `docs/label_logic_audit.md`; `artifacts/chapter1/evaluation/asic/baselines/primary_medians/interpretation_note.md`; `artifacts/chapter1/evaluation/asic/horizon_dependence/final/horizon_interpretation_memo.md`; `reports/ch1_asic_descriptive_viability_memo_draft.md` |

## Bottom-Line Recommendation For The Deck

- Build the main deck around frozen scientific framing, cohort/label contract, calibration-first baseline evaluation, the 24h hard-case comparison, and horizon dependence as a descriptive sensitivity.
- Keep all strong claims explicitly conditional on the observed feature set, proxy labels, local artifact status, and missing external validation.
- Do not let decomposition, SOFA, or appendix sensitivities displace the risk-structure-first story.
