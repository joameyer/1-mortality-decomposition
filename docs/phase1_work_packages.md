# Phase 1 Linear Work Plan — Chapter 1 Mortality Risk Structure (Revised)
Last revision: 16.04.2026

This version updates the original Phase 1 work plan to reflect the current project state.

Rules used in this version:
- Default issues have **high** priority.
- Stretch issues have **low** priority and have an "S" as a prefix before the issue number
- ASIC and MIMIC issues are kept **explicitly separate**
- each issue is written for direct transfer into Linear
- each issue includes **Goal**, **Details**, **Deliverables**, and **Definition of done**
- the plan is stretched to **6 approx. two-week packages** to keep Package 2 realistic and to avoid hiding MIMIC work inside ASIC issues

---

## Package 1 — Chapter 1 startup, framing lock, and ASIC preprocessing rebuild

### 1.1. Lock Chapter 1 operational specification
**Goal**  
Freeze the non-negotiable Chapter 1 analysis definitions before empirical work proceeds further.

**Details**  
- freeze cohort inclusion and exclusion criteria
- freeze mortality endpoint
- freeze index time and observation anchor
- freeze repeat-stay handling
- freeze short-stay handling
- freeze split strategy
- freeze primary and secondary prediction horizons
- freeze reporting template and evaluation outputs
- ensure the frozen spec matches current repo boundaries and current Phase 1 framing

**Deliverables**  
- written Chapter 1 analysis specification
- frozen cohort, outcome, horizon, and split definitions
- frozen evaluation framework

**Definition of done**  
- there is one authoritative Chapter 1 operational specification with no unresolved ambiguity

---

### 1.2. Convert Chapter 1 literature into operational design constraints
**Goal**  
Turn completed literature work into explicit analysis constraints and claim boundaries.

**Details**  
- write a short literature-to-design memo
- extract what literature implies for calibration scrutiny
- extract what literature implies for transportability and external validation
- extract what literature implies for horizon choice and temporal-resolution sensitivity
- extract what literature implies for observation-process sensitivity
- extract what literature implies for treatment-limitation confounding
- extract what literature implies for acceptable wording and negative-result framing

**Deliverables**  
- concise literature-to-design memo
- list of mandatory analyses justified by the literature
- list of explicit claim boundaries and main overreach risks

**Definition of done**  
- a short decision memo exists that directly constrains Phase 1 execution

---

### 1.3. Prepare major supervisor startup check-in for Chapter 1
**Goal**  
Force early agreement on what Chapter 1 is, is not, and what counts as enough.

**Details**  
- summarize bounded Chapter 1 purpose and claim
- summarize explicit non-claims
- define minimum viable chapter criteria
- prepare decision questions on terminology, endpoint, horizon strategy, acceptability of weak or negative results, and whether decomposition is strictly secondary

**Deliverables**  
- 1-page supervisor check-in note
- explicit decision questions
- minimum viable Chapter 1 summary

**Definition of done**  
- the startup check-in package is ready to send or present immediately

---

### 1.4. Rebuild and verify ASIC Chapter 1 preprocessing and model-ready datasets
**Goal**  
Make ASIC Chapter 1 analysis-ready under the frozen operational specification.

**Details**  
- rerun or verify ASIC cohort extraction
- verify 8-hour block construction
- verify label generation for frozen horizons
  - primary: 24h
  - secondary: 48h
  - sensitivities: 8h and 16h
- verify carry-forward and missingness handling behavior
- generate train, validation, and test splits
- confirm ASIC outputs are model-ready inside the Chapter 1 analysis repository
- document preprocessing logic and sanity checks

**Deliverables**  
- model-ready ASIC datasets
- horizon-specific target tables
- split objects
- compact ASIC data-readiness summary with counts and missingness overview

**Definition of done**  
- reproducible model-ready ASIC Chapter 1 datasets exist and core sanity checks are passed

---

### 1.5. Inventory ASIC treatment-limitation and end-of-life proxies
**Goal**  
Determine whether direct treatment-limitation sensitivity testing is possible in ASIC.

**Details**  
- search ASIC for DNR, DNI, palliative, comfort-care, withdrawal, withholding, or similar structured proxies
- document variable availability and reliability
- document whether any direct treatment-limitation sensitivity is feasible
- explicitly document absence if direct testing is not possible

**Deliverables**  
- ASIC treatment-limitation proxy inventory
- ASIC feasibility note
- explicit absence statement if direct testing is not possible

**Definition of done**  
- ASIC treatment-limitation sensitivity status is known clearly as testable, weakly testable, or untestable

---

### 1.6. Update Linear and `context.md` at Package 1 close
**Goal**  
Create a clean and authoritative handover into Package 2.

**Details**  
- update issue status in Linear
- record frozen Chapter 1 definitions
- record ASIC preprocessing status
- record ASIC treatment-limitation proxy status
- record supervisor startup check-in status
- record next Package focus

**Deliverables**  
- updated Linear board
- updated `context.md`
- concise Package 1 closure note

**Definition of done**  
- a new Package 2 chat could start without reconstructing Package 1 from memory

---

### 1.S1 Stretch issue — Prepare annotated reading order for remaining Phase 1 papers
**Goal**  
Create a practical reading order for any remaining Chapter 1 literature work.

**Details**  
- rank the most useful remaining papers
- add a brief note on why each paper matters for execution rather than background only

**Deliverables**  
- ranked reading list
- 1–2 sentence note per paper

**Definition of done**  
- a usable reading order exists for any additional literature time

---

## Package 2 — ASIC observation-process inventory, baseline risk models, and first empirical outputs

### 2.1. Freeze ASIC observation-process and missingness variable set
**Goal**  
Freeze a minimal, reproducible ASIC variable set for later Chapter 1 observation-process sensitivity analysis.

**Details**  
- define ASIC measurement-density variables
- define ASIC missingness-burden variables
- define ASIC time-since-last-measurement or longest-gap variables
- define ASIC block-level completeness indicators
- keep the set minimal and reproducible
- record transfer considerations for later MIMIC implementation without requiring MIMIC completion in this Package

**Deliverables**  
- final ASIC observation-process variable list
- derivation rules
- ASIC feasibility note
- short transfer note for later MIMIC implementation

**Definition of done**  
- a concrete ASIC-derivable observation-process variable set is frozen for later use

---

### 2.2. Implement baseline Chapter 1 risk models on ASIC
**Goal**  
Build the independent baseline risk-modeling pipeline on ASIC.

**Details**  
- implement logistic regression
- implement XGBoost
- add a simple feedforward net only if friction is very low
- save predictions for all frozen horizons
- ensure outputs are reusable for hard-case analysis and later external validation

**Deliverables**  
- trained ASIC logistic regression baseline
- trained ASIC XGBoost baseline
- saved ASIC prediction outputs
- reproducible output structure

**Definition of done**  
- both baseline models run successfully on the frozen ASIC Chapter 1 setup and outputs are stored for reuse

---

### 2.3. Evaluate ASIC discrimination, calibration, and mortality-vs-risk structure
**Goal**  
Produce the first core empirical Chapter 1 outputs on ASIC.

**Details**  
- compute AUROC
- compute AUPRC where relevant
- compute calibration slope and intercept
- generate reliability plots
- generate observed mortality vs predicted-risk plots
- generate horizon comparison view
- generate first site-stratified ASIC sanity check
- write a short first-pass interpretation note

**Deliverables**  
- ASIC evaluation package
- first ASIC mortality-vs-risk figures
- first reviewable interpretation note

**Definition of done**  
- ASIC empirical outputs include calibration checks sufficient to determine whether hard-case analysis is interpretable

---

### 2.4. Update Linear and `context.md` at Package 2 close
**Goal**  
Close the overdue ASIC setup work cleanly and prepare Package 3 hard-case analysis.

**Details**  
- update Linear to reflect completed ASIC setup and modeling work
- record frozen ASIC observation-process variable set
- record baseline model status
- record first ASIC evaluation findings
- record next Package focus on hard-case definition and comparative analysis

**Deliverables**  
- updated Linear board
- updated `context.md`
- concise Package 2 closure note

**Definition of done**  
- Package 3 can start directly from a documented ASIC baseline state

---

### 2.S1. Stretch issue — Run one ASIC early temporal-aggregation preview
**Goal**  
Check whether the ASIC hard-case pattern looks obviously dependent on the 8-hour aggregation choice before formal sensitivity work.

**Details**  
- rerun one alternative aggregation only if Package core work finishes early
- keep cohort, endpoint, splits, horizons, models, and evaluation framework fixed as far as possible
- write a compact note on whether obvious instability is already visible

**Deliverables**  
- one early aggregation preview run
- short comparison note

**Definition of done**  
- an early aggregation preview exists or is dropped with no impact on Package completion

---

## Package 3 — ASIC hard-case characterization and Chapter 1 viability decision

### 3.1. Define low-predicted fatal cases on ASIC
**Goal**  
Freeze the operational hard-case definition on ASIC before comparative analysis begins.

**Details**  
- define what counts as a low-predicted fatal case for each horizon
- decide whether the primary rule is threshold-based, quantile-based, or another simple prespecified rule
- document the rationale
- ensure the rule is fixed before looking at comparative pattern results

**Deliverables**  
- written ASIC hard-case definition
- written thresholding or quantile rule
- horizon-specific implementation note
- rationale note

**Definition of done**  
- the ASIC low-predicted fatal-case definition is fixed and no post hoc threshold shopping is needed later

---

### 3.2. Characterize ASIC low-predicted fatal cases vs other fatal cases
**Goal**  
Run the first main structural Chapter 1 comparison on ASIC.

**Details**  
- compare low-predicted fatal cases against other fatal cases on demographics, admission type, disease-group composition, ICU timing, site distribution, and available organ-support or dysfunction proxies
- produce one core table and one compact visual output
- write a short summary of main differences

**Deliverables**  
- ASIC comparison table for low-predicted vs other fatal cases
- compact ASIC figure or visualization
- summary of main differences observed

**Definition of done**  
- the main ASIC hard-case comparison is complete and at least one output is directly usable in the chapter

---

### 3.3. Analyze horizon dependence of ASIC mortality predictability structure
**Goal**  
Test whether the apparent low-information structure changes materially by prediction horizon on ASIC.

**Details**  
- repeat the core structure analysis for frozen ASIC horizons
- compare low-predicted death share across horizons
- compare mortality-vs-risk shape across horizons
- write a short interpretation of whether hard cases shrink, persist, or change form

**Deliverables**  
- horizon-specific ASIC structure outputs
- comparison of low-predicted death share across horizons
- short interpretation memo

**Definition of done**  
- ASIC horizon dependence is documented clearly enough to guide later sensitivity work

---

### 3.4. Write Package 3 ASIC viability memo for Chapter 1
**Goal**  
Decide whether Chapter 1 already stands on ASIC descriptive structure or still depends too much on later summary modeling.

**Details**  
- summarize ASIC hard-case results and horizon results
- state whether the ASIC analysis already provides a defensible descriptive core
- state whether decomposition should proceed at all
- state main remaining risks and what still depends on later MIMIC work

**Deliverables**  
- 1-page ASIC viability memo
- explicit decomposition go or no-go statement
- main risk summary

**Definition of done**  
- a clear ASIC viability judgment exists and can guide Package 4 and supervisor discussion

---

### 3.5. Run minor supervisor check-in on ASIC hard-case results
**Goal**  
Validate that Chapter 1 still looks acceptable after the first real ASIC structural analysis.

**Details**  
- prepare a short supervisor-facing summary of ASIC hard-case findings and horizon dependence
- ask directly whether the framing still looks right
- ask whether decomposition still deserves to remain in the plan

**Deliverables**  
- short ASIC hard-case findings summary
- short horizon summary
- decision question on decomposition role

**Definition of done**  
- the supervisor check-in is completed or ready to send immediately

---

### 3.6. Update Linear and `context.md` at Package 3 close
**Goal**  
Preserve the ASIC hard-case decision state cleanly.

**Details**  
- update Linear
- update `context.md` with the ASIC hard-case definition, main findings, horizon dependence, and current decomposition go or no-go status
- record what remains open for ASIC sensitivity work and later MIMIC transfer

**Deliverables**  
- updated Linear board
- updated `context.md`
- concise Package 3 closure note

**Definition of done**  
- Package 4 can start without reconstructing the ASIC hard-case decision state

---

### 3.S1. Stretch issue — Add ASIC early-vs-late ICU death subgroup split
**Goal**  
Explore whether low-predicted fatal cases differ meaningfully between early and later ICU deaths.

**Details**  
- run one additional stratified ASIC comparison splitting deaths by ICU timing
- write a short note on whether timing materially changes interpretation

**Deliverables**  
- early-vs-late subgroup comparison
- short interpretation note

**Definition of done**  
- a concise timing subgroup result exists or the issue is dropped without consequence

---

## Package 4 — ASIC sensitivity analyses, stratified structure, and decomposition decision

### 4.1. Derive and analyze ASIC observation-process variables in hard-case comparisons
**Goal**  
Determine whether ASIC hard-case deaths are enriched for sparse or irregular monitoring patterns.

**Details**  
- derive the frozen ASIC observation-process variables
- compare low-predicted fatal cases and other fatal cases on these variables
- summarize effect sizes
- determine whether measurement-process artifacts appear to explain a meaningful share of the hard-case pattern

**Deliverables**  
- derived ASIC observation-process dataset
- comparison analysis
- effect size summary
- short interpretation memo

**Definition of done**  
- the likely contribution of ASIC observation-process artifacts is documented explicitly

---

### 4.2. Record ASIC treatment-limitation sensitivity status formally
**Goal**  
Close the treatment-limitation confounding issue honestly for ASIC.

**Details**  
- document again whether direct treatment-limitation or end-of-life sensitivity is feasible in ASIC
- if direct testing is not possible, write the formal explicit absence note
- state clearly how this limits Chapter 1 interpretation on ASIC

**Deliverables**  
- ASIC treatment-limitation sensitivity note or explicit absence note
- summary of proxy weakness and interpretation risk

**Definition of done**  
- ASIC treatment-limitation sensitivity is either analyzed or formally declared untestable with reasons

---

### 4.3. Run ASIC temporal aggregation sensitivity analysis
**Goal**  
Test whether the ASIC Chapter 1 pattern is materially dependent on temporal aggregation.

**Details**  
- rerun the core ASIC structure analysis using one finer resolution than 8 hours if feasible plus 16-hour and 24-hour blocks
- compare calibration, mortality-vs-risk structure, low-predicted fatal-case prevalence, and overlap or stability of identified hard cases
- write a short interpretation of whether the main ASIC pattern is stable, weakened, or substantially altered

**Deliverables**  
- alternative aggregation ASIC runs
- rerun structure outputs
- aggregation comparison summary
- short interpretation memo

**Definition of done**  
- temporal aggregation dependence is explicitly tested and documented for ASIC

---

### 4.4. Run ASIC disease-stratified predictability-structure analyses
**Goal**  
Assess whether the ASIC Chapter 1 pattern differs across clinically plausible groups.

**Details**  
- run the main structure analysis in the feasible key ASIC strata such as infection-related groups, medical vs surgical, and non-pulmonary comparator groups
- focus on whether direction and strength of predictability heterogeneity are stable across groups
- keep subgroup interpretation weak unless counts and effect sizes are clearly adequate
- document where subgroup sample size or proxy weakness limits interpretation

**Deliverables**  
- ASIC stratified analysis outputs
- summary table or figure
- interpretation note on heterogeneity across groups

**Definition of done**  
- key ASIC disease-stratified comparisons are complete and interpretable enough to judge whether the broad Chapter 1 pattern is clinically heterogeneous or broadly stable across groups

---

### 4.5. Run ASIC site-sensitivity analysis for hard-case structure
**Goal**  
Determine whether the ASIC Chapter 1 pattern is materially driven by site or remains broadly present across centers.

**Details**  
- run an explicit ASIC site-sensitivity analysis focused on whether the hard-case pattern, low-predicted fatal-case share, or main descriptive findings are disproportionately driven by one site
- quantify whether the observed `asic_UK04` enrichment materially changes interpretation or remains modest
- check whether the broad hard-case pattern persists after examining site composition directly
- state clearly whether site/context dependence weakens the bounded Chapter 1 claim or only narrows it

**Deliverables**  
- ASIC site-sensitivity outputs
- site-level comparison summary
- short interpretation note on site dependence

**Definition of done**  
- site dependence has been tested explicitly enough to rule out a single-site-driven Chapter 1 read or to document clearly that the ASIC result is materially context-dependent

---

### 4.6. Decide whether decomposition remains justified for Chapter 1
**Goal**  
Make an explicit gate decision on whether a compact decomposition or weighting summary still deserves to be done.

**Details**  
- review Package 3 viability memo and Package 4 sensitivity findings
- decide whether decomposition should proceed as a secondary summary, be downgraded, or be dropped entirely
- record the rationale in writing

**Deliverables**  
- written decomposition decision
- short rationale note

**Definition of done**  
- there is a documented proceed, downgrade, or drop decision before any decomposition work starts

---

### 4.7. Update Chapter 1 interpretation, resolve documentation conflicts, and freeze ASIC figure/table plan
**Goal**  
Revise the bounded Chapter 1 claim based on what the ASIC sensitivity analyses actually show and close the remaining internal documentation weaknesses before MIMIC.

**Details**  
- rewrite the bounded claim if necessary
- state clearly how much of the hard-case pattern may be explained by observation process, temporal aggregation, treatment-limitation limitations, or site dependence
- reconcile the baseline-package provenance conflict so the interpretation notes match the actual frozen ASIC evaluation outputs
- replace any synthetic or provisional horizon write-up with a real-data interpretation note grounded in the saved horizon tables and overlap results
- choose the final ASIC core figures and tables that carry the argument
- drop decorative or redundant outputs

**Deliverables**  
- revised interpretation memo
- revised bounded claim wording if needed
- reconciled baseline provenance note
- real-data horizon interpretation note
- final ASIC figure list
- final ASIC table list

**Definition of done**  
- an updated defensible ASIC interpretation exists, the baseline and horizon documentation conflicts are closed, and the ASIC figure/table plan is frozen

---

### 4.8. Update Linear and `context.md` at Package 4 close
**Goal**  
Close the near-final ASIC Chapter 1 state cleanly and prepare the MIMIC Package.

**Details**  
- update Linear
- update `context.md` with ASIC sensitivity outcomes, decomposition decision, revised claim wording, and frozen ASIC figure/table plan
- record what must transfer into MIMIC external validation work

**Deliverables**  
- updated Linear board
- updated `context.md`
- concise Package 4 closure note

**Definition of done**  
- Package 5 can start from a clean frozen ASIC Chapter 1 state

---

### 4.S1. Stretch issue — Analyze model-class dependence of hard-case agreement across horizons
**Goal**  
Assess whether the Chapter 1 hard-case structure is broadly model-robust or materially model-dependent across the frozen prediction horizons.

**Details**  
- compare logistic-regression and XGBoost hard-case counts across horizons
- summarize overlap and Jaccard agreement by horizon
- quantify directional overlap in both directions where available
- compare whether horizon-related changes are driven more by logistic stability or XGBoost expansion
- review whether disagreement appears concentrated in specific risk bands or clinically relevant subgroups if low-friction
- write a short interpretation note on whether the Chapter 1 hard-case concept appears model-robust, partially model-dependent, or definition-sensitive
- keep this issue descriptive and bounded; do not introduce new model classes

**Deliverables**  
- cross-model hard-case agreement summary table
- compact figure across horizons
- short interpretation memo

**Definition of done**  
- there is a written judgment on how strongly the Chapter 1 hard-case structure depends on baseline model class across horizons

---

### 4.S2. Stretch issue — Implement a simple feedforward neural-net baseline for Chapter 1 hard-case sensitivity
**Goal**  
Test whether the Chapter 1 hard-case structure changes materially when using a simple nonlinear baseline model beyond logistic regression and XGBoost.

**Details**  
- implement one simple feedforward neural-network baseline using the frozen Chapter 1 feature set and preprocessing
- keep architecture and tuning minimal; this is a bounded sensitivity analysis, not a model-development branch
- generate predictions for the frozen Chapter 1 horizons
- evaluate discrimination and calibration in the same framework used for logistic regression and XGBoost
- derive hard-case counts under the same operational rules used for the other baseline models
- compare hard-case counts, overlap, and horizon behavior against logistic regression and XGBoost
- write a short interpretation note on whether the Chapter 1 hard-case concept looks broadly robust to adding a simple nonlinear model or becomes more definition-sensitive
- do not introduce sequence modeling, new feature engineering, or architecture search

**Deliverables**  
- trained feedforward neural-net baseline
- saved prediction outputs for frozen horizons
- performance and calibration summary
- hard-case overlap comparison against logistic regression and XGBoost
- short interpretation memo

**Definition of done**  
- there is a documented judgment on whether adding a simple feedforward neural-net baseline materially changes the Chapter 1 hard-case interpretation

---

## Package 5 — External validation on MIMIC

### 5.1. Rebuild and verify MIMIC Chapter 1 preprocessing and ASIC–MIMIC alignment
**Goal**  
Make MIMIC Chapter 1 analysis-ready under the frozen Chapter 1 design.

**Details**  
- define the closest feasible MIMIC cohort aligned to the frozen Chapter 1 design
- rebuild or verify MIMIC cohort extraction
- verify block construction and label generation for frozen horizons
- verify carry-forward and missingness handling behavior
- document alignment constraints and unavoidable deviations relative to ASIC
- generate MIMIC analysis tables and splits as needed for external validation

**Deliverables**  
- model-ready MIMIC datasets
- MIMIC horizon-specific target tables
- MIMIC data-readiness summary
- ASIC–MIMIC alignment note with explicit deviations

**Definition of done**  
- reproducible model-ready MIMIC Chapter 1 datasets exist and main alignment constraints are documented explicitly

---

### 5.2. Inventory MIMIC treatment-limitation and end-of-life proxies
**Goal**  
Determine whether direct treatment-limitation sensitivity is feasible in MIMIC and how it compares with ASIC.

**Details**  
- search MIMIC for DNR, DNI, palliative, comfort-care, withdrawal, withholding, and related structured proxies
- document availability and reliability
- document whether these proxies support later hard-case sensitivity work in MIMIC

**Deliverables**  
- MIMIC treatment-limitation proxy inventory
- MIMIC feasibility note
- short ASIC vs MIMIC contrast note

**Definition of done**  
- MIMIC treatment-limitation sensitivity feasibility is known clearly and dataset differences are documented

---

### 5.3. Freeze MIMIC observation-process and missingness variable set
**Goal**  
Freeze the MIMIC observation-process variable set using the ASIC-derived template where possible.

**Details**  
- map the ASIC observation-process variable concept set to MIMIC
- document which variables transfer directly, which require adaptation, and which cannot be reproduced cleanly
- keep the MIMIC set minimal and methodologically aligned with ASIC

**Deliverables**  
- final MIMIC observation-process variable list
- derivation rules
- ASIC–MIMIC transferability note

**Definition of done**  
- a concrete MIMIC-derivable observation-process variable set is frozen for external validation and later sensitivity work

---

### 5.4. Implement baseline Chapter 1 risk models on MIMIC
**Goal**  
Run the frozen baseline risk-modeling framework on MIMIC as the external validation dataset.

**Details**  
- implement logistic regression on MIMIC
- implement XGBoost on MIMIC
- save predictions for frozen horizons
- ensure the output structure is aligned with the ASIC result structure as closely as possible

**Deliverables**  
- trained MIMIC logistic regression baseline
- trained MIMIC XGBoost baseline
- saved MIMIC prediction outputs
- reproducible aligned output structure

**Definition of done**  
- both baseline models run successfully on the frozen MIMIC Chapter 1 setup and outputs are stored for reuse

---

### 5.5. Generate first MIMIC calibration, mortality-vs-risk, and site/context comparison outputs
**Goal**  
Produce the first external-validation empirical outputs for Chapter 1 and check whether the broad structure transports without obvious context-specific reversal.

**Details**  
- compute discrimination and calibration metrics on MIMIC
- generate reliability plots
- generate observed mortality vs predicted-risk plots
- generate first horizon-specific comparison outputs
- write a short first-pass note on whether the broad pattern looks transportable or clearly different
- include an initial comparison note on which ASIC interpretation-sensitive features appear to transport cleanly, weaken, or become cohort/context-specific in MIMIC

**Deliverables**  
- first MIMIC evaluation package
- MIMIC mortality-vs-risk figures
- first transportability note
- initial ASIC–MIMIC context comparison note

**Definition of done**  
- MIMIC empirical outputs exist and are sufficient to begin the final ASIC vs MIMIC comparison Package with the main transportability risks already surfaced

---

### 5.6. Characterize MIMIC low-predicted fatal cases and main sensitivity structure
**Goal**  
Complete the core external-validation structure analysis for Chapter 1 on MIMIC.

**Details**  
- apply the frozen hard-case definition on MIMIC
- compare low-predicted fatal cases against other fatal cases on the key descriptive dimensions feasible in MIMIC
- run MIMIC observation-process comparison where feasible
- include decomposition only if it survived ASIC decision gates and is worth testing externally

**Deliverables**  
- MIMIC hard-case characterization table
- MIMIC compact figure or visualization
- MIMIC observation-process sensitivity summary
- MIMIC decomposition summary if retained

**Definition of done**  
- the core external-validation structure package is complete for MIMIC

---

### 5.7. Compare ASIC and MIMIC Chapter 1 results
**Goal**  
Assess transportability of the main Chapter 1 conclusions.

**Details**  
- compare baseline model performance
- compare mortality-vs-risk structure
- compare hard-case characteristics
- compare observation-process findings where feasible
- compare decomposition behavior if retained
- decide whether the result replicates strongly, partially, or fails materially

**Deliverables**  
- ASIC vs MIMIC comparison summary
- replication classification
- interpretation of transport limitations

**Definition of done**  
- a clear external comparison judgment exists and is ready to be written into the chapter

---

### 5.8. Reconfirm whether decomposition remains worth keeping after MIMIC external-validation
**Goal**  
Run the final narrow decision gate on whether any compact decomposition or weighting summary should remain in the Chapter 1 plan after the first MIMIC external-validation results.

**Details**  
- review the Package 4 ASIC decomposition downgrade decision
- review the first MIMIC external-validation outputs:
  - calibration and discrimination
  - mortality-vs-risk structure
  - first transportability note
  - any early hard-case preview if available
- decide whether MIMIC:
  - materially strengthens the case for retaining a brief optional decomposition summary,
  - leaves decomposition downgraded and optional,
  - or supports dropping decomposition entirely
- keep this as a very narrow keep-vs-drop gate, not a reopening of broad decomposition work
- do not implement new decomposition modeling in this issue
- record the default rule clearly: decomposition stays out unless external validation shows clear added value beyond the descriptive hard-case framework

**Deliverables**  
- short post-MIMIC decomposition gate memo
- explicit keep-optional vs drop-entirely decision
- short rationale note
- brief statement of downstream consequence for Package 6 drafting

**Definition of done**  
- there is a written post-MIMIC keep-vs-drop decision for decomposition before final Chapter 1 synthesis begins

---

### 5.9. Update Linear and `context.md` at Package 5 close
**Goal**  
Close MIMIC setup and first external-validation modeling cleanly before final Chapter 1 synthesis.

**Details**  
- update Linear
- update `context.md` with MIMIC preprocessing status, proxy inventory status, observation-process variable status, baseline model status, and first transportability impressions
- record final Package 6 synthesis tasks

**Deliverables**  
- updated Linear board
- updated `context.md`
- concise Package 5 closure note

**Definition of done**  
- Package 6 can start directly from a documented MIMIC external-validation state

---

### 5.S1. Stretch issue — Fit one exploratory membership model for low-predicted fatal-case status after external validation
**Goal**  
Run one compact exploratory summary model to assess which measured factors are most associated with low-predicted fatal-case status under the final retained Chapter 1 framing after external validation.

**Details**  
- use the frozen post-validation hard-case definition retained for Chapter 1
- fit one simple interpretable model predicting low-predicted fatal-case status among fatal stays
- use only already-derived observation-process variables, timing variables, and other low-friction contextual factors
- decide explicitly whether the model should be fit:
  - on ASIC only as the primary descriptive anchor, or
  - in a matched ASIC/MIMIC formulation if cross-dataset implementation is low-friction
- use the model only as a compact descriptive summary, not as a new main analysis
- write a short interpretation note on which factors appear most associated with hard-case membership and whether they mainly reinforce the existing bounded Chapter 1 interpretation
- do not introduce clustering, latent subtype claims, or new feature engineering

**Deliverables**  
- exploratory membership model
- summary of strongest associated factors
- short interpretation memo

**Definition of done**  
- one compact post-validation descriptive summary exists, or the issue is dropped without affecting Chapter 1 completion

---

## Package 6 — Final Chapter 1 verdict, drafting, and supervisor delivery

### 6.1. Freeze final Chapter 1 verdict
**Goal**  
Lock the final interpretation category for Chapter 1.

**Details**  
- classify the final result as strong positive, weak but usable, or negative bounded result
- justify the classification explicitly
- state what the final verdict means for later chapters and whether any weighting or decomposition output is reusable downstream

**Deliverables**  
- final Chapter 1 verdict memo
- justification note
- downstream implications note

**Definition of done**  
- the final Chapter 1 interpretation is frozen and ready to be written

---

### 6.2. Draft Chapter 1 manuscript package
**Goal**  
Convert the completed Phase 1 analysis into a supervisor-ready Chapter 1 draft.

**Details**  
- draft introduction
- draft methods
- draft results
- draft discussion
- integrate figures and tables
- integrate bounded claim, explicit non-claims, external validation, and named limitations
- ensure the chapter reads as risk-structure-first rather than decomposition-first

**Deliverables**  
- draft introduction
- draft methods
- draft results
- draft discussion
- integrated figures and tables
- integrated limitations and non-claims

**Definition of done**  
- a full Chapter 1 draft exists and is ready to send to the supervisor

---

### 6.3. Send Chapter 1 package to supervisor
**Goal**  
Deliver the completed Chapter 1 package for supervisor review.

**Details**  
- prepare a short cover note
- assemble final draft plus figure and table bundle
- send the package
- log what was sent and when

**Deliverables**  
- supervisor cover note
- final draft package assembled
- package sent
- send date and contents logged

**Definition of done**  
- the Chapter 1 package has been sent to the supervisor

---

### 6.4. Update context for Phase 2 handover
**Goal**  
Close Phase 1 cleanly so Phase 2 can start without ambiguity.

**Details**  
- update `context.md` with final Chapter 1 result
- record which outputs are reusable downstream
- record unresolved risks
- record supervisor feedback status
- record what Phase 2 depends on next

**Deliverables**  
- final Chapter 1 result summary
- reusable downstream outputs summary
- unresolved risks summary
- supervisor feedback status summary
- updated next-phase dependencies in `context.md`

**Definition of done**  
- `context.md` is fully updated for Phase 2 startup

---

### 6.S1. Stretch issue — Prepare concise abstract and anticipated feedback questions
**Goal**  
Prepare a short abstract-style summary and likely supervisor questions.

**Details**  
- write a concise Chapter 1 abstract or summary paragraph
- list likely supervisor objections or revision questions

**Deliverables**  
- short abstract or chapter summary
- anticipated feedback question list

**Definition of done**  
- a concise summary and question list exist or the issue is dropped without consequence
