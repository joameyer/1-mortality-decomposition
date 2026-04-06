# ASIC Viability Memo for Chapter 1

## Question
Does the current ASIC hard-case and horizon artifact set already support Chapter 1 as a descriptive hard-case chapter, and should decomposition remain in scope?

## Evidence summary
The located artifact set documents a frozen hard-case rule (`asic_logistic_last_eligible_nonfatal_q75_v1`), a concrete `24h` fatal-stay comparison package exists with `4` low-predicted fatal stays and `6` other fatal stays out of `10` fatal stays. Located horizon shares among fatal stays are 8h 0.40, 16h 0.40, 24h 0.40, 48h 0.50, 72h 0.60. The saved horizon interpretation memo and figure are `artifacts/chapter1/evaluation/asic/horizon_dependence/final/horizon_interpretation_memo.md` and `artifacts/chapter1/evaluation/asic/horizon_dependence/final/mortality_risk_horizon_comparison.png`. The saved logistic-versus-recalibrated-XGBoost agreement artifact is a caution signal: 24h Jaccard is 0.20. The same hard-case, foundation, overlap, and final-horizon notes explicitly state that the local values come from synthetic stand-in data.

## Does ASIC already provide a defensible descriptive core?
Provisionally yes, as a bounded descriptive core. The hard-case rule is explicit, the low-predicted-versus-other-fatal comparison has already been turned into tables and a figure, and the horizon package shows that the low-risk fatal burden does not vanish immediately when the horizon changes. That is enough to frame Chapter 1 first as a descriptive hard-case chapter. It is not enough for a firm scientific claim yet because the local repo is synthetic and some robustness / sensitivity pieces remain incomplete.

## Decomposition decision
`GO, but secondary only`

The descriptive argument already stands on ASIC hard-case definition, comparison, and horizon structure, so decomposition is not needed to make the chapter work. If retained, decomposition should stay clearly secondary and easy to drop. It should not become the chapter's organizing logic unless the full-data ASIC rerun and later replication materially strengthen the case.

## Main remaining risks
The present readout is provisional because the located local outputs are synthetic implementation-test artifacts. The saved horizon package labels the pattern as changing form rather than a clean horizon-stable subtype. The variable audit says the frozen Issue 3.2 package is not fully ready because exact age is absent, and the SOFA feasibility audit says standard SOFA is not feasible. The agreement artifact also suggests the hard-case signal is definition-sensitive rather than obviously model-invariant.

## What still depends on MIMIC
MIMIC still has to answer whether the ASIC descriptive structure actually replicates, whether the same subgroup contrasts remain visible, whether the horizon pattern still looks like persistence versus change of form, and whether any retained decomposition summary is robust enough to keep. MIMIC should validate or downgrade the ASIC descriptive story; it should not be used to rescue a decomposition that ASIC itself does not clearly need.

## Provisional recommendation
Use the ASIC hard-case comparison and horizon package as the Chapter 1 backbone, keep decomposition explicitly secondary, and rerun this exact review workflow on full ASIC HPC artifacts before treating the memo as a scientific decision. If the full-data ASIC rerun still supports the descriptive core, retain decomposition only as an optional summary layer and drop it quickly if later replication does not materially strengthen the case.
