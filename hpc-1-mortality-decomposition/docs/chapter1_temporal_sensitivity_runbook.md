# Chapter 1 ASIC Temporal Sensitivity Runbook

This issue is a bounded coarsening sensitivity inside the existing Chapter 1 workflow.

Scope:
- frozen primary reference aggregation: `8h`
- coarsened sensitivities: `16h`, `24h`
- primary interpretation anchor: `logistic_regression`, `24h`
- XGBoost remains secondary only
- hard-case rule remains frozen as `asic_logistic_last_eligible_nonfatal_q75_v1`

What the implementation does:
- rebuilds `16h` and `24h` blocked representations inside this repo from the standardized ASIC harmonized dynamic table plus the saved stay-level timing proxy table
- reuses frozen 8h stay-level split assignments where the retained stay remains present
- reruns preprocessing-derived tables, baselines, evaluation, and logistic hard-case outputs for each coarsened aggregation
- writes formal comparison artifacts under `artifacts/chapter1/temporal_sensitivity/asic/`

Cluster entrypoints in the HPC bundle:
- `run_temporal_sensitivity.sh`
- `slurm/submit_chapter1_temporal_sensitivity.slurm`
- `run_chapter1_temporal_sensitivity.py`

Expected artifact tree:
- `artifacts/chapter1/temporal_sensitivity/asic/aggregation_16h/`
- `artifacts/chapter1/temporal_sensitivity/asic/aggregation_24h/`
- `artifacts/chapter1/temporal_sensitivity/asic/comparison/`

Key comparison outputs:
- `reporting_metric_summary.csv`
- `calibration_summary.csv`
- `mortality_risk_structure_summary.csv`
- `hard_case_prevalence_summary.csv`
- `logistic_24h_hard_case_pairwise_overlap.csv`
- `logistic_24h_hard_case_directional_overlap.csv`
- `temporal_aggregation_sensitivity_interpretation.md`
- `provenance_and_limitations.md`
- `interpretation_memo_template.md`

Important limitations to keep explicit in reporting:
- this is not an aggregation search and not a full temporal-resolution study
- finer-than-8h was intentionally excluded
- 16h and 24h are derived inside the Chapter 1 repo rather than rebuilt from upstream raw time series
- the earlier `artifacts/chapter1/temporal_preview/asic/aggregation_16h/` package is superseded once the formal sensitivity run exists
