# Cluster Results

This directory is the local mirror for approved Chapter 1 result exports from the HPC cluster.

## Role

- `cluster-results/chapter1_true_results/` is the authoritative local result root for scientific review
- it is intended for reports, figures, notebooks, and presentation-building
- it is not a working directory for protected-data preprocessing or model training

## What May Be Mirrored Here

Approved derived artifacts only, for example:

- run manifests
- metrics tables
- aggregated summaries
- figures
- interpretation notes
- approved prediction exports if data governance allows them

## What Must Not Be Mirrored Here

Do not copy restricted row-level datasets such as:

- true patient-level static source tables
- blocked filled monitoring data
- raw or harmonized dynamic row-level tables
- model-ready row-level datasets unless explicitly approved
- any artifact that violates the current data-sharing boundary

## Expected Layout

The exported tree should mirror the Chapter 1 cluster artifact structure as closely as possible, for example:

- `chapter1_true_results/baselines/asic/primary_medians/` for approved saved prediction exports
- `chapter1_true_results/recalibration/asic/primary_medians/xgboost/` for approved recalibration exports
- `chapter1_true_results/evaluation/asic/hard_cases/primary_medians/logistic_regression/`
- `chapter1_true_results/evaluation/asic/horizon_dependence/foundation/`
- `chapter1_true_results/evaluation/asic/horizon_dependence/overlap/`
- `chapter1_true_results/evaluation/asic/horizon_dependence/final/`
- `chapter1_true_results/cohort/`
- `chapter1_true_results/splits/`
- `chapter1_true_results/model_ready/`
- `chapter1_true_results/carry_forward/`
- `chapter1_true_results/observation_process/`
- `chapter1_true_results/evaluation/asic/...`
- `chapter1_true_results/temporal_preview/asic/...`

For mixed analysis packages such as `evaluation/asic/hard_cases/.../asic_hard_case_comparison/`,
mirror only the approved aggregate outputs by default, such as summary tables, figures, plot-data
helpers, and manifests. Do not mirror row-level comparison datasets unless that specific export has
been explicitly approved.

For the exact default hard-case-comparison local-review bundle, see
[`docs/asic_hard_case_comparison_local_review_export_contract.md`](/Users/joanameyer/repository/1-mortality-decomposition/docs/asic_hard_case_comparison_local_review_export_contract.md).

For the decision record on the paired variable-audit package, see
[`docs/asic_hard_case_comparison_variable_audit_local_review_decision.md`](/Users/joanameyer/repository/1-mortality-decomposition/docs/asic_hard_case_comparison_variable_audit_local_review_decision.md).

For the exact default local-review contract for the paired variable-audit package, see
[`docs/asic_hard_case_comparison_variable_audit_local_review_export_contract.md`](/Users/joanameyer/repository/1-mortality-decomposition/docs/asic_hard_case_comparison_variable_audit_local_review_export_contract.md).

## Working Rule

If both of these exist locally:

- `artifacts/chapter1/`
- `cluster-results/chapter1_true_results/`

then scientific review should prefer `cluster-results/chapter1_true_results/`, while `artifacts/chapter1/` should be treated as synthetic or development-only output unless explicitly documented otherwise.

## Importing Staged Exports

If you have copied an approved cluster export staging tree into the local repo under
`export-staging/chapter1_true_results/`, import it into the authoritative local mirror with:

```bash
python run_chapter1_import_staged_exports.py
```

This imports only the files listed by each staged `export_manifest.json`, plus the manifest
itself. It does not blindly mirror every extra file that happens to be present in the staging
tree.

This same import step can mirror approved bundles for hard-case comparison, the paired variable
audit, baseline evaluation, XGBoost recalibration, hard-case agreement, the horizon-dependence
packages, temporal preview, and the foundational cohort/splits/model-ready/carry-forward/
observation-process summaries, plus SOFA feasibility and ICD-10 validation review bundles, as long
as they were staged with the corresponding HPC export command first.

Use `--overwrite` only when you intentionally want to replace an already mirrored local bundle.
