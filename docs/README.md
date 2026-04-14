# Docs Scope

Core standalone-repo documents:

- [`chapter1_analysis_spec_frozen_v1.md`](/Users/joanameyer/repository/1-mortality-decomposition/docs/chapter1_analysis_spec_frozen_v1.md): frozen Chapter 1 scientific specification
- [`cluster_artifact_refactor_plan.md`](/Users/joanameyer/repository/1-mortality-decomposition/docs/cluster_artifact_refactor_plan.md): concrete plan for separating cluster-only protected-data computation from local artifact-driven review
- [`asic_hard_case_comparison_local_review_export_contract.md`](/Users/joanameyer/repository/1-mortality-decomposition/docs/asic_hard_case_comparison_local_review_export_contract.md): exact default approved aggregate export set for local review of the ASIC hard-case comparison package
- [`asic_hard_case_comparison_variable_audit_local_review_export_contract.md`](/Users/joanameyer/repository/1-mortality-decomposition/docs/asic_hard_case_comparison_variable_audit_local_review_export_contract.md): exact default approved aggregate export set for local review of the paired hard-case-comparison variable-audit package
- [`asic_hard_case_comparison_variable_audit_local_review_decision.md`](/Users/joanameyer/repository/1-mortality-decomposition/docs/asic_hard_case_comparison_variable_audit_local_review_decision.md): decision record that the variable-audit package should be exported as a local-safe aggregate review artifact rather than kept cluster-only
- [`chapter1_cluster_local_rebuild_manual.md`](/Users/joanameyer/repository/1-mortality-decomposition/docs/chapter1_cluster_local_rebuild_manual.md): step-by-step clean rebuild guide for deleting outputs, rerunning the cluster pipeline, staging approved exports, importing them locally, and reconstructing the local review state
- [`preprocessing_interface.md`](/Users/joanameyer/repository/1-mortality-decomposition/docs/preprocessing_interface.md): current upstream input contract vs repo-local enforcement
- [`label_logic_audit.md`](/Users/joanameyer/repository/1-mortality-decomposition/docs/label_logic_audit.md): audit of the current proxy label implementation
- [`../config/ch1_feature_sets.json`](/Users/joanameyer/repository/1-mortality-decomposition/config/ch1_feature_sets.json): version-controlled Chapter 1 feature-set configuration
- [`../config/ch1_run_config.json`](/Users/joanameyer/repository/1-mortality-decomposition/config/ch1_run_config.json): shared local run configuration for the notebook and CLI, including the canonical split seed
- [`../notebooks/ch1_preprocessing_runbook.ipynb`](/Users/joanameyer/repository/1-mortality-decomposition/notebooks/ch1_preprocessing_runbook.ipynb): thin orchestration notebook for running the Chapter 1 preprocessing package on standardized artifacts, including carry-forward and split QC displays

Retained reference/context documents:

- [`asic_static_harmonization_plan.md`](/Users/joanameyer/repository/1-mortality-decomposition/docs/asic_static_harmonization_plan.md)
- [`context.md`](/Users/joanameyer/repository/1-mortality-decomposition/docs/context.md)
- [`phase1_linear_sprint_plan.md`](/Users/joanameyer/repository/1-mortality-decomposition/docs/phase1_linear_sprint_plan.md)
- [`phase1_working_reference.md`](/Users/joanameyer/repository/1-mortality-decomposition/docs/phase1_working_reference.md)
- [`phd_project_plan.md`](/Users/joanameyer/repository/1-mortality-decomposition/docs/phd_project_plan.md)

These retained reference documents are kept for context only. They are not the active preprocessing contract for this standalone repository.
