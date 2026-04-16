from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT_STAGE_ROOT = PROJECT_ROOT / "export-staging" / "chapter1_true_results"
BASELINE_PREDICTION_RELATIVE_ROOT = (
    Path("baselines") / "asic" / "primary_medians"
)
DEFAULT_BASELINE_PREDICTION_SOURCE_DIR = (
    PROJECT_ROOT / "artifacts" / "chapter1" / BASELINE_PREDICTION_RELATIVE_ROOT
)
BASELINE_EVALUATION_RELATIVE_ROOT = (
    Path("evaluation") / "asic" / "baselines" / "primary_medians"
)
DEFAULT_BASELINE_EVALUATION_SOURCE_DIR = (
    PROJECT_ROOT / "artifacts" / "chapter1" / BASELINE_EVALUATION_RELATIVE_ROOT
)
XGBOOST_RECALIBRATION_RELATIVE_ROOT = (
    Path("recalibration") / "asic" / "primary_medians" / "xgboost"
)
DEFAULT_XGBOOST_RECALIBRATION_SOURCE_DIR = (
    PROJECT_ROOT / "artifacts" / "chapter1" / XGBOOST_RECALIBRATION_RELATIVE_ROOT
)
HARD_CASE_DEFINITION_RELATIVE_ROOT = (
    Path("evaluation")
    / "asic"
    / "hard_cases"
    / "primary_medians"
    / "logistic_regression"
)
DEFAULT_HARD_CASE_DEFINITION_SOURCE_DIR = (
    PROJECT_ROOT / "artifacts" / "chapter1" / HARD_CASE_DEFINITION_RELATIVE_ROOT
)
ASIC_HARD_CASE_COMPARISON_RELATIVE_ROOT = (
    Path("evaluation")
    / "asic"
    / "hard_cases"
    / "primary_medians"
    / "logistic_regression"
    / "asic_hard_case_comparison"
)
DEFAULT_ASIC_HARD_CASE_COMPARISON_SOURCE_DIR = (
    PROJECT_ROOT / "artifacts" / "chapter1" / ASIC_HARD_CASE_COMPARISON_RELATIVE_ROOT
)
ASIC_HARD_CASE_COMPARISON_VARIABLE_AUDIT_RELATIVE_ROOT = (
    Path("evaluation")
    / "asic"
    / "hard_cases"
    / "primary_medians"
    / "logistic_regression"
    / "asic_hard_case_comparison_variable_audit"
)
DEFAULT_ASIC_HARD_CASE_COMPARISON_VARIABLE_AUDIT_SOURCE_DIR = (
    PROJECT_ROOT / "artifacts" / "chapter1" / ASIC_HARD_CASE_COMPARISON_VARIABLE_AUDIT_RELATIVE_ROOT
)
ASIC_SOFA_FEASIBILITY_RELATIVE_ROOT = (
    Path("evaluation")
    / "asic"
    / "hard_cases"
    / "primary_medians"
    / "logistic_regression"
    / "asic_sofa_feasibility_audit"
)
DEFAULT_ASIC_SOFA_FEASIBILITY_SOURCE_DIR = (
    PROJECT_ROOT / "artifacts" / "chapter1" / ASIC_SOFA_FEASIBILITY_RELATIVE_ROOT
)
HARD_CASE_AGREEMENT_RELATIVE_ROOT = (
    Path("evaluation")
    / "asic"
    / "hard_cases"
    / "primary_medians"
    / "agreement"
    / "logistic_regression_vs_xgboost_platt"
)
DEFAULT_HARD_CASE_AGREEMENT_SOURCE_DIR = (
    PROJECT_ROOT / "artifacts" / "chapter1" / HARD_CASE_AGREEMENT_RELATIVE_ROOT
)
HORIZON_DEPENDENCE_FOUNDATION_RELATIVE_ROOT = (
    Path("evaluation") / "asic" / "horizon_dependence" / "foundation"
)
DEFAULT_HORIZON_DEPENDENCE_FOUNDATION_SOURCE_DIR = (
    PROJECT_ROOT / "artifacts" / "chapter1" / HORIZON_DEPENDENCE_FOUNDATION_RELATIVE_ROOT
)
HORIZON_DEPENDENCE_OVERLAP_RELATIVE_ROOT = (
    Path("evaluation") / "asic" / "horizon_dependence" / "overlap"
)
DEFAULT_HORIZON_DEPENDENCE_OVERLAP_SOURCE_DIR = (
    PROJECT_ROOT / "artifacts" / "chapter1" / HORIZON_DEPENDENCE_OVERLAP_RELATIVE_ROOT
)
HORIZON_DEPENDENCE_FINAL_RELATIVE_ROOT = (
    Path("evaluation") / "asic" / "horizon_dependence" / "final"
)
DEFAULT_HORIZON_DEPENDENCE_FINAL_SOURCE_DIR = (
    PROJECT_ROOT / "artifacts" / "chapter1" / HORIZON_DEPENDENCE_FINAL_RELATIVE_ROOT
)
TEMPORAL_PREVIEW_RELATIVE_ROOT = (
    Path("temporal_preview") / "asic" / "aggregation_16h"
)
DEFAULT_TEMPORAL_PREVIEW_SOURCE_DIR = (
    PROJECT_ROOT / "artifacts" / "chapter1" / TEMPORAL_PREVIEW_RELATIVE_ROOT
)
TEMPORAL_SENSITIVITY_RELATIVE_ROOT = (
    Path("temporal_sensitivity") / "asic"
)
DEFAULT_TEMPORAL_SENSITIVITY_SOURCE_DIR = (
    PROJECT_ROOT / "artifacts" / "chapter1" / TEMPORAL_SENSITIVITY_RELATIVE_ROOT
)
ICD10_DISEASE_GROUP_VALIDATION_RELATIVE_ROOT = (
    Path("evaluation") / "asic" / "icd10_disease_group_validation"
)
DEFAULT_ICD10_DISEASE_GROUP_VALIDATION_SOURCE_DIR = (
    PROJECT_ROOT / "artifacts" / "chapter1" / ICD10_DISEASE_GROUP_VALIDATION_RELATIVE_ROOT
)
COHORT_RELATIVE_ROOT = Path("cohort")
DEFAULT_COHORT_SOURCE_DIR = PROJECT_ROOT / "artifacts" / "chapter1" / COHORT_RELATIVE_ROOT
SPLITS_RELATIVE_ROOT = Path("splits")
DEFAULT_SPLITS_SOURCE_DIR = PROJECT_ROOT / "artifacts" / "chapter1" / SPLITS_RELATIVE_ROOT
MODEL_READY_RELATIVE_ROOT = Path("model_ready")
DEFAULT_MODEL_READY_SOURCE_DIR = PROJECT_ROOT / "artifacts" / "chapter1" / MODEL_READY_RELATIVE_ROOT
CARRY_FORWARD_RELATIVE_ROOT = Path("carry_forward")
DEFAULT_CARRY_FORWARD_SOURCE_DIR = (
    PROJECT_ROOT / "artifacts" / "chapter1" / CARRY_FORWARD_RELATIVE_ROOT
)
OBSERVATION_PROCESS_RELATIVE_ROOT = Path("observation_process")
DEFAULT_OBSERVATION_PROCESS_SOURCE_DIR = (
    PROJECT_ROOT / "artifacts" / "chapter1" / OBSERVATION_PROCESS_RELATIVE_ROOT
)

DEFAULT_BASELINE_MODELS = ("logistic_regression", "xgboost")
DEFAULT_BASELINE_HORIZONS = (8, 16, 24, 48, 72)
DEFAULT_RECALIBRATION_HORIZONS = (8, 16, 24, 48, 72)

APPROVED_BASELINE_PREDICTION_EXPORTS = (
    *tuple(
        Path(model_name) / relative_name
        for model_name in DEFAULT_BASELINE_MODELS
        for relative_name in (
            "horizon_run_summary.csv",
            "run_manifest.json",
        )
    ),
    *tuple(
        Path(model_name) / f"horizon_{horizon_h}h" / relative_name
        for model_name in DEFAULT_BASELINE_MODELS
        for horizon_h in DEFAULT_BASELINE_HORIZONS
        for relative_name in (
            "predictions.csv",
            "all_valid_predictions.csv",
            "all_valid_prediction_qc.csv",
            "metrics.csv",
            "metadata.json",
            "selected_feature_columns.json",
        )
    ),
)
EXCLUDED_BASELINE_PREDICTION_EXPORTS = tuple(
    Path(model_name) / f"horizon_{horizon_h}h" / relative_name
    for model_name in DEFAULT_BASELINE_MODELS
    for horizon_h in DEFAULT_BASELINE_HORIZONS
    for relative_name in (
        "preprocessing.pkl",
        "pipeline.pkl",
        "preprocessing_unavailable.json",
        "model_unavailable.json",
        "pipeline_unavailable.json",
        "logistic_regression_model.pkl",
        "xgboost_model.pkl",
    )
)
APPROVED_BASELINE_EVALUATION_EXPORTS = (
    Path("combined_metrics.csv"),
    Path("reporting_split_summary.csv"),
    Path("combined_risk_binned_summary.csv"),
    Path("combined_primary_site_summary.csv"),
    Path("combined_primary_site_risk_binned_summary.csv"),
    Path("interpretation_note.md"),
    Path("run_manifest.json"),
    *tuple(
        Path(model_name) / f"horizon_{horizon_h}h" / relative_name
        for model_name in DEFAULT_BASELINE_MODELS
        for horizon_h in DEFAULT_BASELINE_HORIZONS
        for relative_name in (
            "metrics_by_split.csv",
            "risk_binned_summary.csv",
            "reliability_plot.png",
            "mortality_vs_risk_plot.png",
            "evaluation_metadata.json",
        )
    ),
    *tuple(
        Path(model_name) / relative_name
        for model_name in DEFAULT_BASELINE_MODELS
        for relative_name in (
            "horizon_comparison_metrics.csv",
            "horizon_comparison_plot.png",
            "horizon_risk_structure_grid.png",
            "model_level_evaluation_metadata.json",
            "primary_24h_site_summary.csv",
            "primary_24h_site_risk_binned_summary.csv",
            "primary_24h_site_overview.png",
            "primary_24h_site_risk_structure.png",
            "primary_24h_site_metadata.json",
        )
    ),
)
EXCLUDED_BASELINE_EVALUATION_EXPORTS: tuple[Path, ...] = ()
APPROVED_XGBOOST_RECALIBRATION_EXPORTS = (
    Path("combined_comparison_metrics.csv"),
    Path("combined_test_reliability_binned_summary.csv"),
    Path("test_horizon_calibration_summary.png"),
    Path("interpretation_note.md"),
    Path("run_manifest.json"),
    *tuple(
        Path(f"horizon_{horizon_h}h") / relative_name
        for horizon_h in DEFAULT_RECALIBRATION_HORIZONS
        for relative_name in (
            "comparison_metrics.csv",
            "test_reliability_binned_summary.csv",
            "test_reliability_comparison.png",
            "test_probability_distribution.png",
            "metadata.json",
            "xgboost_platt_canonical_predictions.csv",
            "xgboost_isotonic_canonical_predictions.csv",
        )
    ),
)
EXCLUDED_XGBOOST_RECALIBRATION_EXPORTS = tuple(
    Path(f"horizon_{horizon_h}h") / relative_name
    for horizon_h in DEFAULT_RECALIBRATION_HORIZONS
    for relative_name in (
        "xgboost_raw_canonical_predictions.csv",
        "xgboost_canonical_variant_predictions.csv",
        "platt_predictions.csv",
        "isotonic_predictions.csv",
        "logistic_metrics_by_split.csv",
        "xgboost_raw_metrics_by_split.csv",
    )
)
APPROVED_HARD_CASE_DEFINITION_EXPORTS = (
    Path("stay_level_hard_case_flags.csv"),
    Path("horizon_hard_case_summary.csv"),
    Path("run_manifest.json"),
)
EXCLUDED_HARD_CASE_DEFINITION_EXPORTS: tuple[Path, ...] = ()
APPROVED_ASIC_HARD_CASE_COMPARISON_EXPORTS = (
    Path("comparison_table.csv"),
    Path("effect_size_plot_data.csv"),
    Path("effect_size_figure.png"),
    Path("summary.md"),
    Path("early_vs_late_death_split") / "early_vs_late_fatal_timing_summary.csv",
    Path("early_vs_late_death_split") / "early_vs_late_low_pred_share.png",
    Path("early_vs_late_death_split") / "early_vs_late_interpretation_note.md",
)
EXCLUDED_ASIC_HARD_CASE_COMPARISON_EXPORTS = (
    Path("stay_level_comparison_dataset.csv"),
    Path("standardized_difference_details.csv"),
    Path("run_manifest.json"),
)
APPROVED_ASIC_HARD_CASE_COMPARISON_VARIABLE_AUDIT_EXPORTS = (
    Path("asic_hard_case_comparison_variable_audit_table.csv"),
    Path("asic_hard_case_comparison_variable_audit_memo.md"),
)
EXCLUDED_ASIC_HARD_CASE_COMPARISON_VARIABLE_AUDIT_EXPORTS: tuple[Path, ...] = ()
APPROVED_ASIC_SOFA_FEASIBILITY_EXPORTS = (
    Path("sofa_component_feasibility_table.csv"),
    Path("sofa_feasibility_memo.md"),
)
EXCLUDED_ASIC_SOFA_FEASIBILITY_EXPORTS: tuple[Path, ...] = ()
APPROVED_HARD_CASE_AGREEMENT_EXPORTS = (
    Path("horizon_hard_case_agreement_summary.csv"),
    Path("run_manifest.json"),
)
EXCLUDED_HARD_CASE_AGREEMENT_EXPORTS = (
    Path("fatal_stay_level_hard_case_agreement.csv"),
)
APPROVED_HORIZON_DEPENDENCE_FOUNDATION_EXPORTS = (
    Path("horizon_summary.csv"),
    Path("horizon_summary.md"),
    Path("artifact_foundation_note.md"),
)
EXCLUDED_HORIZON_DEPENDENCE_FOUNDATION_EXPORTS: tuple[Path, ...] = ()
APPROVED_HORIZON_DEPENDENCE_OVERLAP_EXPORTS = (
    Path("pairwise_denominators.csv"),
    Path("pairwise_overlap.csv"),
    Path("directional_overlap.csv"),
    Path("hard_case_persistence.csv"),
    Path("persistence_distribution.csv"),
    Path("jaccard_heatmap.png"),
    Path("directional_overlap_heatmap.png"),
    Path("persistence_barplot.png"),
    Path("overlap_note.md"),
    Path("run_manifest.json"),
)
EXCLUDED_HORIZON_DEPENDENCE_OVERLAP_EXPORTS: tuple[Path, ...] = ()
APPROVED_HORIZON_DEPENDENCE_FINAL_EXPORTS = (
    Path("mortality_risk_horizon_binned_summary.csv"),
    Path("mortality_risk_horizon_comparison.png"),
    Path("horizon_interpretation_memo.md"),
    Path("final_horizon_summary.md"),
    Path("run_manifest.json"),
)
EXCLUDED_HORIZON_DEPENDENCE_FINAL_EXPORTS: tuple[Path, ...] = ()
APPROVED_TEMPORAL_PREVIEW_EXPORTS = (
    Path("comparison") / "aggregation_comparison_metrics.csv",
    Path("comparison") / "preview_note.md",
    Path("comparison") / "preview_review.ipynb",
    Path("comparison") / "logistic_regression_24h_reliability_8h_vs_16h.png",
    Path("comparison") / "logistic_regression_24h_mortality_vs_risk_8h_vs_16h.png",
    Path("comparison") / "xgboost_24h_reliability_8h_vs_16h.png",
    Path("comparison") / "xgboost_24h_mortality_vs_risk_8h_vs_16h.png",
)
EXCLUDED_TEMPORAL_PREVIEW_EXPORTS: tuple[Path, ...] = ()
APPROVED_TEMPORAL_SENSITIVITY_EXPORTS = (
    Path("comparison") / "preprocessing_count_comparison.csv",
    Path("comparison") / "reporting_metric_summary.csv",
    Path("comparison") / "selected_split_summary.csv",
    Path("comparison") / "calibration_summary.csv",
    Path("comparison") / "mortality_risk_structure_summary.csv",
    Path("comparison") / "hard_case_prevalence_summary.csv",
    Path("comparison") / "logistic_24h_hard_case_pairwise_denominators.csv",
    Path("comparison") / "logistic_24h_hard_case_pairwise_overlap.csv",
    Path("comparison") / "logistic_24h_hard_case_directional_overlap.csv",
    Path("comparison") / "logistic_24h_hard_case_persistence.csv",
    Path("comparison") / "logistic_24h_hard_case_persistence_distribution.csv",
    Path("comparison") / "logistic_regression_24h_reliability_8h_vs_16h_vs_24h.png",
    Path("comparison") / "logistic_regression_24h_mortality_vs_risk_8h_vs_16h_vs_24h.png",
    Path("comparison") / "logistic_24h_hard_case_jaccard_heatmap.png",
    Path("comparison") / "logistic_24h_hard_case_directional_overlap_heatmap.png",
    Path("comparison") / "logistic_24h_hard_case_persistence_barplot.png",
    Path("comparison") / "logistic_24h_hard_case_overlap_note.md",
    Path("comparison") / "split_alignment_overview.csv",
    Path("comparison") / "provenance_and_limitations.md",
    Path("comparison") / "supersession_note.md",
    Path("comparison") / "interpretation_memo_template.md",
    Path("comparison") / "run_manifest.json",
    Path("aggregation_16h") / "preprocessing" / "generation_note.md",
    Path("aggregation_16h") / "preprocessing" / "generation_manifest.json",
    Path("aggregation_16h") / "preprocessing" / "splits" / "chapter1_temporal_sensitivity_split_alignment_summary.csv",
    Path("aggregation_16h")
    / "evaluation"
    / "asic"
    / "hard_cases"
    / "primary_medians"
    / "logistic_regression"
    / "horizon_hard_case_summary.csv",
    Path("aggregation_24h") / "preprocessing" / "generation_note.md",
    Path("aggregation_24h") / "preprocessing" / "generation_manifest.json",
    Path("aggregation_24h") / "preprocessing" / "splits" / "chapter1_temporal_sensitivity_split_alignment_summary.csv",
    Path("aggregation_24h")
    / "evaluation"
    / "asic"
    / "hard_cases"
    / "primary_medians"
    / "logistic_regression"
    / "horizon_hard_case_summary.csv",
)
EXCLUDED_TEMPORAL_SENSITIVITY_EXPORTS: tuple[Path, ...] = ()
APPROVED_COHORT_EXPORTS = (
    Path("chapter1_notes.csv"),
    Path("chapter1_site_eligibility.csv"),
    Path("chapter1_site_counts_summary.csv"),
    Path("chapter1_stay_exclusion_summary_by_hospital.csv"),
    Path("chapter1_counts_by_hospital.csv"),
    Path("chapter1_retained_hospitals.csv"),
    Path("chapter1_cohort_summary.csv"),
    Path("chapter1_verification_summary.csv"),
)
EXCLUDED_COHORT_EXPORTS = (
    Path("chapter1_stay_exclusions.csv"),
    Path("chapter1_retained_stays.csv"),
    Path("chapter1_retained_stay_table.csv"),
)
APPROVED_SPLITS_EXPORTS = (
    Path("chapter1_stay_split_summary.csv"),
    Path("chapter1_stay_split_verification_summary.csv"),
    Path("chapter1_primary_split_summary.csv"),
    Path("chapter1_primary_split_verification_summary.csv"),
)
EXCLUDED_SPLITS_EXPORTS = (
    Path("chapter1_stay_split_assignments.csv"),
)
APPROVED_MODEL_READY_EXPORTS = (
    Path("chapter1_primary_readiness_summary.csv"),
    Path("chapter1_primary_feature_availability_by_horizon.csv"),
)
EXCLUDED_MODEL_READY_EXPORTS = (
    Path("chapter1_primary_model_ready_dataset.csv"),
    Path("chapter1_primary_model_ready_with_observation_process.csv"),
)
APPROVED_CARRY_FORWARD_EXPORTS = (
    Path("chapter1_primary_locf_feature_summary.csv"),
    Path("chapter1_primary_ventilator_locf_summary.csv"),
    Path("chapter1_primary_missingness_by_hospital_and_family.csv"),
    Path("chapter1_primary_carry_forward_verification_summary.csv"),
)
EXCLUDED_CARRY_FORWARD_EXPORTS: tuple[Path, ...] = ()
APPROVED_OBSERVATION_PROCESS_EXPORTS = (
    Path("chapter1_observation_process_qc_summary.csv"),
    Path("chapter1_observation_process_verification_summary.csv"),
    Path("chapter1_observation_process_implementation_note.md"),
)
EXCLUDED_OBSERVATION_PROCESS_EXPORTS = (
    Path("chapter1_observation_process_block_features.csv"),
    Path("chapter1_observation_process_spot_check_examples.csv"),
)
APPROVED_ICD10_DISEASE_GROUP_VALIDATION_EXPORTS = (
    Path("final_group_counts.csv"),
    Path("pre_hierarchy_ambiguity_summary.csv"),
    Path("common_multi_match_combinations.csv"),
    Path("sample_rows_by_final_group.csv"),
    Path("other_mixed_uncategorized_examples.csv"),
    Path("non_driver_stem_summary.csv"),
    Path("validation_memo.md"),
)
EXCLUDED_ICD10_DISEASE_GROUP_VALIDATION_EXPORTS = (
    Path("asic_static_icd10_disease_groups.csv"),
)


@dataclass(frozen=True)
class ExportStageResult:
    export_name: str
    source_dir: Path
    stage_root: Path
    staged_dir: Path
    copied_paths: tuple[Path, ...]
    excluded_paths: tuple[Path, ...]
    manifest_path: Path


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve_existing_dir(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Required directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Expected a directory, found: {resolved}")
    return resolved


def _stage_export_bundle(
    *,
    export_name: str,
    relative_root: Path,
    source_dir: Path,
    stage_root: Path,
    approved_exports: tuple[Path, ...],
    excluded_exports: tuple[Path, ...],
    overwrite: bool = False,
    notes: Sequence[str] = (),
) -> ExportStageResult:
    resolved_source_dir = _resolve_existing_dir(source_dir)
    resolved_stage_root = Path(stage_root).expanduser().resolve()
    staged_dir = resolved_stage_root / relative_root

    missing_paths = [
        relative_path
        for relative_path in approved_exports
        if not (resolved_source_dir / relative_path).exists()
    ]
    if missing_paths:
        missing_display = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            f"The {export_name} export package is incomplete. Missing approved artifacts: "
            f"{missing_display}"
        )

    copied_paths: list[Path] = []
    for relative_path in approved_exports:
        source_path = resolved_source_dir / relative_path
        destination_path = staged_dir / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite staged export without --overwrite: {destination_path}"
            )
        shutil.copy2(source_path, destination_path)
        copied_paths.append(destination_path)

    excluded_paths = tuple(
        resolved_source_dir / relative_path
        for relative_path in excluded_exports
        if (resolved_source_dir / relative_path).exists()
    )

    manifest_payload = {
        "timestamp_utc": _utc_timestamp(),
        "export_name": export_name,
        "source_dir": str(resolved_source_dir),
        "stage_root": str(resolved_stage_root),
        "staged_dir": str(staged_dir),
        "copied_relative_paths": [str(path.relative_to(staged_dir)) for path in copied_paths],
        "excluded_relative_paths": [
            str(path.relative_to(resolved_source_dir)) for path in excluded_paths
        ],
        "notes": list(notes),
    }
    manifest_path = staged_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n")

    return ExportStageResult(
        export_name=export_name,
        source_dir=resolved_source_dir,
        stage_root=resolved_stage_root,
        staged_dir=staged_dir,
        copied_paths=tuple(copied_paths),
        excluded_paths=excluded_paths,
        manifest_path=manifest_path,
    )


def stage_asic_hard_case_comparison_exports(
    *,
    source_dir: Path = DEFAULT_ASIC_HARD_CASE_COMPARISON_SOURCE_DIR,
    stage_root: Path = DEFAULT_EXPORT_STAGE_ROOT,
    overwrite: bool = False,
) -> ExportStageResult:
    return _stage_export_bundle(
        export_name="asic_hard_case_comparison_local_review_aggregate_bundle",
        relative_root=ASIC_HARD_CASE_COMPARISON_RELATIVE_ROOT,
        source_dir=source_dir,
        stage_root=stage_root,
        approved_exports=APPROVED_ASIC_HARD_CASE_COMPARISON_EXPORTS,
        excluded_exports=EXCLUDED_ASIC_HARD_CASE_COMPARISON_EXPORTS,
        overwrite=overwrite,
        notes=(
            "This staging step mirrors only the default approved aggregate outputs for local review.",
            "The row-level comparison dataset remains cluster-side unless that export is explicitly approved.",
            "The producer run manifest and detailed standardized-difference table are excluded from the default local-review bundle.",
        ),
    )


def stage_baseline_evaluation_exports(
    *,
    source_dir: Path = DEFAULT_BASELINE_EVALUATION_SOURCE_DIR,
    stage_root: Path = DEFAULT_EXPORT_STAGE_ROOT,
    overwrite: bool = False,
) -> ExportStageResult:
    return _stage_export_bundle(
        export_name="baseline_evaluation_local_review_bundle",
        relative_root=BASELINE_EVALUATION_RELATIVE_ROOT,
        source_dir=source_dir,
        stage_root=stage_root,
        approved_exports=APPROVED_BASELINE_EVALUATION_EXPORTS,
        excluded_exports=EXCLUDED_BASELINE_EVALUATION_EXPORTS,
        overwrite=overwrite,
        notes=(
            "This staging step mirrors the approved baseline evaluation outputs used by local reports and presentations.",
            "The bundle includes top-level summaries plus the per-model and per-horizon figures, metrics, and metadata already consumed under cluster-results.",
        ),
    )


def stage_xgboost_recalibration_exports(
    *,
    source_dir: Path = DEFAULT_XGBOOST_RECALIBRATION_SOURCE_DIR,
    stage_root: Path = DEFAULT_EXPORT_STAGE_ROOT,
    overwrite: bool = False,
) -> ExportStageResult:
    return _stage_export_bundle(
        export_name="xgboost_recalibration_local_review_bundle",
        relative_root=XGBOOST_RECALIBRATION_RELATIVE_ROOT,
        source_dir=source_dir,
        stage_root=stage_root,
        approved_exports=APPROVED_XGBOOST_RECALIBRATION_EXPORTS,
        excluded_exports=EXCLUDED_XGBOOST_RECALIBRATION_EXPORTS,
        overwrite=overwrite,
        notes=(
            "This staging step mirrors the approved recalibration review outputs plus the canonical recalibrated prediction exports needed by local hard-case agreement analysis.",
            "Raw prediction variants and auxiliary combined prediction tables are excluded from the default local-review bundle.",
        ),
    )


def stage_baseline_prediction_exports(
    *,
    source_dir: Path = DEFAULT_BASELINE_PREDICTION_SOURCE_DIR,
    stage_root: Path = DEFAULT_EXPORT_STAGE_ROOT,
    overwrite: bool = False,
) -> ExportStageResult:
    return _stage_export_bundle(
        export_name="baseline_prediction_local_review_bundle",
        relative_root=BASELINE_PREDICTION_RELATIVE_ROOT,
        source_dir=source_dir,
        stage_root=stage_root,
        approved_exports=APPROVED_BASELINE_PREDICTION_EXPORTS,
        excluded_exports=EXCLUDED_BASELINE_PREDICTION_EXPORTS,
        overwrite=overwrite,
        notes=(
            "This staging step mirrors the approved baseline prediction exports used by local trajectory review and downstream artifact consumers.",
            "The bundle includes evaluation-only predictions, all-valid predictions, prediction QC tables, and lightweight per-horizon metadata.",
            "Model pickles and fitted pipeline artifacts are excluded from the default local-review bundle.",
        ),
    )


def stage_hard_case_definition_exports(
    *,
    source_dir: Path = DEFAULT_HARD_CASE_DEFINITION_SOURCE_DIR,
    stage_root: Path = DEFAULT_EXPORT_STAGE_ROOT,
    overwrite: bool = False,
) -> ExportStageResult:
    return _stage_export_bundle(
        export_name="hard_case_definition_local_review_bundle",
        relative_root=HARD_CASE_DEFINITION_RELATIVE_ROOT,
        source_dir=source_dir,
        stage_root=stage_root,
        approved_exports=APPROVED_HARD_CASE_DEFINITION_EXPORTS,
        excluded_exports=EXCLUDED_HARD_CASE_DEFINITION_EXPORTS,
        overwrite=overwrite,
        notes=(
            "This staging step mirrors the approved hard-case review package used by the local hard-case notebook and downstream local-safe summaries.",
            "The bundle includes the stay-level hard-case flags, horizon summary, and producer run manifest.",
        ),
    )


def stage_asic_hard_case_comparison_variable_audit_exports(
    *,
    source_dir: Path = DEFAULT_ASIC_HARD_CASE_COMPARISON_VARIABLE_AUDIT_SOURCE_DIR,
    stage_root: Path = DEFAULT_EXPORT_STAGE_ROOT,
    overwrite: bool = False,
) -> ExportStageResult:
    return _stage_export_bundle(
        export_name="asic_hard_case_comparison_variable_audit_local_review_bundle",
        relative_root=ASIC_HARD_CASE_COMPARISON_VARIABLE_AUDIT_RELATIVE_ROOT,
        source_dir=source_dir,
        stage_root=stage_root,
        approved_exports=APPROVED_ASIC_HARD_CASE_COMPARISON_VARIABLE_AUDIT_EXPORTS,
        excluded_exports=EXCLUDED_ASIC_HARD_CASE_COMPARISON_VARIABLE_AUDIT_EXPORTS,
        overwrite=overwrite,
        notes=(
            "This staging step mirrors only the default approved aggregate outputs for local review.",
            "The variable-audit package remains a cluster-side producer but its memo and table are approved for local review.",
            "No row-level or additional diagnostic audit artifacts are included in the default local-review bundle.",
        ),
    )


def stage_asic_sofa_feasibility_exports(
    *,
    source_dir: Path = DEFAULT_ASIC_SOFA_FEASIBILITY_SOURCE_DIR,
    stage_root: Path = DEFAULT_EXPORT_STAGE_ROOT,
    overwrite: bool = False,
) -> ExportStageResult:
    return _stage_export_bundle(
        export_name="asic_sofa_feasibility_local_review_bundle",
        relative_root=ASIC_SOFA_FEASIBILITY_RELATIVE_ROOT,
        source_dir=source_dir,
        stage_root=stage_root,
        approved_exports=APPROVED_ASIC_SOFA_FEASIBILITY_EXPORTS,
        excluded_exports=EXCLUDED_ASIC_SOFA_FEASIBILITY_EXPORTS,
        overwrite=overwrite,
        notes=(
            "This staging step mirrors the approved SOFA feasibility review outputs for local review.",
        ),
    )


def stage_hard_case_agreement_exports(
    *,
    source_dir: Path = DEFAULT_HARD_CASE_AGREEMENT_SOURCE_DIR,
    stage_root: Path = DEFAULT_EXPORT_STAGE_ROOT,
    overwrite: bool = False,
) -> ExportStageResult:
    return _stage_export_bundle(
        export_name="hard_case_agreement_local_review_bundle",
        relative_root=HARD_CASE_AGREEMENT_RELATIVE_ROOT,
        source_dir=source_dir,
        stage_root=stage_root,
        approved_exports=APPROVED_HARD_CASE_AGREEMENT_EXPORTS,
        excluded_exports=EXCLUDED_HARD_CASE_AGREEMENT_EXPORTS,
        overwrite=overwrite,
        notes=(
            "This staging step mirrors the approved horizon-level hard-case agreement summary for local review.",
            "The fatal stay-level agreement table is excluded from the default local-review bundle.",
        ),
    )


def stage_horizon_dependence_foundation_exports(
    *,
    source_dir: Path = DEFAULT_HORIZON_DEPENDENCE_FOUNDATION_SOURCE_DIR,
    stage_root: Path = DEFAULT_EXPORT_STAGE_ROOT,
    overwrite: bool = False,
) -> ExportStageResult:
    return _stage_export_bundle(
        export_name="asic_horizon_dependence_foundation_local_review_bundle",
        relative_root=HORIZON_DEPENDENCE_FOUNDATION_RELATIVE_ROOT,
        source_dir=source_dir,
        stage_root=stage_root,
        approved_exports=APPROVED_HORIZON_DEPENDENCE_FOUNDATION_EXPORTS,
        excluded_exports=EXCLUDED_HORIZON_DEPENDENCE_FOUNDATION_EXPORTS,
        overwrite=overwrite,
        notes=(
            "This staging step mirrors the approved Package 1 horizon foundation outputs for local review.",
        ),
    )


def stage_horizon_dependence_overlap_exports(
    *,
    source_dir: Path = DEFAULT_HORIZON_DEPENDENCE_OVERLAP_SOURCE_DIR,
    stage_root: Path = DEFAULT_EXPORT_STAGE_ROOT,
    overwrite: bool = False,
) -> ExportStageResult:
    return _stage_export_bundle(
        export_name="asic_horizon_dependence_overlap_local_review_bundle",
        relative_root=HORIZON_DEPENDENCE_OVERLAP_RELATIVE_ROOT,
        source_dir=source_dir,
        stage_root=stage_root,
        approved_exports=APPROVED_HORIZON_DEPENDENCE_OVERLAP_EXPORTS,
        excluded_exports=EXCLUDED_HORIZON_DEPENDENCE_OVERLAP_EXPORTS,
        overwrite=overwrite,
        notes=(
            "This staging step mirrors the approved Package 2 horizon overlap outputs for local review.",
        ),
    )


def stage_horizon_dependence_final_exports(
    *,
    source_dir: Path = DEFAULT_HORIZON_DEPENDENCE_FINAL_SOURCE_DIR,
    stage_root: Path = DEFAULT_EXPORT_STAGE_ROOT,
    overwrite: bool = False,
) -> ExportStageResult:
    return _stage_export_bundle(
        export_name="asic_horizon_dependence_final_local_review_bundle",
        relative_root=HORIZON_DEPENDENCE_FINAL_RELATIVE_ROOT,
        source_dir=source_dir,
        stage_root=stage_root,
        approved_exports=APPROVED_HORIZON_DEPENDENCE_FINAL_EXPORTS,
        excluded_exports=EXCLUDED_HORIZON_DEPENDENCE_FINAL_EXPORTS,
        overwrite=overwrite,
        notes=(
            "This staging step mirrors the approved Package 3 horizon final outputs for local review.",
        ),
    )


def stage_horizon_dependence_exports(
    *,
    foundation_source_dir: Path = DEFAULT_HORIZON_DEPENDENCE_FOUNDATION_SOURCE_DIR,
    overlap_source_dir: Path = DEFAULT_HORIZON_DEPENDENCE_OVERLAP_SOURCE_DIR,
    final_source_dir: Path = DEFAULT_HORIZON_DEPENDENCE_FINAL_SOURCE_DIR,
    stage_root: Path = DEFAULT_EXPORT_STAGE_ROOT,
    overwrite: bool = False,
) -> tuple[ExportStageResult, ExportStageResult, ExportStageResult]:
    return (
        stage_horizon_dependence_foundation_exports(
            source_dir=foundation_source_dir,
            stage_root=stage_root,
            overwrite=overwrite,
        ),
        stage_horizon_dependence_overlap_exports(
            source_dir=overlap_source_dir,
            stage_root=stage_root,
            overwrite=overwrite,
        ),
        stage_horizon_dependence_final_exports(
            source_dir=final_source_dir,
            stage_root=stage_root,
            overwrite=overwrite,
        ),
    )


def stage_temporal_preview_exports(
    *,
    source_dir: Path = DEFAULT_TEMPORAL_PREVIEW_SOURCE_DIR,
    stage_root: Path = DEFAULT_EXPORT_STAGE_ROOT,
    overwrite: bool = False,
) -> ExportStageResult:
    return _stage_export_bundle(
        export_name="temporal_preview_local_review_bundle",
        relative_root=TEMPORAL_PREVIEW_RELATIVE_ROOT,
        source_dir=source_dir,
        stage_root=stage_root,
        approved_exports=APPROVED_TEMPORAL_PREVIEW_EXPORTS,
        excluded_exports=EXCLUDED_TEMPORAL_PREVIEW_EXPORTS,
        overwrite=overwrite,
        notes=(
            "This staging step mirrors the approved 8h-vs-16h temporal preview comparison package for local review.",
        ),
    )


def stage_temporal_sensitivity_exports(
    *,
    source_dir: Path = DEFAULT_TEMPORAL_SENSITIVITY_SOURCE_DIR,
    stage_root: Path = DEFAULT_EXPORT_STAGE_ROOT,
    overwrite: bool = False,
) -> ExportStageResult:
    return _stage_export_bundle(
        export_name="temporal_sensitivity_local_review_bundle",
        relative_root=TEMPORAL_SENSITIVITY_RELATIVE_ROOT,
        source_dir=source_dir,
        stage_root=stage_root,
        approved_exports=APPROVED_TEMPORAL_SENSITIVITY_EXPORTS,
        excluded_exports=EXCLUDED_TEMPORAL_SENSITIVITY_EXPORTS,
        overwrite=overwrite,
        notes=(
            "This staging step mirrors the approved formal Chapter 1 temporal coarsening sensitivity package for local review.",
        ),
    )


def stage_cohort_exports(
    *,
    source_dir: Path = DEFAULT_COHORT_SOURCE_DIR,
    stage_root: Path = DEFAULT_EXPORT_STAGE_ROOT,
    overwrite: bool = False,
) -> ExportStageResult:
    return _stage_export_bundle(
        export_name="cohort_local_review_bundle",
        relative_root=COHORT_RELATIVE_ROOT,
        source_dir=source_dir,
        stage_root=stage_root,
        approved_exports=APPROVED_COHORT_EXPORTS,
        excluded_exports=EXCLUDED_COHORT_EXPORTS,
        overwrite=overwrite,
        notes=(
            "This staging step mirrors the approved cohort summary outputs for local review.",
            "Row-level retained-stay and stay-exclusion tables are excluded from the default local-review bundle.",
        ),
    )


def stage_split_exports(
    *,
    source_dir: Path = DEFAULT_SPLITS_SOURCE_DIR,
    stage_root: Path = DEFAULT_EXPORT_STAGE_ROOT,
    overwrite: bool = False,
) -> ExportStageResult:
    return _stage_export_bundle(
        export_name="split_summary_local_review_bundle",
        relative_root=SPLITS_RELATIVE_ROOT,
        source_dir=source_dir,
        stage_root=stage_root,
        approved_exports=APPROVED_SPLITS_EXPORTS,
        excluded_exports=EXCLUDED_SPLITS_EXPORTS,
        overwrite=overwrite,
        notes=(
            "This staging step mirrors the approved split summary outputs for local review.",
            "Stay-level split assignment tables are excluded from the default local-review bundle.",
        ),
    )


def stage_model_ready_summary_exports(
    *,
    source_dir: Path = DEFAULT_MODEL_READY_SOURCE_DIR,
    stage_root: Path = DEFAULT_EXPORT_STAGE_ROOT,
    overwrite: bool = False,
) -> ExportStageResult:
    return _stage_export_bundle(
        export_name="model_ready_summary_local_review_bundle",
        relative_root=MODEL_READY_RELATIVE_ROOT,
        source_dir=source_dir,
        stage_root=stage_root,
        approved_exports=APPROVED_MODEL_READY_EXPORTS,
        excluded_exports=EXCLUDED_MODEL_READY_EXPORTS,
        overwrite=overwrite,
        notes=(
            "This staging step mirrors the approved model-ready summary outputs for local review.",
            "Model-ready row-level datasets remain excluded from the default local-review bundle.",
        ),
    )


def stage_carry_forward_exports(
    *,
    source_dir: Path = DEFAULT_CARRY_FORWARD_SOURCE_DIR,
    stage_root: Path = DEFAULT_EXPORT_STAGE_ROOT,
    overwrite: bool = False,
) -> ExportStageResult:
    return _stage_export_bundle(
        export_name="carry_forward_summary_local_review_bundle",
        relative_root=CARRY_FORWARD_RELATIVE_ROOT,
        source_dir=source_dir,
        stage_root=stage_root,
        approved_exports=APPROVED_CARRY_FORWARD_EXPORTS,
        excluded_exports=EXCLUDED_CARRY_FORWARD_EXPORTS,
        overwrite=overwrite,
        notes=(
            "This staging step mirrors the approved carry-forward summary outputs for local review.",
        ),
    )


def stage_observation_process_exports(
    *,
    source_dir: Path = DEFAULT_OBSERVATION_PROCESS_SOURCE_DIR,
    stage_root: Path = DEFAULT_EXPORT_STAGE_ROOT,
    overwrite: bool = False,
) -> ExportStageResult:
    return _stage_export_bundle(
        export_name="observation_process_summary_local_review_bundle",
        relative_root=OBSERVATION_PROCESS_RELATIVE_ROOT,
        source_dir=source_dir,
        stage_root=stage_root,
        approved_exports=APPROVED_OBSERVATION_PROCESS_EXPORTS,
        excluded_exports=EXCLUDED_OBSERVATION_PROCESS_EXPORTS,
        overwrite=overwrite,
        notes=(
            "This staging step mirrors the approved observation-process summary outputs for local review.",
            "Block-level feature tables and spot-check examples are excluded from the default local-review bundle.",
        ),
    )


def stage_icd10_disease_group_validation_exports(
    *,
    source_dir: Path = DEFAULT_ICD10_DISEASE_GROUP_VALIDATION_SOURCE_DIR,
    stage_root: Path = DEFAULT_EXPORT_STAGE_ROOT,
    overwrite: bool = False,
) -> ExportStageResult:
    return _stage_export_bundle(
        export_name="icd10_disease_group_validation_local_review_bundle",
        relative_root=ICD10_DISEASE_GROUP_VALIDATION_RELATIVE_ROOT,
        source_dir=source_dir,
        stage_root=stage_root,
        approved_exports=APPROVED_ICD10_DISEASE_GROUP_VALIDATION_EXPORTS,
        excluded_exports=EXCLUDED_ICD10_DISEASE_GROUP_VALIDATION_EXPORTS,
        overwrite=overwrite,
        notes=(
            "This staging step mirrors the approved ICD-10 disease-group validation review outputs for local review.",
            "The stay-level ICD-10 classification table is excluded from the default local-review bundle.",
        ),
    )


def stage_foundational_summary_exports(
    *,
    cohort_source_dir: Path = DEFAULT_COHORT_SOURCE_DIR,
    splits_source_dir: Path = DEFAULT_SPLITS_SOURCE_DIR,
    model_ready_source_dir: Path = DEFAULT_MODEL_READY_SOURCE_DIR,
    carry_forward_source_dir: Path = DEFAULT_CARRY_FORWARD_SOURCE_DIR,
    observation_process_source_dir: Path = DEFAULT_OBSERVATION_PROCESS_SOURCE_DIR,
    stage_root: Path = DEFAULT_EXPORT_STAGE_ROOT,
    overwrite: bool = False,
) -> tuple[ExportStageResult, ExportStageResult, ExportStageResult, ExportStageResult, ExportStageResult]:
    return (
        stage_cohort_exports(
            source_dir=cohort_source_dir,
            stage_root=stage_root,
            overwrite=overwrite,
        ),
        stage_split_exports(
            source_dir=splits_source_dir,
            stage_root=stage_root,
            overwrite=overwrite,
        ),
        stage_model_ready_summary_exports(
            source_dir=model_ready_source_dir,
            stage_root=stage_root,
            overwrite=overwrite,
        ),
        stage_carry_forward_exports(
            source_dir=carry_forward_source_dir,
            stage_root=stage_root,
            overwrite=overwrite,
        ),
        stage_observation_process_exports(
            source_dir=observation_process_source_dir,
            stage_root=stage_root,
            overwrite=overwrite,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage approved Chapter 1 local-review export bundles into an export tree that mirrors "
            "cluster-results/chapter1_true_results."
        )
    )
    parser.add_argument(
        "--include-baseline-predictions",
        action="store_true",
        help="Also stage the baseline prediction local-review bundle.",
    )
    parser.add_argument(
        "--include-variable-audit",
        action="store_true",
        help=(
            "Also stage the paired ASIC hard-case comparison variable-audit bundle "
            "into the mirrored local-review tree."
        ),
    )
    parser.add_argument(
        "--include-sofa-feasibility",
        action="store_true",
        help="Also stage the ASIC SOFA feasibility local-review bundle.",
    )
    parser.add_argument(
        "--include-baseline-evaluation",
        action="store_true",
        help="Also stage the baseline evaluation local-review bundle.",
    )
    parser.add_argument(
        "--include-hard-case-definition",
        action="store_true",
        help="Also stage the hard-case definition local-review bundle.",
    )
    parser.add_argument(
        "--include-xgboost-recalibration",
        action="store_true",
        help="Also stage the XGBoost recalibration local-review bundle.",
    )
    parser.add_argument(
        "--include-hard-case-agreement",
        action="store_true",
        help="Also stage the hard-case agreement local-review bundle.",
    )
    parser.add_argument(
        "--include-horizon-dependence",
        action="store_true",
        help="Also stage the horizon-dependence foundation, overlap, and final local-review bundles.",
    )
    parser.add_argument(
        "--include-temporal-preview",
        action="store_true",
        help="Also stage the temporal preview local-review bundle.",
    )
    parser.add_argument(
        "--include-temporal-sensitivity",
        action="store_true",
        help="Also stage the formal temporal sensitivity local-review bundle.",
    )
    parser.add_argument(
        "--include-foundational-summaries",
        action="store_true",
        help=(
            "Also stage the approved cohort, splits, model-ready summary, carry-forward, "
            "and observation-process local-review bundles."
        ),
    )
    parser.add_argument(
        "--include-icd10-validation",
        action="store_true",
        help="Also stage the ICD-10 disease-group validation local-review bundle.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_ASIC_HARD_CASE_COMPARISON_SOURCE_DIR,
        help="Cluster-side hard-case comparison artifact directory to stage from.",
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=DEFAULT_EXPORT_STAGE_ROOT,
        help=(
            "Export staging root. The command appends the Chapter 1 relative package path "
            "under this directory."
        ),
    )
    parser.add_argument(
        "--variable-audit-source-dir",
        type=Path,
        default=DEFAULT_ASIC_HARD_CASE_COMPARISON_VARIABLE_AUDIT_SOURCE_DIR,
        help="Cluster-side hard-case comparison variable-audit directory to stage from.",
    )
    parser.add_argument(
        "--sofa-feasibility-source-dir",
        type=Path,
        default=DEFAULT_ASIC_SOFA_FEASIBILITY_SOURCE_DIR,
        help="Cluster-side SOFA feasibility artifact directory to stage from.",
    )
    parser.add_argument(
        "--baseline-evaluation-source-dir",
        type=Path,
        default=DEFAULT_BASELINE_EVALUATION_SOURCE_DIR,
        help="Cluster-side baseline evaluation artifact directory to stage from.",
    )
    parser.add_argument(
        "--baseline-prediction-source-dir",
        type=Path,
        default=DEFAULT_BASELINE_PREDICTION_SOURCE_DIR,
        help="Cluster-side baseline prediction artifact directory to stage from.",
    )
    parser.add_argument(
        "--xgboost-recalibration-source-dir",
        type=Path,
        default=DEFAULT_XGBOOST_RECALIBRATION_SOURCE_DIR,
        help="Cluster-side XGBoost recalibration artifact directory to stage from.",
    )
    parser.add_argument(
        "--hard-case-definition-source-dir",
        type=Path,
        default=DEFAULT_HARD_CASE_DEFINITION_SOURCE_DIR,
        help="Cluster-side hard-case definition artifact directory to stage from.",
    )
    parser.add_argument(
        "--hard-case-agreement-source-dir",
        type=Path,
        default=DEFAULT_HARD_CASE_AGREEMENT_SOURCE_DIR,
        help="Cluster-side hard-case agreement artifact directory to stage from.",
    )
    parser.add_argument(
        "--horizon-foundation-source-dir",
        type=Path,
        default=DEFAULT_HORIZON_DEPENDENCE_FOUNDATION_SOURCE_DIR,
        help="Cluster-side horizon foundation artifact directory to stage from.",
    )
    parser.add_argument(
        "--horizon-overlap-source-dir",
        type=Path,
        default=DEFAULT_HORIZON_DEPENDENCE_OVERLAP_SOURCE_DIR,
        help="Cluster-side horizon overlap artifact directory to stage from.",
    )
    parser.add_argument(
        "--horizon-final-source-dir",
        type=Path,
        default=DEFAULT_HORIZON_DEPENDENCE_FINAL_SOURCE_DIR,
        help="Cluster-side horizon final artifact directory to stage from.",
    )
    parser.add_argument(
        "--temporal-preview-source-dir",
        type=Path,
        default=DEFAULT_TEMPORAL_PREVIEW_SOURCE_DIR,
        help="Cluster-side temporal preview artifact directory to stage from.",
    )
    parser.add_argument(
        "--temporal-sensitivity-source-dir",
        type=Path,
        default=DEFAULT_TEMPORAL_SENSITIVITY_SOURCE_DIR,
        help="Cluster-side temporal sensitivity artifact directory to stage from.",
    )
    parser.add_argument(
        "--cohort-source-dir",
        type=Path,
        default=DEFAULT_COHORT_SOURCE_DIR,
        help="Cluster-side cohort artifact directory to stage from.",
    )
    parser.add_argument(
        "--splits-source-dir",
        type=Path,
        default=DEFAULT_SPLITS_SOURCE_DIR,
        help="Cluster-side splits artifact directory to stage from.",
    )
    parser.add_argument(
        "--model-ready-source-dir",
        type=Path,
        default=DEFAULT_MODEL_READY_SOURCE_DIR,
        help="Cluster-side model-ready artifact directory to stage from.",
    )
    parser.add_argument(
        "--carry-forward-source-dir",
        type=Path,
        default=DEFAULT_CARRY_FORWARD_SOURCE_DIR,
        help="Cluster-side carry-forward artifact directory to stage from.",
    )
    parser.add_argument(
        "--observation-process-source-dir",
        type=Path,
        default=DEFAULT_OBSERVATION_PROCESS_SOURCE_DIR,
        help="Cluster-side observation-process artifact directory to stage from.",
    )
    parser.add_argument(
        "--icd10-validation-source-dir",
        type=Path,
        default=DEFAULT_ICD10_DISEASE_GROUP_VALIDATION_SOURCE_DIR,
        help="Cluster-side ICD-10 disease-group validation artifact directory to stage from.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting files that already exist in the staging tree.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    results = [
        stage_asic_hard_case_comparison_exports(
            source_dir=args.source_dir,
            stage_root=args.stage_root,
            overwrite=args.overwrite,
        )
    ]
    if args.include_variable_audit:
        results.append(
            stage_asic_hard_case_comparison_variable_audit_exports(
                source_dir=args.variable_audit_source_dir,
                stage_root=args.stage_root,
                overwrite=args.overwrite,
            )
        )
    if args.include_sofa_feasibility:
        results.append(
            stage_asic_sofa_feasibility_exports(
                source_dir=args.sofa_feasibility_source_dir,
                stage_root=args.stage_root,
                overwrite=args.overwrite,
            )
        )
    if args.include_baseline_evaluation:
        results.append(
            stage_baseline_evaluation_exports(
                source_dir=args.baseline_evaluation_source_dir,
                stage_root=args.stage_root,
                overwrite=args.overwrite,
            )
        )
    if args.include_baseline_predictions:
        results.append(
            stage_baseline_prediction_exports(
                source_dir=args.baseline_prediction_source_dir,
                stage_root=args.stage_root,
                overwrite=args.overwrite,
            )
        )
    if args.include_hard_case_definition:
        results.append(
            stage_hard_case_definition_exports(
                source_dir=args.hard_case_definition_source_dir,
                stage_root=args.stage_root,
                overwrite=args.overwrite,
            )
        )
    if args.include_xgboost_recalibration:
        results.append(
            stage_xgboost_recalibration_exports(
                source_dir=args.xgboost_recalibration_source_dir,
                stage_root=args.stage_root,
                overwrite=args.overwrite,
            )
        )
    if args.include_hard_case_agreement:
        results.append(
            stage_hard_case_agreement_exports(
                source_dir=args.hard_case_agreement_source_dir,
                stage_root=args.stage_root,
                overwrite=args.overwrite,
            )
        )
    if args.include_horizon_dependence:
        results.extend(
            stage_horizon_dependence_exports(
                foundation_source_dir=args.horizon_foundation_source_dir,
                overlap_source_dir=args.horizon_overlap_source_dir,
                final_source_dir=args.horizon_final_source_dir,
                stage_root=args.stage_root,
                overwrite=args.overwrite,
            )
        )
    if args.include_temporal_preview:
        results.append(
            stage_temporal_preview_exports(
                source_dir=args.temporal_preview_source_dir,
                stage_root=args.stage_root,
                overwrite=args.overwrite,
            )
        )
    if args.include_temporal_sensitivity:
        results.append(
            stage_temporal_sensitivity_exports(
                source_dir=args.temporal_sensitivity_source_dir,
                stage_root=args.stage_root,
                overwrite=args.overwrite,
            )
        )
    if args.include_foundational_summaries:
        results.extend(
            stage_foundational_summary_exports(
                cohort_source_dir=args.cohort_source_dir,
                splits_source_dir=args.splits_source_dir,
                model_ready_source_dir=args.model_ready_source_dir,
                carry_forward_source_dir=args.carry_forward_source_dir,
                observation_process_source_dir=args.observation_process_source_dir,
                stage_root=args.stage_root,
                overwrite=args.overwrite,
            )
        )
    if args.include_icd10_validation:
        results.append(
            stage_icd10_disease_group_validation_exports(
                source_dir=args.icd10_validation_source_dir,
                stage_root=args.stage_root,
                overwrite=args.overwrite,
            )
        )

    for result in results:
        print(f"Export name: {result.export_name}")
        print(f"Source directory: {result.source_dir}")
        print(f"Staged directory: {result.staged_dir}")
        print(f"Copied files: {len(result.copied_paths)}")
        if result.excluded_paths:
            print("Excluded files:")
            for path in result.excluded_paths:
                print(f"- {path}")
        print(f"Export manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
