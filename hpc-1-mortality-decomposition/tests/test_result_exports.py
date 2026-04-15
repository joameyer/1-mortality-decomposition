from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "chapter1_mortality_decomposition" / "result_exports.py"
MODULE_SPEC = importlib.util.spec_from_file_location("hpc_result_exports", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise ImportError(f"Could not load module spec from {MODULE_PATH}")
RESULT_EXPORTS = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = RESULT_EXPORTS
MODULE_SPEC.loader.exec_module(RESULT_EXPORTS)


class ResultExportsTests(TestCase):
    def test_stage_baseline_prediction_exports_copies_prediction_bundle_and_excludes_model_pickles(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_dir = (
                tmp_path
                / "artifacts"
                / "chapter1"
                / RESULT_EXPORTS.BASELINE_PREDICTION_RELATIVE_ROOT
            )
            stage_root = tmp_path / "export-staging" / "chapter1_true_results"
            source_dir.mkdir(parents=True)

            for relative_path in RESULT_EXPORTS.APPROVED_BASELINE_PREDICTION_EXPORTS:
                destination = source_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(f"{relative_path}\n")

            excluded_paths = []
            for relative_path in RESULT_EXPORTS.EXCLUDED_BASELINE_PREDICTION_EXPORTS:
                destination = source_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(f"excluded {relative_path}\n")
                excluded_paths.append(destination.resolve())

            result = RESULT_EXPORTS.stage_baseline_prediction_exports(
                source_dir=source_dir,
                stage_root=stage_root,
            )

            staged_dir = stage_root / RESULT_EXPORTS.BASELINE_PREDICTION_RELATIVE_ROOT
            for relative_path in RESULT_EXPORTS.APPROVED_BASELINE_PREDICTION_EXPORTS:
                self.assertTrue((staged_dir / relative_path).exists(), relative_path)
            for relative_path in RESULT_EXPORTS.EXCLUDED_BASELINE_PREDICTION_EXPORTS:
                self.assertFalse((staged_dir / relative_path).exists(), relative_path)

            self.assertEqual(result.excluded_paths, tuple(excluded_paths))
            manifest = json.loads(result.manifest_path.read_text())
            self.assertEqual(
                manifest["export_name"],
                "baseline_prediction_local_review_bundle",
            )

    def test_stage_baseline_evaluation_exports_copies_required_bundle(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_dir = (
                tmp_path
                / "artifacts"
                / "chapter1"
                / RESULT_EXPORTS.BASELINE_EVALUATION_RELATIVE_ROOT
            )
            stage_root = tmp_path / "export-staging" / "chapter1_true_results"
            source_dir.mkdir(parents=True)

            for relative_path in RESULT_EXPORTS.APPROVED_BASELINE_EVALUATION_EXPORTS:
                destination = source_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(f"{relative_path}\n")

            result = RESULT_EXPORTS.stage_baseline_evaluation_exports(
                source_dir=source_dir,
                stage_root=stage_root,
            )

            staged_dir = stage_root / RESULT_EXPORTS.BASELINE_EVALUATION_RELATIVE_ROOT
            for relative_path in RESULT_EXPORTS.APPROVED_BASELINE_EVALUATION_EXPORTS:
                self.assertTrue((staged_dir / relative_path).exists(), relative_path)

            self.assertEqual(result.excluded_paths, ())
            self.assertTrue(result.manifest_path.exists())
            manifest = json.loads(result.manifest_path.read_text())
            self.assertEqual(
                manifest["export_name"],
                "baseline_evaluation_local_review_bundle",
            )
            self.assertEqual(manifest["staged_dir"], str(staged_dir.resolve()))

    def test_stage_asic_hard_case_comparison_exports_copies_only_approved_outputs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_dir = (
                tmp_path
                / "artifacts"
                / "chapter1"
                / RESULT_EXPORTS.ASIC_HARD_CASE_COMPARISON_RELATIVE_ROOT
            )
            stage_root = tmp_path / "export-staging" / "chapter1_true_results"
            source_dir.mkdir(parents=True)

            approved_files = {
                "comparison_table.csv": "variable,level\nsex,M\n",
                "effect_size_plot_data.csv": "variable,standardized_difference\nsex,1.0\n",
                "effect_size_figure.png": "png-bytes-placeholder",
                "summary.md": "# Summary\n",
                "early_vs_late_death_split/early_vs_late_fatal_timing_summary.csv": "group,count\nearly,1\n",
                "early_vs_late_death_split/early_vs_late_low_pred_share.png": "png-bytes-placeholder",
                "early_vs_late_death_split/early_vs_late_interpretation_note.md": "# Note\n",
            }
            for relative_path, content in approved_files.items():
                destination = source_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(content)

            restricted_dataset_path = source_dir / "stay_level_comparison_dataset.csv"
            restricted_dataset_path.write_text("stay_id_global,sex\ns1,M\n")
            details_path = source_dir / "standardized_difference_details.csv"
            details_path.write_text("variable,level,standardized_difference\nsex,M,1.0\n")
            producer_manifest_path = source_dir / "run_manifest.json"
            producer_manifest_path.write_text("{}\n")

            result = RESULT_EXPORTS.stage_asic_hard_case_comparison_exports(
                source_dir=source_dir,
                stage_root=stage_root,
            )

            staged_dir = stage_root / RESULT_EXPORTS.ASIC_HARD_CASE_COMPARISON_RELATIVE_ROOT
            for relative_path in approved_files:
                self.assertTrue((staged_dir / relative_path).exists(), relative_path)

            self.assertFalse((staged_dir / "stay_level_comparison_dataset.csv").exists())
            self.assertFalse((staged_dir / "standardized_difference_details.csv").exists())
            self.assertFalse((staged_dir / "run_manifest.json").exists())
            self.assertEqual(
                result.excluded_paths,
                (
                    restricted_dataset_path.resolve(),
                    details_path.resolve(),
                    producer_manifest_path.resolve(),
                ),
            )
            self.assertTrue(result.manifest_path.exists())

            manifest = json.loads(result.manifest_path.read_text())
            self.assertEqual(manifest["staged_dir"], str(staged_dir.resolve()))
            self.assertIn(
                "stay_level_comparison_dataset.csv",
                manifest["excluded_relative_paths"],
            )
            self.assertIn(
                "standardized_difference_details.csv",
                manifest["excluded_relative_paths"],
            )
            self.assertIn(
                "run_manifest.json",
                manifest["excluded_relative_paths"],
            )

    def test_stage_xgboost_recalibration_exports_copies_review_bundle_and_excludes_raw_auxiliary_files(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_dir = (
                tmp_path
                / "artifacts"
                / "chapter1"
                / RESULT_EXPORTS.XGBOOST_RECALIBRATION_RELATIVE_ROOT
            )
            stage_root = tmp_path / "export-staging" / "chapter1_true_results"
            source_dir.mkdir(parents=True)

            for relative_path in RESULT_EXPORTS.APPROVED_XGBOOST_RECALIBRATION_EXPORTS:
                destination = source_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(f"{relative_path}\n")

            excluded_paths = []
            for relative_path in RESULT_EXPORTS.EXCLUDED_XGBOOST_RECALIBRATION_EXPORTS:
                destination = source_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(f"excluded {relative_path}\n")
                excluded_paths.append(destination.resolve())

            result = RESULT_EXPORTS.stage_xgboost_recalibration_exports(
                source_dir=source_dir,
                stage_root=stage_root,
            )

            staged_dir = stage_root / RESULT_EXPORTS.XGBOOST_RECALIBRATION_RELATIVE_ROOT
            for relative_path in RESULT_EXPORTS.APPROVED_XGBOOST_RECALIBRATION_EXPORTS:
                self.assertTrue((staged_dir / relative_path).exists(), relative_path)
            for relative_path in RESULT_EXPORTS.EXCLUDED_XGBOOST_RECALIBRATION_EXPORTS:
                self.assertFalse((staged_dir / relative_path).exists(), relative_path)

            self.assertEqual(result.excluded_paths, tuple(excluded_paths))
            manifest = json.loads(result.manifest_path.read_text())
            self.assertEqual(
                manifest["export_name"],
                "xgboost_recalibration_local_review_bundle",
            )

    def test_stage_hard_case_definition_exports_copies_review_bundle(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_dir = (
                tmp_path
                / "artifacts"
                / "chapter1"
                / RESULT_EXPORTS.HARD_CASE_DEFINITION_RELATIVE_ROOT
            )
            stage_root = tmp_path / "export-staging" / "chapter1_true_results"
            source_dir.mkdir(parents=True)

            for relative_path in RESULT_EXPORTS.APPROVED_HARD_CASE_DEFINITION_EXPORTS:
                destination = source_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(f"{relative_path}\n")

            result = RESULT_EXPORTS.stage_hard_case_definition_exports(
                source_dir=source_dir,
                stage_root=stage_root,
            )

            staged_dir = stage_root / RESULT_EXPORTS.HARD_CASE_DEFINITION_RELATIVE_ROOT
            for relative_path in RESULT_EXPORTS.APPROVED_HARD_CASE_DEFINITION_EXPORTS:
                self.assertTrue((staged_dir / relative_path).exists(), relative_path)

            self.assertEqual(result.excluded_paths, ())
            manifest = json.loads(result.manifest_path.read_text())
            self.assertEqual(
                manifest["export_name"],
                "hard_case_definition_local_review_bundle",
            )

    def test_stage_hard_case_agreement_exports_excludes_stay_level_table(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_dir = (
                tmp_path
                / "artifacts"
                / "chapter1"
                / RESULT_EXPORTS.HARD_CASE_AGREEMENT_RELATIVE_ROOT
            )
            stage_root = tmp_path / "export-staging" / "chapter1_true_results"
            source_dir.mkdir(parents=True)

            for relative_path in RESULT_EXPORTS.APPROVED_HARD_CASE_AGREEMENT_EXPORTS:
                destination = source_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(f"{relative_path}\n")
            excluded = source_dir / "fatal_stay_level_hard_case_agreement.csv"
            excluded.write_text("stay_id_global,agreement\ns1,match\n")

            result = RESULT_EXPORTS.stage_hard_case_agreement_exports(
                source_dir=source_dir,
                stage_root=stage_root,
            )

            staged_dir = stage_root / RESULT_EXPORTS.HARD_CASE_AGREEMENT_RELATIVE_ROOT
            for relative_path in RESULT_EXPORTS.APPROVED_HARD_CASE_AGREEMENT_EXPORTS:
                self.assertTrue((staged_dir / relative_path).exists(), relative_path)
            self.assertFalse((staged_dir / "fatal_stay_level_hard_case_agreement.csv").exists())
            self.assertEqual(result.excluded_paths, (excluded.resolve(),))

    def test_stage_asic_hard_case_comparison_variable_audit_exports_copies_only_table_and_memo(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_dir = (
                tmp_path
                / "artifacts"
                / "chapter1"
                / RESULT_EXPORTS.ASIC_HARD_CASE_COMPARISON_VARIABLE_AUDIT_RELATIVE_ROOT
            )
            stage_root = tmp_path / "export-staging" / "chapter1_true_results"
            source_dir.mkdir(parents=True)

            approved_files = {
                "asic_hard_case_comparison_variable_audit_table.csv": "variable_family,status\nage,NOT READY\n",
                "asic_hard_case_comparison_variable_audit_memo.md": "# Variable Audit\n",
            }
            for relative_path, content in approved_files.items():
                destination = source_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(content)

            result = RESULT_EXPORTS.stage_asic_hard_case_comparison_variable_audit_exports(
                source_dir=source_dir,
                stage_root=stage_root,
            )

            staged_dir = (
                stage_root
                / RESULT_EXPORTS.ASIC_HARD_CASE_COMPARISON_VARIABLE_AUDIT_RELATIVE_ROOT
            )
            for relative_path in approved_files:
                self.assertTrue((staged_dir / relative_path).exists(), relative_path)

            self.assertEqual(result.excluded_paths, ())
            self.assertTrue(result.manifest_path.exists())

            manifest = json.loads(result.manifest_path.read_text())
            self.assertEqual(
                manifest["export_name"],
                "asic_hard_case_comparison_variable_audit_local_review_bundle",
            )
            self.assertEqual(manifest["staged_dir"], str(staged_dir.resolve()))
            self.assertEqual(
                sorted(manifest["copied_relative_paths"]),
                sorted(approved_files.keys()),
            )
            self.assertEqual(manifest["excluded_relative_paths"], [])

    def test_stage_asic_sofa_feasibility_exports_copies_table_and_memo(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_dir = (
                tmp_path
                / "artifacts"
                / "chapter1"
                / RESULT_EXPORTS.ASIC_SOFA_FEASIBILITY_RELATIVE_ROOT
            )
            stage_root = tmp_path / "export-staging" / "chapter1_true_results"
            source_dir.mkdir(parents=True)

            for relative_path in RESULT_EXPORTS.APPROVED_ASIC_SOFA_FEASIBILITY_EXPORTS:
                destination = source_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(f"{relative_path}\n")

            result = RESULT_EXPORTS.stage_asic_sofa_feasibility_exports(
                source_dir=source_dir,
                stage_root=stage_root,
            )

            staged_dir = stage_root / RESULT_EXPORTS.ASIC_SOFA_FEASIBILITY_RELATIVE_ROOT
            for relative_path in RESULT_EXPORTS.APPROVED_ASIC_SOFA_FEASIBILITY_EXPORTS:
                self.assertTrue((staged_dir / relative_path).exists(), relative_path)
            self.assertEqual(result.excluded_paths, ())

    def test_stage_horizon_dependence_exports_copy_foundation_overlap_and_final(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            foundation_source_dir = (
                tmp_path
                / "artifacts"
                / "chapter1"
                / RESULT_EXPORTS.HORIZON_DEPENDENCE_FOUNDATION_RELATIVE_ROOT
            )
            overlap_source_dir = (
                tmp_path
                / "artifacts"
                / "chapter1"
                / RESULT_EXPORTS.HORIZON_DEPENDENCE_OVERLAP_RELATIVE_ROOT
            )
            final_source_dir = (
                tmp_path
                / "artifacts"
                / "chapter1"
                / RESULT_EXPORTS.HORIZON_DEPENDENCE_FINAL_RELATIVE_ROOT
            )
            stage_root = tmp_path / "export-staging" / "chapter1_true_results"

            for relative_path in RESULT_EXPORTS.APPROVED_HORIZON_DEPENDENCE_FOUNDATION_EXPORTS:
                destination = foundation_source_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(f"{relative_path}\n")
            for relative_path in RESULT_EXPORTS.APPROVED_HORIZON_DEPENDENCE_OVERLAP_EXPORTS:
                destination = overlap_source_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(f"{relative_path}\n")
            for relative_path in RESULT_EXPORTS.APPROVED_HORIZON_DEPENDENCE_FINAL_EXPORTS:
                destination = final_source_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(f"{relative_path}\n")

            results = RESULT_EXPORTS.stage_horizon_dependence_exports(
                foundation_source_dir=foundation_source_dir,
                overlap_source_dir=overlap_source_dir,
                final_source_dir=final_source_dir,
                stage_root=stage_root,
            )

            self.assertEqual(len(results), 3)
            foundation_staged_dir = stage_root / RESULT_EXPORTS.HORIZON_DEPENDENCE_FOUNDATION_RELATIVE_ROOT
            overlap_staged_dir = stage_root / RESULT_EXPORTS.HORIZON_DEPENDENCE_OVERLAP_RELATIVE_ROOT
            final_staged_dir = stage_root / RESULT_EXPORTS.HORIZON_DEPENDENCE_FINAL_RELATIVE_ROOT
            for relative_path in RESULT_EXPORTS.APPROVED_HORIZON_DEPENDENCE_FOUNDATION_EXPORTS:
                self.assertTrue((foundation_staged_dir / relative_path).exists(), relative_path)
            for relative_path in RESULT_EXPORTS.APPROVED_HORIZON_DEPENDENCE_OVERLAP_EXPORTS:
                self.assertTrue((overlap_staged_dir / relative_path).exists(), relative_path)
            for relative_path in RESULT_EXPORTS.APPROVED_HORIZON_DEPENDENCE_FINAL_EXPORTS:
                self.assertTrue((final_staged_dir / relative_path).exists(), relative_path)

    def test_stage_temporal_preview_exports_copies_comparison_package(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_dir = (
                tmp_path
                / "artifacts"
                / "chapter1"
                / RESULT_EXPORTS.TEMPORAL_PREVIEW_RELATIVE_ROOT
            )
            stage_root = tmp_path / "export-staging" / "chapter1_true_results"
            source_dir.mkdir(parents=True)

            for relative_path in RESULT_EXPORTS.APPROVED_TEMPORAL_PREVIEW_EXPORTS:
                destination = source_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(f"{relative_path}\n")

            result = RESULT_EXPORTS.stage_temporal_preview_exports(
                source_dir=source_dir,
                stage_root=stage_root,
            )

            staged_dir = stage_root / RESULT_EXPORTS.TEMPORAL_PREVIEW_RELATIVE_ROOT
            for relative_path in RESULT_EXPORTS.APPROVED_TEMPORAL_PREVIEW_EXPORTS:
                self.assertTrue((staged_dir / relative_path).exists(), relative_path)
            self.assertEqual(result.excluded_paths, ())

    def test_stage_foundational_summary_exports_copy_only_approved_summary_files(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cohort_source_dir = (
                tmp_path / "artifacts" / "chapter1" / RESULT_EXPORTS.COHORT_RELATIVE_ROOT
            )
            splits_source_dir = (
                tmp_path / "artifacts" / "chapter1" / RESULT_EXPORTS.SPLITS_RELATIVE_ROOT
            )
            model_ready_source_dir = (
                tmp_path / "artifacts" / "chapter1" / RESULT_EXPORTS.MODEL_READY_RELATIVE_ROOT
            )
            carry_forward_source_dir = (
                tmp_path / "artifacts" / "chapter1" / RESULT_EXPORTS.CARRY_FORWARD_RELATIVE_ROOT
            )
            observation_process_source_dir = (
                tmp_path
                / "artifacts"
                / "chapter1"
                / RESULT_EXPORTS.OBSERVATION_PROCESS_RELATIVE_ROOT
            )
            stage_root = tmp_path / "export-staging" / "chapter1_true_results"

            for relative_path in RESULT_EXPORTS.APPROVED_COHORT_EXPORTS:
                destination = cohort_source_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(f"{relative_path}\n")
            excluded_cohort = []
            for relative_path in RESULT_EXPORTS.EXCLUDED_COHORT_EXPORTS:
                destination = cohort_source_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(f"excluded {relative_path}\n")
                excluded_cohort.append(destination.resolve())

            for relative_path in RESULT_EXPORTS.APPROVED_SPLITS_EXPORTS:
                destination = splits_source_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(f"{relative_path}\n")
            excluded_split = splits_source_dir / "chapter1_stay_split_assignments.csv"
            excluded_split.parent.mkdir(parents=True, exist_ok=True)
            excluded_split.write_text("stay_id_global,split\ns1,train\n")

            for relative_path in RESULT_EXPORTS.APPROVED_MODEL_READY_EXPORTS:
                destination = model_ready_source_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(f"{relative_path}\n")
            excluded_model_ready = []
            for relative_path in RESULT_EXPORTS.EXCLUDED_MODEL_READY_EXPORTS:
                destination = model_ready_source_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(f"excluded {relative_path}\n")
                excluded_model_ready.append(destination.resolve())

            for relative_path in RESULT_EXPORTS.APPROVED_CARRY_FORWARD_EXPORTS:
                destination = carry_forward_source_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(f"{relative_path}\n")

            for relative_path in RESULT_EXPORTS.APPROVED_OBSERVATION_PROCESS_EXPORTS:
                destination = observation_process_source_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(f"{relative_path}\n")
            excluded_observation_process = []
            for relative_path in RESULT_EXPORTS.EXCLUDED_OBSERVATION_PROCESS_EXPORTS:
                destination = observation_process_source_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(f"excluded {relative_path}\n")
                excluded_observation_process.append(destination.resolve())

            results = RESULT_EXPORTS.stage_foundational_summary_exports(
                cohort_source_dir=cohort_source_dir,
                splits_source_dir=splits_source_dir,
                model_ready_source_dir=model_ready_source_dir,
                carry_forward_source_dir=carry_forward_source_dir,
                observation_process_source_dir=observation_process_source_dir,
                stage_root=stage_root,
            )

            self.assertEqual(len(results), 5)
            cohort_staged_dir = stage_root / RESULT_EXPORTS.COHORT_RELATIVE_ROOT
            splits_staged_dir = stage_root / RESULT_EXPORTS.SPLITS_RELATIVE_ROOT
            model_ready_staged_dir = stage_root / RESULT_EXPORTS.MODEL_READY_RELATIVE_ROOT
            carry_forward_staged_dir = stage_root / RESULT_EXPORTS.CARRY_FORWARD_RELATIVE_ROOT
            observation_process_staged_dir = stage_root / RESULT_EXPORTS.OBSERVATION_PROCESS_RELATIVE_ROOT

            for relative_path in RESULT_EXPORTS.APPROVED_COHORT_EXPORTS:
                self.assertTrue((cohort_staged_dir / relative_path).exists(), relative_path)
            for relative_path in RESULT_EXPORTS.EXCLUDED_COHORT_EXPORTS:
                self.assertFalse((cohort_staged_dir / relative_path).exists(), relative_path)

            for relative_path in RESULT_EXPORTS.APPROVED_SPLITS_EXPORTS:
                self.assertTrue((splits_staged_dir / relative_path).exists(), relative_path)
            self.assertFalse((splits_staged_dir / "chapter1_stay_split_assignments.csv").exists())

            for relative_path in RESULT_EXPORTS.APPROVED_MODEL_READY_EXPORTS:
                self.assertTrue((model_ready_staged_dir / relative_path).exists(), relative_path)
            for relative_path in RESULT_EXPORTS.EXCLUDED_MODEL_READY_EXPORTS:
                self.assertFalse((model_ready_staged_dir / relative_path).exists(), relative_path)

            for relative_path in RESULT_EXPORTS.APPROVED_CARRY_FORWARD_EXPORTS:
                self.assertTrue((carry_forward_staged_dir / relative_path).exists(), relative_path)

            for relative_path in RESULT_EXPORTS.APPROVED_OBSERVATION_PROCESS_EXPORTS:
                self.assertTrue((observation_process_staged_dir / relative_path).exists(), relative_path)
            for relative_path in RESULT_EXPORTS.EXCLUDED_OBSERVATION_PROCESS_EXPORTS:
                self.assertFalse((observation_process_staged_dir / relative_path).exists(), relative_path)

            self.assertEqual(results[0].excluded_paths, tuple(excluded_cohort))
            self.assertEqual(results[1].excluded_paths, (excluded_split.resolve(),))
            self.assertEqual(results[2].excluded_paths, tuple(excluded_model_ready))
            self.assertEqual(results[4].excluded_paths, tuple(excluded_observation_process))

    def test_stage_icd10_validation_exports_excludes_stay_level_table(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_dir = (
                tmp_path
                / "artifacts"
                / "chapter1"
                / RESULT_EXPORTS.ICD10_DISEASE_GROUP_VALIDATION_RELATIVE_ROOT
            )
            stage_root = tmp_path / "export-staging" / "chapter1_true_results"
            source_dir.mkdir(parents=True)

            for relative_path in RESULT_EXPORTS.APPROVED_ICD10_DISEASE_GROUP_VALIDATION_EXPORTS:
                destination = source_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(f"{relative_path}\n")
            excluded = source_dir / "asic_static_icd10_disease_groups.csv"
            excluded.write_text("stay_id_global,final_disease_group\ns1,neurologic\n")

            result = RESULT_EXPORTS.stage_icd10_disease_group_validation_exports(
                source_dir=source_dir,
                stage_root=stage_root,
            )

            staged_dir = stage_root / RESULT_EXPORTS.ICD10_DISEASE_GROUP_VALIDATION_RELATIVE_ROOT
            for relative_path in RESULT_EXPORTS.APPROVED_ICD10_DISEASE_GROUP_VALIDATION_EXPORTS:
                self.assertTrue((staged_dir / relative_path).exists(), relative_path)
            self.assertFalse((staged_dir / "asic_static_icd10_disease_groups.csv").exists())
            self.assertEqual(result.excluded_paths, (excluded.resolve(),))
