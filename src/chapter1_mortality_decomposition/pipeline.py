from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from chapter1_mortality_decomposition.artifacts import (
    DEFAULT_CHAPTER1_OUTPUT_DIR,
    Chapter1InputTables,
    load_chapter1_inputs,
)
from chapter1_mortality_decomposition.cohort import (
    Chapter1CohortResult,
    build_chapter1_cohort,
)
from chapter1_mortality_decomposition.config import (
    Chapter1Config,
    build_chapter1_feature_set_definition,
    default_chapter1_config,
    load_chapter1_feature_set_config,
)
from chapter1_mortality_decomposition.instances import (
    Chapter1ValidInstanceResult,
    build_chapter1_valid_instances,
)
from chapter1_mortality_decomposition.labels import (
    Chapter1LabelResult,
    build_chapter1_proxy_horizon_labels,
)
from chapter1_mortality_decomposition.model_ready import (
    Chapter1ModelReadyResult,
    build_chapter1_model_ready_dataset,
)
from chapter1_mortality_decomposition.utils import ensure_directory, write_dataframe


@dataclass(frozen=True)
class Chapter1Dataset:
    cohort: Chapter1CohortResult
    feature_set_definition: pd.DataFrame
    feature_set_validation_summary: pd.DataFrame
    feature_sets: dict[str, "Chapter1FeatureSetDataset"]


@dataclass(frozen=True)
class Chapter1FeatureSetDataset:
    feature_set_name: str
    valid_instances: Chapter1ValidInstanceResult
    labels: Chapter1LabelResult
    model_ready: Chapter1ModelReadyResult


def _with_feature_set_name(df: pd.DataFrame, feature_set_name: str) -> pd.DataFrame:
    annotated = df.copy()
    annotated.insert(0, "feature_set_name", feature_set_name)
    return annotated


def build_chapter1_dataset(
    inputs: Chapter1InputTables,
    config: Chapter1Config | None = None,
) -> Chapter1Dataset:
    config = config or default_chapter1_config()
    feature_set_config = load_chapter1_feature_set_config(config.feature_set_config_path)
    cohort = build_chapter1_cohort(
        static_harmonized=inputs.static_harmonized,
        dynamic_harmonized=inputs.dynamic_harmonized,
        stay_block_counts=inputs.stay_block_counts,
        config=config,
    )
    feature_set_definition, feature_set_validation_summary = build_chapter1_feature_set_definition(
        inputs.blocked_dynamic_features,
        retained_stays=cohort.retained_stays,
        config=config,
        feature_set_config=feature_set_config,
    )
    feature_sets: dict[str, Chapter1FeatureSetDataset] = {}
    for feature_set_name in feature_set_config.feature_sets:
        feature_set_subset = feature_set_definition[
            feature_set_definition["feature_set_name"].eq(feature_set_name)
        ].reset_index(drop=True)
        valid_instances = build_chapter1_valid_instances(
            retained_cohort=cohort.table,
            block_index=inputs.block_index,
            blocked_dynamic_features=inputs.blocked_dynamic_features,
            feature_set_definition=feature_set_subset,
            config=config,
        )
        labels = build_chapter1_proxy_horizon_labels(
            valid_instances=valid_instances.valid_instances,
            retained_cohort=cohort.table,
        )
        model_ready = build_chapter1_model_ready_dataset(
            usable_labels=labels.usable_labels,
            blocked_dynamic_features=inputs.blocked_dynamic_features,
            feature_set_definition=feature_set_subset,
            feature_set_name=feature_set_name,
        )
        feature_sets[feature_set_name] = Chapter1FeatureSetDataset(
            feature_set_name=feature_set_name,
            valid_instances=valid_instances,
            labels=labels,
            model_ready=model_ready,
        )
    return Chapter1Dataset(
        cohort=cohort,
        feature_set_definition=feature_set_definition,
        feature_set_validation_summary=feature_set_validation_summary,
        feature_sets=feature_sets,
    )


def write_chapter1_dataset(
    dataset: Chapter1Dataset,
    *,
    output_dir: Path = DEFAULT_CHAPTER1_OUTPUT_DIR,
    output_format: str = "csv",
) -> dict[str, Path]:
    extension = "csv" if output_format == "csv" else "parquet"
    output_paths: dict[str, Path] = {}

    cohort_dir = ensure_directory(output_dir / "cohort")
    feature_sets_dir = ensure_directory(output_dir / "feature_sets")
    instances_dir = ensure_directory(output_dir / "instances")
    labels_dir = ensure_directory(output_dir / "labels")
    model_ready_dir = ensure_directory(output_dir / "model_ready")

    cohort_outputs = {
        "chapter1_notes": dataset.cohort.notes,
        "chapter1_core_vital_group_coverage": dataset.cohort.core_vital_group_coverage,
        "chapter1_site_eligibility": dataset.cohort.site_eligibility,
        "chapter1_site_counts_summary": dataset.cohort.site_counts_summary,
        "chapter1_stay_exclusions": dataset.cohort.stay_exclusions,
        "chapter1_stay_exclusion_summary_by_hospital": (
            dataset.cohort.stay_exclusion_summary_by_hospital
        ),
        "chapter1_counts_by_hospital": dataset.cohort.counts_by_hospital,
        "chapter1_retained_hospitals": dataset.cohort.retained_hospitals,
        "chapter1_retained_stays": dataset.cohort.retained_stays,
        "chapter1_retained_stay_table": dataset.cohort.table,
    }
    feature_set_outputs = {
        "chapter1_feature_set_definition": dataset.feature_set_definition,
        "chapter1_feature_set_validation_summary": dataset.feature_set_validation_summary,
    }

    for name, df in cohort_outputs.items():
        path = cohort_dir / f"{name}.{extension}"
        output_paths[f"cohort_{name}"] = write_dataframe(df, path, output_format=output_format)
    for name, df in feature_set_outputs.items():
        path = feature_sets_dir / f"{name}.{extension}"
        output_paths[f"feature_sets_{name}"] = write_dataframe(df, path, output_format=output_format)

    for feature_set_name, feature_set_dataset in dataset.feature_sets.items():
        instance_outputs = {
            f"chapter1_{feature_set_name}_candidate_instances": _with_feature_set_name(
                feature_set_dataset.valid_instances.candidate_instances,
                feature_set_name,
            ),
            f"chapter1_{feature_set_name}_valid_instances": _with_feature_set_name(
                feature_set_dataset.valid_instances.valid_instances,
                feature_set_name,
            ),
            f"chapter1_{feature_set_name}_instance_counts_by_horizon": _with_feature_set_name(
                feature_set_dataset.valid_instances.counts_by_horizon,
                feature_set_name,
            ),
            f"chapter1_{feature_set_name}_instance_exclusion_summary": _with_feature_set_name(
                feature_set_dataset.valid_instances.exclusion_summary,
                feature_set_name,
            ),
        }
        label_outputs = {
            f"chapter1_{feature_set_name}_proxy_horizon_labels": _with_feature_set_name(
                feature_set_dataset.labels.labels,
                feature_set_name,
            ),
            f"chapter1_{feature_set_name}_usable_proxy_horizon_labels": _with_feature_set_name(
                feature_set_dataset.labels.usable_labels,
                feature_set_name,
            ),
            f"chapter1_{feature_set_name}_proxy_label_summary_by_horizon": _with_feature_set_name(
                feature_set_dataset.labels.summary_by_horizon,
                feature_set_name,
            ),
            f"chapter1_{feature_set_name}_proxy_unlabeled_reason_summary": _with_feature_set_name(
                feature_set_dataset.labels.unlabeled_reason_summary,
                feature_set_name,
            ),
            f"chapter1_{feature_set_name}_proxy_label_notes": _with_feature_set_name(
                feature_set_dataset.labels.notes,
                feature_set_name,
            ),
        }
        model_ready_outputs = {
            f"chapter1_{feature_set_name}_model_ready_dataset": feature_set_dataset.model_ready.table,
            f"chapter1_{feature_set_name}_readiness_summary": feature_set_dataset.model_ready.readiness_summary,
            f"chapter1_{feature_set_name}_feature_availability_by_horizon": (
                feature_set_dataset.model_ready.feature_availability_by_horizon
            ),
        }

        for name, df in instance_outputs.items():
            path = instances_dir / f"{name}.{extension}"
            output_paths[f"instances_{name}"] = write_dataframe(
                df,
                path,
                output_format=output_format,
            )
        for name, df in label_outputs.items():
            path = labels_dir / f"{name}.{extension}"
            output_paths[f"labels_{name}"] = write_dataframe(
                df,
                path,
                output_format=output_format,
            )
        for name, df in model_ready_outputs.items():
            path = model_ready_dir / f"{name}.{extension}"
            output_paths[f"model_ready_{name}"] = write_dataframe(
                df,
                path,
                output_format=output_format,
            )

    return output_paths


def build_and_write_chapter1_dataset(
    *,
    input_dir: Path,
    output_dir: Path = DEFAULT_CHAPTER1_OUTPUT_DIR,
    input_format: str = "csv",
    output_format: str = "csv",
    config: Chapter1Config | None = None,
) -> tuple[Chapter1Dataset, dict[str, Path]]:
    inputs = load_chapter1_inputs(input_dir=input_dir, input_format=input_format)
    dataset = build_chapter1_dataset(inputs, config=config)
    output_paths = write_chapter1_dataset(
        dataset,
        output_dir=output_dir,
        output_format=output_format,
    )
    return dataset, output_paths
