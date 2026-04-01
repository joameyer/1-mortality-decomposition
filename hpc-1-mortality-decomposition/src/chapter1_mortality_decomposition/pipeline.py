from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from chapter1_mortality_decomposition.artifacts import (
    DEFAULT_CHAPTER1_OUTPUT_DIR,
    Chapter1InputTables,
    load_chapter1_inputs,
)
from chapter1_mortality_decomposition.carry_forward import INSTANCE_KEY_COLUMNS
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
from chapter1_mortality_decomposition.observation_process import (
    Chapter1ObservationProcessResult,
    build_chapter1_observation_process_features,
    merge_observation_process_into_model_ready,
)
from chapter1_mortality_decomposition.splits import (
    Chapter1StaySplitResult,
    build_chapter1_stay_splits,
)
from chapter1_mortality_decomposition.utils import write_dataframe, write_text


@dataclass(frozen=True)
class Chapter1Dataset:
    cohort: Chapter1CohortResult
    valid_instances: Chapter1ValidInstanceResult
    labels: Chapter1LabelResult
    stay_splits: Chapter1StaySplitResult
    observation_process: Chapter1ObservationProcessResult
    cohort_summary: pd.DataFrame
    verification_summary: pd.DataFrame
    feature_set_definition: pd.DataFrame
    feature_set_validation_summary: pd.DataFrame
    feature_sets: dict[str, "Chapter1FeatureSetDataset"]


@dataclass(frozen=True)
class Chapter1FeatureSetDataset:
    feature_set_name: str
    model_ready: Chapter1ModelReadyResult


def _build_chapter1_cohort_summary(
    cohort: Chapter1CohortResult,
    valid_instances: Chapter1ValidInstanceResult,
    labels: Chapter1LabelResult,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {
            "summary_group": "cohort",
            "metric": "input_hospitals",
            "horizon_h": pd.NA,
            "value": int(cohort.site_eligibility["hospital_id"].nunique(dropna=True)),
        },
        {
            "summary_group": "cohort",
            "metric": "retained_hospitals",
            "horizon_h": pd.NA,
            "value": int(cohort.retained_hospitals["hospital_id"].nunique(dropna=True)),
        },
        {
            "summary_group": "cohort",
            "metric": "input_stays",
            "horizon_h": pd.NA,
            "value": int(cohort.stay_exclusions["stay_id_global"].nunique(dropna=True)),
        },
        {
            "summary_group": "cohort",
            "metric": "retained_stays",
            "horizon_h": pd.NA,
            "value": int(cohort.table["stay_id_global"].nunique(dropna=True)),
        },
        {
            "summary_group": "instances",
            "metric": "valid_prediction_instances_total",
            "horizon_h": pd.NA,
            "value": int(valid_instances.valid_instances.shape[0]),
        },
    ]

    for row in valid_instances.counts_by_horizon.itertuples(index=False):
        rows.append(
            {
                "summary_group": "instances",
                "metric": "valid_prediction_instances",
                "horizon_h": int(row.horizon_h),
                "value": int(row.valid_instances),
            }
        )

    for row in labels.summary_by_horizon.itertuples(index=False):
        rows.extend(
            [
                {
                    "summary_group": "labels",
                    "metric": "labelable_instances",
                    "horizon_h": int(row.horizon_h),
                    "value": int(row.labelable_instances),
                },
                {
                    "summary_group": "labels",
                    "metric": "positive_labels",
                    "horizon_h": int(row.horizon_h),
                    "value": int(row.positive_labels),
                },
                {
                    "summary_group": "labels",
                    "metric": "negative_labels",
                    "horizon_h": int(row.horizon_h),
                    "value": int(row.negative_labels),
                },
                {
                    "summary_group": "labels",
                    "metric": "unlabeled_instances",
                    "horizon_h": int(row.horizon_h),
                    "value": int(row.unlabeled_instances),
                },
            ]
        )

    return pd.DataFrame(rows)


def _build_chapter1_verification_summary(
    cohort: Chapter1CohortResult,
    valid_instances: Chapter1ValidInstanceResult,
    labels: Chapter1LabelResult,
) -> pd.DataFrame:
    retained_hospitals = set(cohort.retained_hospitals["hospital_id"].astype("string").tolist())
    retained_stays = set(cohort.table["stay_id_global"].astype("string").tolist())
    retained_valid_instances = valid_instances.valid_instances.copy()

    valid_hospitals = set(retained_valid_instances["hospital_id"].astype("string").tolist())
    valid_stays = set(retained_valid_instances["stay_id_global"].astype("string").tolist())

    label_summary = labels.summary_by_horizon.set_index("horizon_h") if not labels.summary_by_horizon.empty else pd.DataFrame()
    valid_counts = valid_instances.counts_by_horizon.set_index("horizon_h") if not valid_instances.counts_by_horizon.empty else pd.DataFrame()
    label_consistency = True
    if not label_summary.empty and not valid_counts.empty:
        for horizon_h in valid_counts.index.tolist():
            labelable = int(label_summary.at[horizon_h, "labelable_instances"])
            unlabeled = int(label_summary.at[horizon_h, "unlabeled_instances"])
            total_valid = int(valid_counts.at[horizon_h, "valid_instances"])
            if labelable + unlabeled != total_valid:
                label_consistency = False
                break

    rows = [
        {
            "check_id": "no_excluded_hospital_contributes_retained_stays",
            "passed": valid_hospitals.issubset(retained_hospitals)
            and set(cohort.table["hospital_id"].astype("string").tolist()).issubset(retained_hospitals),
            "detail": (
                "All retained stays and valid prediction instances come from site-eligible hospitals."
            ),
        },
        {
            "check_id": "no_mech_vent_failed_stay_retained",
            "passed": bool(cohort.table["mech_vent_ge_24h_qc"].eq(True).all()) if not cohort.table.empty else True,
            "detail": "No retained stay fails the upstream mech_vent_ge_24h_qc contract.",
        },
        {
            "check_id": "no_missing_or_readmission_flagged_stay_retained",
            "passed": bool(cohort.table["readmission"].eq(0).all()) if not cohort.table.empty else True,
            "detail": "No retained stay has missing readmission or readmission == 1.",
        },
        {
            "check_id": "valid_instances_restricted_to_retained_stays",
            "passed": valid_stays.issubset(retained_stays),
            "detail": "Valid prediction instances are computed only on retained stays.",
        },
        {
            "check_id": "proxy_label_counts_consistent_with_valid_instances",
            "passed": label_consistency and int(labels.usable_labels.shape[0]) == int(
                labels.summary_by_horizon["labelable_instances"].sum()
            ),
            "detail": (
                "Per-horizon labelable and unlabeled counts sum to the valid-instance counts, "
                "and usable label rows match the total labelable count."
            ),
        },
    ]
    return pd.DataFrame(rows)


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
        mech_vent_stay_level_qc=inputs.mech_vent_stay_level_qc,
        config=config,
    )
    valid_instances = build_chapter1_valid_instances(
        retained_cohort=cohort.table,
        block_index=inputs.block_index,
        blocked_dynamic_features=inputs.blocked_dynamic_features,
        config=config,
    )
    labels = build_chapter1_proxy_horizon_labels(
        valid_instances=valid_instances.valid_instances,
        retained_cohort=cohort.table,
    )
    stay_splits = build_chapter1_stay_splits(
        retained_cohort=cohort.table,
        config=config,
    )
    observation_process = build_chapter1_observation_process_features(
        instance_index=labels.usable_labels[INSTANCE_KEY_COLUMNS],
        dynamic_harmonized=inputs.dynamic_harmonized,
    )
    cohort_summary = _build_chapter1_cohort_summary(cohort, valid_instances, labels)
    verification_summary = _build_chapter1_verification_summary(cohort, valid_instances, labels)

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
        model_ready = build_chapter1_model_ready_dataset(
            usable_labels=labels.usable_labels,
            blocked_dynamic_features=inputs.blocked_dynamic_features,
            feature_set_definition=feature_set_subset,
            feature_set_name=feature_set_name,
            mech_vent_episode_level=inputs.mech_vent_episode_level,
            stay_split_assignments=stay_splits.stay_assignments,
            config=config,
        )
        feature_sets[feature_set_name] = Chapter1FeatureSetDataset(
            feature_set_name=feature_set_name,
            model_ready=model_ready,
        )
    return Chapter1Dataset(
        cohort=cohort,
        valid_instances=valid_instances,
        labels=labels,
        stay_splits=stay_splits,
        observation_process=observation_process,
        cohort_summary=cohort_summary,
        verification_summary=verification_summary,
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
        "chapter1_cohort_summary": dataset.cohort_summary,
        "chapter1_verification_summary": dataset.verification_summary,
    }
    feature_set_outputs = {
        "chapter1_feature_set_definition": dataset.feature_set_definition,
        "chapter1_feature_set_validation_summary": dataset.feature_set_validation_summary,
    }
    instance_outputs = {
        "chapter1_candidate_instances": dataset.valid_instances.candidate_instances,
        "chapter1_valid_instances": dataset.valid_instances.valid_instances,
        "chapter1_instance_counts_by_horizon": dataset.valid_instances.counts_by_horizon,
        "chapter1_instance_exclusion_summary": dataset.valid_instances.exclusion_summary,
    }
    label_outputs = {
        "chapter1_proxy_horizon_labels": dataset.labels.labels,
        "chapter1_usable_proxy_horizon_labels": dataset.labels.usable_labels,
        "chapter1_proxy_label_summary_by_horizon": dataset.labels.summary_by_horizon,
        "chapter1_proxy_unlabeled_reason_summary": dataset.labels.unlabeled_reason_summary,
        "chapter1_proxy_label_notes": dataset.labels.notes,
    }
    split_outputs = {
        "chapter1_stay_split_assignments": dataset.stay_splits.stay_assignments,
        "chapter1_stay_split_summary": dataset.stay_splits.stay_summary,
        "chapter1_stay_split_verification_summary": dataset.stay_splits.verification_summary,
    }
    observation_process_outputs = {
        "chapter1_observation_process_block_features": dataset.observation_process.block_features,
        "chapter1_observation_process_qc_summary": dataset.observation_process.qc_summary,
        "chapter1_observation_process_verification_summary": (
            dataset.observation_process.verification_summary
        ),
        "chapter1_observation_process_spot_check_examples": (
            dataset.observation_process.spot_check_examples
        ),
    }

    for group_name, outputs in {
        "cohort": cohort_outputs,
        "feature_sets": feature_set_outputs,
        "instances": instance_outputs,
        "labels": label_outputs,
        "splits": split_outputs,
        "observation_process": observation_process_outputs,
    }.items():
        for name, df in outputs.items():
            path = output_dir / group_name / f"{name}.{extension}"
            output_paths[f"{group_name}_{name}"] = write_dataframe(
                df,
                path,
                output_format=output_format,
            )

    implementation_note_path = (
        output_dir / "observation_process" / "chapter1_observation_process_implementation_note.md"
    )
    output_paths["observation_process_implementation_note"] = write_text(
        dataset.observation_process.implementation_note_markdown,
        implementation_note_path,
    )

    for feature_set_name, feature_set_dataset in dataset.feature_sets.items():
        model_ready_outputs = {
            f"chapter1_{feature_set_name}_model_ready_dataset": feature_set_dataset.model_ready.table,
            f"chapter1_{feature_set_name}_model_ready_with_observation_process": (
                merge_observation_process_into_model_ready(
                    feature_set_dataset.model_ready.table,
                    dataset.observation_process.block_features,
                )
            ),
            f"chapter1_{feature_set_name}_readiness_summary": feature_set_dataset.model_ready.readiness_summary,
            f"chapter1_{feature_set_name}_feature_availability_by_horizon": (
                feature_set_dataset.model_ready.feature_availability_by_horizon
            ),
        }
        carry_forward_outputs = {
            f"chapter1_{feature_set_name}_locf_feature_summary": (
                feature_set_dataset.model_ready.locf_feature_summary
            ),
            f"chapter1_{feature_set_name}_ventilator_locf_summary": (
                feature_set_dataset.model_ready.ventilator_locf_summary
            ),
            f"chapter1_{feature_set_name}_missingness_by_hospital_and_family": (
                feature_set_dataset.model_ready.missingness_by_hospital_and_family
            ),
            f"chapter1_{feature_set_name}_carry_forward_verification_summary": (
                feature_set_dataset.model_ready.carry_forward_verification_summary
            ),
        }
        split_model_ready_outputs = {
            f"chapter1_{feature_set_name}_split_summary": feature_set_dataset.model_ready.split_summary,
            f"chapter1_{feature_set_name}_split_verification_summary": (
                feature_set_dataset.model_ready.split_verification_summary
            ),
        }
        for name, df in model_ready_outputs.items():
            path = output_dir / "model_ready" / f"{name}.{extension}"
            output_paths[f"model_ready_{name}"] = write_dataframe(
                df,
                path,
                output_format=output_format,
            )
        for name, df in split_model_ready_outputs.items():
            path = output_dir / "splits" / f"{name}.{extension}"
            output_paths[f"splits_{name}"] = write_dataframe(
                df,
                path,
                output_format=output_format,
            )
        for name, df in carry_forward_outputs.items():
            path = output_dir / "carry_forward" / f"{name}.{extension}"
            output_paths[f"carry_forward_{name}"] = write_dataframe(
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
