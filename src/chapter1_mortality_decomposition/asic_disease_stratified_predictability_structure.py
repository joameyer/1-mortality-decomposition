from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    if "MPLCONFIGDIR" not in os.environ:
        matplotlib_cache_dir = Path("/tmp") / "chapter1_mortality_decomposition_matplotlib"
        matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(matplotlib_cache_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - environment dependency branch
    plt = None
    MATPLOTLIB_IMPORT_ERROR = exc
else:  # pragma: no cover - trivial assignment
    MATPLOTLIB_IMPORT_ERROR = None

from chapter1_mortality_decomposition.asic_hard_case_comparison import (
    HARD_CASE_RULE,
    LOW_PREDICTED_FATAL_GROUP,
    OTHER_FATAL_GROUP,
    _continuous_standardized_difference,
    _proportion_standardized_difference,
)
from chapter1_mortality_decomposition.icd10_disease_groups import (
    FROZEN_DISEASE_GROUP_HIERARCHY,
    derive_icd10_disease_group,
)
from chapter1_mortality_decomposition.utils import (
    ensure_directory,
    normalize_boolean_codes,
    read_dataframe,
    require_columns,
    write_dataframe,
    write_text,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts"
    / "chapter1"
    / "evaluation"
    / "asic"
    / "hard_cases"
    / "primary_medians"
    / "logistic_regression"
    / "asic_disease_stratified_predictability_structure"
)
DEFAULT_COMPARISON_DATASET_PATH = (
    REPO_ROOT
    / "cluster-results"
    / "chapter1_true_results"
    / "evaluation"
    / "asic"
    / "hard_cases"
    / "primary_medians"
    / "logistic_regression"
    / "asic_hard_case_comparison"
    / "stay_level_comparison_dataset.csv"
)
DEFAULT_HARD_CASE_PATH = (
    REPO_ROOT
    / "cluster-results"
    / "chapter1_true_results"
    / "evaluation"
    / "asic"
    / "hard_cases"
    / "primary_medians"
    / "logistic_regression"
    / "stay_level_hard_case_flags.csv"
)
DEFAULT_DISEASE_GROUP_PATH = (
    REPO_ROOT
    / "cluster-results"
    / "chapter1_true_results"
    / "evaluation"
    / "asic"
    / "icd10_disease_group_validation"
    / "asic_static_icd10_disease_groups.csv"
)
DEFAULT_DISEASE_GROUP_COUNTS_PATH = (
    REPO_ROOT
    / "cluster-results"
    / "chapter1_true_results"
    / "evaluation"
    / "asic"
    / "icd10_disease_group_validation"
    / "final_group_counts.csv"
)
STATIC_RELATIVE_PATH = Path("static") / "harmonized.csv"

TARGET_HORIZON_H = 24
EARLY_ICU_DEATH_THRESHOLD_HOURS = 48.0

ASSIGNMENT_QC_FILENAME = "asic_disease_group_assignment_qc.md"
GROUP_COUNTS_FILENAME = "asic_disease_group_summary.csv"
MAIN_SUMMARY_FILENAME = "asic_disease_stratified_hardcase_summary.csv"
CONTRAST_PANEL_FILENAME = "asic_disease_stratified_contrast_panel.csv"
FIGURE_FILENAME = "asic_disease_stratified_hardcase_share.png"
MEMO_FILENAME = "asic_disease_stratified_interpretation_memo.md"
MANIFEST_FILENAME = "run_manifest.json"

ADEQUATE_FATAL_MIN = 100
ADEQUATE_LOW_MIN = 20
BORDERLINE_FATAL_MIN = 30
BORDERLINE_LOW_MIN = 10
CONTINUOUS_MIN_PER_ARM = 5
BINARY_MIN_PER_ARM = 5
MIN_ASSESSABLE_FOR_ADEQUATE = 3
MIN_ASSESSABLE_FOR_BORDERLINE = 2
NEAR_NULL_EFFECT_SIZE = 0.10
VISIBLY_DIFFERENT_SHARE = 0.15
MAJOR_COLLAPSE_SHARE = 0.05

KEY_COLUMNS = ["stay_id_global", "hospital_id"]
ANCHOR_COLUMNS = [
    "stay_id_global",
    "hospital_id",
    "horizon_h",
    "label_value",
    "instance_id",
    "block_index",
    "prediction_time_h",
    "hard_case_flag",
    "hard_case_rule",
]
COMPARISON_REQUIRED_COLUMNS = {
    "stay_id_global",
    "instance_id",
    "hard_case_flag",
    "hard_case_group",
    "disease_group",
    "prediction_time_h",
    "icu_end_time_proxy_hours",
    "hospital_id",
    "pf_ratio_last",
    "map_last",
    "peep_last",
}

VARIABLE_SPECS = (
    {
        "name": "early_icu_death_flag",
        "label": "Early ICU death (<=48h)",
        "kind": "binary",
    },
    {
        "name": "map_last",
        "label": "MAP",
        "kind": "continuous",
    },
    {
        "name": "pf_ratio_last",
        "label": "PF ratio",
        "kind": "continuous",
    },
    {
        "name": "peep_last",
        "label": "PEEP",
        "kind": "continuous",
    },
)


@dataclass(frozen=True)
class DiseaseGroupSource:
    assignments: pd.DataFrame
    group_counts: pd.DataFrame
    source_mode: str
    source_path: Path
    source_note: str


@dataclass(frozen=True)
class ASICDiseaseStratifiedArtifacts:
    assignment_qc_path: Path
    disease_group_summary_path: Path
    hardcase_summary_path: Path
    contrast_panel_path: Path
    figure_path: Path
    memo_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class ASICDiseaseStratifiedRunResult:
    merged_dataset: pd.DataFrame
    disease_group_summary: pd.DataFrame
    hardcase_summary: pd.DataFrame
    contrast_panel: pd.DataFrame
    final_judgment: str
    wording_needs_narrowing: bool
    artifacts: ASICDiseaseStratifiedArtifacts


def _require_matplotlib() -> None:
    if plt is None:
        raise ImportError(
            "matplotlib is required for the ASIC disease-stratified hard-case figure."
        ) from MATPLOTLIB_IMPORT_ERROR


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _write_json(payload: dict[str, object], path: Path) -> Path:
    def _json_default(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if value is pd.NA or pd.isna(value):
            return None
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    return write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        path,
    )


def _resolve_existing_path(path: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    alternate_suffix = ".parquet" if candidate.suffix == ".csv" else ".csv"
    alternate = candidate.with_suffix(alternate_suffix)
    if alternate.exists():
        return alternate.resolve()
    raise FileNotFoundError(f"Required artifact is missing: {candidate}")


def _candidate_asic_input_roots() -> list[Path]:
    candidates: list[Path] = []
    env_value = os.environ.get("ASIC_INPUT_ROOT")
    if env_value:
        candidates.append(Path(env_value).expanduser())

    upstream_artifact_root = REPO_ROOT.parent / "icu-data-platform" / "artifacts"
    candidates.extend(
        [
            upstream_artifact_root / "asic_harmonized",
            upstream_artifact_root / "asic_harmonized_full",
        ]
    )

    deduplicated: list[Path] = []
    for candidate in candidates:
        if candidate not in deduplicated:
            deduplicated.append(candidate)
    return deduplicated


def _resolve_asic_input_root(input_root: Path | None) -> Path:
    if input_root is not None:
        resolved = Path(input_root).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"ASIC input root does not exist: {resolved}")
        return resolved

    for candidate in _candidate_asic_input_roots():
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate the authoritative local ASIC harmonized artifacts. Checked: "
        + ", ".join(str(path) for path in _candidate_asic_input_roots())
    )


def _load_saved_comparison_dataset(comparison_dataset_path: Path) -> pd.DataFrame:
    resolved_path = _resolve_existing_path(comparison_dataset_path)
    comparison_dataset = read_dataframe(resolved_path)
    require_columns(
        comparison_dataset,
        COMPARISON_REQUIRED_COLUMNS,
        "saved ASIC hard-case comparison dataset",
    )

    normalized = comparison_dataset.copy()
    for column in (
        "stay_id_global",
        "instance_id",
        "hard_case_group",
        "disease_group",
        "hospital_id",
    ):
        normalized[column] = normalized[column].astype("string")
    normalized["hard_case_flag"] = normalize_boolean_codes(normalized["hard_case_flag"])
    for column in ("prediction_time_h", "icu_end_time_proxy_hours", "pf_ratio_last", "map_last", "peep_last"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if normalized["hard_case_flag"].isna().any():
        raise ValueError(
            f"Saved comparison dataset {resolved_path} contains missing hard_case_flag values."
        )

    expected_group = pd.Series(
        np.where(
            normalized["hard_case_flag"].astype(bool),
            LOW_PREDICTED_FATAL_GROUP,
            OTHER_FATAL_GROUP,
        ),
        index=normalized.index,
        dtype="string",
    )
    if not normalized["hard_case_group"].eq(expected_group).all():
        raise ValueError(
            f"Saved comparison dataset {resolved_path} contains hard_case_group labels "
            "that are inconsistent with hard_case_flag."
        )

    if int(normalized.duplicated(subset=["stay_id_global", "hospital_id", "instance_id"]).sum()) > 0:
        raise ValueError(
            f"Saved comparison dataset {resolved_path} contains duplicated fatal stay rows."
        )

    normalized["hard_case_flag"] = normalized["hard_case_flag"].astype(bool)
    normalized["early_icu_death_flag"] = normalized["icu_end_time_proxy_hours"].le(
        EARLY_ICU_DEATH_THRESHOLD_HOURS
    )
    return normalized.sort_values(
        ["hard_case_flag", "hospital_id", "stay_id_global"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def _load_fatal_hard_case_anchor(hard_case_path: Path) -> pd.DataFrame:
    resolved_path = _resolve_existing_path(hard_case_path)
    hard_case = read_dataframe(resolved_path)
    require_columns(hard_case, set(ANCHOR_COLUMNS), "saved stay-level hard-case artifact")

    anchor = hard_case.loc[:, ANCHOR_COLUMNS].copy()
    for column in ("stay_id_global", "hospital_id", "instance_id", "hard_case_rule"):
        anchor[column] = anchor[column].astype("string")
    for column in ("horizon_h", "label_value", "block_index", "prediction_time_h"):
        anchor[column] = pd.to_numeric(anchor[column], errors="coerce")
    anchor["hard_case_flag"] = normalize_boolean_codes(anchor["hard_case_flag"])

    if anchor["hard_case_flag"].isna().any():
        raise ValueError(f"Saved hard-case artifact {resolved_path} contains missing hard_case_flag values.")

    anchor = anchor[
        anchor["horizon_h"].eq(TARGET_HORIZON_H) & anchor["label_value"].eq(1)
    ].copy()
    if anchor.empty:
        raise ValueError("The authoritative 24h fatal hard-case anchor is empty.")
    if not anchor["hard_case_rule"].eq(HARD_CASE_RULE).all():
        raise ValueError("The saved hard-case artifact does not match the frozen logistic hard-case rule.")
    duplicate_rows = int(anchor.duplicated(subset=["stay_id_global", "hospital_id", "instance_id"]).sum())
    if duplicate_rows:
        raise ValueError(
            f"The authoritative 24h fatal hard-case anchor contains {duplicate_rows} duplicated rows."
        )
    anchor["hard_case_flag"] = anchor["hard_case_flag"].astype(bool)
    return anchor.sort_values(
        ["hard_case_flag", "hospital_id", "stay_id_global"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def _build_group_counts(assignments: pd.DataFrame) -> pd.DataFrame:
    total = int(assignments.shape[0])
    summary = (
        assignments["disease_group"]
        .value_counts(dropna=False)
        .reindex(FROZEN_DISEASE_GROUP_HIERARCHY, fill_value=0)
        .rename_axis("disease_group")
        .reset_index(name="total_stays")
    )
    summary["total_stay_share"] = summary["total_stays"].map(
        lambda count: float(count / total) if total else np.nan
    )
    return summary


def _load_saved_group_counts(path: Path) -> pd.DataFrame:
    resolved_path = _resolve_existing_path(path)
    frame = read_dataframe(resolved_path)
    require_columns(frame, {"final_disease_group", "stay_count"}, "saved disease-group count summary")
    summary = frame.loc[:, ["final_disease_group", "stay_count"]].copy()
    summary = summary.rename(
        columns={"final_disease_group": "disease_group", "stay_count": "total_stays"}
    )
    summary["disease_group"] = summary["disease_group"].astype("string")
    summary["total_stays"] = pd.to_numeric(summary["total_stays"], errors="coerce")
    summary = summary.set_index("disease_group").reindex(FROZEN_DISEASE_GROUP_HIERARCHY).reset_index()
    if summary["total_stays"].isna().any():
        raise ValueError(f"Saved disease-group counts {resolved_path} are missing one or more frozen groups.")
    total = int(summary["total_stays"].sum())
    summary["total_stay_share"] = summary["total_stays"].map(
        lambda count: float(count / total) if total else np.nan
    )
    return summary


def _load_disease_group_source(
    *,
    comparison_dataset: pd.DataFrame,
    disease_group_path: Path = DEFAULT_DISEASE_GROUP_PATH,
    disease_group_counts_path: Path = DEFAULT_DISEASE_GROUP_COUNTS_PATH,
    asic_input_root: Path | None = None,
) -> DiseaseGroupSource:
    candidate_path = Path(disease_group_path).expanduser()
    if candidate_path.exists() or candidate_path.with_suffix(".parquet").exists():
        resolved_path = _resolve_existing_path(candidate_path)
        assignments = read_dataframe(resolved_path)
        require_columns(
            assignments,
            {"stay_id_global", "hospital_id", "final_disease_group"},
            "saved ICD-10 disease-group assignments",
        )
        normalized = assignments.loc[:, ["stay_id_global", "hospital_id", "final_disease_group"]].copy()
        normalized = normalized.rename(columns={"final_disease_group": "disease_group"})
        source_mode = "saved_disease_group_file"
        source_note = "Saved stay-level ICD-10 disease-group file reused directly."
        source_path = resolved_path
        group_counts = _build_group_counts(normalized)
    elif Path(disease_group_counts_path).expanduser().exists():
        normalized = (
            comparison_dataset.loc[:, ["stay_id_global", "hospital_id", "disease_group"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        source_mode = "comparison_dataset_embedded_assignments_with_validation_counts"
        source_note = (
            "The local review bundle excluded the saved stay-level ICD-10 disease-group file, so the "
            "issue reuses the authoritative disease_group labels already embedded in the frozen 24h "
            "comparison dataset for anchor-aligned assignment and pairs them with the frozen "
            "final_group_counts.csv validation output for cohort-level totals."
        )
        source_path = _resolve_existing_path(disease_group_counts_path)
        group_counts = _load_saved_group_counts(disease_group_counts_path)
    else:
        resolved_input_root = _resolve_asic_input_root(asic_input_root)
        static_path = _resolve_existing_path(resolved_input_root / STATIC_RELATIVE_PATH)
        static = pd.read_csv(static_path, usecols=["stay_id_global", "hospital_id", "icd10_codes"])
        normalized = static.loc[:, ["stay_id_global", "hospital_id"]].copy()
        normalized["disease_group"] = static["icd10_codes"].map(
            lambda raw_codes: derive_icd10_disease_group(raw_codes).final_group
        )
        source_mode = "derived_from_authoritative_static_fallback"
        source_note = (
            "The saved stay-level ICD-10 disease-group file was not present in the local review bundle, "
            "so assignments were regenerated from the authoritative local ASIC static harmonized table "
            "using the same frozen deterministic hierarchy."
        )
        source_path = static_path
        group_counts = _build_group_counts(normalized)

    for column in KEY_COLUMNS:
        normalized[column] = normalized[column].astype("string")
    normalized["disease_group"] = normalized["disease_group"].astype("string")
    duplicate_rows = int(normalized.duplicated(subset=KEY_COLUMNS).sum())
    if duplicate_rows:
        raise ValueError(
            f"Disease-group assignments contain {duplicate_rows} duplicated stay keys."
        )
    invalid_groups = sorted(set(normalized["disease_group"].dropna()) - set(FROZEN_DISEASE_GROUP_HIERARCHY))
    if invalid_groups:
        raise ValueError(f"Disease-group assignments contain unexpected groups: {invalid_groups}")

    normalized = normalized.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)
    return DiseaseGroupSource(
        assignments=normalized,
        group_counts=group_counts,
        source_mode=source_mode,
        source_path=source_path,
        source_note=source_note,
    )


def _format_count_pct(count: int, total: int) -> str:
    pct = (100.0 * count / total) if total else 0.0
    return f"{count}/{total} ({pct:.1f}%)"


def _format_pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{100.0 * float(value):.1f}%"


def _markdown_table(frame: pd.DataFrame) -> str:
    rendered = frame.astype(object).where(frame.notna(), "").astype(str)
    header = "| " + " | ".join(rendered.columns.tolist()) + " |"
    separator = "| " + " | ".join(["---"] * len(rendered.columns)) + " |"
    rows = [
        "| " + " | ".join(row) + " |"
        for row in rendered.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _format_continuous_summary(series: pd.Series, *, decimals: int = 1) -> str:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return "NA (n=0)"
    q1 = float(numeric.quantile(0.25))
    median = float(numeric.quantile(0.50))
    q3 = float(numeric.quantile(0.75))
    return f"{median:.{decimals}f} [{q1:.{decimals}f}, {q3:.{decimals}f}] (n={int(numeric.shape[0])})"


def _format_binary_summary(series: pd.Series) -> str:
    values = series.astype(bool)
    total = int(values.shape[0])
    early = int(values.sum())
    late = total - early
    return (
        f"early ICU death (<=48h): {_format_count_pct(early, total)}; "
        f"late ICU death (>48h): {_format_count_pct(late, total)}"
    )


def _build_merged_dataset(
    *,
    comparison_dataset: pd.DataFrame,
    fatal_anchor: pd.DataFrame,
    disease_group_source: DiseaseGroupSource,
) -> tuple[pd.DataFrame, dict[str, object]]:
    compare_cols = ["stay_id_global", "hospital_id", "instance_id", "prediction_time_h", "hard_case_flag"]
    anchor_compare = fatal_anchor.loc[:, compare_cols].copy()
    merged_anchor = anchor_compare.merge(
        comparison_dataset.loc[:, compare_cols],
        on=["stay_id_global", "hospital_id", "instance_id", "prediction_time_h"],
        how="outer",
        indicator=True,
        suffixes=("_anchor", "_comparison"),
    )
    if not merged_anchor["_merge"].eq("both").all():
        raise ValueError("Saved comparison dataset does not align cleanly to the authoritative 24h fatal anchor.")
    if not merged_anchor["hard_case_flag_anchor"].eq(merged_anchor["hard_case_flag_comparison"]).all():
        raise ValueError("Saved comparison dataset hard_case_flag values disagree with the authoritative anchor.")

    merged = fatal_anchor.merge(
        disease_group_source.assignments,
        on=KEY_COLUMNS,
        how="left",
        indicator=True,
        validate="many_to_one",
    )
    unmatched_rows = int(merged["_merge"].ne("both").sum())
    if unmatched_rows:
        raise ValueError(
            f"{unmatched_rows} fatal anchor rows could not be linked to the disease-group assignments."
        )
    merged = merged.drop(columns="_merge")

    comparison_check = comparison_dataset.merge(
        merged.loc[:, ["stay_id_global", "hospital_id", "instance_id", "disease_group"]],
        on=["stay_id_global", "hospital_id", "instance_id"],
        how="left",
        suffixes=("_comparison", "_assignment"),
        validate="one_to_one",
    )
    mismatched_groups = int(
        comparison_check["disease_group_comparison"].astype("string").ne(
            comparison_check["disease_group_assignment"].astype("string")
        ).sum()
    )
    if mismatched_groups:
        raise ValueError(
            f"{mismatched_groups} saved comparison-dataset rows disagree with the disease-group assignments."
        )

    analysis_dataset = comparison_dataset.drop(columns="disease_group").merge(
        merged.loc[:, ["stay_id_global", "hospital_id", "instance_id", "disease_group"]],
        on=["stay_id_global", "hospital_id", "instance_id"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    if not analysis_dataset["_merge"].eq("both").all():
        raise ValueError("Disease-group assignments could not be linked back to the comparison dataset.")
    analysis_dataset = analysis_dataset.drop(columns="_merge")

    empty_groups = [
        group
        for group in FROZEN_DISEASE_GROUP_HIERARCHY
        if int(analysis_dataset["disease_group"].eq(group).sum()) == 0
    ]
    fatal_group_counts = (
        analysis_dataset["disease_group"]
        .value_counts()
        .reindex(FROZEN_DISEASE_GROUP_HIERARCHY, fill_value=0)
    )
    dominant_group = fatal_group_counts.idxmax()
    dominant_group_share = float(fatal_group_counts.max() / analysis_dataset.shape[0])

    qc_summary = {
        "fatal_anchor_rows": int(fatal_anchor.shape[0]),
        "comparison_dataset_rows": int(comparison_dataset.shape[0]),
        "disease_assignment_rows": int(disease_group_source.assignments.shape[0]),
        "empty_groups_among_fatal_anchor": empty_groups,
        "dominant_fatal_group": dominant_group,
        "dominant_fatal_group_share": dominant_group_share,
    }
    return analysis_dataset, qc_summary


def _build_pooled_reference(dataset: pd.DataFrame) -> dict[str, float]:
    pooled: dict[str, float] = {}
    hard_group = dataset[dataset["hard_case_flag"]].copy()
    other_group = dataset[~dataset["hard_case_flag"]].copy()
    for variable in VARIABLE_SPECS:
        name = variable["name"]
        if variable["kind"] == "binary":
            pooled[name] = _proportion_standardized_difference(
                float(hard_group[name].mean()),
                float(other_group[name].mean()),
            )
            continue
        pooled[name], _ = _continuous_standardized_difference(hard_group[name], other_group[name])
    return pooled


def _classify_direction(
    *,
    pooled_effect: float,
    group_effect: float,
    assessable: bool,
) -> str:
    if not assessable or not np.isfinite(group_effect):
        return "not_assessable"
    if not np.isfinite(pooled_effect) or abs(pooled_effect) < NEAR_NULL_EFFECT_SIZE:
        return "near_null_or_unclear"
    if abs(group_effect) < NEAR_NULL_EFFECT_SIZE:
        return "near_null_or_unclear"
    if np.sign(group_effect) == np.sign(pooled_effect):
        return "same_direction"
    return "opposite_direction"


def _summarize_effect_size(value: float) -> str:
    if not np.isfinite(value):
        return "not assessable"
    return f"standardized difference = {value:.3f}"


def _build_contrast_row(
    *,
    disease_group: str,
    group_dataset: pd.DataFrame,
    pooled_reference: dict[str, float],
) -> dict[str, object]:
    low_group = group_dataset[group_dataset["hard_case_flag"]].copy()
    other_group = group_dataset[~group_dataset["hard_case_flag"]].copy()
    rows: list[dict[str, object]] = []

    for variable in VARIABLE_SPECS:
        name = str(variable["name"])
        label = str(variable["label"])
        kind = str(variable["kind"])

        if kind == "binary":
            assessable = low_group.shape[0] >= BINARY_MIN_PER_ARM and other_group.shape[0] >= BINARY_MIN_PER_ARM
            group_effect = _proportion_standardized_difference(
                float(low_group[name].mean()),
                float(other_group[name].mean()),
            ) if assessable else float("nan")
            low_summary = _format_binary_summary(low_group[name])
            other_summary = _format_binary_summary(other_group[name])
            low_nonmissing = int(low_group.shape[0])
            other_nonmissing = int(other_group.shape[0])
        else:
            low_nonmissing = int(pd.to_numeric(low_group[name], errors="coerce").notna().sum())
            other_nonmissing = int(pd.to_numeric(other_group[name], errors="coerce").notna().sum())
            assessable = (
                low_nonmissing >= CONTINUOUS_MIN_PER_ARM and other_nonmissing >= CONTINUOUS_MIN_PER_ARM
            )
            group_effect, _ = _continuous_standardized_difference(low_group[name], other_group[name])
            if not assessable:
                group_effect = float("nan")
            low_summary = _format_continuous_summary(low_group[name])
            other_summary = _format_continuous_summary(other_group[name])

        rows.append(
            {
                "disease_group": disease_group,
                "variable": name,
                "variable_label": label,
                "low_predicted_fatal_value_summary": low_summary,
                "other_fatal_value_summary": other_summary,
                "effect_size_contrast_summary": _summarize_effect_size(group_effect),
                "direction_classification": _classify_direction(
                    pooled_effect=pooled_reference[name],
                    group_effect=group_effect,
                    assessable=assessable,
                ),
                "pooled_standardized_difference": pooled_reference[name],
                "group_standardized_difference": group_effect,
                "low_predicted_fatal_nonmissing": low_nonmissing,
                "other_fatal_nonmissing": other_nonmissing,
                "assessable": assessable,
            }
        )

    frame = pd.DataFrame(rows)
    frame["pooled_standardized_difference"] = frame["pooled_standardized_difference"].round(3)
    frame["group_standardized_difference"] = frame["group_standardized_difference"].round(3)
    return frame


def _assign_adequacy_flag(
    *,
    fatal_stays: int,
    low_predicted_fatal_stays: int,
    assessable_variable_count: int,
) -> str:
    if (
        fatal_stays >= ADEQUATE_FATAL_MIN
        and low_predicted_fatal_stays >= ADEQUATE_LOW_MIN
        and assessable_variable_count >= MIN_ASSESSABLE_FOR_ADEQUATE
    ):
        return "adequate"
    if (
        fatal_stays >= BORDERLINE_FATAL_MIN
        and low_predicted_fatal_stays >= BORDERLINE_LOW_MIN
        and assessable_variable_count >= MIN_ASSESSABLE_FOR_BORDERLINE
    ):
        return "borderline"
    return "inadequate"


def _assign_direction_summary(group_panel: pd.DataFrame) -> str:
    assessable = group_panel[group_panel["direction_classification"].ne("not_assessable")].copy()
    if assessable.empty or int(assessable.shape[0]) < 2:
        return "unclear"

    counts = assessable["direction_classification"].value_counts()
    same = int(counts.get("same_direction", 0))
    opposite = int(counts.get("opposite_direction", 0))
    near_null = int(counts.get("near_null_or_unclear", 0))

    if opposite >= 2:
        return "discordant"
    if same >= 2 and opposite == 0:
        return "concordant"
    if same == 0 and near_null == int(assessable.shape[0]):
        return "unclear"
    return "mixed"


def _assign_strength_summary(
    *,
    adequacy_flag: str,
    hard_case_share: float,
    pooled_share: float,
    direction_summary: str,
    mean_assessable_abs_effect: float,
) -> str:
    if adequacy_flag == "inadequate":
        return "unstable"
    if (
        adequacy_flag == "adequate"
        and hard_case_share >= max(0.15, pooled_share * 0.75)
        and direction_summary == "concordant"
        and np.isfinite(mean_assessable_abs_effect)
        and mean_assessable_abs_effect >= 0.35
    ):
        return "clear"
    if hard_case_share < 0.12 and (
        not np.isfinite(mean_assessable_abs_effect) or mean_assessable_abs_effect < 0.20
    ):
        return "minimal"
    return "modest"


def _build_issue_note(
    *,
    adequacy_flag: str,
    fatal_stays: int,
    low_predicted_fatal_stays: int,
    hard_case_share: float,
    assessable_variable_count: int,
) -> str:
    share_text = _format_pct(hard_case_share)
    if adequacy_flag == "adequate":
        return (
            f"Fatal {fatal_stays}, low-predicted fatal {low_predicted_fatal_stays}, hard-case share {share_text}; "
            f"{assessable_variable_count}/4 panel variables assessable."
        )
    if adequacy_flag == "borderline":
        return (
            f"Fatal {fatal_stays}, low-predicted fatal {low_predicted_fatal_stays}, hard-case share {share_text}; "
            "descriptive-only subgroup because counts are limited."
        )
    return (
        f"Fatal {fatal_stays}, low-predicted fatal {low_predicted_fatal_stays}, hard-case share {share_text}; "
        "counts too sparse for substantive subgroup interpretation."
    )


def build_disease_stratified_outputs(
    *,
    comparison_dataset: pd.DataFrame,
    disease_group_source: DiseaseGroupSource,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled_reference = _build_pooled_reference(comparison_dataset)
    pooled_share = float(comparison_dataset["hard_case_flag"].mean())

    contrast_frames: list[pd.DataFrame] = []
    main_rows: list[dict[str, object]] = []
    disease_group_summary = disease_group_source.group_counts.copy()

    for disease_group in FROZEN_DISEASE_GROUP_HIERARCHY:
        group_dataset = comparison_dataset[comparison_dataset["disease_group"].eq(disease_group)].copy()
        group_panel = _build_contrast_row(
            disease_group=disease_group,
            group_dataset=group_dataset,
            pooled_reference=pooled_reference,
        )
        contrast_frames.append(group_panel)

        fatal_stays = int(group_dataset.shape[0])
        low_predicted_fatal_stays = int(group_dataset["hard_case_flag"].sum())
        hard_case_share = (
            float(low_predicted_fatal_stays / fatal_stays) if fatal_stays else np.nan
        )
        assessable_variable_count = int(group_panel["assessable"].sum())
        adequacy_flag = _assign_adequacy_flag(
            fatal_stays=fatal_stays,
            low_predicted_fatal_stays=low_predicted_fatal_stays,
            assessable_variable_count=assessable_variable_count,
        )
        direction_summary = _assign_direction_summary(group_panel)
        assessable_effects = group_panel.loc[group_panel["assessable"], "group_standardized_difference"]
        mean_assessable_abs_effect = float(
            pd.to_numeric(assessable_effects, errors="coerce").abs().mean()
        ) if assessable_variable_count else float("nan")
        strength_summary = _assign_strength_summary(
            adequacy_flag=adequacy_flag,
            hard_case_share=hard_case_share,
            pooled_share=pooled_share,
            direction_summary=direction_summary,
            mean_assessable_abs_effect=mean_assessable_abs_effect,
        )

        total_row = disease_group_summary[disease_group_summary["disease_group"].eq(disease_group)]
        total_stays = int(total_row["total_stays"].iloc[0]) if not total_row.empty else 0
        main_rows.append(
            {
                "disease_group": disease_group,
                "total_stays": total_stays,
                "fatal_stays": fatal_stays,
                "low_predicted_fatal_stays": low_predicted_fatal_stays,
                "hard_case_share_among_fatal": round(hard_case_share, 3) if np.isfinite(hard_case_share) else np.nan,
                "adequacy_flag": adequacy_flag,
                "direction_summary": direction_summary,
                "strength_summary": strength_summary,
                "note": _build_issue_note(
                    adequacy_flag=adequacy_flag,
                    fatal_stays=fatal_stays,
                    low_predicted_fatal_stays=low_predicted_fatal_stays,
                    hard_case_share=hard_case_share,
                    assessable_variable_count=assessable_variable_count,
                ),
            }
        )

    contrast_panel = pd.concat(contrast_frames, ignore_index=True)
    contrast_panel["variable"] = pd.Categorical(
        contrast_panel["variable"],
        categories=[spec["name"] for spec in VARIABLE_SPECS],
        ordered=True,
    )
    contrast_panel = contrast_panel.sort_values(
        ["disease_group", "variable"],
        kind="stable",
    ).reset_index(drop=True)

    main_summary = pd.DataFrame(main_rows)
    main_summary["disease_group"] = pd.Categorical(
        main_summary["disease_group"],
        categories=list(FROZEN_DISEASE_GROUP_HIERARCHY),
        ordered=True,
    )
    main_summary = main_summary.sort_values("disease_group", kind="stable").reset_index(drop=True)
    return disease_group_summary, main_summary, contrast_panel


def _build_assignment_qc_note(
    *,
    comparison_dataset_path: Path,
    hard_case_path: Path,
    disease_group_source: DiseaseGroupSource,
    qc_summary: dict[str, object],
    disease_group_summary: pd.DataFrame,
    main_summary: pd.DataFrame,
) -> str:
    duplicate_keys = int(disease_group_source.assignments.duplicated(subset=KEY_COLUMNS).sum())
    empty_groups = qc_summary["empty_groups_among_fatal_anchor"]
    dominant_group = str(qc_summary["dominant_fatal_group"])
    dominant_share = float(qc_summary["dominant_fatal_group_share"])

    lines = [
        "# ASIC Disease-Group Assignment QC",
        "",
        "## Inputs",
        "",
        f"- Hard-case anchor: `{hard_case_path}` filtered to the frozen 24h fatal logistic slice.",
        f"- Comparison dataset: `{comparison_dataset_path}` reused for the fixed contrast panel.",
        f"- Disease-group source mode: `{disease_group_source.source_mode}`.",
        f"- Disease-group source path: `{disease_group_source.source_path}`.",
        f"- Source note: {disease_group_source.source_note}",
        "",
        "## Merge Checks",
        "",
        f"- Fatal anchor rows: `{qc_summary['fatal_anchor_rows']}`.",
        f"- Saved comparison dataset rows: `{qc_summary['comparison_dataset_rows']}`.",
        f"- Disease-group assignment rows: `{qc_summary['disease_assignment_rows']}`.",
        f"- Duplicate disease-group stay keys: `{duplicate_keys}`.",
        "- Unmatched fatal anchor rows after disease-group merge: `0`.",
        f"- Empty disease groups among fatal stays: `{', '.join(empty_groups) if empty_groups else 'none'}`.",
        "",
        "## Count Review",
        "",
        _markdown_table(disease_group_summary),
        "",
        _markdown_table(
            main_summary.loc[
                :,
                [
                    "disease_group",
                    "fatal_stays",
                    "low_predicted_fatal_stays",
                    "hard_case_share_among_fatal",
                ],
            ]
        ),
        "",
        "## Interpretation Boundary",
        "",
        (
            f"- No mechanical merge failures were detected. The fatal anchor remains front-loaded toward "
            f"`{dominant_group}` ({100.0 * dominant_share:.1f}% of fatal stays), which is not a parser error "
            "but does limit how much weight the smaller disease strata can carry."
        ),
        "- Disease groups are ICD-10-derived hierarchy-based proxy strata only, not etiologic truth.",
        "",
    ]
    return "\n".join(lines)


def _plot_hardcase_share(main_summary: pd.DataFrame, *, output_path: Path) -> Path:
    _require_matplotlib()
    ensure_directory(output_path.parent)

    plot_frame = main_summary.copy()
    pooled_share = float(plot_frame["low_predicted_fatal_stays"].sum() / plot_frame["fatal_stays"].sum())
    shares = plot_frame["hard_case_share_among_fatal"].astype(float).to_numpy()
    labels = plot_frame["disease_group"].astype(str).tolist()
    y_positions = np.arange(len(labels))

    fig, axis = plt.subplots(figsize=(9.5, 4.8))
    axis.barh(y_positions, shares, color="#2f6c8f", alpha=0.9)
    axis.axvline(pooled_share, color="#b4462a", linestyle="--", linewidth=1.5, label="pooled ASIC")
    axis.set_yticks(y_positions)
    axis.set_yticklabels(labels, fontsize=9)
    axis.set_xlim(0, max(1.0, float(np.nanmax(shares) * 1.12)))
    axis.set_xlabel("Hard-case share among fatal stays", fontsize=10)
    axis.set_title("ASIC 24h hard-case share among fatal stays by disease group", fontsize=12)
    axis.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.5)
    axis.invert_yaxis()

    for index, row in plot_frame.iterrows():
        axis.text(
            float(row["hard_case_share_among_fatal"]) + 0.015,
            y_positions[index],
            f"{int(row['low_predicted_fatal_stays'])}/{int(row['fatal_stays'])}",
            va="center",
            fontsize=8,
        )

    axis.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _assign_final_judgment(main_summary: pd.DataFrame) -> tuple[str, bool]:
    interpretable = main_summary[main_summary["adequacy_flag"].isin(["adequate", "borderline"])].copy()
    adequate = main_summary[main_summary["adequacy_flag"].eq("adequate")].copy()
    if interpretable.shape[0] < 2:
        return "uninterpretable", False

    pooled_share = float(main_summary["low_predicted_fatal_stays"].sum() / main_summary["fatal_stays"].sum())
    visible_share_spread = bool(
        (interpretable["hard_case_share_among_fatal"].astype(float) - pooled_share).abs().ge(VISIBLY_DIFFERENT_SHARE).any()
    )
    major_collapse = bool(
        adequate["hard_case_share_among_fatal"].astype(float).lt(MAJOR_COLLAPSE_SHARE).any()
    ) if not adequate.empty else False
    major_reversal = bool(adequate["direction_summary"].eq("discordant").any()) if not adequate.empty else False
    burden_present_in_interpretable_groups = int(
        interpretable["hard_case_share_among_fatal"].astype(float).ge(0.10).sum()
    )

    if major_collapse and (major_reversal or burden_present_in_interpretable_groups <= 1):
        return "materially_heterogeneous", True
    if major_reversal and burden_present_in_interpretable_groups <= 1:
        return "materially_heterogeneous", True
    if visible_share_spread:
        return "suggestive_heterogeneity", False
    if burden_present_in_interpretable_groups >= max(2, int(interpretable.shape[0]) - 1):
        return "broadly_stable", False
    return "suggestive_heterogeneity", False


def _build_interpretation_memo(
    *,
    main_summary: pd.DataFrame,
    final_judgment: str,
    wording_needs_narrowing: bool,
) -> str:
    interpretable = main_summary[main_summary["adequacy_flag"].isin(["adequate", "borderline"])].copy()
    adequate_groups = main_summary.loc[
        main_summary["adequacy_flag"].eq("adequate"), "disease_group"
    ].astype(str).tolist()
    borderline_groups = main_summary.loc[
        main_summary["adequacy_flag"].eq("borderline"), "disease_group"
    ].astype(str).tolist()
    inadequate_groups = main_summary.loc[
        main_summary["adequacy_flag"].eq("inadequate"), "disease_group"
    ].astype(str).tolist()

    strong_group_source = interpretable if not interpretable.empty else main_summary
    strong_group = strong_group_source.sort_values(
        ["hard_case_share_among_fatal", "fatal_stays"],
        ascending=[False, False],
        kind="stable",
    ).iloc[0]
    pooled_share = float(main_summary["low_predicted_fatal_stays"].sum() / main_summary["fatal_stays"].sum())

    lines = [
        "# ASIC Disease-Stratified Interpretation Memo",
        "",
        f"- Final judgment: `{final_judgment}`.",
        (
            f"- Pooled ASIC hard-case share among fatal stays is `{100.0 * pooled_share:.1f}%`. "
            f"The highest share among interpretable groups is "
            f"`{100.0 * float(strong_group['hard_case_share_among_fatal']):.1f}%` "
            f"in `{strong_group['disease_group']}`."
        ),
        "- Disease-group labels are ICD-10-derived hierarchy-based proxy strata, not biological or etiologic truth.",
        "",
        "## Interpretability",
        "",
        f"- Adequate: `{', '.join(adequate_groups) if adequate_groups else 'none'}`.",
        f"- Borderline: `{', '.join(borderline_groups) if borderline_groups else 'none'}`.",
        f"- Inadequate: `{', '.join(inadequate_groups) if inadequate_groups else 'none'}`.",
        "",
        "## Bottom Line",
        "",
    ]

    if final_judgment == "broadly_stable":
        lines.append(
            "- The hard-case burden remains present across the interpretable disease groups, and the fixed-panel contrasts are mostly directionally aligned with the pooled ASIC pattern where assessable."
        )
    elif final_judgment == "suggestive_heterogeneity":
        lines.append(
            "- The pooled ASIC pattern still appears in more than one interpretable disease group, but subgroup strength is not uniform. Cardiovascular fatal stays show a higher hard-case share than the pooled anchor, while the largest surgical and respiratory groups remain directionally aligned with the pooled panel."
        )
    elif final_judgment == "materially_heterogeneous":
        lines.append(
            "- Disease-stratified results narrow the pooled story enough that broad Chapter 1 wording would be misleading without explicit subgroup qualification."
        )
    else:
        lines.append(
            "- Sparse subgroup counts and proxy-assignment limits prevent meaningful disease-stratified interpretation beyond descriptive counts."
        )

    if wording_needs_narrowing:
        lines.append("- Pooled Chapter 1 wording should be narrowed explicitly when reporting subgroup structure.")
    else:
        lines.append(
            "- Pooled Chapter 1 wording does not need formal narrowing on this issue, but any subgroup mention should stay cautious and note that only surgical, respiratory, and borderline cardiovascular strata are even modestly interpretable."
        )

    lines.extend(
        [
            "- No biological subtype or causal claims are supported here; these are bounded robustness checks on pragmatic proxy strata only.",
            "",
        ]
    )
    return "\n".join(lines)


def run_asic_disease_stratified_predictability_structure(
    *,
    comparison_dataset_path: Path = DEFAULT_COMPARISON_DATASET_PATH,
    hard_case_path: Path = DEFAULT_HARD_CASE_PATH,
    disease_group_path: Path = DEFAULT_DISEASE_GROUP_PATH,
    disease_group_counts_path: Path = DEFAULT_DISEASE_GROUP_COUNTS_PATH,
    asic_input_root: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> ASICDiseaseStratifiedRunResult:
    comparison_dataset = _load_saved_comparison_dataset(comparison_dataset_path)
    fatal_anchor = _load_fatal_hard_case_anchor(hard_case_path)
    disease_group_source = _load_disease_group_source(
        comparison_dataset=comparison_dataset,
        disease_group_path=disease_group_path,
        disease_group_counts_path=disease_group_counts_path,
        asic_input_root=asic_input_root,
    )
    merged_dataset, qc_summary = _build_merged_dataset(
        comparison_dataset=comparison_dataset,
        fatal_anchor=fatal_anchor,
        disease_group_source=disease_group_source,
    )
    disease_group_summary, main_summary, contrast_panel = build_disease_stratified_outputs(
        comparison_dataset=merged_dataset,
        disease_group_source=disease_group_source,
    )
    final_judgment, wording_needs_narrowing = _assign_final_judgment(main_summary)
    assignment_qc_note = _build_assignment_qc_note(
        comparison_dataset_path=_resolve_existing_path(comparison_dataset_path),
        hard_case_path=_resolve_existing_path(hard_case_path),
        disease_group_source=disease_group_source,
        qc_summary=qc_summary,
        disease_group_summary=disease_group_summary,
        main_summary=main_summary,
    )
    interpretation_memo = _build_interpretation_memo(
        main_summary=main_summary,
        final_judgment=final_judgment,
        wording_needs_narrowing=wording_needs_narrowing,
    )

    ensure_directory(output_dir)
    assignment_qc_path = write_text(assignment_qc_note + "\n", output_dir / ASSIGNMENT_QC_FILENAME)
    disease_group_summary_path = write_dataframe(
        disease_group_summary,
        output_dir / GROUP_COUNTS_FILENAME,
        output_format="csv",
    )
    main_summary_path = write_dataframe(
        main_summary,
        output_dir / MAIN_SUMMARY_FILENAME,
        output_format="csv",
    )
    contrast_panel_path = write_dataframe(
        contrast_panel,
        output_dir / CONTRAST_PANEL_FILENAME,
        output_format="csv",
    )
    figure_path = _plot_hardcase_share(
        main_summary,
        output_path=output_dir / FIGURE_FILENAME,
    )
    memo_path = write_text(interpretation_memo + "\n", output_dir / MEMO_FILENAME)
    manifest_path = _write_json(
        {
            "issue_id": "phase1_chapter1_sprint4_issue_4_4",
            "target_population": "ASIC 24h logistic frozen fatal hard-case anchor",
            "hard_case_rule": HARD_CASE_RULE,
            "comparison_dataset_path": str(_resolve_existing_path(comparison_dataset_path)),
            "hard_case_path": str(_resolve_existing_path(hard_case_path)),
            "disease_group_source_mode": disease_group_source.source_mode,
            "disease_group_source_path": str(disease_group_source.source_path),
            "disease_group_counts_path": str(_resolve_existing_path(disease_group_counts_path))
            if Path(disease_group_counts_path).expanduser().exists()
            else None,
            "final_judgment": final_judgment,
            "wording_needs_narrowing": wording_needs_narrowing,
            "timestamp_utc": _utc_timestamp(),
            "artifacts": {
                "assignment_qc_note": str(assignment_qc_path.resolve()),
                "disease_group_summary": str(disease_group_summary_path.resolve()),
                "hardcase_summary": str(main_summary_path.resolve()),
                "contrast_panel": str(contrast_panel_path.resolve()),
                "figure": str(figure_path.resolve()),
                "interpretation_memo": str(memo_path.resolve()),
            },
        },
        output_dir / MANIFEST_FILENAME,
    )

    return ASICDiseaseStratifiedRunResult(
        merged_dataset=merged_dataset,
        disease_group_summary=disease_group_summary,
        hardcase_summary=main_summary,
        contrast_panel=contrast_panel,
        final_judgment=final_judgment,
        wording_needs_narrowing=wording_needs_narrowing,
        artifacts=ASICDiseaseStratifiedArtifacts(
            assignment_qc_path=assignment_qc_path,
            disease_group_summary_path=disease_group_summary_path,
            hardcase_summary_path=main_summary_path,
            contrast_panel_path=contrast_panel_path,
            figure_path=figure_path,
            memo_path=memo_path,
            manifest_path=manifest_path,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ASIC disease-stratified predictability-structure robustness analysis "
            "anchored to the frozen 24h logistic hard-case bundle."
        )
    )
    parser.add_argument(
        "--comparison-dataset-path",
        type=Path,
        default=DEFAULT_COMPARISON_DATASET_PATH,
        help="Saved frozen ASIC 24h hard-case comparison dataset.",
    )
    parser.add_argument(
        "--hard-case-path",
        type=Path,
        default=DEFAULT_HARD_CASE_PATH,
        help="Saved frozen stay-level hard-case flags.",
    )
    parser.add_argument(
        "--disease-group-path",
        type=Path,
        default=DEFAULT_DISEASE_GROUP_PATH,
        help=(
            "Saved stay-level ICD-10 disease-group assignment file. If unavailable locally, "
            "the analysis falls back to authoritative ASIC static harmonized codes."
        ),
    )
    parser.add_argument(
        "--disease-group-counts-path",
        type=Path,
        default=DEFAULT_DISEASE_GROUP_COUNTS_PATH,
        help=(
            "Frozen disease-group count summary used for cohort-level totals when the stay-level "
            "disease-group file is not available locally."
        ),
    )
    parser.add_argument(
        "--asic-input-root",
        type=Path,
        help="Optional override for the authoritative local ASIC harmonized artifact root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where Issue 4.4 artifacts will be written.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_asic_disease_stratified_predictability_structure(
        comparison_dataset_path=args.comparison_dataset_path,
        hard_case_path=args.hard_case_path,
        disease_group_path=args.disease_group_path,
        disease_group_counts_path=args.disease_group_counts_path,
        asic_input_root=args.asic_input_root,
        output_dir=args.output_dir,
    )
    print(f"Final judgment: {result.final_judgment}")
    print(f"Assignment QC: {result.artifacts.assignment_qc_path}")
    print(f"Hard-case summary: {result.artifacts.hardcase_summary_path}")
    print(f"Contrast panel: {result.artifacts.contrast_panel_path}")
    print(f"Figure: {result.artifacts.figure_path}")
    print(f"Memo: {result.artifacts.memo_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
