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

from chapter1_mortality_decomposition.hard_case_definition import (
    DEFAULT_HARD_CASE_OUTPUT_DIR,
    HARD_CASE_RULE,
)
from chapter1_mortality_decomposition.icd10_disease_groups import (
    FROZEN_DISEASE_GROUP_HIERARCHY,
    derive_icd10_disease_group,
)
from chapter1_mortality_decomposition.utils import (
    normalize_boolean_codes,
    read_dataframe,
    require_columns,
    write_dataframe,
    write_text,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_CONFIG_PATH = REPO_ROOT / "config" / "ch1_run_config.json"
DEFAULT_HARD_CASE_PATH = DEFAULT_HARD_CASE_OUTPUT_DIR / "stay_level_hard_case_flags.csv"
DEFAULT_MODEL_READY_PATH = (
    REPO_ROOT / "artifacts" / "chapter1" / "model_ready" / "chapter1_primary_model_ready_dataset.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_HARD_CASE_OUTPUT_DIR / "issue_3_2_asic_hardcase_comparison"
STATIC_RELATIVE_PATH = Path("static") / "harmonized.csv"

ISSUE_ID = "phase1_chapter1_sprint3_issue_3_2"
TARGET_HORIZON_H = 24
LOW_PREDICTED_FATAL_GROUP = "low-predicted fatal stays"
OTHER_FATAL_GROUP = "other fatal stays"
CSV_CHUNKSIZE = 100_000

MERGE_KEYS = [
    "instance_id",
    "stay_id_global",
    "hospital_id",
    "block_index",
    "prediction_time_h",
    "horizon_h",
    "label_value",
]
STATIC_JOIN_KEYS = ["stay_id_global", "hospital_id"]
STRING_KEY_COLUMNS = {"instance_id", "stay_id_global", "hospital_id"}
NUMERIC_KEY_COLUMNS = {"block_index", "prediction_time_h", "horizon_h", "label_value"}

COMPARISON_DATASET_COLUMNS = [
    "stay_id_global",
    "instance_id",
    "hard_case_flag",
    "hard_case_group",
    "age_group",
    "sex",
    "disease_group",
    "prediction_time_h",
    "hospital_id",
    "pf_ratio_last",
    "map_last",
    "creatinine_last",
    "peep_last",
]

CATEGORICAL_VARIABLES = ("age_group", "sex", "disease_group", "hospital_id")
CONTINUOUS_VARIABLES = (
    "prediction_time_h",
    "pf_ratio_last",
    "map_last",
    "creatinine_last",
    "peep_last",
)
VARIABLE_ORDER = [*CATEGORICAL_VARIABLES, *CONTINUOUS_VARIABLES]

VARIABLE_LABELS = {
    "age_group": "Age group",
    "sex": "Sex",
    "disease_group": "Disease group",
    "hospital_id": "Hospital",
    "prediction_time_h": "Prediction time from ICU admission (h)",
    "pf_ratio_last": "PF ratio",
    "map_last": "MAP",
    "creatinine_last": "Creatinine",
    "peep_last": "PEEP",
}
SUMMARY_LABELS = {
    "prediction_time_h": "prediction time",
    "pf_ratio_last": "PF ratio",
    "map_last": "MAP",
    "creatinine_last": "creatinine",
    "peep_last": "PEEP",
}
ORDERED_CATEGORICAL_LEVELS = {
    "age_group": ("<70", "70-79", "80-130"),
    "sex": ("F", "M"),
    "disease_group": FROZEN_DISEASE_GROUP_HIERARCHY,
}


@dataclass(frozen=True)
class ASICHardCaseComparisonArtifacts:
    comparison_dataset_path: Path
    comparison_table_path: Path
    effect_size_plot_data_path: Path
    standardized_difference_details_path: Path
    figure_path: Path
    summary_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class ASICHardCaseComparisonRunResult:
    output_dir: Path
    artifacts: ASICHardCaseComparisonArtifacts
    comparison_dataset: pd.DataFrame
    comparison_table: pd.DataFrame
    effect_size_plot_data: pd.DataFrame
    standardized_difference_details: pd.DataFrame
    summary_markdown: str


def _require_matplotlib() -> None:
    if plt is None:
        raise ImportError(
            "matplotlib is required for the Chapter 1 ASIC hard-case comparison."
        ) from MATPLOTLIB_IMPORT_ERROR


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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

    if DEFAULT_RUN_CONFIG_PATH.exists():
        run_config = json.loads(DEFAULT_RUN_CONFIG_PATH.read_text())
        input_dir = run_config.get("input_dir")
        if input_dir:
            candidates.append(Path(str(input_dir)).expanduser())

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

    candidates = _candidate_asic_input_roots()
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate the local ASIC harmonized artifacts. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _normalize_key_columns(frame: pd.DataFrame, *, key_columns: Sequence[str]) -> pd.DataFrame:
    normalized = frame.copy()
    for column in key_columns:
        if column in STRING_KEY_COLUMNS:
            normalized[column] = normalized[column].astype("string")
        elif column in NUMERIC_KEY_COLUMNS:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


def _filter_table_to_target(
    path: Path,
    *,
    usecols: Sequence[str],
    key_columns: Sequence[str],
    target_keys: pd.DataFrame,
) -> pd.DataFrame:
    resolved_path = _resolve_existing_path(path)
    deduplicated_usecols = list(dict.fromkeys(usecols))
    normalized_target_keys = _normalize_key_columns(
        target_keys.loc[:, key_columns].drop_duplicates().reset_index(drop=True),
        key_columns=key_columns,
    )
    target_index = pd.MultiIndex.from_frame(normalized_target_keys)

    if resolved_path.suffix == ".csv":
        parts: list[pd.DataFrame] = []
        for chunk in pd.read_csv(
            resolved_path,
            usecols=deduplicated_usecols,
            chunksize=CSV_CHUNKSIZE,
        ):
            normalized_chunk = _normalize_key_columns(chunk, key_columns=key_columns)
            mask = pd.MultiIndex.from_frame(normalized_chunk.loc[:, key_columns]).isin(target_index)
            if bool(mask.any()):
                parts.append(normalized_chunk.loc[mask].copy())
        if parts:
            return pd.concat(parts, ignore_index=True)
        return pd.read_csv(resolved_path, usecols=deduplicated_usecols, nrows=0)

    if resolved_path.suffix == ".parquet":
        table = pd.read_parquet(resolved_path, columns=deduplicated_usecols)
        normalized_table = _normalize_key_columns(table, key_columns=key_columns)
        mask = pd.MultiIndex.from_frame(normalized_table.loc[:, key_columns]).isin(target_index)
        return normalized_table.loc[mask].reset_index(drop=True)

    return read_dataframe(resolved_path).loc[:, deduplicated_usecols]


def _ordered_levels(series: pd.Series, *, variable_name: str) -> list[str]:
    base_order = ORDERED_CATEGORICAL_LEVELS.get(variable_name)
    observed_values = [
        str(value)
        for value in series.dropna().astype("string").tolist()
        if str(value).strip()
    ]
    observed_set = set(observed_values)
    levels: list[str] = []

    if base_order is not None:
        levels.extend([level for level in base_order if level in observed_set])
        levels.extend(sorted(observed_set - set(levels)))
    else:
        levels.extend(sorted(observed_set))

    if int(series.isna().sum()) > 0:
        levels.append("Missing")
    return levels


def _categorical_level_mask(series: pd.Series, level: str) -> pd.Series:
    if level == "Missing":
        return series.isna()
    return series.astype("string").eq(level)


def _format_count_pct(count: int, total: int) -> str:
    pct = (100.0 * count / total) if total else 0.0
    return f"{count} ({pct:.0f}%)"


def _format_numeric_value(value: float, *, variable_name: str) -> str:
    if pd.isna(value):
        return "NA"
    if variable_name == "prediction_time_h":
        return f"{float(value):.0f}"
    return f"{float(value):.1f}"


def _format_continuous_summary(series: pd.Series, *, variable_name: str) -> str:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return "NA (n=0)"

    q1 = float(numeric.quantile(0.25))
    median = float(numeric.quantile(0.50))
    q3 = float(numeric.quantile(0.75))
    return (
        f"{_format_numeric_value(median, variable_name=variable_name)} "
        f"[{_format_numeric_value(q1, variable_name=variable_name)}, "
        f"{_format_numeric_value(q3, variable_name=variable_name)}] "
        f"(n={int(numeric.shape[0])})"
    )


def _continuous_standardized_difference(
    hard_values: pd.Series,
    other_values: pd.Series,
) -> tuple[float, dict[str, float | int]]:
    hard_numeric = pd.to_numeric(hard_values, errors="coerce").dropna()
    other_numeric = pd.to_numeric(other_values, errors="coerce").dropna()

    details: dict[str, float | int] = {
        "hard_nonmissing": int(hard_numeric.shape[0]),
        "other_nonmissing": int(other_numeric.shape[0]),
    }
    if hard_numeric.empty or other_numeric.empty:
        return float("nan"), details

    hard_mean = float(hard_numeric.mean())
    other_mean = float(other_numeric.mean())
    hard_variance = float(hard_numeric.var(ddof=0))
    other_variance = float(other_numeric.var(ddof=0))
    pooled_sd = float(np.sqrt((hard_variance + other_variance) / 2.0))

    details.update(
        {
            "hard_mean": hard_mean,
            "other_mean": other_mean,
            "hard_sd": float(np.sqrt(hard_variance)),
            "other_sd": float(np.sqrt(other_variance)),
            "pooled_sd": pooled_sd,
        }
    )
    if pooled_sd == 0.0:
        if np.isclose(hard_mean, other_mean):
            return 0.0, details
        return float("nan"), details
    return (hard_mean - other_mean) / pooled_sd, details


def _proportion_standardized_difference(hard_fraction: float, other_fraction: float) -> float:
    pooled_sd = float(
        np.sqrt(
            (
                hard_fraction * (1.0 - hard_fraction)
                + other_fraction * (1.0 - other_fraction)
            )
            / 2.0
        )
    )
    if pooled_sd == 0.0:
        if np.isclose(hard_fraction, other_fraction):
            return 0.0
        return float("nan")
    return (hard_fraction - other_fraction) / pooled_sd


def build_stay_level_comparison_dataset(
    *,
    hard_case_path: Path = DEFAULT_HARD_CASE_PATH,
    model_ready_path: Path = DEFAULT_MODEL_READY_PATH,
    asic_input_root: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    resolved_hard_case_path = _resolve_existing_path(hard_case_path)
    resolved_model_ready_path = _resolve_existing_path(model_ready_path)
    resolved_input_root = _resolve_asic_input_root(asic_input_root)
    resolved_static_path = _resolve_existing_path(resolved_input_root / STATIC_RELATIVE_PATH)

    hard_case_source = read_dataframe(resolved_hard_case_path)
    require_columns(
        hard_case_source,
        set(MERGE_KEYS) | {"hard_case_flag", "hard_case_rule"},
        "saved stay-level hard-case artifact",
    )
    hard_case_source = _normalize_key_columns(hard_case_source, key_columns=MERGE_KEYS)
    hard_case_source["hard_case_flag"] = normalize_boolean_codes(hard_case_source["hard_case_flag"])
    hard_case_slice = hard_case_source[
        hard_case_source["horizon_h"].eq(TARGET_HORIZON_H)
        & hard_case_source["label_value"].eq(1)
    ].copy()
    if hard_case_slice.empty:
        raise ValueError("The frozen 24h fatal hard-case target population is empty.")
    if not hard_case_slice["hard_case_rule"].astype("string").eq(HARD_CASE_RULE).all():
        raise ValueError(
            "The saved stay-level hard-case artifact does not match the frozen logistic hard-case rule."
        )

    duplicate_target_rows = int(hard_case_slice.duplicated(subset=MERGE_KEYS).sum())
    if duplicate_target_rows:
        raise ValueError(
            f"The saved stay-level hard-case artifact contains {duplicate_target_rows} duplicated target rows."
        )

    model_ready = _filter_table_to_target(
        resolved_model_ready_path,
        usecols=[
            *MERGE_KEYS,
            "pf_ratio_last",
            "map_last",
            "creatinine_last",
            "peep_last",
        ],
        key_columns=MERGE_KEYS,
        target_keys=hard_case_slice.loc[:, MERGE_KEYS],
    )
    model_ready = _normalize_key_columns(model_ready, key_columns=MERGE_KEYS)
    merged = hard_case_slice.merge(
        model_ready,
        on=MERGE_KEYS,
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError(
            "Some 24h fatal hard-case rows could not be linked to the Chapter 1 model-ready dataset."
        )
    merged = merged.drop(columns="_merge")

    static = _filter_table_to_target(
        resolved_static_path,
        usecols=[*STATIC_JOIN_KEYS, "age_group", "sex", "icd10_codes"],
        key_columns=STATIC_JOIN_KEYS,
        target_keys=hard_case_slice.loc[:, STATIC_JOIN_KEYS],
    )
    static = _normalize_key_columns(static, key_columns=STATIC_JOIN_KEYS)
    static_duplicate_rows = int(static.duplicated(subset=STATIC_JOIN_KEYS).sum())
    if static_duplicate_rows:
        raise ValueError(
            f"The ASIC static harmonized table contains {static_duplicate_rows} duplicated stay rows."
        )

    merged = merged.merge(
        static,
        on=STATIC_JOIN_KEYS,
        how="left",
        indicator=True,
        validate="one_to_one",
        suffixes=("", "_static"),
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError(
            "Some 24h fatal hard-case rows could not be linked to the ASIC static harmonized table."
        )
    merged = merged.drop(columns="_merge")

    merged["disease_group"] = merged["icd10_codes"].map(
        lambda raw_codes: derive_icd10_disease_group(raw_codes).final_group
    )
    merged["hard_case_flag"] = merged["hard_case_flag"].astype(bool)
    merged["hard_case_group"] = np.where(
        merged["hard_case_flag"],
        LOW_PREDICTED_FATAL_GROUP,
        OTHER_FATAL_GROUP,
    )

    comparison_dataset = merged.loc[:, COMPARISON_DATASET_COLUMNS].copy()
    comparison_dataset = comparison_dataset.sort_values(
        ["hard_case_flag", "hospital_id", "stay_id_global"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)

    join_metadata = {
        "issue_id": ISSUE_ID,
        "target_horizon_h": TARGET_HORIZON_H,
        "target_population_definition": (
            "ASIC only, 24h horizon only, fatal stays only, one last eligible prediction instance per stay"
        ),
        "hard_case_rule": HARD_CASE_RULE,
        "source_paths": {
            "hard_case_path": resolved_hard_case_path,
            "model_ready_path": resolved_model_ready_path,
            "asic_input_root": resolved_input_root,
            "static_path": resolved_static_path,
        },
        "join_logic": [
            {
                "anchor": "saved stay-level hard-case artifact",
                "path": resolved_hard_case_path,
                "filter": "horizon_h == 24 and label_value == 1",
                "selected_columns": [*MERGE_KEYS, "hard_case_flag", "hard_case_rule"],
            },
            {
                "path": resolved_model_ready_path,
                "join_type": "left",
                "join_keys": MERGE_KEYS,
                "selected_columns": [*MERGE_KEYS, "pf_ratio_last", "map_last", "creatinine_last", "peep_last"],
                "validation": "one_to_one",
            },
            {
                "path": resolved_static_path,
                "join_type": "left",
                "join_keys": STATIC_JOIN_KEYS,
                "selected_columns": [*STATIC_JOIN_KEYS, "age_group", "sex", "icd10_codes"],
                "derived_columns": ["disease_group = derive_icd10_disease_group(icd10_codes).final_group"],
                "validation": "one_to_one",
            },
        ],
    }
    return comparison_dataset, join_metadata


def build_comparison_outputs(
    comparison_dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hard_mask = comparison_dataset["hard_case_flag"].astype(bool)
    hard_group = comparison_dataset.loc[hard_mask].copy()
    other_group = comparison_dataset.loc[~hard_mask].copy()
    n_hard = int(hard_group.shape[0])
    n_other = int(other_group.shape[0])
    if n_hard == 0 or n_other == 0:
        raise ValueError("Both hard-case groups must be non-empty for the comparison output.")

    comparison_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for variable_name in VARIABLE_ORDER:
        if variable_name in CATEGORICAL_VARIABLES:
            levels = _ordered_levels(comparison_dataset[variable_name], variable_name=variable_name)
            hard_summary_parts: list[str] = []
            other_summary_parts: list[str] = []
            variable_level_details: list[dict[str, object]] = []

            for level in levels:
                hard_count = int(_categorical_level_mask(hard_group[variable_name], level).sum())
                other_count = int(_categorical_level_mask(other_group[variable_name], level).sum())
                hard_fraction = float(hard_count / n_hard)
                other_fraction = float(other_count / n_other)
                standardized_difference = _proportion_standardized_difference(
                    hard_fraction,
                    other_fraction,
                )
                detail_row = {
                    "variable": variable_name,
                    "variable_label": VARIABLE_LABELS[variable_name],
                    "variable_kind": "categorical",
                    "level": level,
                    "hard_count": hard_count,
                    "hard_total": n_hard,
                    "hard_fraction": hard_fraction,
                    "other_count": other_count,
                    "other_total": n_other,
                    "other_fraction": other_fraction,
                    "standardized_difference": standardized_difference,
                    "abs_standardized_difference": abs(standardized_difference)
                    if np.isfinite(standardized_difference)
                    else np.nan,
                }
                variable_level_details.append(detail_row)
                detail_rows.append(detail_row)
                hard_summary_parts.append(f"{level}: {_format_count_pct(hard_count, n_hard)}")
                other_summary_parts.append(f"{level}: {_format_count_pct(other_count, n_other)}")

            variable_level_details_frame = pd.DataFrame(variable_level_details)
            variable_level_details_frame = variable_level_details_frame.sort_values(
                ["abs_standardized_difference", "level"],
                ascending=[False, True],
                na_position="last",
                kind="stable",
            ).reset_index(drop=True)
            dominant_row = variable_level_details_frame.iloc[0]

            comparison_rows.append(
                {
                    "variable": variable_name,
                    "variable_label": VARIABLE_LABELS[variable_name],
                    "low_predicted_fatal_stays": "; ".join(hard_summary_parts),
                    "other_fatal_stays": "; ".join(other_summary_parts),
                    "effect_size_type": "max absolute level-specific standardized difference",
                    "effect_size_basis": dominant_row["level"],
                    "standardized_difference": dominant_row["standardized_difference"],
                    "absolute_standardized_difference": dominant_row["abs_standardized_difference"],
                }
            )
            continue

        standardized_difference, details = _continuous_standardized_difference(
            hard_group[variable_name],
            other_group[variable_name],
        )
        detail_rows.append(
            {
                "variable": variable_name,
                "variable_label": VARIABLE_LABELS[variable_name],
                "variable_kind": "continuous",
                "level": pd.NA,
                "hard_count": details.get("hard_nonmissing"),
                "hard_total": n_hard,
                "hard_fraction": (
                    float(details["hard_nonmissing"]) / n_hard if n_hard else np.nan
                ),
                "other_count": details.get("other_nonmissing"),
                "other_total": n_other,
                "other_fraction": (
                    float(details["other_nonmissing"]) / n_other if n_other else np.nan
                ),
                "hard_mean": details.get("hard_mean"),
                "other_mean": details.get("other_mean"),
                "hard_sd": details.get("hard_sd"),
                "other_sd": details.get("other_sd"),
                "pooled_sd": details.get("pooled_sd"),
                "standardized_difference": standardized_difference,
                "abs_standardized_difference": abs(standardized_difference)
                if np.isfinite(standardized_difference)
                else np.nan,
            }
        )
        comparison_rows.append(
            {
                "variable": variable_name,
                "variable_label": VARIABLE_LABELS[variable_name],
                "low_predicted_fatal_stays": _format_continuous_summary(
                    hard_group[variable_name],
                    variable_name=variable_name,
                ),
                "other_fatal_stays": _format_continuous_summary(
                    other_group[variable_name],
                    variable_name=variable_name,
                ),
                "effect_size_type": "continuous pooled-SD standardized mean difference",
                "effect_size_basis": "available values",
                "standardized_difference": standardized_difference,
                "absolute_standardized_difference": abs(standardized_difference)
                if np.isfinite(standardized_difference)
                else np.nan,
            }
        )

    comparison_table = pd.DataFrame(comparison_rows)
    comparison_table["variable"] = pd.Categorical(
        comparison_table["variable"],
        categories=VARIABLE_ORDER,
        ordered=True,
    )
    comparison_table = comparison_table.sort_values("variable", kind="stable").reset_index(drop=True)
    comparison_table["figure_label"] = comparison_table.apply(
        lambda row: (
            row["variable_label"]
            if row["variable"] in CONTINUOUS_VARIABLES
            else f"{row['variable_label']} ({row['effect_size_basis']})"
        ),
        axis=1,
    )

    effect_size_plot_data = comparison_table.loc[
        :,
        [
            "variable",
            "variable_label",
            "figure_label",
            "effect_size_type",
            "effect_size_basis",
            "standardized_difference",
            "absolute_standardized_difference",
        ],
    ].copy()
    effect_size_plot_data = effect_size_plot_data.sort_values(
        "absolute_standardized_difference",
        ascending=False,
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)

    standardized_difference_details = pd.DataFrame(detail_rows)
    standardized_difference_details["variable"] = pd.Categorical(
        standardized_difference_details["variable"],
        categories=VARIABLE_ORDER,
        ordered=True,
    )
    standardized_difference_details = standardized_difference_details.sort_values(
        ["variable", "variable_kind", "level"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    numeric_table_columns = ["standardized_difference", "absolute_standardized_difference"]
    comparison_table.loc[:, numeric_table_columns] = comparison_table.loc[:, numeric_table_columns].round(3)
    effect_size_plot_data.loc[:, numeric_table_columns] = (
        effect_size_plot_data.loc[:, numeric_table_columns].round(3)
    )
    standardized_difference_details.loc[
        :,
        [column for column in standardized_difference_details.columns if "difference" in str(column) or "mean" in str(column) or column.endswith("_sd") or column == "pooled_sd"],
    ] = standardized_difference_details.loc[
        :,
        [column for column in standardized_difference_details.columns if "difference" in str(column) or "mean" in str(column) or column.endswith("_sd") or column == "pooled_sd"],
    ].round(6)

    return comparison_table, effect_size_plot_data, standardized_difference_details


def _modifier_from_abs_effect_size(abs_effect_size: float) -> str:
    if abs_effect_size >= 0.8:
        return ""
    return "modestly "


def build_summary_markdown(
    comparison_dataset: pd.DataFrame,
    comparison_table: pd.DataFrame,
    standardized_difference_details: pd.DataFrame,
) -> str:
    n_hard = int(comparison_dataset["hard_case_flag"].astype(bool).sum())
    n_total = int(comparison_dataset.shape[0])
    n_other = n_total - n_hard

    detail_lookup = standardized_difference_details.copy()
    detail_lookup["variable"] = detail_lookup["variable"].astype("string")
    comparison_lookup = comparison_table.copy()
    comparison_lookup["variable"] = comparison_lookup["variable"].astype("string")

    summary_lines = [
        "# ASIC 24h Fatal-Stay Hard-Case Comparison",
        "",
        "## Cohort",
        f"- Fatal 24h stay-level comparison dataset: `{n_total}` stays.",
        f"- Low-predicted fatal stays: `{n_hard}`. Other fatal stays: `{n_other}`.",
        f"- Hard-case anchor: `{HARD_CASE_RULE}` from the saved stay-level hard-case artifact.",
        "",
        "## Main Differences",
    ]

    hospital_details = detail_lookup[detail_lookup["variable"].eq("hospital_id")].copy()
    hospital_details = hospital_details.sort_values(
        "abs_standardized_difference",
        ascending=False,
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    if not hospital_details.empty and pd.notna(hospital_details.at[0, "abs_standardized_difference"]):
        top_hospital = hospital_details.iloc[0]
        hospital_modifier = _modifier_from_abs_effect_size(float(top_hospital["abs_standardized_difference"]))
        if float(top_hospital["standardized_difference"]) >= 0:
            hospital_text = (
                f"- Low-predicted fatal stays were {hospital_modifier}more common among "
                f"`{top_hospital['level']}` fatal stays "
                f"({int(top_hospital['hard_count'])}/{int(top_hospital['hard_total'])}, "
                f"{100.0 * float(top_hospital['hard_fraction']):.0f}%) than among other fatal stays "
                f"({int(top_hospital['other_count'])}/{int(top_hospital['other_total'])}, "
                f"{100.0 * float(top_hospital['other_fraction']):.0f}%)."
            )
        else:
            hospital_text = (
                f"- Low-predicted fatal stays were {hospital_modifier}less common among "
                f"`{top_hospital['level']}` fatal stays "
                f"({int(top_hospital['hard_count'])}/{int(top_hospital['hard_total'])}, "
                f"{100.0 * float(top_hospital['hard_fraction']):.0f}%) than among other fatal stays "
                f"({int(top_hospital['other_count'])}/{int(top_hospital['other_total'])}, "
                f"{100.0 * float(top_hospital['other_fraction']):.0f}%)."
            )
        summary_lines.append(hospital_text)

    disease_details = detail_lookup[detail_lookup["variable"].eq("disease_group")].copy()
    disease_details = disease_details.sort_values(
        "abs_standardized_difference",
        ascending=False,
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    if not disease_details.empty and pd.notna(disease_details.at[0, "abs_standardized_difference"]):
        top_disease = disease_details.iloc[0]
        disease_modifier = _modifier_from_abs_effect_size(float(top_disease["abs_standardized_difference"]))
        if float(top_disease["standardized_difference"]) >= 0:
            summary_lines.append(
                f"- Low-predicted fatal stays were {disease_modifier}enriched in "
                f"`{top_disease['level']}` disease-group assignments "
                f"({int(top_disease['hard_count'])}/{int(top_disease['hard_total'])}, "
                f"{100.0 * float(top_disease['hard_fraction']):.0f}% vs "
                f"{100.0 * float(top_disease['other_fraction']):.0f}%)."
            )
        else:
            summary_lines.append(
                f"- `{top_disease['level']}` disease-group assignments were {disease_modifier}less common "
                f"among low-predicted fatal stays "
                f"({int(top_disease['hard_count'])}/{int(top_disease['hard_total'])}, "
                f"{100.0 * float(top_disease['hard_fraction']):.0f}% vs "
                f"{100.0 * float(top_disease['other_fraction']):.0f}%)."
            )

    continuous_rows = comparison_lookup[
        comparison_lookup["variable"].isin(list(CONTINUOUS_VARIABLES))
        & comparison_lookup["absolute_standardized_difference"].notna()
    ].copy()
    continuous_rows = continuous_rows.sort_values(
        "absolute_standardized_difference",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    prominent_continuous = continuous_rows[
        continuous_rows["absolute_standardized_difference"].ge(0.2)
    ].head(3)
    if prominent_continuous.empty:
        summary_lines.append(
            "- Prediction timing and the frozen physiologic proxies showed only modest differences overall."
        )
    else:
        phrases: list[str] = []
        for row in prominent_continuous.itertuples(index=False):
            modifier = _modifier_from_abs_effect_size(float(row.absolute_standardized_difference))
            direction = "higher" if float(row.standardized_difference) >= 0 else "lower"
            phrases.append(f"{SUMMARY_LABELS[str(row.variable)]} was {modifier}{direction}")
        summary_lines.append(
            "- Among the frozen timing and physiologic proxies, "
            + ", ".join(phrases[:-1])
            + (", and " if len(phrases) > 1 else "")
            + phrases[-1]
            + " among low-predicted fatal stays."
        )

    summary_lines.extend(
        [
            "",
            "## Caveat",
            (
                f"- This is a bounded descriptive comparison on a small 24h fatal-only slice "
                f"(`{n_hard}` vs `{n_other}` stays), so the outputs should be read as chapter-oriented "
                "structure rather than stable subgroup estimates."
            ),
        ]
    )
    return "\n".join(summary_lines) + "\n"


def _plot_effect_sizes(effect_size_plot_data: pd.DataFrame, *, output_path: Path) -> Path:
    _require_matplotlib()
    plot_frame = effect_size_plot_data.copy()
    plot_frame = plot_frame[np.isfinite(plot_frame["absolute_standardized_difference"])].copy()
    if plot_frame.empty:
        raise ValueError("No finite standardized-difference values are available for plotting.")

    plot_frame = plot_frame.sort_values(
        "absolute_standardized_difference",
        ascending=True,
        kind="stable",
    ).reset_index(drop=True)
    figure_height = max(3.6, 0.48 * plot_frame.shape[0] + 1.2)
    figure, axis = plt.subplots(figsize=(8.4, figure_height))
    axis.barh(
        np.arange(plot_frame.shape[0]),
        plot_frame["absolute_standardized_difference"],
        color="#4c78a8",
        edgecolor="#1f3a5f",
        linewidth=0.7,
    )
    axis.set_yticks(np.arange(plot_frame.shape[0]))
    axis.set_yticklabels(plot_frame["figure_label"].tolist(), fontsize=9)
    axis.set_xlabel("Absolute standardized difference")
    axis.set_title("ASIC 24h fatal stays: low-predicted vs other fatal stays", fontsize=12)
    axis.grid(axis="x", color="#d9d9d9", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.text(
        0.99,
        -0.12,
        "Categorical variables use the maximum absolute level-specific standardized difference.",
        fontsize=8,
        color="#555555",
        ha="right",
        va="top",
        transform=axis.transAxes,
    )
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def run_asic_hardcase_comparison(
    *,
    hard_case_path: Path = DEFAULT_HARD_CASE_PATH,
    model_ready_path: Path = DEFAULT_MODEL_READY_PATH,
    asic_input_root: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> ASICHardCaseComparisonRunResult:
    comparison_dataset, join_metadata = build_stay_level_comparison_dataset(
        hard_case_path=hard_case_path,
        model_ready_path=model_ready_path,
        asic_input_root=asic_input_root,
    )
    comparison_table, effect_size_plot_data, standardized_difference_details = (
        build_comparison_outputs(comparison_dataset)
    )
    summary_markdown = build_summary_markdown(
        comparison_dataset,
        comparison_table,
        standardized_difference_details,
    )

    resolved_output_dir = Path(output_dir)
    comparison_dataset_path = write_dataframe(
        comparison_dataset,
        resolved_output_dir / "stay_level_comparison_dataset.csv",
        output_format="csv",
    )
    comparison_table_path = write_dataframe(
        comparison_table.drop(columns="figure_label"),
        resolved_output_dir / "comparison_table.csv",
        output_format="csv",
    )
    effect_size_plot_data_path = write_dataframe(
        effect_size_plot_data,
        resolved_output_dir / "effect_size_plot_data.csv",
        output_format="csv",
    )
    standardized_difference_details_path = write_dataframe(
        standardized_difference_details,
        resolved_output_dir / "standardized_difference_details.csv",
        output_format="csv",
    )
    figure_path = _plot_effect_sizes(
        effect_size_plot_data,
        output_path=resolved_output_dir / "effect_size_figure.png",
    )
    summary_path = write_text(summary_markdown, resolved_output_dir / "summary.md")

    manifest_payload = {
        "timestamp_utc": _utc_timestamp(),
        "issue_id": ISSUE_ID,
        "target_horizon_h": TARGET_HORIZON_H,
        "hard_case_rule": HARD_CASE_RULE,
        "comparison_variables": [
            "age_group",
            "sex",
            "disease_group",
            "prediction_time_h",
            "hospital_id",
            "pf_ratio_last",
            "map_last",
            "creatinine_last",
            "peep_last",
        ],
        "group_counts": {
            LOW_PREDICTED_FATAL_GROUP: int(comparison_dataset["hard_case_flag"].astype(bool).sum()),
            OTHER_FATAL_GROUP: int((~comparison_dataset["hard_case_flag"].astype(bool)).sum()),
            "total_fatal_stays": int(comparison_dataset.shape[0]),
        },
        "source_paths": {
            key: str(Path(value).resolve())
            for key, value in join_metadata["source_paths"].items()
        },
        "join_logic": [
            {
                **step,
                "path": str(Path(step["path"]).resolve()),
            }
            for step in join_metadata["join_logic"]
        ],
        "effect_size_definitions": {
            "continuous": (
                "standardized mean difference using the difference in means divided by the pooled "
                "within-group SD across available values"
            ),
            "categorical": (
                "level-specific standardized difference in proportions; the core table and figure "
                "report the level with the largest absolute standardized difference for each variable"
            ),
        },
        "output_paths": {
            "comparison_dataset": str(Path(comparison_dataset_path).resolve()),
            "comparison_table": str(Path(comparison_table_path).resolve()),
            "effect_size_plot_data": str(Path(effect_size_plot_data_path).resolve()),
            "standardized_difference_details": str(Path(standardized_difference_details_path).resolve()),
            "figure": str(Path(figure_path).resolve()),
            "summary_markdown": str(Path(summary_path).resolve()),
        },
    }
    manifest_path = _write_json(manifest_payload, resolved_output_dir / "run_manifest.json")

    return ASICHardCaseComparisonRunResult(
        output_dir=resolved_output_dir,
        artifacts=ASICHardCaseComparisonArtifacts(
            comparison_dataset_path=comparison_dataset_path,
            comparison_table_path=comparison_table_path,
            effect_size_plot_data_path=effect_size_plot_data_path,
            standardized_difference_details_path=standardized_difference_details_path,
            figure_path=figure_path,
            summary_path=summary_path,
            manifest_path=manifest_path,
        ),
        comparison_dataset=comparison_dataset,
        comparison_table=comparison_table.drop(columns="figure_label"),
        effect_size_plot_data=effect_size_plot_data,
        standardized_difference_details=standardized_difference_details,
        summary_markdown=summary_markdown,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble the frozen ASIC 24h fatal-stay hard-case comparison package for "
            "Phase 1 / Chapter 1 / Sprint 3 / Issue 3.2."
        )
    )
    parser.add_argument(
        "--hard-case-path",
        type=Path,
        default=DEFAULT_HARD_CASE_PATH,
        help="Saved stay-level logistic hard-case artifact path.",
    )
    parser.add_argument(
        "--model-ready-path",
        type=Path,
        default=DEFAULT_MODEL_READY_PATH,
        help="Chapter 1 primary model-ready dataset path.",
    )
    parser.add_argument(
        "--asic-input-root",
        type=Path,
        help="Optional override for the standardized ASIC harmonized artifact root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the Issue 3.2 comparison package will be written.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = run_asic_hardcase_comparison(
        hard_case_path=args.hard_case_path,
        model_ready_path=args.model_ready_path,
        asic_input_root=args.asic_input_root,
        output_dir=args.output_dir,
    )

    print(f"Output directory: {result.output_dir}")
    print(f"Comparison dataset: {result.artifacts.comparison_dataset_path}")
    print(f"Comparison table: {result.artifacts.comparison_table_path}")
    print(f"Effect-size plot data: {result.artifacts.effect_size_plot_data_path}")
    print(f"Standardized-difference details: {result.artifacts.standardized_difference_details_path}")
    print(f"Figure: {result.artifacts.figure_path}")
    print(f"Summary markdown: {result.artifacts.summary_path}")
    print(f"Manifest: {result.artifacts.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
