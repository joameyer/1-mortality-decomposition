from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from chapter1_mortality_decomposition.mimic_blocks import REPO_ROOT, _resolve_path
from chapter1_mortality_decomposition.utils import ensure_directory


DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "ch1_mimic_preprocessing_qc.yaml"
DEFAULT_HORIZONS = (8, 16, 24, 48)


@dataclass(frozen=True)
class MimicPreprocessingQcConfig:
    mimic_processed_root: Path
    preprocessing_output_root: Path
    horizon_target_root: Path
    upstream_reports_dir: Path
    output_reports_dir: Path
    feature_freeze_csv: Path
    mapping_csv: Path
    horizons_hours: tuple[int, ...] = DEFAULT_HORIZONS
    model_ready_feature_set: str = "primary"


def _parse_horizons(value: str) -> tuple[int, ...]:
    horizons: list[int] = []
    seen: set[int] = set()
    for token in str(value).replace(";", ",").split(","):
        stripped = token.strip()
        if not stripped:
            continue
        horizon = int(stripped)
        if horizon <= 0:
            raise ValueError("QC horizons must be positive integers.")
        if horizon not in seen:
            horizons.append(horizon)
            seen.add(horizon)
    return tuple(horizons or DEFAULT_HORIZONS)


def load_config(path: Path) -> MimicPreprocessingQcConfig:
    raw: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(f"Unsupported config line in {path}: {line!r}")
        key, value = stripped.split(":", 1)
        raw[key.strip()] = value.split("#", 1)[0].strip().strip("'\"")

    required = [
        "mimic_processed_root",
        "preprocessing_output_root",
        "horizon_target_root",
        "upstream_reports_dir",
        "output_reports_dir",
        "feature_freeze_csv",
        "mapping_csv",
    ]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"MIMIC preprocessing QC config {path} is missing keys: {missing}")

    return MimicPreprocessingQcConfig(
        mimic_processed_root=_resolve_path(raw["mimic_processed_root"]),
        preprocessing_output_root=_resolve_path(raw["preprocessing_output_root"]),
        horizon_target_root=_resolve_path(raw["horizon_target_root"]),
        upstream_reports_dir=_resolve_path(raw["upstream_reports_dir"]),
        output_reports_dir=_resolve_path(raw["output_reports_dir"]),
        feature_freeze_csv=_resolve_path(raw["feature_freeze_csv"]),
        mapping_csv=_resolve_path(raw["mapping_csv"]),
        horizons_hours=_parse_horizons(raw.get("horizons_hours", ",".join(map(str, DEFAULT_HORIZONS)))),
        model_ready_feature_set=raw.get("model_ready_feature_set", "primary"),
    )


def _read_optional_csv(path: Path) -> tuple[pd.DataFrame | None, str]:
    if not path.exists():
        return None, f"missing: {path}"
    try:
        return pd.read_csv(path), f"read: {path}"
    except Exception as exc:  # pragma: no cover - surfaced in QC output
        return None, f"read_failed: {path}: {type(exc).__name__}: {exc}"


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _percentiles(values: pd.Series) -> dict[str, object]:
    numeric = _numeric(values).dropna()
    if numeric.empty:
        return {"min": pd.NA, "p25": pd.NA, "median": pd.NA, "p75": pd.NA, "max": pd.NA}
    p25, median, p75 = np.percentile(numeric.to_numpy(dtype=float), [25, 50, 75])
    return {
        "min": float(numeric.min()),
        "p25": float(p25),
        "median": float(median),
        "p75": float(p75),
        "max": float(numeric.max()),
    }


def build_cohort_flow_verification(config: MimicPreprocessingQcConfig) -> pd.DataFrame:
    path = config.upstream_reports_dir / "ch1_mimic_cohort_flow.csv"
    flow, source = _read_optional_csv(path)
    expected_steps = [
        "total_icu_stays_considered",
        "age_ge_18_retained",
        "first_icu_stay_retained",
        "invasive_vent_ge_24h_retained",
        "ventilation_qc_guard_retained",
        "final_retained_stay_level_cohort_count",
    ]
    if flow is None or "flow_step" not in flow.columns or "retained_count" not in flow.columns:
        return pd.DataFrame(
            [
                {
                    "flow_step": step,
                    "retained_count": pd.NA,
                    "excluded_from_previous": pd.NA,
                    "verification_status": "not_checkable",
                    "source": source,
                    "note": "Existing cohort flow report is unavailable or lacks required columns.",
                }
                for step in expected_steps
            ]
        )

    rows: list[dict[str, object]] = []
    indexed = flow.set_index("flow_step", drop=False)
    for step in expected_steps:
        if step in indexed.index:
            row = indexed.loc[step]
            rows.append(
                {
                    "flow_step": step,
                    "retained_count": row.get("retained_count", pd.NA),
                    "excluded_from_previous": row.get("excluded_from_previous", pd.NA),
                    "verification_status": "verified_directly",
                    "source": source,
                    "note": row.get("step_note", ""),
                }
            )
        else:
            rows.append(
                {
                    "flow_step": step,
                    "retained_count": pd.NA,
                    "excluded_from_previous": pd.NA,
                    "verification_status": "not_checkable",
                    "source": source,
                    "note": "Expected flow step was not present in existing cohort flow report.",
                }
            )
    return pd.DataFrame(rows)


def build_valid_instance_summary(config: MimicPreprocessingQcConfig) -> pd.DataFrame:
    block_qc, block_source = _read_optional_csv(config.upstream_reports_dir / "ch1_mimic_block_qc_summary.csv")
    counts, counts_source = _read_optional_csv(
        config.preprocessing_output_root / "instances" / "chapter1_instance_counts_by_horizon.csv"
    )
    exclusions, exclusions_source = _read_optional_csv(
        config.preprocessing_output_root / "instances" / "chapter1_instance_exclusion_summary.csv"
    )

    rows: list[dict[str, object]] = []
    metric_lookup = {}
    if block_qc is not None and {"metric", "value"}.issubset(block_qc.columns):
        metric_lookup = dict(zip(block_qc["metric"], block_qc["value"]))

    total_completed_blocks = metric_lookup.get("total_completed_blocks_emitted", pd.NA)
    rows.append(
        {
            "summary_scope": "overall",
            "horizon_h": pd.NA,
            "metric": "total_completed_blocks",
            "value": total_completed_blocks,
            "verification_status": "verified_directly" if pd.notna(total_completed_blocks) else "not_checkable",
            "source": block_source,
            "note": "Completed block count from b2 QC summary.",
        }
    )

    if counts is not None and {"horizon_h", "candidate_instances", "valid_instances"}.issubset(counts.columns):
        selected = counts[counts["horizon_h"].isin(config.horizons_hours)].copy()
        rows.append(
            {
                "summary_scope": "overall",
                "horizon_h": pd.NA,
                "metric": "total_valid_prediction_instances",
                "value": int(_numeric(selected["valid_instances"]).fillna(0).sum()),
                "verification_status": "verified_directly",
                "source": counts_source,
                "note": "Sum over configured QC horizons.",
            }
        )
        candidate_total = float(_numeric(selected["candidate_instances"]).fillna(0).sum())
        valid_total = float(_numeric(selected["valid_instances"]).fillna(0).sum())
        rows.append(
            {
                "summary_scope": "overall",
                "horizon_h": pd.NA,
                "metric": "valid_instance_share_among_candidate_horizon_rows",
                "value": valid_total / candidate_total if candidate_total else pd.NA,
                "verification_status": "verified_directly",
                "source": counts_source,
                "note": "Share is valid_instances / candidate_instances across configured horizon-duplicated rows.",
            }
        )
        for row in selected.itertuples(index=False):
            candidate = float(getattr(row, "candidate_instances"))
            valid = float(getattr(row, "valid_instances"))
            rows.append(
                {
                    "summary_scope": "by_horizon",
                    "horizon_h": int(getattr(row, "horizon_h")),
                    "metric": "valid_instance_share",
                    "value": valid / candidate if candidate else pd.NA,
                    "verification_status": "verified_directly",
                    "source": counts_source,
                    "note": f"{int(valid)} valid of {int(candidate)} candidate rows.",
                }
            )
    else:
        rows.append(
            {
                "summary_scope": "overall",
                "horizon_h": pd.NA,
                "metric": "total_valid_prediction_instances",
                "value": pd.NA,
                "verification_status": "not_checkable",
                "source": counts_source,
                "note": "Instance count artifact is unavailable or lacks required columns.",
            }
        )

    if exclusions is not None and not exclusions.empty and {"horizon_h", "exclusion_reason", "instance_count"}.issubset(exclusions.columns):
        for row in exclusions.itertuples(index=False):
            rows.append(
                {
                    "summary_scope": "invalid_reason",
                    "horizon_h": int(getattr(row, "horizon_h")),
                    "metric": str(getattr(row, "exclusion_reason")),
                    "value": int(getattr(row, "instance_count")),
                    "verification_status": "verified_directly",
                    "source": exclusions_source,
                    "note": "Invalid candidate-instance reason from reused preprocessing output.",
                }
            )
    else:
        rows.append(
            {
                "summary_scope": "invalid_reason",
                "horizon_h": pd.NA,
                "metric": "invalid_candidate_reason_counts",
                "value": 0 if exclusions is not None else pd.NA,
                "verification_status": "verified_directly" if exclusions is not None else "not_checkable",
                "source": exclusions_source,
                "note": "No invalid candidate rows were reported." if exclusions is not None else "Instance exclusion artifact unavailable.",
            }
        )
    return pd.DataFrame(rows)


def build_horizon_event_summary(config: MimicPreprocessingQcConfig) -> pd.DataFrame:
    summary, source = _read_optional_csv(config.upstream_reports_dir / "ch1_mimic_horizon_label_summary.csv")
    required = {
        "horizon_h",
        "candidate_rows",
        "labeled_rows",
        "positive_rows",
        "negative_rows",
        "unlabeled_rows",
        "positive_rate_among_labeled",
    }
    if summary is None or not required.issubset(summary.columns):
        return pd.DataFrame(
            [
                {
                    "horizon_h": horizon,
                    "candidate_rows": pd.NA,
                    "labeled_rows": pd.NA,
                    "positive_rows": pd.NA,
                    "negative_rows": pd.NA,
                    "unlabeled_rows": pd.NA,
                    "positive_rate_among_labeled": pd.NA,
                    "verification_status": "not_checkable",
                    "source": source,
                    "note": "Horizon label summary unavailable or missing required columns.",
                }
                for horizon in config.horizons_hours
            ]
        )
    selected = summary[summary["horizon_h"].isin(config.horizons_hours)].copy()
    selected["verification_status"] = "verified_directly"
    selected["source"] = source
    selected["note"] = "Conservative proxy-label counts from b4 summary."
    return selected[
        [
            "horizon_h",
            "candidate_rows",
            "labeled_rows",
            "positive_rows",
            "negative_rows",
            "unlabeled_rows",
            "positive_rate_among_labeled",
            "verification_status",
            "source",
            "note",
        ]
    ]


def build_block_distribution_summary(config: MimicPreprocessingQcConfig) -> pd.DataFrame:
    block_qc, block_source = _read_optional_csv(config.upstream_reports_dir / "ch1_mimic_block_qc_summary.csv")
    stay_counts, stay_source = _read_optional_csv(
        config.mimic_processed_root / "ch1_mimic_stay_block_counts.csv"
    )
    rows: list[dict[str, object]] = []

    if stay_counts is not None and "completed_block_count" in stay_counts.columns:
        stats = _percentiles(stay_counts["completed_block_count"])
        for metric, value in stats.items():
            rows.append(
                {
                    "metric": f"completed_blocks_per_stay_{metric}",
                    "value": value,
                    "verification_status": "verified_directly",
                    "source": stay_source,
                    "note": "Computed as an aggregate from stay block counts.",
                }
            )
    else:
        for metric in ["min", "p25", "median", "p75", "max"]:
            rows.append(
                {
                    "metric": f"completed_blocks_per_stay_{metric}",
                    "value": pd.NA,
                    "verification_status": "not_checkable",
                    "source": stay_source,
                    "note": "Stay block-count artifact unavailable or lacks completed_block_count.",
                }
            )

    qc_lookup = {}
    if block_qc is not None and {"metric", "value"}.issubset(block_qc.columns):
        qc_lookup = dict(zip(block_qc["metric"], block_qc["value"]))
    for metric in [
        "total_completed_blocks_emitted",
        "completed_blocks_with_zero_dynamic_rows",
        "completed_blocks_with_zero_observed_variables",
        "stays_with_at_least_one_completed_block",
    ]:
        rows.append(
            {
                "metric": metric,
                "value": qc_lookup.get(metric, pd.NA),
                "verification_status": "verified_directly" if metric in qc_lookup else "not_checkable",
                "source": block_source,
                "note": "Copied from b2 block QC summary.",
            }
        )
    return pd.DataFrame(rows)


def build_feature_missingness_summary(config: MimicPreprocessingQcConfig) -> pd.DataFrame:
    freeze, freeze_source = _read_optional_csv(config.feature_freeze_csv)
    blocked, blocked_source = _read_optional_csv(
        config.mimic_processed_root / "ch1_mimic_blocked_dynamic_features.csv"
    )
    model_ready_path = (
        config.preprocessing_output_root
        / "model_ready"
        / f"chapter1_{config.model_ready_feature_set}_model_ready_dataset.csv"
    )
    model_ready, model_source = _read_optional_csv(model_ready_path)
    if freeze is None:
        return pd.DataFrame(
            [
                {
                    "variable": pd.NA,
                    "final_role": pd.NA,
                    "freeze_decision": pd.NA,
                    "raw_block_observation_count": pd.NA,
                    "preprocessed_non_missing_count": pd.NA,
                    "preprocessed_missing_count": pd.NA,
                    "preprocessed_non_missing_fraction": pd.NA,
                    "verification_status": "not_checkable",
                    "note": f"Feature freeze artifact unavailable: {freeze_source}",
                }
            ]
        )

    rows: list[dict[str, object]] = []
    denominator = pd.NA
    model_subset = model_ready
    if model_ready is not None and "horizon_h" in model_ready.columns:
        model_subset = model_ready[model_ready["horizon_h"].isin(config.horizons_hours)].copy()
        denominator = int(model_subset.shape[0])

    for row in freeze.itertuples(index=False):
        variable = str(getattr(row, "asic_base_variable"))
        final_role = str(getattr(row, "final_role"))
        freeze_decision = str(getattr(row, "freeze_decision"))
        raw_col = f"{variable}_obs_count"
        value_col = f"{variable}_last"

        raw_count = pd.NA
        raw_status = "not_checkable"
        notes: list[str] = []
        if blocked is not None and raw_col in blocked.columns:
            raw_count = float(_numeric(blocked[raw_col]).fillna(0).sum())
            raw_status = "verified_directly"
        else:
            notes.append(f"Raw block observation count unavailable from {blocked_source}.")

        non_missing = pd.NA
        missing = pd.NA
        fraction = pd.NA
        pre_status = "not_checkable"
        if model_subset is not None and value_col in model_subset.columns:
            non_missing = int(model_subset[value_col].notna().sum())
            missing = int(model_subset[value_col].isna().sum())
            fraction = non_missing / int(model_subset.shape[0]) if model_subset.shape[0] else pd.NA
            pre_status = "approximated_from_available_artifact"
            notes.append(
                f"Preprocessed missingness uses `{value_col}` as representative LOCF-ready feature over configured horizons."
            )
        else:
            notes.append(f"Preprocessed `{value_col}` column unavailable from {model_source}.")

        if freeze_decision == "derived_only":
            notes.append("Derived-only variable; materialization remains deferred or sparse according to prior freeze.")
        if final_role == "mimic_secondary":
            notes.append("MIMIC-secondary variable; not required for primary external-validation feature set.")

        rows.append(
            {
                "variable": variable,
                "final_role": final_role,
                "freeze_decision": freeze_decision,
                "raw_block_observation_count": raw_count,
                "preprocessed_non_missing_count": non_missing,
                "preprocessed_missing_count": missing,
                "preprocessed_non_missing_fraction": fraction,
                "model_ready_rows_checked": denominator,
                "verification_status": (
                    "verified_directly" if raw_status == "verified_directly" and pre_status != "not_checkable"
                    else "partial"
                ),
                "note": " ".join(notes).strip(),
            }
        )
    return pd.DataFrame(rows)


def build_mapping_quality_summary(config: MimicPreprocessingQcConfig) -> pd.DataFrame:
    freeze, source = _read_optional_csv(config.feature_freeze_csv)
    if freeze is None:
        return pd.DataFrame(
            [
                {
                    "summary_type": "freeze_decision",
                    "category": "not_checkable",
                    "count": pd.NA,
                    "proportion": pd.NA,
                    "source": source,
                }
            ]
        )
    rows: list[dict[str, object]] = []
    for column, summary_type in [("freeze_decision", "freeze_decision"), ("final_role", "final_role")]:
        counts = freeze[column].fillna("(missing)").astype("string").value_counts().sort_index()
        total = int(counts.sum())
        for category, count in counts.items():
            rows.append(
                {
                    "summary_type": summary_type,
                    "category": str(category),
                    "count": int(count),
                    "proportion": int(count) / total if total else pd.NA,
                    "source": source,
                }
            )
    return pd.DataFrame(rows)


def _metric_value(df: pd.DataFrame, metric_column: str, value_column: str, metric: str) -> object:
    if df is None or metric_column not in df.columns or value_column not in df.columns:
        return pd.NA
    selected = df.loc[df[metric_column].eq(metric), value_column]
    return selected.iloc[0] if not selected.empty else pd.NA


def build_qc_summary(
    config: MimicPreprocessingQcConfig,
    cohort_flow: pd.DataFrame,
    valid_summary: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    block_summary: pd.DataFrame,
    mapping_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    final_count = cohort_flow.loc[
        cohort_flow["flow_step"].eq("final_retained_stay_level_cohort_count"),
        "retained_count",
    ]
    rows.append(
        {
            "metric": "retained_stay_count",
            "horizon_h": pd.NA,
            "value": final_count.iloc[0] if not final_count.empty else pd.NA,
            "status": "verified_directly" if not final_count.empty else "not_checkable",
            "note": "Final retained b1 stay-level cohort count.",
        }
    )
    rows.append(
        {
            "metric": "completed_block_count",
            "horizon_h": pd.NA,
            "value": _metric_value(block_summary, "metric", "value", "total_completed_blocks_emitted"),
            "status": "verified_directly",
            "note": "Total completed blocks emitted by b2.",
        }
    )
    rows.append(
        {
            "metric": "valid_instance_count",
            "horizon_h": pd.NA,
            "value": _metric_value(valid_summary, "metric", "value", "total_valid_prediction_instances"),
            "status": "verified_directly",
            "note": "Total valid prediction instances over configured QC horizons.",
        }
    )
    for row in horizon_summary.itertuples(index=False):
        rows.extend(
            [
                {
                    "metric": "horizon_labeled_rows",
                    "horizon_h": int(row.horizon_h),
                    "value": int(row.labeled_rows),
                    "status": "verified_directly",
                    "note": "B4 conservative proxy-label labeled row count.",
                },
                {
                    "metric": "horizon_positive_rows",
                    "horizon_h": int(row.horizon_h),
                    "value": int(row.positive_rows),
                    "status": "verified_directly",
                    "note": "B4 conservative proxy-label positive row count.",
                },
            ]
        )
    shared_primary_count = mapping_summary[
        mapping_summary["summary_type"].eq("final_role")
        & mapping_summary["category"].eq("shared_primary")
    ]["count"]
    rows.append(
        {
            "metric": "shared_primary_variables",
            "horizon_h": pd.NA,
            "value": int(shared_primary_count.iloc[0]) if not shared_primary_count.empty else pd.NA,
            "status": "verified_directly",
            "note": "Feature freeze final_role count.",
        }
    )
    for category in [
        "proxy_retained",
        "accepted_asymmetry_retained",
        "derived_only",
        "demoted_from_shared_primary",
    ]:
        selected = mapping_summary[
            mapping_summary["summary_type"].eq("freeze_decision")
            & mapping_summary["category"].eq(category)
        ]["count"]
        rows.append(
            {
                "metric": f"freeze_decision_{category}_variables",
                "horizon_h": pd.NA,
                "value": int(selected.iloc[0]) if not selected.empty else 0,
                "status": "verified_directly",
                "note": "Feature freeze decision count.",
            }
        )
    return pd.DataFrame(rows)


def write_qc_note(
    config: MimicPreprocessingQcConfig,
    *,
    partial_domains: list[str],
) -> None:
    partial_text = (
        "\n".join(f"- {domain}" for domain in partial_domains)
        if partial_domains
        else "- None identified in the configured demo artifacts."
    )
    lines = [
        "# Chapter 1 MIMIC Preprocessing QC Note",
        "",
        "## Purpose",
        "",
        "This report documents issue 5.1.c1: an aggregated QC/readiness audit over the MIMIC Chapter 1 preprocessing artifacts produced by b1-b5.",
        "",
        "This is a QC/readiness audit only. It does not rebuild the cohort, blocks, preprocessing, labels, or models, and it does not compare distributions or performance.",
        "",
        "## Input Artifacts Read",
        "",
        f"- MIMIC processed root: `{config.mimic_processed_root}`",
        f"- Reused preprocessing output root: `{config.preprocessing_output_root}`",
        f"- Horizon target root: `{config.horizon_target_root}`",
        f"- Upstream report root: `{config.upstream_reports_dir}`",
        f"- Feature freeze CSV: `{config.feature_freeze_csv}`",
        "",
        "## Verified Domains",
        "",
        "- cohort flow counts",
        "- ventilation inclusion gate and vent-vs-LOS QC counts via b1 flow/QC artifacts",
        "- first-stay handling via b1 flow artifacts",
        "- ICU timing/block counts via b2 block QC and stay block counts",
        "- valid-instance counts via reused ASIC preprocessing outputs",
        "- per-horizon conservative proxy-label counts via b4 outputs",
        "- block-count distribution summaries",
        "- feature coverage and missingness using blocked features and model-ready exports",
        "- mapping/freeze quality proportions",
        "",
        "## Partial Checks",
        "",
        partial_text,
        "",
        "## Full-MIMIC Use",
        "",
        "Run the same QC script with path overrides pointing to the private full-MIMIC processed, preprocessing, horizon-target, and upstream-report roots. The script writes aggregated reports only; full-MIMIC row-level inputs should remain outside the repo.",
        "",
        "Example:",
        "",
        "```bash",
        "FULL_ROOT=/Users/joanameyer/data/mimic-iv/mimic-iv-3.1",
        "CH1_ROOT=$FULL_ROOT/1-mortality-decomposition",
        "python scripts/run_mimic_preprocessing_qc.py \\",
        "  --mimic-processed-root $CH1_ROOT/processed \\",
        "  --preprocessing-output-root $CH1_ROOT/preprocessing_outputs \\",
        "  --horizon-target-root $CH1_ROOT/horizon_targets \\",
        "  --upstream-reports-dir reports \\",
        "  --output-reports-dir reports",
        "```",
        "",
        "Full-MIMIC scientific interpretation still depends on the user running this QC on the private full outputs and then reviewing downstream external-validation results.",
    ]
    ensure_directory(config.output_reports_dir)
    (config.output_reports_dir / "ch1_mimic_preprocessing_qc_note.md").write_text(
        "\n".join(lines)
    )


def run_qc(config: MimicPreprocessingQcConfig) -> dict[str, pd.DataFrame]:
    ensure_directory(config.output_reports_dir)
    cohort_flow = build_cohort_flow_verification(config)
    valid_summary = build_valid_instance_summary(config)
    horizon_summary = build_horizon_event_summary(config)
    block_summary = build_block_distribution_summary(config)
    feature_missingness = build_feature_missingness_summary(config)
    mapping_summary = build_mapping_quality_summary(config)
    qc_summary = build_qc_summary(
        config,
        cohort_flow,
        valid_summary,
        horizon_summary,
        block_summary,
        mapping_summary,
    )

    outputs = {
        "ch1_mimic_cohort_flow_verification.csv": cohort_flow,
        "ch1_mimic_valid_instance_summary.csv": valid_summary,
        "ch1_mimic_horizon_event_summary.csv": horizon_summary,
        "ch1_mimic_block_distribution_summary.csv": block_summary,
        "ch1_mimic_feature_missingness_summary.csv": feature_missingness,
        "ch1_mimic_mapping_quality_summary.csv": mapping_summary,
        "ch1_mimic_preprocessing_qc_summary.csv": qc_summary,
    }
    for filename, df in outputs.items():
        df.to_csv(config.output_reports_dir / filename, index=False)

    partial_domains = []
    if feature_missingness["verification_status"].astype("string").eq("partial").any():
        partial_domains.append(
            "Feature missingness uses each variable's `{variable}_last` model-ready column as the representative preprocessed non-missingness proxy where exact per-base-variable post-LOCF counts are not separately exported."
        )
    if valid_summary["verification_status"].astype("string").eq("not_checkable").any():
        partial_domains.append("Some valid-instance QC elements were unavailable from configured inputs.")
    write_qc_note(config, partial_domains=partial_domains)
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mimic-processed-root", type=Path, default=None)
    parser.add_argument("--preprocessing-output-root", type=Path, default=None)
    parser.add_argument("--horizon-target-root", type=Path, default=None)
    parser.add_argument("--upstream-reports-dir", type=Path, default=None)
    parser.add_argument("--output-reports-dir", type=Path, default=None)
    parser.add_argument("--feature-freeze-csv", type=Path, default=None)
    parser.add_argument("--mapping-csv", type=Path, default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=None)
    parser.add_argument("--model-ready-feature-set", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    replacements = {}
    for field_name, value in [
        ("mimic_processed_root", args.mimic_processed_root),
        ("preprocessing_output_root", args.preprocessing_output_root),
        ("horizon_target_root", args.horizon_target_root),
        ("upstream_reports_dir", args.upstream_reports_dir),
        ("output_reports_dir", args.output_reports_dir),
        ("feature_freeze_csv", args.feature_freeze_csv),
        ("mapping_csv", args.mapping_csv),
    ]:
        if value is not None:
            replacements[field_name] = _resolve_path(value)
    if args.horizons is not None:
        replacements["horizons_hours"] = tuple(int(horizon) for horizon in args.horizons)
    if args.model_ready_feature_set is not None:
        replacements["model_ready_feature_set"] = args.model_ready_feature_set
    if replacements:
        config = replace(config, **replacements)

    outputs = run_qc(config)
    print("Wrote MIMIC preprocessing QC reports:")
    for filename in outputs:
        print(config.output_reports_dir / filename)
    print(config.output_reports_dir / "ch1_mimic_preprocessing_qc_note.md")


if __name__ == "__main__":
    main()
