from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from chapter1_mortality_decomposition.utils import ensure_directory, write_text


ORDERED_CATEGORIES = (
    "hard-case definition artifacts",
    "comparison artifacts",
    "horizon artifacts",
    "interpretation notes",
)

DISCOVERY_ROOTS = (
    Path("artifacts") / "chapter1" / "evaluation" / "asic",
    Path("artifacts") / "chapter1" / "temporal_preview" / "asic",
    Path("docs"),
    Path("notebooks"),
    Path("scripts"),
    Path("reports"),
)

SUPPORTED_SUFFIXES = {
    ".csv",
    ".feather",
    ".ipynb",
    ".jpeg",
    ".jpg",
    ".json",
    ".md",
    ".parquet",
    ".pdf",
    ".png",
    ".txt",
}

EXCLUDED_DIR_NAMES = {
    ".git",
    ".ipynb_checkpoints",
    ".venv",
    "__pycache__",
    "artifacts_old",
    "logs",
}

SELF_OUTPUT_NAMES = {
    "ch1_asic_descriptive_viability_review.ipynb",
    "ch1_asic_descriptive_viability_evidence_pack.md",
    "ch1_asic_descriptive_viability_memo_draft.md",
}

KEY_ARTIFACT_SUFFIXES: dict[str, tuple[str, ...]] = {
    "hard_case_manifest": (
        "artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/run_manifest.json",
    ),
    "hard_case_summary": (
        "artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/horizon_hard_case_summary.csv",
    ),
    "stay_level_flags": (
        "artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/stay_level_hard_case_flags.csv",
    ),
    "comparison_manifest": (
        "artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/run_manifest.json",
    ),
    "comparison_table": (
        "artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/comparison_table.csv",
    ),
    "comparison_summary": (
        "artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/summary.md",
    ),
    "comparison_figure": (
        "artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/effect_size_figure.png",
    ),
    "comparison_details": (
        "artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/standardized_difference_details.csv",
    ),
    "variable_audit_memo": (
        "artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison_variable_audit/asic_hard_case_comparison_variable_audit_memo.md",
    ),
    "variable_audit_table": (
        "artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison_variable_audit/asic_hard_case_comparison_variable_audit_table.csv",
    ),
    "sofa_feasibility_memo": (
        "artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_sofa_feasibility_audit/sofa_feasibility_memo.md",
    ),
    "agreement_summary": (
        "artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/agreement/logistic_regression_vs_xgboost_platt/horizon_hard_case_agreement_summary.csv",
    ),
    "horizon_foundation_summary": (
        "artifacts/chapter1/evaluation/asic/horizon_dependence/foundation/horizon_summary.csv",
    ),
    "horizon_foundation_note": (
        "artifacts/chapter1/evaluation/asic/horizon_dependence/foundation/artifact_foundation_note.md",
    ),
    "overlap_note": (
        "artifacts/chapter1/evaluation/asic/horizon_dependence/overlap/overlap_note.md",
    ),
    "pairwise_overlap": (
        "artifacts/chapter1/evaluation/asic/horizon_dependence/overlap/pairwise_overlap.csv",
    ),
    "pairwise_denominators": (
        "artifacts/chapter1/evaluation/asic/horizon_dependence/overlap/pairwise_denominators.csv",
    ),
    "hard_case_persistence": (
        "artifacts/chapter1/evaluation/asic/horizon_dependence/overlap/hard_case_persistence.csv",
    ),
    "horizon_final_manifest": (
        "artifacts/chapter1/evaluation/asic/horizon_dependence/final/run_manifest.json",
    ),
    "horizon_final_summary": (
        "artifacts/chapter1/evaluation/asic/horizon_dependence/final/final_horizon_summary.md",
    ),
    "horizon_memo": (
        "artifacts/chapter1/evaluation/asic/horizon_dependence/final/horizon_interpretation_memo.md",
    ),
    "horizon_figure": (
        "artifacts/chapter1/evaluation/asic/horizon_dependence/final/mortality_risk_horizon_comparison.png",
    ),
    "horizon_binned_summary": (
        "artifacts/chapter1/evaluation/asic/horizon_dependence/final/mortality_risk_horizon_binned_summary.csv",
    ),
}

EXPECTED_EVIDENCE_ITEMS = (
    ("Hard-case run manifest", "hard_case_manifest"),
    ("Hard-case horizon summary", "hard_case_summary"),
    ("Stay-level hard-case flags", "stay_level_flags"),
    ("24h fatal comparison table", "comparison_table"),
    ("24h fatal comparison summary note", "comparison_summary"),
    ("24h fatal comparison figure", "comparison_figure"),
    ("Variable-audit memo", "variable_audit_memo"),
    ("SOFA feasibility memo", "sofa_feasibility_memo"),
    ("Horizon summary table", "horizon_foundation_summary"),
    ("Pairwise horizon overlap table", "pairwise_overlap"),
    ("Final horizon interpretation memo", "horizon_memo"),
    ("Final horizon comparison figure", "horizon_figure"),
)

MISSING_SEARCH_DESCRIPTIONS = (
    (
        "No dedicated treatment-limitation or end-of-life proxy artifact was found in the searched ASIC Sprint 3 roots.",
        ("treatment", "limitation", "dnr", "dni", "palliative", "comfort", "withdraw"),
    ),
    (
        "No pre-existing ASIC viability memo artifact was found before this workflow; the decision state still had to be reconstructed from comparison and horizon notes.",
        ("viability", "memo"),
    ),
)


@dataclass(frozen=True)
class ReviewContext:
    repo_root: Path
    inventory: pd.DataFrame
    key_artifacts: dict[str, Path | None]
    search_roots: tuple[str, ...]
    hard_case_rule: str | None
    target_horizon_label: str | None
    comparison_group_counts: dict[str, int]
    horizon_share_table: pd.DataFrame
    comparison_highlights: tuple[str, ...]
    horizon_highlights: tuple[str, ...]
    risk_highlights: tuple[str, ...]
    decision_findings: tuple[str, ...]
    open_evidence_gaps: tuple[str, ...]
    synthetic_local_outputs: bool
    descriptive_core_assessment: str
    decomposition_decision: str


def _display_path(path: Path | None, repo_root: Path) -> str:
    if path is None:
        return "(not found)"
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _has_excluded_part(path: Path) -> bool:
    return any(part in EXCLUDED_DIR_NAMES for part in path.parts)


def _is_relevant_path(relative_path: str) -> bool:
    path_lower = relative_path.lower()
    if any(path_lower.endswith(name) for name in SELF_OUTPUT_NAMES):
        return False

    if path_lower.startswith("artifacts/chapter1/evaluation/asic/hard_cases/"):
        return any(
            token in path_lower
            for token in (
                "hard_case",
                "hardcase",
                "comparison",
                "fatal",
                "horizon",
                "memo",
                "note",
                "summary",
                "agreement",
                "audit",
                "sofa",
            )
        )

    if path_lower.startswith("artifacts/chapter1/evaluation/asic/horizon_dependence/"):
        return any(
            token in path_lower
            for token in (
                "horizon",
                "overlap",
                "memo",
                "note",
                "summary",
                "figure",
                "mortality_risk",
            )
        )

    if path_lower.startswith("artifacts/chapter1/evaluation/asic/baselines/"):
        name_lower = Path(relative_path).name.lower()
        return name_lower in {
            "interpretation_note.md",
            "horizon_comparison_metrics.csv",
            "horizon_comparison_plot.png",
        }

    if path_lower.startswith("artifacts/chapter1/temporal_preview/asic/"):
        return "/comparison/" in path_lower and any(
            token in path_lower for token in ("comparison", "preview_note", "8h_vs_16h")
        )

    if path_lower.startswith(("docs/", "notebooks/", "scripts/", "reports/")):
        return any(token in path_lower for token in ("asic", "sprint3")) and any(
            token in path_lower
            for token in ("hard", "horizon", "review", "comparison", "memo", "note", "fatal", "audit", "sprint3")
        )

    return False


def _infer_category(relative_path: str) -> str:
    path_lower = relative_path.lower()
    name_lower = Path(relative_path).name.lower()

    if Path(relative_path).suffix.lower() in {".md", ".txt"} and any(
        token in name_lower for token in ("note", "memo", "summary")
    ):
        return "interpretation notes"

    if any(
        token in path_lower
        for token in (
            "horizon_dependence",
            "horizon_comparison",
            "mortality_risk_horizon",
            "pairwise_overlap",
            "directional_overlap",
            "hard_case_persistence",
            "persistence_distribution",
        )
    ):
        return "horizon artifacts"

    if any(
        token in path_lower
        for token in ("asic_hard_case_comparison", "variable_audit", "sofa_feasibility", "/agreement/")
    ):
        return "comparison artifacts"

    if any(token in path_lower for token in ("hard_cases", "hard_case", "hardcase")):
        return "hard-case definition artifacts"

    return "interpretation notes"


def _preview_priority(relative_path: str) -> int:
    name_lower = Path(relative_path).name.lower()
    if name_lower in {
        "comparison_table.csv",
        "horizon_hard_case_summary.csv",
        "horizon_summary.csv",
        "pairwise_overlap.csv",
    }:
        return 0
    if name_lower in {
        "summary.md",
        "horizon_interpretation_memo.md",
        "final_horizon_summary.md",
        "artifact_foundation_note.md",
        "overlap_note.md",
    }:
        return 1
    if name_lower in {
        "effect_size_figure.png",
        "mortality_risk_horizon_comparison.png",
    }:
        return 2
    if name_lower.endswith(".png"):
        return 3
    if name_lower.endswith(".csv"):
        return 4
    if name_lower.endswith(".json"):
        return 5
    return 6


def discover_viability_artifacts(repo_root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for root in DISCOVERY_ROOTS:
        absolute_root = repo_root / root
        if not absolute_root.exists():
            continue
        for path in absolute_root.rglob("*"):
            if not path.is_file():
                continue
            if _has_excluded_part(path.relative_to(repo_root)):
                continue
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue

            relative_path = path.relative_to(repo_root).as_posix()
            if not _is_relevant_path(relative_path):
                continue

            modified_at = datetime.fromtimestamp(
                path.stat().st_mtime,
                tz=timezone.utc,
            ).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            category = _infer_category(relative_path)
            records.append(
                {
                    "artifact_path": relative_path,
                    "inferred_category": category,
                    "file_type": path.suffix.lower().lstrip("."),
                    "last_modified": modified_at,
                    "parent_dir": path.parent.relative_to(repo_root).as_posix(),
                    "preview_priority": _preview_priority(relative_path),
                }
            )

    inventory = pd.DataFrame.from_records(records)
    if inventory.empty:
        return pd.DataFrame(
            columns=[
                "artifact_path",
                "inferred_category",
                "file_type",
                "last_modified",
                "parent_dir",
                "preview_priority",
            ]
        )

    return inventory.sort_values(
        ["inferred_category", "preview_priority", "artifact_path"],
        kind="stable",
    ).reset_index(drop=True)


def category_inventory(inventory: pd.DataFrame, category: str) -> pd.DataFrame:
    if inventory.empty:
        return inventory.copy()
    category_frame = inventory.loc[inventory["inferred_category"].eq(category)].copy()
    return category_frame[
        ["artifact_path", "file_type", "last_modified", "parent_dir"]
    ].reset_index(drop=True)


def summarize_directory_groups(inventory: pd.DataFrame, category: str) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame(columns=["parent_dir", "file_count", "sample_artifacts"])

    category_frame = inventory.loc[inventory["inferred_category"].eq(category)].copy()
    if category_frame.empty:
        return pd.DataFrame(columns=["parent_dir", "file_count", "sample_artifacts"])

    grouped = (
        category_frame.groupby("parent_dir", dropna=False)["artifact_path"]
        .agg(
            file_count="size",
            sample_artifacts=lambda values: "; ".join(list(values)[:3]),
        )
        .reset_index()
        .sort_values(["file_count", "parent_dir"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    return grouped


def select_preview_inventory(
    inventory: pd.DataFrame,
    category: str,
    *,
    max_per_parent: int = 4,
) -> pd.DataFrame:
    if inventory.empty:
        return inventory.copy()

    category_frame = inventory.loc[inventory["inferred_category"].eq(category)].copy()
    if category_frame.empty:
        return category_frame

    category_frame = category_frame.sort_values(
        ["preview_priority", "artifact_path"],
        kind="stable",
    ).reset_index(drop=True)

    selected_frames: list[pd.DataFrame] = []
    for _, parent_frame in category_frame.groupby("parent_dir", sort=False):
        if len(parent_frame) <= max_per_parent:
            selected_frames.append(parent_frame)
            continue
        selected_frames.append(parent_frame.head(max_per_parent))

    selected = pd.concat(selected_frames, ignore_index=True)
    return selected.sort_values(
        ["preview_priority", "artifact_path"],
        kind="stable",
    ).reset_index(drop=True)


def _locate_key_artifacts(repo_root: Path, inventory: pd.DataFrame) -> dict[str, Path | None]:
    located: dict[str, Path | None] = {}
    if inventory.empty:
        return {name: None for name in KEY_ARTIFACT_SUFFIXES}

    paths = inventory["artifact_path"].tolist()
    for key, suffixes in KEY_ARTIFACT_SUFFIXES.items():
        match: Path | None = None
        for suffix in suffixes:
            for candidate in paths:
                if candidate.endswith(suffix):
                    match = repo_root / candidate
                    break
            if match is not None:
                break
        located[key] = match
    return located


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def _read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text()


def _read_table(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    raise ValueError(f"Unsupported table preview extension: {path.suffix}")


def _extract_backticked_value(text: str, label: str) -> str | None:
    match = re.search(rf"{re.escape(label)}:\s*`([^`]+)`", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _extract_bullets(text: str) -> list[str]:
    bullets: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            bullets.append(stripped.removeprefix("- ").strip())
    return bullets


def _comparison_group_counts(
    comparison_manifest: dict[str, Any],
    comparison_summary_text: str,
) -> dict[str, int]:
    group_counts = comparison_manifest.get("group_counts", {})
    if group_counts:
        return {
            key: int(value)
            for key, value in group_counts.items()
            if isinstance(value, (int, float))
        }

    counts: dict[str, int] = {}
    cohort_match = re.search(r"Fatal 24h stay-level comparison dataset:\s*`(\d+)`\s*stays", comparison_summary_text)
    hard_match = re.search(r"Low-predicted fatal stays:\s*`(\d+)`", comparison_summary_text)
    other_match = re.search(r"Other fatal stays:\s*`(\d+)`", comparison_summary_text)
    if cohort_match:
        counts["total_fatal_stays"] = int(cohort_match.group(1))
    if hard_match:
        counts["low-predicted fatal stays"] = int(hard_match.group(1))
    if other_match:
        counts["other fatal stays"] = int(other_match.group(1))
    return counts


def _normalized_horizon_share_table(
    hard_case_summary: pd.DataFrame,
    horizon_summary: pd.DataFrame,
) -> pd.DataFrame:
    if not horizon_summary.empty:
        if "horizon" in horizon_summary.columns:
            table = horizon_summary.copy()
            table["horizon_label"] = table["horizon"].astype(str)
        else:
            table = horizon_summary.copy()
            table["horizon_label"] = table["horizon_h"].astype(int).map(lambda value: f"{value}h")

        rename_map = {
            "hard_case_n": "hard_case_n",
            "n_hard_cases": "hard_case_n",
            "hard_case_share_among_fatal": "hard_case_share_among_fatal",
            "pct_fatal_hard_cases": "hard_case_share_among_fatal",
            "fatal_last_n": "fatal_last_n",
            "n_fatal_last_points": "fatal_last_n",
            "nonfatal_last_n": "nonfatal_last_n",
            "n_nonfatal_last_points": "nonfatal_last_n",
            "nonfatal_q75_threshold": "nonfatal_q75_threshold",
        }
        available = {
            column: rename_map[column]
            for column in table.columns
            if column in rename_map
        }
        normalized = table.rename(columns=available)
        ordered_columns = [
            "horizon_label",
            "nonfatal_last_n",
            "fatal_last_n",
            "nonfatal_q75_threshold",
            "hard_case_n",
            "hard_case_share_among_fatal",
        ]
        for column in ordered_columns:
            if column not in normalized.columns:
                normalized[column] = pd.NA
        return normalized[ordered_columns].reset_index(drop=True)

    if hard_case_summary.empty:
        return pd.DataFrame(
            columns=[
                "horizon_label",
                "nonfatal_last_n",
                "fatal_last_n",
                "nonfatal_q75_threshold",
                "hard_case_n",
                "hard_case_share_among_fatal",
            ]
        )

    table = hard_case_summary.copy()
    table["horizon_label"] = table["horizon_h"].astype(int).map(lambda value: f"{value}h")
    normalized = table.rename(
        columns={
            "n_nonfatal_last_points": "nonfatal_last_n",
            "n_fatal_last_points": "fatal_last_n",
            "n_hard_cases": "hard_case_n",
            "pct_fatal_hard_cases": "hard_case_share_among_fatal",
        }
    )
    return normalized[
        [
            "horizon_label",
            "nonfatal_last_n",
            "fatal_last_n",
            "nonfatal_q75_threshold",
            "hard_case_n",
            "hard_case_share_among_fatal",
        ]
    ].reset_index(drop=True)


def _top_comparison_highlights(
    comparison_summary_text: str,
    comparison_table: pd.DataFrame,
) -> tuple[str, ...]:
    summary_bullets = [
        bullet
        for bullet in _extract_bullets(comparison_summary_text)
        if "more common" in bullet.lower()
        or "enriched" in bullet.lower()
        or "higher" in bullet.lower()
        or "lower" in bullet.lower()
    ]
    if summary_bullets:
        return tuple(summary_bullets[:3])

    if comparison_table.empty or "absolute_standardized_difference" not in comparison_table.columns:
        return tuple()

    ranked = comparison_table.sort_values(
        "absolute_standardized_difference",
        ascending=False,
        kind="stable",
    ).head(3)
    highlights = []
    for row in ranked.itertuples(index=False):
        highlights.append(
            f"{row.variable_label}: absolute standardized difference {float(row.absolute_standardized_difference):.3f}."
        )
    return tuple(highlights)


def _top_horizon_highlights(
    horizon_memo_text: str,
    horizon_share_table: pd.DataFrame,
    pairwise_overlap: pd.DataFrame,
) -> tuple[str, ...]:
    memo_bullets = _extract_bullets(horizon_memo_text)
    if memo_bullets:
        return tuple(memo_bullets[:4])

    highlights: list[str] = []
    if not horizon_share_table.empty:
        share_text = ", ".join(
            f"{row.horizon_label} {float(row.hard_case_share_among_fatal):.2f}"
            for row in horizon_share_table.itertuples(index=False)
            if pd.notna(row.hard_case_share_among_fatal)
        )
        if share_text:
            highlights.append(f"Hard-case share among fatal stays by horizon: {share_text}.")

    if not pairwise_overlap.empty:
        anchor_pair = pairwise_overlap.loc[
            pairwise_overlap["horizon_a"].astype(str).eq("24h")
            & pairwise_overlap["horizon_b"].astype(str).eq("48h")
        ]
        if not anchor_pair.empty:
            row = anchor_pair.iloc[0]
            highlights.append(
                "24h vs 48h hard-case overlap: "
                f"Jaccard {float(row['jaccard_index']):.3f} on matched fatal denominator {int(row['matched_fatal_n'])}."
            )
    return tuple(highlights[:4])


def _risk_highlights(
    *,
    synthetic_local_outputs: bool,
    variable_audit_text: str,
    sofa_text: str,
    agreement_summary: pd.DataFrame,
    horizon_final_manifest: dict[str, Any],
) -> tuple[str, ...]:
    risks: list[str] = []
    if synthetic_local_outputs:
        risks.append(
            "All discovered hard-case and horizon notes explicitly describe the local values as synthetic implementation-test outputs, so the current readout is workflow-valid but not scientifically interpretable."
        )

    variable_judgement = _extract_backticked_value(variable_audit_text, "Overall judgement")
    blocking_family = _extract_backticked_value(variable_audit_text, "Blocking variable family")
    if variable_judgement:
        message = f"Frozen Issue 3.2 variable-package status: {variable_judgement}."
        if blocking_family:
            message += f" Blocking family: {blocking_family}."
        risks.append(message)

    sofa_classification = _extract_backticked_value(sofa_text, "Final classification")
    if sofa_classification:
        risks.append(f"SOFA feasibility audit result: {sofa_classification}.")

    if not agreement_summary.empty:
        anchor_row = agreement_summary.loc[agreement_summary["horizon_h"].astype(int).eq(24)]
        if not anchor_row.empty:
            row = anchor_row.iloc[0]
            risks.append(
                "Cross-model hard-case agreement is limited at 24h: "
                f"logistic vs recalibrated XGBoost Jaccard {float(row['jaccard_hard_case_overlap']):.2f} "
                f"with logistic-hard confirmation by XGBoost {float(row['pct_logistic_hard_confirmed_by_xgb']):.2f}."
            )

    interpretation_label = horizon_final_manifest.get("interpretation_label")
    if interpretation_label:
        risks.append(
            f"The saved horizon package labels the pattern '{interpretation_label}', which is weaker than a clean single-form persistence story."
        )

    return tuple(risks[:5])


def _decision_findings(
    *,
    hard_case_rule: str | None,
    target_horizon_label: str | None,
    comparison_group_counts: dict[str, int],
    horizon_share_table: pd.DataFrame,
    key_artifacts: dict[str, Path | None],
    repo_root: Path,
) -> tuple[str, ...]:
    findings: list[str] = []
    if hard_case_rule:
        findings.append(f"Hard-case definition used: `{hard_case_rule}`.")

    low_count = comparison_group_counts.get("low-predicted fatal stays")
    other_count = comparison_group_counts.get("other fatal stays")
    total_count = comparison_group_counts.get("total_fatal_stays")
    if low_count is not None and other_count is not None:
        label = target_horizon_label or "24h"
        if total_count is not None:
            findings.append(
                f"Key counts for the {label} fatal comparison slice: `{low_count}` low-predicted fatal vs `{other_count}` other fatal stays (total `{total_count}`)."
            )
        else:
            findings.append(
                f"Key counts for the {label} fatal comparison slice: `{low_count}` low-predicted fatal vs `{other_count}` other fatal stays."
            )

    if not horizon_share_table.empty:
        share_text = ", ".join(
            f"`{row.horizon_label}` `{float(row.hard_case_share_among_fatal):.2f}`"
            for row in horizon_share_table.itertuples(index=False)
            if pd.notna(row.hard_case_share_among_fatal)
        )
        if share_text:
            findings.append(f"Horizon-specific low-predicted death shares located: {share_text}.")

    comparison_paths = [
        _display_path(key_artifacts.get(name), repo_root)
        for name in ("comparison_table", "comparison_details", "variable_audit_table")
        if key_artifacts.get(name) is not None
    ]
    if comparison_paths:
        findings.append(
            "Main comparison table paths: " + ", ".join(f"`{path}`" for path in comparison_paths) + "."
        )

    figure_paths = [
        _display_path(key_artifacts.get(name), repo_root)
        for name in ("comparison_figure", "horizon_figure")
        if key_artifacts.get(name) is not None
    ]
    if figure_paths:
        findings.append("Main figure paths: " + ", ".join(f"`{path}`" for path in figure_paths) + ".")

    return tuple(findings)


def _open_evidence_gaps(
    *,
    inventory: pd.DataFrame,
    key_artifacts: dict[str, Path | None],
    synthetic_local_outputs: bool,
) -> tuple[str, ...]:
    gaps: list[str] = []
    for label, artifact_key in EXPECTED_EVIDENCE_ITEMS:
        if key_artifacts.get(artifact_key) is None:
            gaps.append(f"{label} was not found in the searched Sprint 3 artifact roots.")

    artifact_paths_lower = inventory["artifact_path"].str.lower().tolist() if not inventory.empty else []
    for gap_text, search_tokens in MISSING_SEARCH_DESCRIPTIONS:
        if not any(all(token in path for token in search_tokens[:1]) for path in artifact_paths_lower):
            if gap_text.startswith("No dedicated treatment-limitation"):
                if not any(any(token in path for token in search_tokens) for path in artifact_paths_lower):
                    gaps.append(gap_text)
            elif gap_text.startswith("No pre-existing ASIC viability memo"):
                if not any("viability" in path and "memo" in path for path in artifact_paths_lower):
                    gaps.append(gap_text)

    if synthetic_local_outputs:
        gaps.append(
            "Only local synthetic stand-in outputs were located here. The same review must be rerun on full ASIC HPC artifacts before treating the memo as a scientific decision."
        )

    return tuple(gaps)


def build_review_context(repo_root: Path) -> ReviewContext:
    repo_root = repo_root.resolve()
    inventory = discover_viability_artifacts(repo_root)
    key_artifacts = _locate_key_artifacts(repo_root, inventory)

    hard_case_manifest = _read_json(key_artifacts.get("hard_case_manifest"))
    comparison_manifest = _read_json(key_artifacts.get("comparison_manifest"))
    horizon_final_manifest = _read_json(key_artifacts.get("horizon_final_manifest"))

    hard_case_summary = _read_table(key_artifacts.get("hard_case_summary"))
    horizon_summary = _read_table(key_artifacts.get("horizon_foundation_summary"))
    comparison_table = _read_table(key_artifacts.get("comparison_table"))
    pairwise_overlap = _read_table(key_artifacts.get("pairwise_overlap"))
    agreement_summary = _read_table(key_artifacts.get("agreement_summary"))

    comparison_summary_text = _read_text(key_artifacts.get("comparison_summary"))
    horizon_memo_text = _read_text(key_artifacts.get("horizon_memo"))
    foundation_note_text = _read_text(key_artifacts.get("horizon_foundation_note"))
    overlap_note_text = _read_text(key_artifacts.get("overlap_note"))
    variable_audit_text = _read_text(key_artifacts.get("variable_audit_memo"))
    sofa_text = _read_text(key_artifacts.get("sofa_feasibility_memo"))
    final_horizon_summary_text = _read_text(key_artifacts.get("horizon_final_summary"))

    synthetic_local_outputs = any(
        "synthetic" in text.lower()
        for text in (
            comparison_summary_text,
            horizon_memo_text,
            foundation_note_text,
            overlap_note_text,
            variable_audit_text,
            sofa_text,
            final_horizon_summary_text,
        )
        if text
    )

    hard_case_rule = (
        hard_case_manifest.get("hard_case_rule")
        or comparison_manifest.get("hard_case_rule")
        or None
    )

    target_horizon_h = comparison_manifest.get("target_horizon_h")
    target_horizon_label = f"{int(target_horizon_h)}h" if target_horizon_h is not None else "24h"

    comparison_group_counts = _comparison_group_counts(comparison_manifest, comparison_summary_text)
    horizon_share_table = _normalized_horizon_share_table(hard_case_summary, horizon_summary)

    comparison_highlights = _top_comparison_highlights(comparison_summary_text, comparison_table)
    horizon_highlights = _top_horizon_highlights(horizon_memo_text, horizon_share_table, pairwise_overlap)
    risk_highlights = _risk_highlights(
        synthetic_local_outputs=synthetic_local_outputs,
        variable_audit_text=variable_audit_text,
        sofa_text=sofa_text,
        agreement_summary=agreement_summary,
        horizon_final_manifest=horizon_final_manifest,
    )

    core_artifact_count = sum(
        key_artifacts.get(name) is not None
        for name in ("hard_case_manifest", "comparison_table", "horizon_foundation_summary", "horizon_memo")
    )
    descriptive_core_assessment = (
        "Provisionally yes, as a bounded ASIC descriptive core, but only after a full-data ASIC rerun."
        if core_artifact_count >= 3
        else "Not yet; the located artifact set is too incomplete to support a descriptive-core claim."
    )
    decomposition_decision = "GO, but secondary only" if core_artifact_count >= 3 else "NO-GO for now"

    decision_findings = _decision_findings(
        hard_case_rule=hard_case_rule,
        target_horizon_label=target_horizon_label,
        comparison_group_counts=comparison_group_counts,
        horizon_share_table=horizon_share_table,
        key_artifacts=key_artifacts,
        repo_root=repo_root,
    )
    open_evidence_gaps = _open_evidence_gaps(
        inventory=inventory,
        key_artifacts=key_artifacts,
        synthetic_local_outputs=synthetic_local_outputs,
    )

    existing_search_roots = tuple(
        root.as_posix() for root in DISCOVERY_ROOTS if (repo_root / root).exists()
    )

    return ReviewContext(
        repo_root=repo_root,
        inventory=inventory,
        key_artifacts=key_artifacts,
        search_roots=existing_search_roots,
        hard_case_rule=hard_case_rule,
        target_horizon_label=target_horizon_label,
        comparison_group_counts=comparison_group_counts,
        horizon_share_table=horizon_share_table,
        comparison_highlights=comparison_highlights,
        horizon_highlights=horizon_highlights,
        risk_highlights=risk_highlights,
        decision_findings=decision_findings,
        open_evidence_gaps=open_evidence_gaps,
        synthetic_local_outputs=synthetic_local_outputs,
        descriptive_core_assessment=descriptive_core_assessment,
        decomposition_decision=decomposition_decision,
    )


def decision_findings_markdown(context: ReviewContext) -> str:
    lines = ["## Decision-relevant findings to inspect", ""]
    if context.decision_findings:
        lines.extend(f"- {line}" for line in context.decision_findings)
    else:
        lines.append("- No decision-relevant findings could be extracted from the currently discovered artifacts.")
    return "\n".join(lines)


def open_evidence_gaps_markdown(context: ReviewContext) -> str:
    lines = ["## Open evidence gaps", ""]
    if context.open_evidence_gaps:
        lines.extend(f"- {line}" for line in context.open_evidence_gaps)
    else:
        lines.append("- No expected-item gaps were detected in the searched artifact roots.")
    return "\n".join(lines)


def render_artifact_preview(
    repo_root: Path,
    artifact_path: str,
    *,
    max_rows: int = 8,
    max_text_chars: int = 12000,
) -> None:
    from IPython.display import IFrame, Image, Markdown, display

    path = repo_root / artifact_path
    display(Markdown(f"### `{artifact_path}`"))
    suffix = path.suffix.lower()

    if suffix in {".md", ".txt"}:
        text = path.read_text()
        if len(text) > max_text_chars:
            text = text[:max_text_chars].rstrip() + "\n\n...\n"
        display(Markdown(text))
        return

    if suffix in {".csv", ".parquet", ".feather"}:
        try:
            table = _read_table(path)
        except Exception as exc:  # pragma: no cover - notebook-only fallback
            display(Markdown(f"- Could not preview `{artifact_path}`: `{exc}`"))
            return
        if table.empty:
            display(Markdown("- Table is empty."))
            return
        if len(table) <= max_rows and len(table.columns) <= 12:
            display(table)
            return
        display(
            Markdown(
                f"- Previewing the first `{min(len(table), max_rows)}` rows out of `{len(table)}` total rows."
            )
        )
        display(table.head(max_rows))
        return

    if suffix in {".png", ".jpg", ".jpeg"}:
        display(Image(filename=str(path)))
        return

    if suffix == ".pdf":
        display(IFrame(src=str(path), width="100%", height=720))
        return

    if suffix == ".json":
        payload = _read_json(path)
        rendered = json.dumps(payload, indent=2)
        if len(rendered) > max_text_chars:
            rendered = rendered[:max_text_chars].rstrip() + "\n..."
        display(Markdown(f"```json\n{rendered}\n```"))
        return

    if suffix == ".ipynb":
        display(Markdown("- Notebook artifact located. Open it directly for interactive inspection."))
        return

    display(Markdown(f"- Preview not implemented for `{artifact_path}`."))


def _table_to_markdown(table: pd.DataFrame) -> str:
    if table.empty:
        return "(not found)"

    normalized = table.fillna("").astype(str)
    headers = list(normalized.columns)
    separator = ["---"] * len(headers)
    rows = normalized.values.tolist()

    markdown_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    markdown_lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(markdown_lines)


def _share_table_for_markdown(context: ReviewContext) -> str:
    if context.horizon_share_table.empty:
        return "(not found)"
    share_table = context.horizon_share_table.copy()
    numeric_columns = [
        "nonfatal_q75_threshold",
        "hard_case_share_among_fatal",
    ]
    for column in numeric_columns:
        if column in share_table.columns:
            share_table[column] = pd.to_numeric(share_table[column], errors="coerce").map(
                lambda value: f"{value:.3f}" if pd.notna(value) else ""
            )
    return _table_to_markdown(share_table)


def render_evidence_pack(context: ReviewContext) -> str:
    repo_root = context.repo_root
    evidence_paths = [
        context.key_artifacts.get("hard_case_manifest"),
        context.key_artifacts.get("hard_case_summary"),
        context.key_artifacts.get("comparison_table"),
        context.key_artifacts.get("comparison_summary"),
        context.key_artifacts.get("variable_audit_memo"),
        context.key_artifacts.get("agreement_summary"),
        context.key_artifacts.get("horizon_foundation_summary"),
        context.key_artifacts.get("pairwise_overlap"),
        context.key_artifacts.get("horizon_memo"),
        context.key_artifacts.get("horizon_figure"),
    ]
    seen: set[str] = set()
    evidence_lines = []
    for path in evidence_paths:
        rendered = _display_path(path, repo_root)
        if rendered in seen or rendered == "(not found)":
            continue
        seen.add(rendered)
        evidence_lines.append(f"- `{rendered}`")

    low_count = context.comparison_group_counts.get("low-predicted fatal stays")
    other_count = context.comparison_group_counts.get("other fatal stays")
    total_count = context.comparison_group_counts.get("total_fatal_stays")

    definition_lines = []
    if context.hard_case_rule:
        definition_lines.append(f"- Frozen rule located: `{context.hard_case_rule}`.")
    if low_count is not None and other_count is not None:
        if total_count is not None:
            definition_lines.append(
                f"- `{context.target_horizon_label}` fatal comparison slice located: `{low_count}` low-predicted fatal vs `{other_count}` other fatal stays (total `{total_count}`)."
            )
        else:
            definition_lines.append(
                f"- `{context.target_horizon_label}` fatal comparison slice located: `{low_count}` low-predicted fatal vs `{other_count}` other fatal stays."
            )
    if context.key_artifacts.get("hard_case_summary") is not None:
        definition_lines.append(
            f"- Hard-case count / threshold source: `{_display_path(context.key_artifacts['hard_case_summary'], repo_root)}`."
        )
    if context.horizon_share_table.empty:
        definition_lines.append("- No horizon-share table could be extracted automatically.")
    else:
        definition_lines.append("- Horizon-specific q75 thresholds and hard-case shares were extracted:")
        definition_lines.append("")
        definition_lines.append(_share_table_for_markdown(context))

    comparison_lines = []
    if context.comparison_highlights:
        comparison_lines.extend(f"- {line}" for line in context.comparison_highlights)
    else:
        comparison_lines.append("- No concise comparison highlight could be extracted automatically.")
    if context.key_artifacts.get("comparison_table") is not None:
        comparison_lines.append(
            f"- Main comparison table: `{_display_path(context.key_artifacts['comparison_table'], repo_root)}`."
        )
    if context.key_artifacts.get("comparison_figure") is not None:
        comparison_lines.append(
            f"- Main comparison figure: `{_display_path(context.key_artifacts['comparison_figure'], repo_root)}`."
        )

    horizon_lines = []
    if context.horizon_highlights:
        horizon_lines.extend(f"- {line}" for line in context.horizon_highlights)
    else:
        horizon_lines.append("- No concise horizon-dependence highlight could be extracted automatically.")
    if context.key_artifacts.get("pairwise_overlap") is not None:
        horizon_lines.append(
            f"- Main overlap table: `{_display_path(context.key_artifacts['pairwise_overlap'], repo_root)}`."
        )
    if context.key_artifacts.get("horizon_memo") is not None:
        horizon_lines.append(
            f"- Main interpretation memo: `{_display_path(context.key_artifacts['horizon_memo'], repo_root)}`."
        )

    strengthen_descriptive = [
        "- The hard-case rule is frozen and recoverable from a saved manifest rather than implied retrospectively.",
        "- A concrete low-predicted-versus-other-fatal comparison package exists with tables, a figure, and a short summary note.",
        "- Horizon dependence was materialized into summary tables, overlap tables, and a final interpretation memo rather than left implicit.",
    ]
    weaken_descriptive = [
        "- The located notes repeatedly say the local values are synthetic implementation-test outputs, so the current readout is not a scientific claim yet.",
        "- The local comparison slice is very small and explicitly flagged as a bounded descriptive comparison.",
        "- The final horizon package labels the pattern as changing form rather than a clean stable subtype."
    ]
    strengthen_decomposition = [
        "- There is at least some recurring low-risk fatal structure across horizons, so a secondary summary device is not obviously pointless.",
        "- The artifact set is now organized enough to ask whether decomposition adds anything beyond the descriptive hard-case story.",
    ]
    weaken_decomposition = [
        "- The descriptive story already has its own rule, comparison table, and horizon memo, so decomposition is not needed to make Chapter 1 legible.",
        "- Cross-model hard-case agreement is limited in the saved agreement summary, which weakens confidence in a fragile summary-model layer.",
        "- Key sensitivity pieces remain incomplete or negative, including the variable-package readiness gap and the non-feasible SOFA route.",
    ]

    risk_lines = [f"- {line}" for line in context.risk_highlights] or [
        "- No additional risk highlights were extracted automatically."
    ]

    missing_lines = [f"- {line}" for line in context.open_evidence_gaps] or [
        "- No expected-item gaps were detected in the searched roots."
    ]

    purpose_paragraph = (
        "This document compresses the existing ASIC Sprint 3 hard-case and horizon-dependence artifacts into a short review pack for Issue 3.4. "
        "It is intentionally bounded to located artifacts and does not rerun the analyses. "
        "The discovered notes explicitly mark the local numbers as synthetic stand-in outputs, so this pack is for workflow and argument structure rather than scientific inference."
    )

    sections = [
        "# Sprint 3 ASIC Viability Evidence Pack",
        "",
        "## Purpose",
        purpose_paragraph,
        "",
        "## Evidence located",
        *evidence_lines,
        "",
        "## Hard-case definition summary",
        *definition_lines,
        "",
        "## ASIC hard-case comparison summary",
        *comparison_lines,
        "",
        "## Horizon dependence summary",
        *horizon_lines,
        "",
        "## Preliminary decision-relevant interpretation",
        "- What strengthens descriptive viability:",
        *strengthen_descriptive,
        "- What weakens descriptive viability:",
        *weaken_descriptive,
        "- What strengthens decomposition:",
        *strengthen_decomposition,
        "- What weakens decomposition:",
        *weaken_decomposition,
        "",
        "## Main remaining risks",
        *risk_lines,
        "",
        "## Missing evidence / unresolved items",
        *missing_lines,
        "",
    ]
    return "\n".join(sections)


def render_memo_draft(context: ReviewContext) -> str:
    repo_root = context.repo_root
    low_count = context.comparison_group_counts.get("low-predicted fatal stays")
    other_count = context.comparison_group_counts.get("other fatal stays")
    total_count = context.comparison_group_counts.get("total_fatal_stays")

    counts_text = "a concrete fatal-stay comparison package exists."
    if low_count is not None and other_count is not None:
        counts_text = (
            f"a concrete `{context.target_horizon_label}` fatal-stay comparison package exists with `{low_count}` "
            f"low-predicted fatal stays and `{other_count}` other fatal stays"
        )
        if total_count is not None:
            counts_text += f" out of `{total_count}` fatal stays"
        counts_text += "."

    horizon_text = "A horizon summary package was located."
    if not context.horizon_share_table.empty:
        share_text = ", ".join(
            f"{row.horizon_label} {float(row.hard_case_share_among_fatal):.2f}"
            for row in context.horizon_share_table.itertuples(index=False)
            if pd.notna(row.hard_case_share_among_fatal)
        )
        if share_text:
            horizon_text = f"Located horizon shares among fatal stays are {share_text}."

    agreement_summary = _read_table(context.key_artifacts.get("agreement_summary"))
    agreement_sentence = ""
    if not agreement_summary.empty:
        anchor_row = agreement_summary.loc[agreement_summary["horizon_h"].astype(int).eq(24)]
        if not anchor_row.empty:
            row = anchor_row.iloc[0]
            agreement_sentence = (
                " The saved logistic-versus-recalibrated-XGBoost agreement artifact is a caution signal: "
                f"24h Jaccard is {float(row['jaccard_hard_case_overlap']):.2f}."
            )

    question = (
        "Does the current ASIC hard-case and horizon artifact set already support Chapter 1 as a descriptive hard-case chapter, "
        "and should decomposition remain in scope?"
    )
    evidence_summary = (
        f"The located artifact set documents a frozen hard-case rule (`{context.hard_case_rule or 'not found'}`), {counts_text} "
        f"{horizon_text} The saved horizon interpretation memo and figure are `{_display_path(context.key_artifacts.get('horizon_memo'), repo_root)}` "
        f"and `{_display_path(context.key_artifacts.get('horizon_figure'), repo_root)}`.{agreement_sentence} "
        "The same hard-case, foundation, overlap, and final-horizon notes explicitly state that the local values come from synthetic stand-in data."
    )
    descriptive_core = (
        "Provisionally yes, as a bounded descriptive core. "
        "The hard-case rule is explicit, the low-predicted-versus-other-fatal comparison has already been turned into tables and a figure, "
        "and the horizon package shows that the low-risk fatal burden does not vanish immediately when the horizon changes. "
        "That is enough to frame Chapter 1 first as a descriptive hard-case chapter. "
        "It is not enough for a firm scientific claim yet because the local repo is synthetic and some robustness / sensitivity pieces remain incomplete."
    )
    decomposition = (
        f"`{context.decomposition_decision}`\n\n"
        "The descriptive argument already stands on ASIC hard-case definition, comparison, and horizon structure, so decomposition is not needed to make the chapter work. "
        "If retained, decomposition should stay clearly secondary and easy to drop. "
        "It should not become the chapter's organizing logic unless the full-data ASIC rerun and later replication materially strengthen the case."
    )
    main_risks = (
        "The present readout is provisional because the located local outputs are synthetic implementation-test artifacts. "
        "The saved horizon package labels the pattern as changing form rather than a clean horizon-stable subtype. "
        "The variable audit says the frozen Issue 3.2 package is not fully ready because exact age is absent, and the SOFA feasibility audit says standard SOFA is not feasible. "
        "The agreement artifact also suggests the hard-case signal is definition-sensitive rather than obviously model-invariant."
    )
    mimic_dependency = (
        "MIMIC still has to answer whether the ASIC descriptive structure actually replicates, whether the same subgroup contrasts remain visible, "
        "whether the horizon pattern still looks like persistence versus change of form, and whether any retained decomposition summary is robust enough to keep. "
        "MIMIC should validate or downgrade the ASIC descriptive story; it should not be used to rescue a decomposition that ASIC itself does not clearly need."
    )
    recommendation = (
        "Use the ASIC hard-case comparison and horizon package as the Chapter 1 backbone, keep decomposition explicitly secondary, "
        "and rerun this exact review workflow on full ASIC HPC artifacts before treating the memo as a scientific decision. "
        "If the full-data ASIC rerun still supports the descriptive core, retain decomposition only as an optional summary layer and drop it quickly if later replication does not materially strengthen the case."
    )

    sections = [
        "# ASIC Viability Memo for Chapter 1",
        "",
        "## Question",
        question,
        "",
        "## Evidence summary",
        evidence_summary,
        "",
        "## Does ASIC already provide a defensible descriptive core?",
        descriptive_core,
        "",
        "## Decomposition decision",
        decomposition,
        "",
        "## Main remaining risks",
        main_risks,
        "",
        "## What still depends on MIMIC",
        mimic_dependency,
        "",
        "## Provisional recommendation",
        recommendation,
        "",
    ]
    return "\n".join(sections)


def _source_lines(text: str) -> list[str]:
    return [line + "\n" for line in text.splitlines()]


def build_notebook_payload() -> dict[str, Any]:
    intro_markdown = """# Sprint 3 ASIC Viability Review

This notebook is a lightweight artifact-review notebook for Chapter 1 Sprint 3 Issue 3.4.
It discovers the existing ASIC hard-case and horizon-dependence artifacts, previews them, and highlights the evidence that matters for the descriptive-core versus decomposition decision.

Important caveat: this notebook reviews saved artifacts only. It does not rerun the Sprint 3 analyses.
If the located artifacts are from the local synthetic stand-in data, treat the numbers as workflow-valid implementation checks rather than scientific findings."""

    setup_code = """from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from IPython.display import Markdown, display


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src" / "chapter1_mortality_decomposition").exists():
            return candidate
    raise FileNotFoundError("Could not locate the Chapter 1 repo root from the current working directory.")


REPO_ROOT = find_repo_root(Path.cwd().resolve())
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chapter1_mortality_decomposition.ch1_asic_descriptive_viability import (
    ORDERED_CATEGORIES,
    build_review_context,
    category_inventory,
    decision_findings_markdown,
    open_evidence_gaps_markdown,
    render_artifact_preview,
    select_preview_inventory,
    summarize_directory_groups,
)

context = build_review_context(REPO_ROOT)
inventory = context.inventory.copy()

display(Markdown(f"**Repo root:** `{REPO_ROOT}`"))
display(Markdown("**Search roots:** " + ", ".join(f"`{root}`" for root in context.search_roots)))
display(Markdown(f"**Discovered candidate artifacts:** `{len(inventory)}`"))
"""

    grouped_listing_code = """for category in ORDERED_CATEGORIES:
    display(Markdown(f"## {category}"))
    category_table = category_inventory(inventory, category)
    if category_table.empty:
        display(Markdown("- No artifacts found in this category."))
        continue
    display(category_table)
    grouped = summarize_directory_groups(inventory, category)
    if not grouped.empty:
        display(Markdown("Filtered directory / grouped view"))
        display(grouped)
"""

    inventory_code = """inventory[["artifact_path", "inferred_category", "file_type", "last_modified"]]"""

    findings_code = """display(Markdown(decision_findings_markdown(context)))"""

    preview_code = """for category in ORDERED_CATEGORIES:
    preview_table = select_preview_inventory(inventory, category)
    if preview_table.empty:
        continue
    full_category_table = category_inventory(inventory, category)
    display(Markdown(f"## Preview subset: {category}"))
    if len(preview_table) < len(full_category_table):
        display(
            Markdown(
                f"Showing a filtered preview of `{len(preview_table)}` artifacts out of `{len(full_category_table)}` in this category to avoid dumping large sets of near-duplicate files."
            )
        )
    for row in preview_table.itertuples(index=False):
        render_artifact_preview(REPO_ROOT, row.artifact_path)
"""

    gap_code = """display(Markdown(open_evidence_gaps_markdown(context)))"""

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": _source_lines(intro_markdown),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": _source_lines(setup_code),
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": _source_lines("## Discovered artifact groups"),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": _source_lines(grouped_listing_code),
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": _source_lines("## Artifact inventory"),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": _source_lines(inventory_code),
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": _source_lines("## Decision-relevant findings to inspect"),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": _source_lines(findings_code),
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": _source_lines("## Artifact previews"),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": _source_lines(preview_code),
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": _source_lines("## Open evidence gaps"),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": _source_lines(gap_code),
        },
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def generate_ch1_asic_descriptive_viability(
    *,
    repo_root: Path,
    notebook_path: Path,
    evidence_pack_path: Path,
    memo_path: Path,
) -> ReviewContext:
    repo_root = repo_root.resolve()
    context = build_review_context(repo_root)
    notebook_payload = build_notebook_payload()

    ensure_directory(notebook_path.parent)
    ensure_directory(evidence_pack_path.parent)
    ensure_directory(memo_path.parent)

    write_text(json.dumps(notebook_payload, indent=2), notebook_path)
    write_text(render_evidence_pack(context), evidence_pack_path)
    write_text(render_memo_draft(context), memo_path)
    return context


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Sprint 3 ASIC viability review notebook and markdown outputs from existing artifacts."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing src/, notebooks/, reports/, and artifacts/.",
    )
    parser.add_argument(
        "--notebook-path",
        type=Path,
        default=Path("notebooks") / "ch1_asic_descriptive_viability_review.ipynb",
        help="Output path for the review notebook.",
    )
    parser.add_argument(
        "--evidence-pack-path",
        type=Path,
        default=Path("reports") / "ch1_asic_descriptive_viability_evidence_pack.md",
        help="Output path for the evidence-pack markdown.",
    )
    parser.add_argument(
        "--memo-path",
        type=Path,
        default=Path("reports") / "ch1_asic_descriptive_viability_memo_draft.md",
        help="Output path for the memo-draft markdown.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()

    notebook_path = (repo_root / args.notebook_path).resolve()
    evidence_pack_path = (repo_root / args.evidence_pack_path).resolve()
    memo_path = (repo_root / args.memo_path).resolve()

    context = generate_ch1_asic_descriptive_viability(
        repo_root=repo_root,
        notebook_path=notebook_path,
        evidence_pack_path=evidence_pack_path,
        memo_path=memo_path,
    )

    print(f"Discovered {len(context.inventory)} candidate artifacts.")
    print(f"Notebook: {notebook_path}")
    print(f"Evidence pack: {evidence_pack_path}")
    print(f"Memo draft: {memo_path}")
    print(f"Decomposition decision: {context.decomposition_decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
