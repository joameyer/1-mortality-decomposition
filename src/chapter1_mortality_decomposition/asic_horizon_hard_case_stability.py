from __future__ import annotations

import argparse
import itertools
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

from chapter1_mortality_decomposition.utils import (
    ensure_directory,
    normalize_boolean_codes,
    read_dataframe,
    require_columns,
    write_dataframe,
    write_text,
)


DEFAULT_HARD_CASE_DIR = (
    Path("artifacts")
    / "chapter1"
    / "evaluation"
    / "asic"
    / "hard_cases"
    / "primary_medians"
    / "logistic_regression"
)
DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "chapter1"
    / "evaluation"
    / "asic"
    / "horizon_dependence"
    / "overlap"
)
DEFAULT_HORIZONS = (8, 16, 24, 48, 72)
REQUIRED_STAY_LEVEL_COLUMNS = {
    "stay_id_global",
    "hospital_id",
    "horizon_h",
    "label_value",
    "hard_case_flag",
}
HORIZON_ORDER = tuple(int(horizon) for horizon in DEFAULT_HORIZONS)


@dataclass(frozen=True)
class HorizonHardCaseStabilityArtifacts:
    pairwise_denominators_path: Path
    pairwise_overlap_path: Path
    directional_overlap_path: Path
    jaccard_heatmap_path: Path
    directional_overlap_heatmap_path: Path
    hard_case_persistence_path: Path
    persistence_distribution_path: Path
    persistence_barplot_path: Path
    overlap_note_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class HorizonHardCaseStabilityRunResult:
    hard_case_dir: Path
    output_dir: Path
    horizons: tuple[int, ...]
    harmonized_stay_level: pd.DataFrame
    pairwise_denominators: pd.DataFrame
    pairwise_overlap: pd.DataFrame
    directional_overlap: pd.DataFrame
    hard_case_persistence: pd.DataFrame
    persistence_distribution: pd.DataFrame
    artifacts: HorizonHardCaseStabilityArtifacts
    package3_ready: bool


def _require_matplotlib() -> None:
    if plt is None:
        raise ImportError(
            "matplotlib is required for the Chapter 1 ASIC horizon hard-case stability package."
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


def _normalize_horizons(horizons: Sequence[int] | None) -> tuple[int, ...]:
    values = tuple(sorted({int(horizon) for horizon in (horizons or DEFAULT_HORIZONS)}))
    unsupported = sorted(set(values) - set(DEFAULT_HORIZONS))
    if unsupported:
        raise ValueError(f"Unsupported horizons requested: {unsupported}")
    return values


def _horizon_label(horizon_h: int) -> str:
    return f"{int(horizon_h)}h"


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def _markdown_table(rows: list[dict[str, object]], columns: Sequence[str]) -> str:
    if not rows:
        return ""

    rendered_rows = []
    widths = {column: len(column) for column in columns}
    for row in rows:
        rendered = {column: str(row.get(column, "")) for column in columns}
        rendered_rows.append(rendered)
        for column, value in rendered.items():
            widths[column] = max(widths[column], len(value))

    header = "| " + " | ".join(column.ljust(widths[column]) for column in columns) + " |"
    divider = "| " + " | ".join("-" * widths[column] for column in columns) + " |"
    body = [
        "| " + " | ".join(row[column].ljust(widths[column]) for column in columns) + " |"
        for row in rendered_rows
    ]
    return "\n".join([header, divider, *body])


def _load_manifest(hard_case_dir: Path) -> dict[str, object]:
    manifest_path = hard_case_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Expected hard-case manifest at {manifest_path}")
    return json.loads(manifest_path.read_text())


def _load_stay_level_artifact(
    hard_case_dir: Path,
    *,
    horizons: tuple[int, ...],
) -> tuple[pd.DataFrame, Path, dict[str, object]]:
    manifest = _load_manifest(hard_case_dir)
    if "stay_level_artifact" not in manifest:
        raise KeyError(f"{hard_case_dir / 'run_manifest.json'} is missing stay_level_artifact.")

    stay_level_path = Path(str(manifest["stay_level_artifact"]))
    stay_level = read_dataframe(stay_level_path)
    require_columns(stay_level, REQUIRED_STAY_LEVEL_COLUMNS, str(stay_level_path))

    harmonized = stay_level.copy()
    harmonized["stay_id"] = harmonized["stay_id_global"].astype("string")
    harmonized["hospital_id"] = harmonized["hospital_id"].astype("string")
    harmonized["horizon_h"] = pd.to_numeric(harmonized["horizon_h"], errors="coerce").astype("Int64")
    harmonized["fatal_flag"] = pd.to_numeric(
        harmonized["label_value"],
        errors="coerce",
    ).astype("Int64").eq(1)
    harmonized["hard_case_flag"] = normalize_boolean_codes(harmonized["hard_case_flag"]).fillna(False).astype(bool)
    harmonized["available_flag"] = True

    filtered = harmonized[harmonized["horizon_h"].isin(list(horizons))].copy()
    observed_horizons = tuple(sorted(int(value) for value in filtered["horizon_h"].dropna().unique()))
    if observed_horizons != horizons:
        raise ValueError(
            f"Stay-level artifact {stay_level_path} covered horizons {observed_horizons}, expected {horizons}."
        )

    duplicate_count = int(filtered.duplicated(["stay_id", "horizon_h"]).sum())
    if duplicate_count:
        raise ValueError(
            f"Stay-level artifact {stay_level_path} contains {duplicate_count} duplicated stay_id/horizon rows."
        )

    hospital_conflicts = filtered.groupby("stay_id", observed=True)["hospital_id"].nunique(dropna=False).gt(1)
    if bool(hospital_conflicts.any()):
        conflicted = hospital_conflicts[hospital_conflicts].index.tolist()
        raise ValueError(
            "Stable stay identifier validation failed because hospital_id changed across horizons for: "
            f"{conflicted[:5]}"
        )

    invalid_hard_cases = filtered["hard_case_flag"] & ~filtered["fatal_flag"]
    if bool(invalid_hard_cases.any()):
        examples = filtered.loc[invalid_hard_cases, ["stay_id", "horizon_h"]].head(5)
        raise ValueError(
            "Found hard_case_flag=True rows outside the fatal population. "
            f"Examples: {examples.to_dict(orient='records')}"
        )

    output = filtered[
        [
            "stay_id",
            "hospital_id",
            "horizon_h",
            "fatal_flag",
            "hard_case_flag",
            "available_flag",
        ]
    ].sort_values(["horizon_h", "hospital_id", "stay_id"], kind="stable").reset_index(drop=True)
    return output, stay_level_path, manifest


def _build_pairwise_tables(
    harmonized_stay_level: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairwise_denominator_rows: list[dict[str, object]] = []
    pairwise_overlap_rows: list[dict[str, object]] = []

    fatal_by_horizon = {
        int(horizon): harmonized_stay_level[
            harmonized_stay_level["horizon_h"].astype(int).eq(int(horizon))
            & harmonized_stay_level["fatal_flag"].astype(bool)
        ][["stay_id", "hospital_id", "hard_case_flag"]].copy()
        for horizon in horizons
    }

    for horizon_a, horizon_b in itertools.combinations(horizons, 2):
        fatal_a = fatal_by_horizon[int(horizon_a)].rename(
            columns={
                "hospital_id": "hospital_id_a",
                "hard_case_flag": "hard_case_flag_a",
            }
        )
        fatal_b = fatal_by_horizon[int(horizon_b)].rename(
            columns={
                "hospital_id": "hospital_id_b",
                "hard_case_flag": "hard_case_flag_b",
            }
        )
        matched = fatal_a.merge(fatal_b, on="stay_id", how="inner")

        hospital_mismatch = matched[
            matched["hospital_id_a"].astype("string").ne(matched["hospital_id_b"].astype("string"))
        ]
        if not hospital_mismatch.empty:
            raise ValueError(
                "Hospital ID mismatch detected while matching fatal stays across horizons. "
                f"Examples: {hospital_mismatch[['stay_id', 'hospital_id_a', 'hospital_id_b']].head(5).to_dict(orient='records')}"
            )

        matched_fatal_n = int(matched.shape[0])
        fatal_n_a = int(fatal_a.shape[0])
        fatal_n_b = int(fatal_b.shape[0])
        dropped_fatal_a_unmatched = int(fatal_n_a - matched_fatal_n)
        dropped_fatal_b_unmatched = int(fatal_n_b - matched_fatal_n)

        pairwise_denominator_rows.append(
            {
                "horizon_a": _horizon_label(horizon_a),
                "horizon_b": _horizon_label(horizon_b),
                "fatal_n_horizon_a": fatal_n_a,
                "fatal_n_horizon_b": fatal_n_b,
                "matched_fatal_n": matched_fatal_n,
                "dropped_fatal_a_unmatched": dropped_fatal_a_unmatched,
                "dropped_fatal_b_unmatched": dropped_fatal_b_unmatched,
                "matched_share_of_horizon_a_fatal": float(matched_fatal_n / fatal_n_a) if fatal_n_a else np.nan,
                "matched_share_of_horizon_b_fatal": float(matched_fatal_n / fatal_n_b) if fatal_n_b else np.nan,
            }
        )

        hard_case_flag_a = matched["hard_case_flag_a"].astype(bool)
        hard_case_flag_b = matched["hard_case_flag_b"].astype(bool)
        hard_n_a = int(hard_case_flag_a.sum())
        hard_n_b = int(hard_case_flag_b.sum())
        intersection_n = int((hard_case_flag_a & hard_case_flag_b).sum())
        union_n = int((hard_case_flag_a | hard_case_flag_b).sum())
        jaccard_index = float(intersection_n / union_n) if union_n else np.nan

        pairwise_overlap_rows.append(
            {
                "horizon_a": _horizon_label(horizon_a),
                "horizon_b": _horizon_label(horizon_b),
                "matched_fatal_n": matched_fatal_n,
                "hard_n_horizon_a": hard_n_a,
                "hard_n_horizon_b": hard_n_b,
                "intersection_n": intersection_n,
                "union_n": union_n,
                "jaccard_index": jaccard_index,
            }
        )

    pairwise_denominators = pd.DataFrame(pairwise_denominator_rows)
    pairwise_overlap = pd.DataFrame(pairwise_overlap_rows)
    return pairwise_denominators, pairwise_overlap


def _build_directional_overlap(pairwise_overlap: pd.DataFrame) -> pd.DataFrame:
    directional_rows: list[dict[str, object]] = []
    for row in pairwise_overlap.itertuples(index=False):
        overlap_a_to_b = (
            float(row.intersection_n / row.hard_n_horizon_a)
            if int(row.hard_n_horizon_a) > 0
            else np.nan
        )
        overlap_b_to_a = (
            float(row.intersection_n / row.hard_n_horizon_b)
            if int(row.hard_n_horizon_b) > 0
            else np.nan
        )
        directional_rows.append(
            {
                "horizon_from": str(row.horizon_a),
                "horizon_to": str(row.horizon_b),
                "matched_fatal_n": int(row.matched_fatal_n),
                "hard_n_from": int(row.hard_n_horizon_a),
                "hard_n_to": int(row.hard_n_horizon_b),
                "intersection_n": int(row.intersection_n),
                "overlap_from_A_to_B": overlap_a_to_b,
            }
        )
        directional_rows.append(
            {
                "horizon_from": str(row.horizon_b),
                "horizon_to": str(row.horizon_a),
                "matched_fatal_n": int(row.matched_fatal_n),
                "hard_n_from": int(row.hard_n_horizon_b),
                "hard_n_to": int(row.hard_n_horizon_a),
                "intersection_n": int(row.intersection_n),
                "overlap_from_A_to_B": overlap_b_to_a,
            }
        )
    directional_overlap = pd.DataFrame(directional_rows)
    order_lookup = {
        _horizon_label(horizon): position
        for position, horizon in enumerate(HORIZON_ORDER)
    }
    directional_overlap["_from_order"] = directional_overlap["horizon_from"].map(order_lookup)
    directional_overlap["_to_order"] = directional_overlap["horizon_to"].map(order_lookup)
    directional_overlap = directional_overlap.sort_values(
        ["_from_order", "_to_order"],
        kind="stable",
    ).drop(columns=["_from_order", "_to_order"])
    return directional_overlap.reset_index(drop=True)


def _build_heatmap_matrix(
    pairwise_overlap: pd.DataFrame,
    directional_overlap: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = [_horizon_label(horizon) for horizon in horizons]
    jaccard_matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    directional_matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)

    for label in labels:
        jaccard_matrix.loc[label, label] = 1.0
        directional_matrix.loc[label, label] = 1.0

    for row in pairwise_overlap.itertuples(index=False):
        jaccard_matrix.loc[row.horizon_a, row.horizon_b] = row.jaccard_index
        jaccard_matrix.loc[row.horizon_b, row.horizon_a] = row.jaccard_index

    for row in directional_overlap.itertuples(index=False):
        directional_matrix.loc[row.horizon_from, row.horizon_to] = row.overlap_from_A_to_B

    return jaccard_matrix, directional_matrix


def _annotate_heatmap(matrix: pd.DataFrame, axis: plt.Axes) -> None:
    for row_index, row_label in enumerate(matrix.index):
        for column_index, column_label in enumerate(matrix.columns):
            value = matrix.loc[row_label, column_label]
            if pd.isna(value):
                axis.text(column_index, row_index, "NA", ha="center", va="center", color="#555555", fontsize=9)
            else:
                color = "white" if float(value) >= 0.5 else "#222222"
                axis.text(
                    column_index,
                    row_index,
                    f"{float(value):.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                )


def _plot_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str,
    x_label: str,
    y_label: str,
    colorbar_label: str,
    output_path: Path,
) -> Path:
    _require_matplotlib()
    ensure_directory(output_path.parent)
    figure, axis = plt.subplots(figsize=(7.2, 6.0))
    image = axis.imshow(matrix.to_numpy(dtype=float), cmap="Blues", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    _annotate_heatmap(matrix, axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _build_persistence_tables(
    harmonized_stay_level: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fatal_any_ids = (
        harmonized_stay_level.loc[harmonized_stay_level["fatal_flag"].astype(bool), "stay_id"]
        .astype("string")
        .drop_duplicates()
        .sort_values(kind="stable")
        .tolist()
    )
    if not fatal_any_ids:
        raise ValueError("No fatal stays were available for cross-horizon persistence analysis.")

    hospital_lookup = (
        harmonized_stay_level[["stay_id", "hospital_id"]]
        .drop_duplicates(subset=["stay_id"], keep="first")
        .set_index("stay_id")["hospital_id"]
    )
    persistence = pd.DataFrame({"stay_id": fatal_any_ids})
    persistence["hospital_id"] = persistence["stay_id"].map(hospital_lookup).astype("string")

    for horizon in horizons:
        horizon_label = _horizon_label(horizon)
        horizon_df = harmonized_stay_level[
            harmonized_stay_level["horizon_h"].astype(int).eq(int(horizon))
        ][["stay_id", "available_flag", "fatal_flag", "hard_case_flag"]].copy()
        horizon_df = horizon_df.set_index("stay_id")
        persistence[f"available_{horizon_label}"] = (
            persistence["stay_id"].map(horizon_df["available_flag"]).fillna(False).astype(bool)
        )
        persistence[f"fatal_{horizon_label}"] = (
            persistence["stay_id"].map(horizon_df["fatal_flag"]).fillna(False).astype(bool)
        )
        persistence[f"hard_case_{horizon_label}"] = (
            persistence["stay_id"].map(horizon_df["hard_case_flag"]).fillna(False).astype(bool)
        )

    available_columns = [f"available_{_horizon_label(horizon)}" for horizon in horizons]
    fatal_columns = [f"fatal_{_horizon_label(horizon)}" for horizon in horizons]
    hard_case_columns = [f"hard_case_{_horizon_label(horizon)}" for horizon in horizons]
    persistence["available_horizon_n"] = persistence[available_columns].sum(axis=1).astype(int)
    persistence["fatal_horizon_n"] = persistence[fatal_columns].sum(axis=1).astype(int)
    persistence["hard_case_horizon_n"] = persistence[hard_case_columns].sum(axis=1).astype(int)
    persistence["hard_case_share_among_fatal_horizons"] = (
        persistence["hard_case_horizon_n"] / persistence["fatal_horizon_n"]
    ).astype(float)

    distribution = (
        persistence["hard_case_horizon_n"]
        .value_counts(dropna=False)
        .reindex(range(0, len(horizons) + 1), fill_value=0)
        .rename_axis("hard_case_horizon_n")
        .reset_index(name="fatal_stay_count")
    )
    distribution["fatal_stay_share"] = distribution["fatal_stay_count"] / int(persistence.shape[0])
    return persistence, distribution


def _plot_persistence_barplot(
    persistence_distribution: pd.DataFrame,
    *,
    output_path: Path,
) -> Path:
    _require_matplotlib()
    ensure_directory(output_path.parent)
    figure, axis = plt.subplots(figsize=(7.0, 4.6))
    x_positions = persistence_distribution["hard_case_horizon_n"].astype(int).tolist()
    counts = persistence_distribution["fatal_stay_count"].astype(int).tolist()
    axis.bar(x_positions, counts, color="#4c78a8", edgecolor="#2f4a6d", width=0.75)
    axis.set_xlabel("Number of horizons labeled hard")
    axis.set_ylabel("Fatal stays")
    axis.set_title("Hard-Case Persistence Across Horizons")
    axis.set_xticks(x_positions)
    axis.set_xlim(-0.5, max(x_positions) + 0.5)
    ymax = max(counts) if counts else 0
    axis.set_ylim(0, max(1, ymax * 1.15))
    for x_position, count in zip(x_positions, counts):
        axis.text(x_position, count + max(0.02, ymax * 0.02), str(count), ha="center", va="bottom", fontsize=9)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _build_overlap_note(
    *,
    hard_case_dir: Path,
    stay_level_path: Path,
    pairwise_denominators: pd.DataFrame,
    pairwise_overlap: pd.DataFrame,
    directional_overlap: pd.DataFrame,
    hard_case_persistence: pd.DataFrame,
    persistence_distribution: pd.DataFrame,
) -> str:
    jaccard_nonmissing = pairwise_overlap["jaccard_index"].dropna()
    directional_nonmissing = directional_overlap["overlap_from_A_to_B"].dropna()
    all_pairs_fully_matched = bool(
        pairwise_denominators["dropped_fatal_a_unmatched"].eq(0).all()
        and pairwise_denominators["dropped_fatal_b_unmatched"].eq(0).all()
    )
    strongest_pair = pairwise_overlap.sort_values(
        ["jaccard_index", "matched_fatal_n", "horizon_a", "horizon_b"],
        ascending=[False, False, True, True],
        kind="stable",
    ).iloc[0]
    weakest_pair = pairwise_overlap.sort_values(
        ["jaccard_index", "matched_fatal_n", "horizon_a", "horizon_b"],
        ascending=[True, False, True, True],
        kind="stable",
    ).iloc[0]

    one_horizon = int(
        persistence_distribution.loc[
            persistence_distribution["hard_case_horizon_n"].astype(int).eq(1),
            "fatal_stay_count",
        ].iloc[0]
    )
    two_plus_horizons = int(
        persistence_distribution.loc[
            persistence_distribution["hard_case_horizon_n"].astype(int).ge(2),
            "fatal_stay_count",
        ].sum()
    )
    if two_plus_horizons > one_horizon:
        persistence_read = "more persistent than purely horizon-specific"
    elif two_plus_horizons < one_horizon:
        persistence_read = "more horizon-specific rotation than stable persistence"
    else:
        persistence_read = "mixed between persistence and horizon-specific rotation"

    denominator_rows = [
        {
            "horizon_a": row.horizon_a,
            "horizon_b": row.horizon_b,
            "fatal_n_horizon_a": int(row.fatal_n_horizon_a),
            "fatal_n_horizon_b": int(row.fatal_n_horizon_b),
            "matched_fatal_n": int(row.matched_fatal_n),
        }
        for row in pairwise_denominators.itertuples(index=False)
    ]
    overlap_rows = [
        {
            "horizon_a": row.horizon_a,
            "horizon_b": row.horizon_b,
            "matched_fatal_n": int(row.matched_fatal_n),
            "hard_n_horizon_a": int(row.hard_n_horizon_a),
            "hard_n_horizon_b": int(row.hard_n_horizon_b),
            "intersection_n": int(row.intersection_n),
            "union_n": int(row.union_n),
            "jaccard_index": "NA" if pd.isna(row.jaccard_index) else f"{float(row.jaccard_index):.3f}",
        }
        for row in pairwise_overlap.itertuples(index=False)
    ]
    persistence_rows = [
        {
            "hard_case_horizon_n": int(row.hard_case_horizon_n),
            "fatal_stay_count": int(row.fatal_stay_count),
            "fatal_stay_share": f"{float(row.fatal_stay_share):.3f}",
        }
        for row in persistence_distribution.itertuples(index=False)
    ]

    lines = [
        "# Overlap Note",
        "",
        "These outputs use the validated ASIC logistic stay-level hard-case artifact from Package 1.",
        "All local counts and overlap values come from the repository's small synthetic stand-in data and are only for implementation testing.",
        "",
        "## Input Artifact",
        "",
        f"- Hard-case source directory: `{_display_path(hard_case_dir)}`",
        f"- Stay-level source artifact: `{_display_path(stay_level_path)}`",
        "- Matching key: `stay_id` harmonized from `stay_id_global`, with `hospital_id` checked for consistency across horizons.",
        "",
        "## Matched-Denominator Logic",
        "",
        "- For each unordered horizon pair, the denominator is the intersection of fatal stay IDs present in both horizons.",
        "- Pairwise overlap metrics do not use all stays and do not use each horizon's raw fatal count as the overlap denominator.",
        "- Directional overlap uses the same matched fatal set for the pair, then divides the intersection by the hard-case count in the source horizon.",
        (
            "- In this local synthetic run, every horizon pair retained the full fatal population on both sides."
            if all_pairs_fully_matched
            else "- Some horizon pairs lost fatal stays during matching; see the denominator table above."
        ),
        "",
        _markdown_table(
            denominator_rows,
            columns=[
                "horizon_a",
                "horizon_b",
                "fatal_n_horizon_a",
                "fatal_n_horizon_b",
                "matched_fatal_n",
            ],
        ),
        "",
        "## Pairwise Overlap Summary",
        "",
        f"- Mean Jaccard across horizon pairs: `{float(jaccard_nonmissing.mean()):.3f}`",
        f"- Jaccard range across horizon pairs: `{float(jaccard_nonmissing.min()):.3f}` to `{float(jaccard_nonmissing.max()):.3f}`",
        f"- Strongest pair: `{strongest_pair.horizon_a}` vs `{strongest_pair.horizon_b}` with Jaccard `{float(strongest_pair.jaccard_index):.3f}` on matched fatal denominator `{int(strongest_pair.matched_fatal_n)}`.",
        f"- Weakest pair: `{weakest_pair.horizon_a}` vs `{weakest_pair.horizon_b}` with Jaccard `{float(weakest_pair.jaccard_index):.3f}` on matched fatal denominator `{int(weakest_pair.matched_fatal_n)}`.",
        f"- Mean directional overlap across ordered pairs: `{float(directional_nonmissing.mean()):.3f}`",
        "",
        _markdown_table(
            overlap_rows,
            columns=[
                "horizon_a",
                "horizon_b",
                "matched_fatal_n",
                "hard_n_horizon_a",
                "hard_n_horizon_b",
                "intersection_n",
                "union_n",
                "jaccard_index",
            ],
        ),
        "",
        "## Persistence Summary",
        "",
        f"- Fatal-stay union across horizons: `{int(hard_case_persistence.shape[0])}` stays.",
        "- Persistence table includes separate `available_*`, `fatal_*`, and `hard_case_*` columns so nonfatal or unavailable horizons are not silently mixed with matched fatal denominators.",
        f"- Based on the synthetic persistence distribution, hard-case membership looks `{persistence_read}`.",
        "",
        _markdown_table(
            persistence_rows,
            columns=[
                "hard_case_horizon_n",
                "fatal_stay_count",
                "fatal_stay_share",
            ],
        ),
        "",
        "## Caveats",
        "",
        "- Later horizons can have smaller stay availability, so matched fatal denominators may be smaller than horizon-specific fatal totals.",
        "- Heatmaps show diagonal self-overlap as `1.00` for readability; the CSV overlap tables only contain cross-horizon pairs.",
        "- Local overlap values are synthetic small-sample outputs and should not be used for substantive interpretation.",
        "",
    ]
    return "\n".join(lines)


def run_asic_horizon_hard_case_stability(
    *,
    hard_case_dir: Path = DEFAULT_HARD_CASE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    horizons: Sequence[int] | None = None,
    output_format: str = "csv",
) -> HorizonHardCaseStabilityRunResult:
    if output_format != "csv":
        raise ValueError(f"Unsupported output format: {output_format}")

    normalized_horizons = _normalize_horizons(horizons)
    resolved_hard_case_dir = Path(hard_case_dir)
    resolved_output_dir = Path(output_dir)

    harmonized_stay_level, stay_level_path, manifest = _load_stay_level_artifact(
        resolved_hard_case_dir,
        horizons=normalized_horizons,
    )
    pairwise_denominators, pairwise_overlap = _build_pairwise_tables(
        harmonized_stay_level,
        horizons=normalized_horizons,
    )
    directional_overlap = _build_directional_overlap(pairwise_overlap)
    hard_case_persistence, persistence_distribution = _build_persistence_tables(
        harmonized_stay_level,
        horizons=normalized_horizons,
    )
    jaccard_matrix, directional_matrix = _build_heatmap_matrix(
        pairwise_overlap,
        directional_overlap,
        horizons=normalized_horizons,
    )

    pairwise_denominators_path = write_dataframe(
        pairwise_denominators,
        resolved_output_dir / "pairwise_denominators.csv",
        output_format=output_format,
    )
    pairwise_overlap_path = write_dataframe(
        pairwise_overlap,
        resolved_output_dir / "pairwise_overlap.csv",
        output_format=output_format,
    )
    directional_overlap_path = write_dataframe(
        directional_overlap,
        resolved_output_dir / "directional_overlap.csv",
        output_format=output_format,
    )
    hard_case_persistence_path = write_dataframe(
        hard_case_persistence,
        resolved_output_dir / "hard_case_persistence.csv",
        output_format=output_format,
    )
    persistence_distribution_path = write_dataframe(
        persistence_distribution,
        resolved_output_dir / "persistence_distribution.csv",
        output_format=output_format,
    )
    jaccard_heatmap_path = _plot_heatmap(
        jaccard_matrix,
        title="Pairwise Hard-Case Jaccard Overlap",
        x_label="Horizon",
        y_label="Horizon",
        colorbar_label="Jaccard Index",
        output_path=resolved_output_dir / "jaccard_heatmap.png",
    )
    directional_overlap_heatmap_path = _plot_heatmap(
        directional_matrix,
        title="Directional Hard-Case Overlap",
        x_label="Target Horizon",
        y_label="Source Horizon",
        colorbar_label="Directional Overlap",
        output_path=resolved_output_dir / "directional_overlap_heatmap.png",
    )
    persistence_barplot_path = _plot_persistence_barplot(
        persistence_distribution,
        output_path=resolved_output_dir / "persistence_barplot.png",
    )
    overlap_note_path = write_text(
        _build_overlap_note(
            hard_case_dir=resolved_hard_case_dir,
            stay_level_path=stay_level_path,
            pairwise_denominators=pairwise_denominators,
            pairwise_overlap=pairwise_overlap,
            directional_overlap=directional_overlap,
            hard_case_persistence=hard_case_persistence,
            persistence_distribution=persistence_distribution,
        ),
        resolved_output_dir / "overlap_note.md",
    )
    manifest_path = _write_json(
        {
            "timestamp_utc": _utc_timestamp(),
            "hard_case_dir": str(resolved_hard_case_dir.resolve()),
            "hard_case_rule": manifest.get("hard_case_rule"),
            "stay_level_artifact": str(stay_level_path.resolve()),
            "output_dir": str(resolved_output_dir.resolve()),
            "horizons": list(normalized_horizons),
            "pairwise_denominators_artifact": str(pairwise_denominators_path.resolve()),
            "pairwise_overlap_artifact": str(pairwise_overlap_path.resolve()),
            "directional_overlap_artifact": str(directional_overlap_path.resolve()),
            "jaccard_heatmap_artifact": str(jaccard_heatmap_path.resolve()),
            "directional_overlap_heatmap_artifact": str(directional_overlap_heatmap_path.resolve()),
            "hard_case_persistence_artifact": str(hard_case_persistence_path.resolve()),
            "persistence_distribution_artifact": str(persistence_distribution_path.resolve()),
            "persistence_barplot_artifact": str(persistence_barplot_path.resolve()),
            "overlap_note_artifact": str(overlap_note_path.resolve()),
        },
        resolved_output_dir / "run_manifest.json",
    )

    package3_ready = bool(
        not pairwise_denominators["matched_fatal_n"].isna().any()
        and not pairwise_overlap.empty
        and not directional_overlap.empty
        and not hard_case_persistence.empty
    )

    return HorizonHardCaseStabilityRunResult(
        hard_case_dir=resolved_hard_case_dir,
        output_dir=resolved_output_dir,
        horizons=normalized_horizons,
        harmonized_stay_level=harmonized_stay_level,
        pairwise_denominators=pairwise_denominators,
        pairwise_overlap=pairwise_overlap,
        directional_overlap=directional_overlap,
        hard_case_persistence=hard_case_persistence,
        persistence_distribution=persistence_distribution,
        artifacts=HorizonHardCaseStabilityArtifacts(
            pairwise_denominators_path=pairwise_denominators_path,
            pairwise_overlap_path=pairwise_overlap_path,
            directional_overlap_path=directional_overlap_path,
            jaccard_heatmap_path=jaccard_heatmap_path,
            directional_overlap_heatmap_path=directional_overlap_heatmap_path,
            hard_case_persistence_path=hard_case_persistence_path,
            persistence_distribution_path=persistence_distribution_path,
            persistence_barplot_path=persistence_barplot_path,
            overlap_note_path=overlap_note_path,
            manifest_path=manifest_path,
        ),
        package3_ready=package3_ready,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute cross-horizon ASIC logistic hard-case overlap and persistence metrics "
            "from the frozen stay-level hard-case artifact."
        )
    )
    parser.add_argument(
        "--hard-case-dir",
        type=Path,
        default=DEFAULT_HARD_CASE_DIR,
        help="Directory containing the saved logistic hard-case stay-level artifact and manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where cross-horizon overlap outputs will be written.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        help="Optional subset of frozen horizons to analyze.",
    )
    parser.add_argument(
        "--output-format",
        choices=("csv",),
        default="csv",
        help="Output format for tabular artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = run_asic_horizon_hard_case_stability(
        hard_case_dir=args.hard_case_dir,
        output_dir=args.output_dir,
        horizons=args.horizons,
        output_format=args.output_format,
    )

    print(f"Hard-case directory: {result.hard_case_dir}")
    print(f"Output directory: {result.output_dir}")
    print(f"Package 3 ready: {'yes' if result.package3_ready else 'no'}")
    for row in result.pairwise_overlap.itertuples(index=False):
        jaccard_text = "NA" if pd.isna(row.jaccard_index) else f"{float(row.jaccard_index):.3f}"
        print(
            f"{row.horizon_a} vs {row.horizon_b}: matched_fatal_n={int(row.matched_fatal_n)}, "
            f"intersection_n={int(row.intersection_n)}, union_n={int(row.union_n)}, "
            f"jaccard_index={jaccard_text}"
        )
    print(f"Wrote {result.artifacts.pairwise_denominators_path}")
    print(f"Wrote {result.artifacts.pairwise_overlap_path}")
    print(f"Wrote {result.artifacts.directional_overlap_path}")
    print(f"Wrote {result.artifacts.jaccard_heatmap_path}")
    print(f"Wrote {result.artifacts.directional_overlap_heatmap_path}")
    print(f"Wrote {result.artifacts.hard_case_persistence_path}")
    print(f"Wrote {result.artifacts.persistence_distribution_path}")
    print(f"Wrote {result.artifacts.persistence_barplot_path}")
    print(f"Wrote {result.artifacts.overlap_note_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
