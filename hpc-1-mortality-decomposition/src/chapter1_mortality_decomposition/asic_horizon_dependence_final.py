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
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, MaxNLocator
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
DEFAULT_FOUNDATION_DIR = (
    Path("artifacts")
    / "chapter1"
    / "evaluation"
    / "asic"
    / "horizon_dependence"
    / "foundation"
)
DEFAULT_OVERLAP_DIR = (
    Path("artifacts")
    / "chapter1"
    / "evaluation"
    / "asic"
    / "horizon_dependence"
    / "overlap"
)
DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "chapter1"
    / "evaluation"
    / "asic"
    / "horizon_dependence"
    / "final"
)
DEFAULT_HORIZONS = (8, 16, 24, 48, 72)
DEFAULT_BIN_COUNT = 6
BUBBLE_AREA_MIN = 28.0
BUBBLE_AREA_MAX = 260.0
ANCHOR_HORIZON = 24
MAIN_CONTRAST_HORIZON = 48
REQUIRED_STAY_LEVEL_COLUMNS = {
    "stay_id_global",
    "hospital_id",
    "horizon_h",
    "label_value",
    "predicted_probability",
    "nonfatal_q75_threshold",
    "hard_case_flag",
}
REQUIRED_FOUNDATION_SUMMARY_COLUMNS = {
    "horizon",
    "nonfatal_last_n",
    "fatal_last_n",
    "nonfatal_q75_threshold",
    "hard_case_n",
    "hard_case_share_among_fatal",
}
REQUIRED_PAIRWISE_DENOMINATOR_COLUMNS = {
    "horizon_a",
    "horizon_b",
    "fatal_n_horizon_a",
    "fatal_n_horizon_b",
    "matched_fatal_n",
}
REQUIRED_PAIRWISE_OVERLAP_COLUMNS = {
    "horizon_a",
    "horizon_b",
    "matched_fatal_n",
    "hard_n_horizon_a",
    "hard_n_horizon_b",
    "intersection_n",
    "union_n",
    "jaccard_index",
}
REQUIRED_DIRECTIONAL_OVERLAP_COLUMNS = {
    "horizon_from",
    "horizon_to",
    "matched_fatal_n",
    "hard_n_from",
    "hard_n_to",
    "intersection_n",
    "overlap_from_A_to_B",
}
REQUIRED_PERSISTENCE_DISTRIBUTION_COLUMNS = {
    "hard_case_horizon_n",
    "fatal_stay_count",
    "fatal_stay_share",
}


@dataclass(frozen=True)
class HorizonFinalArtifacts:
    figure_path: Path
    binned_summary_path: Path
    interpretation_memo_path: Path
    final_summary_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class HorizonFinalRunResult:
    hard_case_dir: Path
    foundation_dir: Path
    overlap_dir: Path
    output_dir: Path
    horizons: tuple[int, ...]
    horizon_summary: pd.DataFrame
    binned_summary: pd.DataFrame
    consistency_issues: tuple[str, ...]
    interpretation_label: str
    artifacts: HorizonFinalArtifacts


def _require_matplotlib() -> None:
    if plt is None:
        raise ImportError(
            "matplotlib is required for the Chapter 1 ASIC horizon comparison final package."
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


def _horizon_from_label(label: str) -> int:
    return int(str(label).removesuffix("h"))


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


def _format_probability(value: float) -> str:
    return f"{float(value):.3f}"


def _load_manifest(hard_case_dir: Path) -> dict[str, object]:
    manifest_path = hard_case_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Expected hard-case manifest at {manifest_path}")
    return json.loads(manifest_path.read_text())


def _load_stay_level(
    hard_case_dir: Path,
    *,
    horizons: tuple[int, ...],
) -> tuple[pd.DataFrame, Path]:
    manifest = _load_manifest(hard_case_dir)
    stay_level_path = Path(str(manifest["stay_level_artifact"]))
    stay_level = read_dataframe(stay_level_path)
    require_columns(stay_level, REQUIRED_STAY_LEVEL_COLUMNS, str(stay_level_path))

    harmonized = stay_level.copy()
    harmonized["stay_id"] = harmonized["stay_id_global"].astype("string")
    harmonized["hospital_id"] = harmonized["hospital_id"].astype("string")
    harmonized["horizon_h"] = pd.to_numeric(harmonized["horizon_h"], errors="coerce").astype("Int64")
    harmonized["fatal_flag"] = pd.to_numeric(harmonized["label_value"], errors="coerce").astype("Int64").eq(1)
    harmonized["predicted_probability"] = pd.to_numeric(
        harmonized["predicted_probability"],
        errors="coerce",
    )
    harmonized["nonfatal_q75_threshold"] = pd.to_numeric(
        harmonized["nonfatal_q75_threshold"],
        errors="coerce",
    )
    harmonized["hard_case_flag"] = normalize_boolean_codes(harmonized["hard_case_flag"]).fillna(False).astype(bool)

    filtered = harmonized[harmonized["horizon_h"].isin(list(horizons))].copy()
    observed_horizons = tuple(sorted(int(value) for value in filtered["horizon_h"].dropna().unique()))
    if observed_horizons != horizons:
        raise ValueError(
            f"Stay-level artifact {stay_level_path} covered horizons {observed_horizons}, expected {horizons}."
        )
    duplicate_count = int(filtered.duplicated(["stay_id", "horizon_h"]).sum())
    if duplicate_count:
        raise ValueError(
            f"Stay-level artifact {stay_level_path} contains {duplicate_count} duplicated stay/horizon rows."
        )
    return filtered.reset_index(drop=True), stay_level_path


def _load_foundation_summary(
    foundation_dir: Path,
    *,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    path = foundation_dir / "horizon_summary.csv"
    summary = read_dataframe(path)
    require_columns(summary, REQUIRED_FOUNDATION_SUMMARY_COLUMNS, str(path))
    validated = summary.copy()
    validated["horizon_h"] = validated["horizon"].astype("string").map(_horizon_from_label).astype(int)
    filtered = validated[validated["horizon_h"].isin(list(horizons))].copy()
    observed_horizons = tuple(sorted(filtered["horizon_h"].astype(int).unique().tolist()))
    if observed_horizons != horizons:
        raise ValueError(
            f"Foundation summary {path} covered horizons {observed_horizons}, expected {horizons}."
        )
    return filtered.sort_values("horizon_h", kind="stable").reset_index(drop=True)


def _load_overlap_artifacts(
    overlap_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pairwise_denominators_path = overlap_dir / "pairwise_denominators.csv"
    pairwise_overlap_path = overlap_dir / "pairwise_overlap.csv"
    directional_overlap_path = overlap_dir / "directional_overlap.csv"
    persistence_distribution_path = overlap_dir / "persistence_distribution.csv"

    pairwise_denominators = read_dataframe(pairwise_denominators_path)
    require_columns(pairwise_denominators, REQUIRED_PAIRWISE_DENOMINATOR_COLUMNS, str(pairwise_denominators_path))
    pairwise_overlap = read_dataframe(pairwise_overlap_path)
    require_columns(pairwise_overlap, REQUIRED_PAIRWISE_OVERLAP_COLUMNS, str(pairwise_overlap_path))
    directional_overlap = read_dataframe(directional_overlap_path)
    require_columns(directional_overlap, REQUIRED_DIRECTIONAL_OVERLAP_COLUMNS, str(directional_overlap_path))
    persistence_distribution = read_dataframe(persistence_distribution_path)
    require_columns(
        persistence_distribution,
        REQUIRED_PERSISTENCE_DISTRIBUTION_COLUMNS,
        str(persistence_distribution_path),
    )
    return pairwise_denominators, pairwise_overlap, directional_overlap, persistence_distribution


def _build_horizon_summary_from_stay_level(stay_level: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon_h, horizon_df in stay_level.groupby("horizon_h", sort=True):
        horizon_key = int(horizon_h)
        fatal_n = int(horizon_df["fatal_flag"].sum())
        hard_case_n = int(horizon_df["hard_case_flag"].sum())
        rows.append(
            {
                "horizon_h": horizon_key,
                "horizon": _horizon_label(horizon_key),
                "last_stay_n": int(horizon_df.shape[0]),
                "nonfatal_last_n": int((~horizon_df["fatal_flag"]).sum()),
                "fatal_last_n": fatal_n,
                "nonfatal_q75_threshold": float(horizon_df["nonfatal_q75_threshold"].iloc[0]),
                "hard_case_n": hard_case_n,
                "hard_case_share_among_fatal": float(hard_case_n / fatal_n) if fatal_n else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("horizon_h", kind="stable").reset_index(drop=True)


def _compare_with_foundation_summary(
    stay_summary: pd.DataFrame,
    foundation_summary: pd.DataFrame,
) -> list[str]:
    merged = stay_summary.merge(
        foundation_summary[
            [
                "horizon_h",
                "nonfatal_last_n",
                "fatal_last_n",
                "nonfatal_q75_threshold",
                "hard_case_n",
                "hard_case_share_among_fatal",
            ]
        ],
        on="horizon_h",
        suffixes=("_figure", "_foundation"),
        how="inner",
    )
    issues: list[str] = []
    for row in merged.itertuples(index=False):
        for field in ("nonfatal_last_n", "fatal_last_n", "hard_case_n"):
            figure_value = int(getattr(row, f"{field}_figure"))
            foundation_value = int(getattr(row, f"{field}_foundation"))
            if figure_value != foundation_value:
                issues.append(
                    f"{_horizon_label(int(row.horizon_h))}: figure-derived {field}={figure_value} "
                    f"but foundation summary has {foundation_value}."
                )

        threshold_diff = abs(
            float(row.nonfatal_q75_threshold_figure) - float(row.nonfatal_q75_threshold_foundation)
        )
        if threshold_diff > 1e-12:
            issues.append(
                f"{_horizon_label(int(row.horizon_h))}: figure-derived nonfatal_q75_threshold="
                f"{float(row.nonfatal_q75_threshold_figure):.12f} but foundation summary has "
                f"{float(row.nonfatal_q75_threshold_foundation):.12f}."
            )

        share_diff = abs(
            float(row.hard_case_share_among_fatal_figure)
            - float(row.hard_case_share_among_fatal_foundation)
        )
        if share_diff > 1e-12:
            issues.append(
                f"{_horizon_label(int(row.horizon_h))}: figure-derived hard_case_share_among_fatal="
                f"{float(row.hard_case_share_among_fatal_figure):.12f} but foundation summary has "
                f"{float(row.hard_case_share_among_fatal_foundation):.12f}."
            )
    return issues


def _compare_with_overlap_outputs(
    stay_summary: pd.DataFrame,
    pairwise_denominators: pd.DataFrame,
    pairwise_overlap: pd.DataFrame,
) -> list[str]:
    summary_lookup = stay_summary.set_index("horizon")
    denominator_lookup = pairwise_denominators.set_index(["horizon_a", "horizon_b"])
    issues: list[str] = []

    for row in pairwise_denominators.itertuples(index=False):
        fatal_a_expected = int(summary_lookup.at[row.horizon_a, "fatal_last_n"])
        fatal_b_expected = int(summary_lookup.at[row.horizon_b, "fatal_last_n"])
        if int(row.fatal_n_horizon_a) != fatal_a_expected:
            issues.append(
                f"{row.horizon_a} vs {row.horizon_b}: pairwise_denominators fatal_n_horizon_a="
                f"{int(row.fatal_n_horizon_a)} but horizon summary has {fatal_a_expected}."
            )
        if int(row.fatal_n_horizon_b) != fatal_b_expected:
            issues.append(
                f"{row.horizon_a} vs {row.horizon_b}: pairwise_denominators fatal_n_horizon_b="
                f"{int(row.fatal_n_horizon_b)} but horizon summary has {fatal_b_expected}."
            )
        if int(row.matched_fatal_n) > min(fatal_a_expected, fatal_b_expected):
            issues.append(
                f"{row.horizon_a} vs {row.horizon_b}: matched_fatal_n={int(row.matched_fatal_n)} "
                "exceeds the available fatal denominator."
            )

    for row in pairwise_overlap.itertuples(index=False):
        overall_hard_a = int(summary_lookup.at[row.horizon_a, "hard_case_n"])
        overall_hard_b = int(summary_lookup.at[row.horizon_b, "hard_case_n"])
        matched_fatal_n = int(denominator_lookup.at[(row.horizon_a, row.horizon_b), "matched_fatal_n"])
        fatal_a_total = int(summary_lookup.at[row.horizon_a, "fatal_last_n"])
        fatal_b_total = int(summary_lookup.at[row.horizon_b, "fatal_last_n"])

        if int(row.hard_n_horizon_a) > overall_hard_a:
            issues.append(
                f"{row.horizon_a} vs {row.horizon_b}: pairwise_overlap hard_n_horizon_a="
                f"{int(row.hard_n_horizon_a)} exceeds overall hard_case_n={overall_hard_a}."
            )
        if int(row.hard_n_horizon_b) > overall_hard_b:
            issues.append(
                f"{row.horizon_a} vs {row.horizon_b}: pairwise_overlap hard_n_horizon_b="
                f"{int(row.hard_n_horizon_b)} exceeds overall hard_case_n={overall_hard_b}."
            )
        if matched_fatal_n == fatal_a_total and int(row.hard_n_horizon_a) != overall_hard_a:
            issues.append(
                f"{row.horizon_a} vs {row.horizon_b}: matched fatal set covers all {row.horizon_a} fatal stays, "
                f"but hard_n_horizon_a={int(row.hard_n_horizon_a)} does not equal overall hard_case_n={overall_hard_a}."
            )
        if matched_fatal_n == fatal_b_total and int(row.hard_n_horizon_b) != overall_hard_b:
            issues.append(
                f"{row.horizon_a} vs {row.horizon_b}: matched fatal set covers all {row.horizon_b} fatal stays, "
                f"but hard_n_horizon_b={int(row.hard_n_horizon_b)} does not equal overall hard_case_n={overall_hard_b}."
            )
    return issues


def _build_common_probability_edges(
    stay_level: pd.DataFrame,
    *,
    requested_bin_count: int,
) -> np.ndarray:
    finite = stay_level[np.isfinite(stay_level["predicted_probability"])].copy()
    if finite.empty:
        raise ValueError("No finite predicted_probability values were available for the final horizon figure.")

    quantiles = np.linspace(0.0, 1.0, requested_bin_count + 1)
    edges = np.quantile(finite["predicted_probability"].to_numpy(dtype=float), quantiles)
    edges = np.unique(np.asarray(edges, dtype=float))

    if edges.size < 2:
        only_value = float(finite["predicted_probability"].iloc[0])
        lower = max(0.0, only_value - 1e-6)
        upper = min(1.0, only_value + 1e-6)
        edges = np.array([lower, upper], dtype=float)

    edges[0] = min(edges[0], float(finite["predicted_probability"].min()))
    edges[-1] = max(edges[-1], float(finite["predicted_probability"].max()))
    return edges


def _build_binned_summary(
    stay_level: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    *,
    requested_bin_count: int,
) -> pd.DataFrame:
    edges = _build_common_probability_edges(stay_level, requested_bin_count=requested_bin_count)
    bin_count = int(edges.size - 1)
    complete_bins = pd.DataFrame(
        {
            "bin_index": np.arange(1, bin_count + 1, dtype=int),
            "bin_low": edges[:-1],
            "bin_high": edges[1:],
        }
    )
    complete_bins["bin_center"] = (complete_bins["bin_low"] + complete_bins["bin_high"]) / 2.0
    complete_bins["bin_width"] = complete_bins["bin_high"] - complete_bins["bin_low"]
    complete_bins["bin_label"] = complete_bins["bin_index"].map(lambda value: f"B{int(value):02d}")

    summary_lookup = horizon_summary.set_index("horizon_h")
    rows: list[pd.DataFrame] = []

    for horizon_h, horizon_df in stay_level.groupby("horizon_h", sort=True):
        horizon_key = int(horizon_h)
        assigned = horizon_df.copy()
        assigned["bin_index"] = pd.cut(
            assigned["predicted_probability"],
            bins=edges,
            labels=False,
            include_lowest=True,
            right=True,
        )
        assigned["bin_index"] = assigned["bin_index"].astype("Int64") + 1

        grouped = (
            assigned.groupby("bin_index", observed=True)
            .agg(
                sample_count=("stay_id", "size"),
                fatal_count=("fatal_flag", lambda values: int(pd.Series(values).astype(bool).sum())),
                predicted_probability_mean=("predicted_probability", "mean"),
                predicted_probability_min=("predicted_probability", "min"),
                predicted_probability_max=("predicted_probability", "max"),
            )
            .reset_index()
        )
        grouped["bin_index"] = grouped["bin_index"].astype(int)
        grouped["nonfatal_count"] = grouped["sample_count"] - grouped["fatal_count"]
        grouped["observed_mortality"] = grouped["fatal_count"] / grouped["sample_count"]
        grouped["sample_fraction_of_horizon"] = grouped["sample_count"] / int(horizon_df.shape[0])
        total_fatal = int(horizon_df["fatal_flag"].sum())
        grouped["fatal_fraction_of_horizon_fatal"] = (
            grouped["fatal_count"] / total_fatal if total_fatal > 0 else np.nan
        )

        merged = complete_bins.merge(grouped, on="bin_index", how="left")
        merged["horizon_h"] = horizon_key
        merged["horizon"] = _horizon_label(horizon_key)
        merged["sample_count"] = merged["sample_count"].fillna(0).astype(int)
        merged["fatal_count"] = merged["fatal_count"].fillna(0).astype(int)
        merged["nonfatal_count"] = merged["nonfatal_count"].fillna(0).astype(int)
        merged["sample_fraction_of_horizon"] = merged["sample_fraction_of_horizon"].fillna(0.0).astype(float)
        merged["fatal_fraction_of_horizon_fatal"] = merged["fatal_fraction_of_horizon_fatal"].fillna(0.0).astype(float)

        merged["nonfatal_q75_threshold"] = float(summary_lookup.at[horizon_key, "nonfatal_q75_threshold"])
        merged["hard_case_n"] = int(summary_lookup.at[horizon_key, "hard_case_n"])
        merged["fatal_last_n"] = int(summary_lookup.at[horizon_key, "fatal_last_n"])
        merged["nonfatal_last_n"] = int(summary_lookup.at[horizon_key, "nonfatal_last_n"])
        merged["hard_case_share_among_fatal"] = float(
            summary_lookup.at[horizon_key, "hard_case_share_among_fatal"]
        )
        rows.append(merged)

    return pd.concat(rows, ignore_index=True).sort_values(
        ["horizon_h", "bin_index"],
        kind="stable",
    ).reset_index(drop=True)


def _shape_distance(
    binned_summary: pd.DataFrame,
    *,
    horizon_a: int,
    horizon_b: int,
) -> float:
    summary_a = binned_summary[binned_summary["horizon_h"].astype(int).eq(int(horizon_a))].copy()
    summary_b = binned_summary[binned_summary["horizon_h"].astype(int).eq(int(horizon_b))].copy()
    merged = summary_a[
        ["bin_index", "observed_mortality", "sample_fraction_of_horizon"]
    ].merge(
        summary_b[["bin_index", "observed_mortality", "sample_fraction_of_horizon"]],
        on="bin_index",
        how="outer",
        suffixes=("_a", "_b"),
    )
    merged["observed_mortality_a"] = merged["observed_mortality_a"].fillna(0.0)
    merged["observed_mortality_b"] = merged["observed_mortality_b"].fillna(0.0)
    merged["sample_fraction_of_horizon_a"] = merged["sample_fraction_of_horizon_a"].fillna(0.0)
    merged["sample_fraction_of_horizon_b"] = merged["sample_fraction_of_horizon_b"].fillna(0.0)
    merged["weight"] = (
        merged["sample_fraction_of_horizon_a"] + merged["sample_fraction_of_horizon_b"]
    ) / 2.0
    weight_sum = float(merged["weight"].sum())
    if weight_sum == 0:
        return 0.0
    return float(
        (merged["weight"] * (merged["observed_mortality_a"] - merged["observed_mortality_b"]).abs()).sum()
        / weight_sum
    )


def _compute_bubble_areas(sample_fractions: pd.Series) -> np.ndarray:
    fractions = pd.to_numeric(sample_fractions, errors="coerce").fillna(0.0).clip(lower=0.0)
    if fractions.empty:
        return np.array([], dtype=float)

    max_fraction = float(fractions.max())
    if max_fraction <= 0.0:
        return np.full(fractions.shape[0], BUBBLE_AREA_MIN, dtype=float)

    scaled = np.sqrt(fractions.to_numpy(dtype=float) / max_fraction)
    return BUBBLE_AREA_MIN + scaled * (BUBBLE_AREA_MAX - BUBBLE_AREA_MIN)


def _plot_mortality_risk_horizon_comparison(
    binned_summary: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    *,
    output_path: Path,
) -> Path:
    _require_matplotlib()
    ensure_directory(output_path.parent)

    ordered_horizons = horizon_summary["horizon_h"].astype(int).tolist()
    x_max = max(
        float(binned_summary["bin_high"].max()),
        float(horizon_summary["nonfatal_q75_threshold"].max()),
    )
    x_max = min(max(x_max * 1.08, 0.05), 1.0)
    y_max = min(
        max(
            0.10,
            float(binned_summary["observed_mortality"].dropna().max()) if binned_summary["observed_mortality"].notna().any() else 0.10,
            float(binned_summary["predicted_probability_max"].dropna().max()) if binned_summary["predicted_probability_max"].notna().any() else 0.10,
        )
        * 1.08,
        1.0,
    )
    count_y_max = max(1.0, float(binned_summary["sample_count"].max()) * 1.15)

    figure, axes = plt.subplots(
        2,
        len(ordered_horizons),
        figsize=(18.5, 6.5),
        sharex="col",
        sharey="row",
        gridspec_kw={"height_ratios": [1.0, 1.2], "wspace": 0.12, "hspace": 0.06},
    )
    if len(ordered_horizons) == 1:
        axes = np.array(axes).reshape(2, 1)

    scaled_summary = binned_summary.copy()
    scaled_summary["bubble_area"] = _compute_bubble_areas(
        scaled_summary["sample_fraction_of_horizon"]
    )
    percent_formatter = FuncFormatter(lambda value, _: f"{value:.0%}")

    for column_index, horizon_h in enumerate(ordered_horizons):
        horizon_panel = scaled_summary[scaled_summary["horizon_h"].astype(int).eq(int(horizon_h))].copy()
        top_axis = axes[0, column_index]
        bottom_axis = axes[1, column_index]
        summary_row = horizon_summary[horizon_summary["horizon_h"].astype(int).eq(int(horizon_h))].iloc[0]
        threshold = float(summary_row["nonfatal_q75_threshold"])

        for axis in (top_axis, bottom_axis):
            axis.axvspan(0.0, threshold, color="#fee8c8", alpha=0.35, zorder=0)
            axis.axvline(threshold, color="#d95f0e", linestyle="--", linewidth=1.2, zorder=2)

        top_axis.bar(
            horizon_panel["bin_center"],
            horizon_panel["sample_count"],
            width=np.maximum(horizon_panel["bin_width"] * 0.92, x_max * 0.015),
            color="#cfd8dc",
            edgecolor="#78909c",
            linewidth=0.8,
            zorder=1,
        )
        top_axis.bar(
            horizon_panel["bin_center"],
            horizon_panel["fatal_count"],
            width=np.maximum(horizon_panel["bin_width"] * 0.55, x_max * 0.009),
            color="#d6604d",
            edgecolor="#8c2d26",
            linewidth=0.8,
            zorder=3,
        )
        top_axis.set_title(_horizon_label(horizon_h), fontsize=12)
        top_axis.set_ylim(0.0, count_y_max)
        top_axis.grid(axis="y", alpha=0.25, linewidth=0.6)
        top_axis.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        top_axis.spines["top"].set_visible(False)
        top_axis.spines["right"].set_visible(False)
        top_axis.text(
            0.02,
            0.97,
            (
                f"last={int(summary_row['nonfatal_last_n']) + int(summary_row['fatal_last_n'])}\n"
                f"fatal={int(summary_row['fatal_last_n'])}\n"
                f"hard={int(summary_row['hard_case_n'])} ({float(summary_row['hard_case_share_among_fatal']):.0%})\n"
                f"q75={_format_probability(threshold)}"
            ),
            transform=top_axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.8,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#dddddd", "pad": 2.5},
        )

        valid_points = horizon_panel[horizon_panel["sample_count"].gt(0)].copy()
        bottom_axis.plot(
            [0.0, x_max],
            [0.0, x_max],
            linestyle="--",
            color="#4d4d4d",
            linewidth=1.0,
            zorder=1,
        )
        if not valid_points.empty:
            bottom_axis.plot(
                valid_points["predicted_probability_mean"],
                valid_points["observed_mortality"],
                linewidth=2.0,
                color="#2b6cb0",
                zorder=3,
            )
            bottom_axis.scatter(
                valid_points["predicted_probability_mean"],
                valid_points["observed_mortality"],
                s=valid_points["bubble_area"],
                color="#2b6cb0",
                edgecolor="white",
                linewidth=0.7,
                alpha=0.88,
                zorder=4,
            )
        bottom_axis.set_xlim(0.0, x_max)
        bottom_axis.set_ylim(0.0, y_max)
        bottom_axis.grid(alpha=0.25, linewidth=0.6)
        bottom_axis.spines["top"].set_visible(False)
        bottom_axis.spines["right"].set_visible(False)
        bottom_axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        bottom_axis.xaxis.set_major_formatter(percent_formatter)
        bottom_axis.yaxis.set_major_formatter(percent_formatter)

        if column_index == 0:
            top_axis.set_ylabel("Stay count")
            bottom_axis.set_ylabel("Observed mortality")
        else:
            top_axis.tick_params(labelleft=False)
            bottom_axis.tick_params(labelleft=False)

        bottom_axis.set_xlabel("Predicted risk")

    legend_handles = [
        Patch(facecolor="#cfd8dc", edgecolor="#78909c", label="All last-eligible stays"),
        Patch(facecolor="#d6604d", edgecolor="#8c2d26", label="Fatal stays"),
        Patch(facecolor="#fee8c8", edgecolor="none", label="Risk <= horizon q75 threshold"),
        Line2D([0], [0], color="#d95f0e", linestyle="--", linewidth=1.2, label="Horizon q75 threshold"),
        Line2D([0], [0], color="#2b6cb0", marker="o", linewidth=2.0, label="Binned mortality vs risk"),
        Line2D([0], [0], color="#4d4d4d", linestyle="--", linewidth=1.0, label="Identity line"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=9,
    )
    figure.suptitle(
        "ASIC logistic last-eligible stay-level mortality vs risk across horizons",
        fontsize=14,
        y=1.08,
    )
    figure.text(
        0.5,
        -0.01,
        "Common pooled risk bins across all horizons. Shaded region marks each horizon's nonfatal q75 hard-case threshold.",
        ha="center",
        va="top",
        fontsize=9,
        color="#555555",
    )
    figure.subplots_adjust(top=0.77, bottom=0.17, left=0.06, right=0.995, wspace=0.12, hspace=0.08)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _select_interpretation_label(
    horizon_summary: pd.DataFrame,
    pairwise_overlap: pd.DataFrame,
    directional_overlap: pd.DataFrame,
    binned_summary: pd.DataFrame,
) -> tuple[str, dict[str, float]]:
    share_lookup = horizon_summary.set_index("horizon_h")["hard_case_share_among_fatal"]
    share_range = float(share_lookup.max() - share_lookup.min())
    anchor_share = float(share_lookup.loc[ANCHOR_HORIZON])
    contrast_share = float(share_lookup.loc[MAIN_CONTRAST_HORIZON])
    late_share = float(share_lookup.loc[max(share_lookup.index.astype(int).tolist())])

    pairwise_key = (
        pairwise_overlap[
            pairwise_overlap["horizon_a"].eq(_horizon_label(ANCHOR_HORIZON))
            & pairwise_overlap["horizon_b"].eq(_horizon_label(MAIN_CONTRAST_HORIZON))
        ]
        .iloc[0]
    )
    directional_anchor_to_contrast = float(
        directional_overlap[
            directional_overlap["horizon_from"].eq(_horizon_label(ANCHOR_HORIZON))
            & directional_overlap["horizon_to"].eq(_horizon_label(MAIN_CONTRAST_HORIZON))
        ]["overlap_from_A_to_B"].iloc[0]
    )
    directional_contrast_to_anchor = float(
        directional_overlap[
            directional_overlap["horizon_from"].eq(_horizon_label(MAIN_CONTRAST_HORIZON))
            & directional_overlap["horizon_to"].eq(_horizon_label(ANCHOR_HORIZON))
        ]["overlap_from_A_to_B"].iloc[0]
    )
    mean_jaccard = float(pairwise_overlap["jaccard_index"].dropna().mean())
    shape_distance_anchor_to_contrast = _shape_distance(
        binned_summary,
        horizon_a=ANCHOR_HORIZON,
        horizon_b=MAIN_CONTRAST_HORIZON,
    )

    share_stable = bool(share_range <= 0.20 and abs(contrast_share - anchor_share) <= 0.15)
    membership_substantial = bool(
        mean_jaccard >= 0.50
        and float(pairwise_key["jaccard_index"]) >= 0.45
        and directional_anchor_to_contrast >= 0.60
        and directional_contrast_to_anchor >= 0.50
    )
    shape_broadly_similar = bool(shape_distance_anchor_to_contrast <= 0.22)
    share_clearly_decreases = bool(
        contrast_share <= anchor_share - 0.10 and late_share <= anchor_share - 0.10
    )

    if share_clearly_decreases:
        label = "shrink"
    elif share_stable and membership_substantial and shape_broadly_similar:
        label = "persist"
    else:
        label = "change form"

    return label, {
        "share_range": share_range,
        "anchor_share": anchor_share,
        "contrast_share": contrast_share,
        "late_share": late_share,
        "mean_jaccard": mean_jaccard,
        "anchor_contrast_jaccard": float(pairwise_key["jaccard_index"]),
        "directional_anchor_to_contrast": directional_anchor_to_contrast,
        "directional_contrast_to_anchor": directional_contrast_to_anchor,
        "shape_distance_anchor_to_contrast": shape_distance_anchor_to_contrast,
    }


def _build_interpretation_memo(
    *,
    horizon_summary: pd.DataFrame,
    pairwise_overlap: pd.DataFrame,
    directional_overlap: pd.DataFrame,
    binned_summary: pd.DataFrame,
    persistence_distribution: pd.DataFrame,
    consistency_issues: Sequence[str],
    interpretation_label: str,
    interpretation_metrics: dict[str, float],
) -> str:
    share_lookup = horizon_summary.set_index("horizon_h")["hard_case_share_among_fatal"]
    pair_24_48 = pairwise_overlap[
        pairwise_overlap["horizon_a"].eq(_horizon_label(ANCHOR_HORIZON))
        & pairwise_overlap["horizon_b"].eq(_horizon_label(MAIN_CONTRAST_HORIZON))
    ].iloc[0]
    one_horizon_count = int(
        persistence_distribution.loc[
            persistence_distribution["hard_case_horizon_n"].astype(int).eq(1),
            "fatal_stay_count",
        ].iloc[0]
    )
    multi_horizon_count = int(
        persistence_distribution.loc[
            persistence_distribution["hard_case_horizon_n"].astype(int).ge(2),
            "fatal_stay_count",
        ].sum()
    )

    if consistency_issues:
        consistency_text = (
            "Consistency check detected mismatches between the plotted horizon summaries and the saved Package 1/2 artifacts; those mismatches should be resolved before chapter use."
        )
    else:
        consistency_text = (
            "The plotted horizon-level counts, thresholds, and overlap denominators were consistent with the saved Package 1 and Package 2 artifacts."
        )

    if interpretation_label == "persist":
        label_reason = (
            "On the local synthetic run, the low-risk fatal burden stays present across horizons, cross-horizon membership overlap is substantial, and the 24h vs 48h mortality-vs-risk panels remain broadly similar."
        )
    elif interpretation_label == "shrink":
        label_reason = (
            "On the local synthetic run, the low-risk fatal burden contracts at longer horizons and the hard-case share drops visibly enough to support a shrink reading."
        )
    else:
        label_reason = (
            "On the local synthetic run, the balance between share, membership overlap, and the binned mortality-vs-risk panels is not cleanly captured by a simple persistence story, so change form is the closest label."
        )

    if interpretation_metrics["shape_distance_anchor_to_contrast"] <= 0.22:
        shape_read = (
            f"the five binned panels keep the same pooled risk axis and remain broadly similar in overall upward mortality-with-risk structure. "
            f"The weighted 24h vs 48h shape distance is `{interpretation_metrics['shape_distance_anchor_to_contrast']:.3f}`, which is small enough for a descriptive similarity read."
        )
    else:
        shape_read = (
            f"the five binned panels still show mortality increasing with risk, but the pooled-bin profile shifts enough between 24h and 48h to count as a material descriptive change in this local run. "
            f"The weighted 24h vs 48h shape distance is `{interpretation_metrics['shape_distance_anchor_to_contrast']:.3f}`."
        )

    lines = [
        "# Horizon Interpretation Memo",
        "",
        "This memo is intentionally bounded to the frozen Chapter 1 ASIC logistic last-eligible stay-level design.",
        "All numeric values below come from the repository's synthetic stand-in data and are implementation-test outputs only.",
        "",
        "## Readout",
        "",
        f"- Hard-case share: `{_horizon_label(8)}` to `{_horizon_label(24)}` stays at `{float(share_lookup.loc[8]):.2f}` to `{float(share_lookup.loc[24]):.2f}`, then is higher at `{_horizon_label(48)}` `{float(share_lookup.loc[48]):.2f}` and `{_horizon_label(72)}` `{float(share_lookup.loc[72]):.2f}`. For the narrative anchor and main contrast, `{_horizon_label(ANCHOR_HORIZON)}` is `{interpretation_metrics['anchor_share']:.2f}` and `{_horizon_label(MAIN_CONTRAST_HORIZON)}` is `{interpretation_metrics['contrast_share']:.2f}`.",
        f"- Hard-case membership: overlap is substantial but incomplete. Mean pairwise Jaccard is `{interpretation_metrics['mean_jaccard']:.3f}`; `{_horizon_label(ANCHOR_HORIZON)}` vs `{_horizon_label(MAIN_CONTRAST_HORIZON)}` has Jaccard `{interpretation_metrics['anchor_contrast_jaccard']:.3f}`, with directional overlap `{_horizon_label(ANCHOR_HORIZON)} -> {_horizon_label(MAIN_CONTRAST_HORIZON)}` `{interpretation_metrics['directional_anchor_to_contrast']:.3f}` and the reverse `{interpretation_metrics['directional_contrast_to_anchor']:.3f}`.",
        f"- Mortality-vs-risk shape: {shape_read}",
        f"- Overall label: `{interpretation_label}`. {label_reason}",
        "",
        "## Answers To The Four Questions",
        "",
        f"1. Hard-case share stable across horizons? `{ 'roughly yes, but modestly higher at longer horizons' if interpretation_metrics['share_range'] <= 0.20 else 'no' }`.",
        f"2. Hard-case membership stable across horizons? `{ 'substantial but not perfect overlap' if interpretation_metrics['mean_jaccard'] >= 0.50 else 'limited overlap' }`.",
        f"3. Mortality-vs-risk shape shift materially? `{ 'not materially in this local descriptive run' if interpretation_metrics['shape_distance_anchor_to_contrast'] <= 0.22 else 'yes, enough to look materially different' }`.",
        (
            f"4. Do hard cases persist, shrink, or change form? `{interpretation_label}`. "
            f"On the synthetic run, `{multi_horizon_count}` fatal stays are hard in at least two horizons versus `{one_horizon_count}` in exactly one horizon, which is more compatible with persistence than with pure horizon-specific rotation."
            if interpretation_label == "persist"
            else f"4. Do hard cases persist, shrink, or change form? `{interpretation_label}`."
        ),
        "",
        "## Caveat",
        "",
        f"- {consistency_text}",
        "- Any chapter interpretation on real data must remain conditional on the saved horizon-specific q75 thresholds, the last-eligible stay-level representation, and the exact matched-denominator logic from Package 2.",
        "- Local synthetic values in this memo are not scientifically interpretable.",
        "",
    ]
    return "\n".join(lines)


def _build_final_summary_note(
    *,
    hard_case_dir: Path,
    foundation_dir: Path,
    overlap_dir: Path,
    output_dir: Path,
    consistency_issues: Sequence[str],
    interpretation_label: str,
) -> str:
    output_rows = [
        {"output": "mortality_risk_horizon_comparison.png", "path": _display_path(output_dir / "mortality_risk_horizon_comparison.png")},
        {"output": "horizon_interpretation_memo.md", "path": _display_path(output_dir / "horizon_interpretation_memo.md")},
        {"output": "final_horizon_summary.md", "path": _display_path(output_dir / "final_horizon_summary.md")},
        {"output": "mortality_risk_horizon_binned_summary.csv", "path": _display_path(output_dir / "mortality_risk_horizon_binned_summary.csv")},
    ]
    input_rows = [
        {"input": "Saved stay-level hard-case artifact", "path": _display_path(hard_case_dir / "stay_level_hard_case_flags.csv")},
        {"input": "Package 1 horizon summary", "path": _display_path(foundation_dir / "horizon_summary.csv")},
        {"input": "Package 1 foundation note", "path": _display_path(foundation_dir / "artifact_foundation_note.md")},
        {"input": "Package 2 pairwise denominators", "path": _display_path(overlap_dir / "pairwise_denominators.csv")},
        {"input": "Package 2 pairwise overlap", "path": _display_path(overlap_dir / "pairwise_overlap.csv")},
        {"input": "Package 2 directional overlap", "path": _display_path(overlap_dir / "directional_overlap.csv")},
        {"input": "Package 2 persistence distribution", "path": _display_path(overlap_dir / "persistence_distribution.csv")},
        {"input": "Package 2 overlap note", "path": _display_path(overlap_dir / "overlap_note.md")},
    ]

    lines = [
        "# Final Horizon Summary",
        "",
        "## Inputs Used",
        "",
        _markdown_table(input_rows, columns=["input", "path"]),
        "",
        "## Package 3 Outputs",
        "",
        _markdown_table(output_rows, columns=["output", "path"]),
        "",
        "## Assumptions",
        "",
        "- The final figure uses the saved logistic stay-level hard-case artifact as the last-eligible stay-level prediction source for all five horizons.",
        "- For direct visual comparability, the figure uses a common pooled probability binning scheme across horizons while keeping the horizon-specific q75 threshold line from Package 1.",
        f"- The memo keeps `{_horizon_label(ANCHOR_HORIZON)}` as the narrative anchor and `{_horizon_label(MAIN_CONTRAST_HORIZON)}` as the main contrast, consistent with the sprint brief.",
        f"- The current descriptive label is `{interpretation_label}`, but only as a local implementation-test readout.",
        "",
        "## Caveats For Chapter Write-Up",
        "",
        (
            "- No consistency mismatches were detected between the plotted horizon-level counts/thresholds and the saved Package 1/2 outputs."
            if not consistency_issues
            else "- Consistency mismatches were detected and should be resolved before chapter use:"
        ),
    ]
    if consistency_issues:
        lines.extend(f"- {issue}" for issue in consistency_issues)
    lines.extend(
        [
            "- The mortality-vs-risk figure is descriptive only; it does not reopen or replace the frozen hard-case definition.",
            "- Any chapter narrative should remain bounded to representation-level risk structure under the last-eligible stay design and should avoid causal or subtype claims.",
            "- Current local outputs are implementation-test outputs from synthetic data only and are not scientifically interpretable.",
            "",
        ]
    )
    return "\n".join(lines)


def run_asic_horizon_dependence_final(
    *,
    hard_case_dir: Path = DEFAULT_HARD_CASE_DIR,
    foundation_dir: Path = DEFAULT_FOUNDATION_DIR,
    overlap_dir: Path = DEFAULT_OVERLAP_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    horizons: Sequence[int] | None = None,
    bin_count: int = DEFAULT_BIN_COUNT,
    output_format: str = "csv",
) -> HorizonFinalRunResult:
    if output_format != "csv":
        raise ValueError(f"Unsupported output format: {output_format}")

    normalized_horizons = _normalize_horizons(horizons)
    resolved_hard_case_dir = Path(hard_case_dir)
    resolved_foundation_dir = Path(foundation_dir)
    resolved_overlap_dir = Path(overlap_dir)
    resolved_output_dir = Path(output_dir)

    stay_level, stay_level_path = _load_stay_level(resolved_hard_case_dir, horizons=normalized_horizons)
    foundation_summary = _load_foundation_summary(resolved_foundation_dir, horizons=normalized_horizons)
    pairwise_denominators, pairwise_overlap, directional_overlap, persistence_distribution = _load_overlap_artifacts(
        resolved_overlap_dir
    )
    horizon_summary = _build_horizon_summary_from_stay_level(stay_level)

    consistency_issues = [
        *_compare_with_foundation_summary(horizon_summary, foundation_summary),
        *_compare_with_overlap_outputs(horizon_summary, pairwise_denominators, pairwise_overlap),
    ]

    binned_summary = _build_binned_summary(
        stay_level,
        horizon_summary,
        requested_bin_count=bin_count,
    )
    interpretation_label, interpretation_metrics = _select_interpretation_label(
        horizon_summary,
        pairwise_overlap,
        directional_overlap,
        binned_summary,
    )

    binned_summary_path = write_dataframe(
        binned_summary,
        resolved_output_dir / "mortality_risk_horizon_binned_summary.csv",
        output_format=output_format,
    )
    figure_path = _plot_mortality_risk_horizon_comparison(
        binned_summary,
        horizon_summary,
        output_path=resolved_output_dir / "mortality_risk_horizon_comparison.png",
    )
    memo_path = write_text(
        _build_interpretation_memo(
            horizon_summary=horizon_summary,
            pairwise_overlap=pairwise_overlap,
            directional_overlap=directional_overlap,
            binned_summary=binned_summary,
            persistence_distribution=persistence_distribution,
            consistency_issues=consistency_issues,
            interpretation_label=interpretation_label,
            interpretation_metrics=interpretation_metrics,
        ),
        resolved_output_dir / "horizon_interpretation_memo.md",
    )
    final_summary_path = write_text(
        _build_final_summary_note(
            hard_case_dir=resolved_hard_case_dir,
            foundation_dir=resolved_foundation_dir,
            overlap_dir=resolved_overlap_dir,
            output_dir=resolved_output_dir,
            consistency_issues=consistency_issues,
            interpretation_label=interpretation_label,
        ),
        resolved_output_dir / "final_horizon_summary.md",
    )
    manifest_path = _write_json(
        {
            "timestamp_utc": _utc_timestamp(),
            "hard_case_dir": str(resolved_hard_case_dir.resolve()),
            "foundation_dir": str(resolved_foundation_dir.resolve()),
            "overlap_dir": str(resolved_overlap_dir.resolve()),
            "output_dir": str(resolved_output_dir.resolve()),
            "horizons": list(normalized_horizons),
            "bin_count": int(bin_count),
            "consistency_issue_count": int(len(consistency_issues)),
            "consistency_issues": list(consistency_issues),
            "interpretation_label": interpretation_label,
            "artifacts": {
                "binned_summary": str(binned_summary_path.resolve()),
                "figure": str(figure_path.resolve()),
                "interpretation_memo": str(memo_path.resolve()),
                "final_summary_note": str(final_summary_path.resolve()),
            },
        },
        resolved_output_dir / "run_manifest.json",
    )

    return HorizonFinalRunResult(
        hard_case_dir=resolved_hard_case_dir,
        foundation_dir=resolved_foundation_dir,
        overlap_dir=resolved_overlap_dir,
        output_dir=resolved_output_dir,
        horizons=normalized_horizons,
        horizon_summary=horizon_summary,
        binned_summary=binned_summary,
        consistency_issues=tuple(consistency_issues),
        interpretation_label=interpretation_label,
        artifacts=HorizonFinalArtifacts(
            figure_path=figure_path,
            binned_summary_path=binned_summary_path,
            interpretation_memo_path=memo_path,
            final_summary_path=final_summary_path,
            manifest_path=manifest_path,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the final ASIC horizon-comparison figure and short interpretation memo "
            "from the saved Package 1 and Package 2 artifacts."
        )
    )
    parser.add_argument(
        "--hard-case-dir",
        type=Path,
        default=DEFAULT_HARD_CASE_DIR,
        help="Directory containing the saved logistic stay-level hard-case artifact and manifest.",
    )
    parser.add_argument(
        "--foundation-dir",
        type=Path,
        default=DEFAULT_FOUNDATION_DIR,
        help="Directory containing Package 1 horizon summary outputs.",
    )
    parser.add_argument(
        "--overlap-dir",
        type=Path,
        default=DEFAULT_OVERLAP_DIR,
        help="Directory containing Package 2 overlap outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where final horizon-comparison outputs will be written.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        help="Optional subset of frozen horizons to include.",
    )
    parser.add_argument(
        "--bin-count",
        type=int,
        default=DEFAULT_BIN_COUNT,
        help="Requested number of common pooled probability bins for the comparison figure.",
    )
    parser.add_argument(
        "--output-format",
        choices=("csv",),
        default="csv",
        help="Output format for the supporting binned summary artifact.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = run_asic_horizon_dependence_final(
        hard_case_dir=args.hard_case_dir,
        foundation_dir=args.foundation_dir,
        overlap_dir=args.overlap_dir,
        output_dir=args.output_dir,
        horizons=args.horizons,
        bin_count=args.bin_count,
        output_format=args.output_format,
    )

    print(f"Hard-case directory: {result.hard_case_dir}")
    print(f"Foundation directory: {result.foundation_dir}")
    print(f"Overlap directory: {result.overlap_dir}")
    print(f"Output directory: {result.output_dir}")
    print(f"Interpretation label: {result.interpretation_label}")
    if result.consistency_issues:
        print("Consistency issues:")
        for issue in result.consistency_issues:
            print(f"- {issue}")
    else:
        print("Consistency check: no mismatches detected")
    print(f"Wrote {result.artifacts.figure_path}")
    print(f"Wrote {result.artifacts.interpretation_memo_path}")
    print(f"Wrote {result.artifacts.final_summary_path}")
    print(f"Wrote {result.artifacts.binned_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
