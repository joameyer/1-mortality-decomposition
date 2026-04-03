from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
RUN_CONFIG_PATH = REPO_ROOT / "config" / "ch1_run_config.json"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chapter1_mortality_decomposition.icd10_disease_groups import (  # noqa: E402
    FROZEN_DISEASE_GROUP_HIERARCHY,
    OTHER_MIXED_UNCATEGORIZED_GROUP,
    derive_icd10_disease_group,
    match_candidate_disease_groups,
)
from chapter1_mortality_decomposition.utils import (  # noqa: E402
    ensure_directory,
    write_dataframe,
    write_text,
)


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts" / "chapter1" / "evaluation" / "asic" / "icd10_disease_group_validation"
)


def _candidate_asic_input_roots() -> list[Path]:
    candidates: list[Path] = []

    env_value = os.environ.get("ASIC_INPUT_ROOT")
    if env_value:
        candidates.append(Path(env_value).expanduser())

    if RUN_CONFIG_PATH.exists():
        payload = json.loads(RUN_CONFIG_PATH.read_text(encoding="utf-8"))
        config_input = payload.get("input_dir")
        if isinstance(config_input, str) and config_input.strip():
            config_path = Path(config_input).expanduser()
            if not config_path.is_absolute():
                config_path = (REPO_ROOT / config_path).resolve()
            candidates.append(config_path)

    upstream_artifact_root = REPO_ROOT.parent / "hpc-icu-data-platform" / "artifacts"
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
        return input_root.expanduser().resolve()

    candidates = _candidate_asic_input_roots()
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate the upstream ASIC harmonized artifacts. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _fmt_count(count: int, total: int) -> str:
    pct = (100.0 * count / total) if total else 0.0
    return f"{count}/{total} ({pct:.0f}%)"


def _markdown_table(frame: pd.DataFrame) -> str:
    rendered = frame.fillna("").astype(str)
    header = "| " + " | ".join(rendered.columns.tolist()) + " |"
    separator = "| " + " | ".join(["---"] * len(rendered.columns)) + " |"
    rows = [
        "| " + " | ".join(row) + " |"
        for row in rendered.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _candidate_group_label(candidate_groups: tuple[str, ...]) -> str:
    return " + ".join(candidate_groups) if candidate_groups else "no explicit group match"


def _preview_stems(stems: tuple[str, ...], *, limit: int = 24) -> str:
    if not stems:
        return ""
    if len(stems) <= limit:
        return ", ".join(stems)
    return ", ".join(stems[:limit]) + f", ... (+{len(stems) - limit} more)"


def _multi_match_bucket(candidate_group_count: int) -> str:
    if candidate_group_count == 1:
        return "1 group"
    if candidate_group_count >= 3:
        return "3+ groups"
    return f"{candidate_group_count} groups"


def _build_stay_level_output(static: pd.DataFrame) -> pd.DataFrame:
    matches = static["icd10_codes"].map(derive_icd10_disease_group)

    stay_level = static.copy()
    stay_level["normalized_stems"] = matches.map(lambda match: "|".join(match.normalized_stems))
    stay_level["normalized_stem_count"] = matches.map(lambda match: len(match.normalized_stems))
    stay_level["candidate_groups"] = matches.map(
        lambda match: "|".join(match.candidate_groups)
    )
    stay_level["candidate_group_count"] = matches.map(lambda match: len(match.candidate_groups))
    stay_level["candidate_group_combination"] = matches.map(
        lambda match: _candidate_group_label(match.candidate_groups)
    )
    stay_level["final_disease_group"] = matches.map(lambda match: match.final_group)
    stay_level["normalized_stems_preview"] = matches.map(
        lambda match: _preview_stems(match.normalized_stems)
    )
    return stay_level


def _build_group_counts(stay_level: pd.DataFrame) -> pd.DataFrame:
    total = int(stay_level.shape[0])
    counts = (
        stay_level["final_disease_group"]
        .value_counts(dropna=False)
        .reindex(FROZEN_DISEASE_GROUP_HIERARCHY, fill_value=0)
        .rename_axis("final_disease_group")
        .reset_index(name="stay_count")
    )
    counts["stay_pct"] = counts["stay_count"].map(lambda count: round((100.0 * count / total), 1))
    return counts


def _build_ambiguity_summary(stay_level: pd.DataFrame) -> pd.DataFrame:
    total = int(stay_level.shape[0])
    summary = (
        stay_level["candidate_group_count"]
        .map(_multi_match_bucket)
        .value_counts()
        .reindex(["0 groups", "1 group", "2 groups", "3+ groups"], fill_value=0)
        .rename_axis("pre_hierarchy_match_bucket")
        .reset_index(name="stay_count")
    )
    summary["stay_pct"] = summary["stay_count"].map(lambda count: round((100.0 * count / total), 1))
    return summary


def _build_combination_summary(stay_level: pd.DataFrame) -> pd.DataFrame:
    multi_match = stay_level[stay_level["candidate_group_count"].ge(2)].copy()
    total = int(stay_level.shape[0])
    if multi_match.empty:
        return pd.DataFrame(
            columns=[
                "candidate_group_combination",
                "stay_count",
                "stay_pct",
            ]
        )

    summary = (
        multi_match["candidate_group_combination"]
        .value_counts()
        .rename_axis("candidate_group_combination")
        .reset_index(name="stay_count")
    )
    summary["stay_pct"] = summary["stay_count"].map(lambda count: round((100.0 * count / total), 1))
    return summary


def _build_sample_rows(stay_level: pd.DataFrame, *, n_per_group: int = 5) -> pd.DataFrame:
    sample_columns = [
        "stay_id_global",
        "hospital_id",
        "raw_icd10_codes",
        "normalized_stems_preview",
        "candidate_group_combination",
        "final_disease_group",
    ]
    sampling_frame = stay_level.rename(columns={"icd10_codes": "raw_icd10_codes"}).copy()
    samples: list[pd.DataFrame] = []
    for group in FROZEN_DISEASE_GROUP_HIERARCHY:
        subset = sampling_frame[sampling_frame["final_disease_group"].eq(group)].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(
            by=["candidate_group_count", "normalized_stem_count", "stay_id_global"],
            ascending=[False, False, True],
        ).head(n_per_group)
        samples.append(subset[sample_columns])
    if not samples:
        return pd.DataFrame(columns=sample_columns)
    return pd.concat(samples, ignore_index=True)


def _build_other_examples(stay_level: pd.DataFrame) -> pd.DataFrame:
    sample_columns = [
        "stay_id_global",
        "hospital_id",
        "raw_icd10_codes",
        "normalized_stems_preview",
        "candidate_group_combination",
        "final_disease_group",
    ]
    other_rows = stay_level.rename(columns={"icd10_codes": "raw_icd10_codes"}).copy()
    other_rows = other_rows[
        other_rows["final_disease_group"].eq(OTHER_MIXED_UNCATEGORIZED_GROUP)
    ].copy()
    other_rows = other_rows.sort_values(
        by=["normalized_stem_count", "stay_id_global"],
        ascending=[False, True],
    )
    return other_rows[sample_columns]


def _build_non_driver_stem_summary(stay_level: pd.DataFrame) -> pd.DataFrame:
    counts: dict[str, int] = {}

    for stems_blob in stay_level["normalized_stems"]:
        stems = [stem for stem in str(stems_blob).split("|") if stem]
        non_driver_stems = [
            stem for stem in stems if not match_candidate_disease_groups((stem,))
        ]
        for stem in non_driver_stems:
            counts[stem] = counts.get(stem, 0) + 1

    summary = pd.DataFrame(
        sorted(counts.items(), key=lambda item: (-item[1], item[0])),
        columns=["stem", "stays_with_stem"],
    )
    if summary.empty:
        return summary

    summary["chapter_letter"] = summary["stem"].str[0]
    return summary


def _build_validation_memo(
    *,
    input_root: Path,
    static_path: Path,
    stay_level: pd.DataFrame,
    group_counts: pd.DataFrame,
    ambiguity_summary: pd.DataFrame,
    combination_summary: pd.DataFrame,
    non_driver_stems: pd.DataFrame,
    output_dir: Path,
) -> str:
    total = int(stay_level.shape[0])
    surgical_count = int(
        group_counts.loc[
            group_counts["final_disease_group"].eq(FROZEN_DISEASE_GROUP_HIERARCHY[0]),
            "stay_count",
        ].iloc[0]
    )
    respiratory_count = int(
        group_counts.loc[
            group_counts["final_disease_group"].eq(FROZEN_DISEASE_GROUP_HIERARCHY[1]),
            "stay_count",
        ].iloc[0]
    )
    infection_count = int(
        group_counts.loc[
            group_counts["final_disease_group"].eq(FROZEN_DISEASE_GROUP_HIERARCHY[2]),
            "stay_count",
        ].iloc[0]
    )
    cardiovascular_count = int(
        group_counts.loc[
            group_counts["final_disease_group"].eq(FROZEN_DISEASE_GROUP_HIERARCHY[3]),
            "stay_count",
        ].iloc[0]
    )
    neurologic_count = int(
        group_counts.loc[
            group_counts["final_disease_group"].eq(FROZEN_DISEASE_GROUP_HIERARCHY[4]),
            "stay_count",
        ].iloc[0]
    )
    other_count = int(
        group_counts.loc[
            group_counts["final_disease_group"].eq(OTHER_MIXED_UNCATEGORIZED_GROUP),
            "stay_count",
        ].iloc[0]
    )
    multi_match_count = int(stay_level["candidate_group_count"].ge(2).sum())
    no_match_count = int(stay_level["candidate_group_count"].eq(0).sum())

    top_non_driver = non_driver_stems.head(12)
    top_combinations = combination_summary.head(10)

    lines = [
        "# ASIC ICD-10 Disease-Group Validation Memo",
        "",
        "## Implementation",
        "",
        f"- Disease-group logic module: `{_display_path(REPO_ROOT / 'src' / 'chapter1_mortality_decomposition' / 'icd10_disease_groups.py')}`",
        f"- Inspection script: `{_display_path(REPO_ROOT / 'scripts' / 'ch1_asic_icd10_disease_group_inspection.py')}`",
        f"- Local ASIC input root used: `{input_root}`",
        f"- Static table inspected: `{static_path}`",
        f"- Output directory: `{_display_path(output_dir)}`",
        "",
        "## Parsing Behavior",
        "",
        "- Split `icd10_codes` on comma, strip whitespace, uppercase, remove internal spaces, strip trailing exclamation marks, collapse decimals to normalized 3-character stems, deduplicate within stay, then match on the normalized stem set.",
        "- The frozen first-match hierarchy is applied exactly in this order: surgical / postoperative / trauma-related, respiratory / pulmonary, infection / sepsis non-pulmonary, cardiovascular, neurologic, other / mixed / uncategorized.",
        "",
        "## Final Counts",
        "",
        _markdown_table(group_counts),
        "",
        "## Ambiguity Burden",
        "",
        _markdown_table(ambiguity_summary),
        "",
        f"- Multi-match burden before hierarchy resolution: `{_fmt_count(multi_match_count, total)}`.",
        f"- No explicit group match before fallback: `{_fmt_count(no_match_count, total)}`.",
        f"- Fallback bucket size after hierarchy assignment: `{_fmt_count(other_count, total)}`.",
        "",
        "## Common Multi-Match Combinations",
        "",
    ]

    if top_combinations.empty:
        lines.extend(["- No 2+ group combinations were observed.", ""])
    else:
        lines.extend([_markdown_table(top_combinations), ""])

    lines.extend(
        [
            "## Edge Cases",
            "",
            "- The local artifact still behaves like a diagnosis bag rather than a principal-diagnosis field, so the hierarchy is doing real work.",
            "- `J95` and `I97` are kept inside the earlier surgical/postoperative rule instead of being absorbed into respiratory or cardiovascular matching.",
            "- `N39` is intentionally not used as a standalone infection trigger, and auxiliary stems such as `U80`, `U81`, `Z22`, and `Z29` do not drive assignment on their own.",
            (
                f"- Final group distribution is strongly front-loaded: surgical `{_fmt_count(surgical_count, total)}`, "
                f"respiratory `{_fmt_count(respiratory_count, total)}`, infection `{_fmt_count(infection_count, total)}`, "
                f"cardiovascular `{_fmt_count(cardiovascular_count, total)}`, neurologic `{_fmt_count(neurologic_count, total)}`, "
                f"other `{_fmt_count(other_count, total)}`."
            ),
            "",
        ]
    )

    if top_non_driver.empty:
        lines.extend(["- No frequent non-driver stems were observed after normalization.", ""])
    else:
        lines.extend(
            [
                "Most frequent stems that do not directly drive any target group in the current rules:",
                "",
                _markdown_table(top_non_driver),
                "",
            ]
        )

    readiness_line = (
        "- Judgment: the parser and implementation look clean enough for full-dataset execution, but the local artifact should be treated as a substantive review checkpoint because the frozen hierarchy strongly compresses final assignment toward the earliest buckets."
    )
    if no_match_count > max(5, int(0.25 * total)):
        readiness_line = (
            "- Judgment: the mapping needs revision before full-dataset HPC execution because the fallback burden is too high in the local artifact."
        )
    elif multi_match_count >= int(0.8 * total) and (infection_count + neurologic_count) <= max(3, int(0.05 * total)):
        readiness_line = (
            "- Judgment: the implementation is technically ready, but there is a real scientific review point before freezing for HPC: in the local artifact, the bag-of-codes field plus the frozen first-match hierarchy collapses most stays into surgical or respiratory, leaving very small final infection and neurologic buckets. That is not a parser failure, but you should confirm that this degree of hierarchy compression is acceptable for the intended descriptive use."
        )

    lines.extend(
        [
            "## Judgment",
            "",
            readiness_line,
            "",
        ]
    )
    return "\n".join(lines)


def build_outputs(
    *,
    input_root: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    resolved_input_root = _resolve_asic_input_root(input_root)
    static_path = resolved_input_root / "static" / "harmonized.csv"
    if not static_path.exists():
        raise FileNotFoundError(f"Missing ASIC static harmonized table: {static_path}")

    ensure_directory(output_dir)
    static = pd.read_csv(
        static_path,
        usecols=["stay_id_global", "hospital_id", "icd10_codes"],
    )

    stay_level = _build_stay_level_output(static)
    group_counts = _build_group_counts(stay_level)
    ambiguity_summary = _build_ambiguity_summary(stay_level)
    combination_summary = _build_combination_summary(stay_level)
    sample_rows = _build_sample_rows(stay_level)
    other_examples = _build_other_examples(stay_level)
    non_driver_stems = _build_non_driver_stem_summary(stay_level)
    memo_text = _build_validation_memo(
        input_root=resolved_input_root,
        static_path=static_path,
        stay_level=stay_level,
        group_counts=group_counts,
        ambiguity_summary=ambiguity_summary,
        combination_summary=combination_summary,
        non_driver_stems=non_driver_stems,
        output_dir=output_dir,
    )

    output_paths = {
        "stay_level": write_dataframe(
            stay_level,
            output_dir / "asic_static_icd10_disease_groups.csv",
            output_format="csv",
        ),
        "group_counts": write_dataframe(
            group_counts,
            output_dir / "final_group_counts.csv",
            output_format="csv",
        ),
        "ambiguity_summary": write_dataframe(
            ambiguity_summary,
            output_dir / "pre_hierarchy_ambiguity_summary.csv",
            output_format="csv",
        ),
        "combination_summary": write_dataframe(
            combination_summary,
            output_dir / "common_multi_match_combinations.csv",
            output_format="csv",
        ),
        "sample_rows": write_dataframe(
            sample_rows,
            output_dir / "sample_rows_by_final_group.csv",
            output_format="csv",
        ),
        "other_examples": write_dataframe(
            other_examples,
            output_dir / "other_mixed_uncategorized_examples.csv",
            output_format="csv",
        ),
        "non_driver_stems": write_dataframe(
            non_driver_stems,
            output_dir / "non_driver_stem_summary.csv",
            output_format="csv",
        ),
        "memo": write_text(
            memo_text + "\n",
            output_dir / "validation_memo.md",
        ),
    }
    return output_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the frozen ASIC ICD-10 disease-group mapping on the locally available "
            "ASIC static harmonized table."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        help="Optional override for the local ASIC harmonized artifact root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where validation outputs will be written.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_paths = build_outputs(input_root=args.input_root, output_dir=args.output_dir)
    print(f"Wrote {len(output_paths)} ICD-10 disease-group inspection artifacts.")
    for name, path in output_paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
