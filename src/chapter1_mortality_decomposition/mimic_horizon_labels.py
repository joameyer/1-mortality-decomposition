from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from chapter1_mortality_decomposition.mimic_blocks import (
    REPO_ROOT,
    _is_demo_mimic_root,
    _is_relative_to,
    _resolve_path,
)
from chapter1_mortality_decomposition.utils import ensure_directory


DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "ch1_mimic_horizon_labels.yaml"
DEFAULT_HORIZONS = (8, 16, 24, 48)
LABEL_FILE_STEM = "chapter1_proxy_horizon_labels"
USABLE_LABEL_FILE_STEM = "chapter1_usable_proxy_horizon_labels"
RETAINED_COHORT_FILE_STEM = "chapter1_retained_stay_table"


@dataclass(frozen=True)
class MimicHorizonLabelConfig:
    mimic_root: Path
    preprocessing_output_root: Path
    target_output_root: Path
    reports_dir: Path
    horizons_hours: tuple[int, ...] = DEFAULT_HORIZONS
    output_format: str = "csv"

    @property
    def label_path(self) -> Path:
        return self.preprocessing_output_root / "labels" / f"{LABEL_FILE_STEM}.{self.output_format}"

    @property
    def usable_label_path(self) -> Path:
        return (
            self.preprocessing_output_root
            / "labels"
            / f"{USABLE_LABEL_FILE_STEM}.{self.output_format}"
        )

    @property
    def retained_cohort_path(self) -> Path:
        return (
            self.preprocessing_output_root
            / "cohort"
            / f"{RETAINED_COHORT_FILE_STEM}.{self.output_format}"
        )


def _parse_horizons(value: str) -> tuple[int, ...]:
    horizons: list[int] = []
    seen: set[int] = set()
    for token in str(value).replace(";", ",").split(","):
        stripped = token.strip()
        if not stripped:
            continue
        horizon = int(stripped)
        if horizon <= 0:
            raise ValueError("MIMIC horizon labels require positive hour horizons.")
        if horizon not in seen:
            horizons.append(horizon)
            seen.add(horizon)
    if not horizons:
        raise ValueError("At least one horizon must be configured.")
    return tuple(horizons)


def load_config(path: Path) -> MimicHorizonLabelConfig:
    raw: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(f"Unsupported config line in {path}: {line!r}")
        key, value = stripped.split(":", 1)
        raw[key.strip()] = value.split("#", 1)[0].strip().strip("'\"")

    required = ["mimic_root", "preprocessing_output_root", "target_output_root", "reports_dir"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"MIMIC horizon label config {path} is missing keys: {missing}")

    return MimicHorizonLabelConfig(
        mimic_root=_resolve_path(raw["mimic_root"]),
        preprocessing_output_root=_resolve_path(raw["preprocessing_output_root"]),
        target_output_root=_resolve_path(raw["target_output_root"]),
        reports_dir=_resolve_path(raw["reports_dir"]),
        horizons_hours=_parse_horizons(raw.get("horizons_hours", ",".join(map(str, DEFAULT_HORIZONS)))),
        output_format=raw.get("output_format", "csv"),
    )


def enforce_storage_policy(config: MimicHorizonLabelConfig) -> None:
    if _is_demo_mimic_root(config.mimic_root):
        return

    violations = [
        f"{label}={path}"
        for label, path in [
            ("preprocessing_output_root", config.preprocessing_output_root),
            ("target_output_root", config.target_output_root),
        ]
        if _is_relative_to(path, REPO_ROOT)
    ]
    if violations:
        raise ValueError(
            "Unsafe full-MIMIC horizon-label path inside the project repo. Full-MIMIC "
            "row-level preprocessing and target outputs must live outside the repo. "
            "Violations: " + "; ".join(violations)
        )


def _read_table(path: Path, output_format: str) -> pd.DataFrame:
    if output_format == "csv":
        return pd.read_csv(path)
    if output_format == "parquet":
        return pd.read_parquet(path)
    raise ValueError("output_format must be csv or parquet.")


def _write_table(df: pd.DataFrame, path: Path, output_format: str) -> None:
    ensure_directory(path.parent)
    if output_format == "csv":
        df.to_csv(path, index=False)
    elif output_format == "parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError("output_format must be csv or parquet.")


def validate_inputs(config: MimicHorizonLabelConfig) -> None:
    enforce_storage_policy(config)
    required_paths = [config.label_path, config.usable_label_path, config.retained_cohort_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required MIMIC horizon-label inputs: " + ", ".join(missing))


def _require_columns(df: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_label_inputs(config: MimicHorizonLabelConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = _read_table(config.label_path, config.output_format)
    usable = _read_table(config.usable_label_path, config.output_format)
    cohort = _read_table(config.retained_cohort_path, config.output_format)
    _require_columns(
        labels,
        {
            "stay_id_global",
            "hospital_id",
            "horizon_h",
            "prediction_time_h",
            "future_window_end_h",
            "icu_end_time_proxy_hours",
            "event_time_proxy_h",
            "proxy_horizon_labelable",
            "label_value",
            "unlabeled_reason",
        },
        "chapter1_proxy_horizon_labels",
    )
    _require_columns(
        usable,
        {"stay_id_global", "hospital_id", "horizon_h", "label_value"},
        "chapter1_usable_proxy_horizon_labels",
    )
    _require_columns(
        cohort,
        {"stay_id_global", "hospital_id", "icu_mortality"},
        "chapter1_retained_stay_table",
    )
    return labels, usable, cohort


def _normalize_label_frame(labels: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    normalized = labels.copy()
    normalized["stay_id_global"] = normalized["stay_id_global"].astype("string")
    normalized["hospital_id"] = normalized["hospital_id"].astype("string")
    for column in [
        "horizon_h",
        "prediction_time_h",
        "future_window_end_h",
        "icu_end_time_proxy_hours",
        "event_time_proxy_h",
        "label_value",
    ]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["proxy_horizon_labelable"] = normalized["proxy_horizon_labelable"].astype("boolean")

    cohort_lookup = cohort[["stay_id_global", "hospital_id", "icu_mortality"]].copy()
    cohort_lookup["stay_id_global"] = cohort_lookup["stay_id_global"].astype("string")
    cohort_lookup["hospital_id"] = cohort_lookup["hospital_id"].astype("string")
    cohort_lookup["icu_mortality"] = pd.to_numeric(cohort_lookup["icu_mortality"], errors="coerce")
    return normalized.merge(cohort_lookup, on=["stay_id_global", "hospital_id"], how="left")


def semantic_violation_counts(labels_with_mortality: pd.DataFrame) -> dict[str, int]:
    labelable = labels_with_mortality["proxy_horizon_labelable"].fillna(False)
    label_value = labels_with_mortality["label_value"]
    positive = labelable & label_value.eq(1)
    negative = labelable & label_value.eq(0)
    unlabeled = ~labelable

    expected_positive = (
        labels_with_mortality["icu_mortality"].eq(1)
        & labels_with_mortality["event_time_proxy_h"].gt(labels_with_mortality["prediction_time_h"])
        & labels_with_mortality["event_time_proxy_h"].le(labels_with_mortality["future_window_end_h"])
    )
    expected_negative = (
        labels_with_mortality["icu_mortality"].eq(0)
        & labels_with_mortality["event_time_proxy_h"].ge(labels_with_mortality["future_window_end_h"])
    )
    expected_labelable = expected_positive | expected_negative

    return {
        "positive_semantic_violations": int((positive & ~expected_positive).sum()),
        "negative_semantic_violations": int((negative & ~expected_negative).sum()),
        "unlabeled_semantic_violations": int((unlabeled & expected_labelable).sum()),
        "eventual_non_survivor_outside_horizon_labeled_negative": int(
            (
                labels_with_mortality["icu_mortality"].eq(1)
                & ~expected_positive
                & label_value.eq(0)
            ).sum()
        ),
        "early_discharged_survivor_labeled_negative": int(
            (
                labels_with_mortality["icu_mortality"].eq(0)
                & labels_with_mortality["event_time_proxy_h"].lt(
                    labels_with_mortality["future_window_end_h"]
                )
                & label_value.eq(0)
            ).sum()
        ),
    }


def build_label_summary(labels_with_mortality: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        subset = labels_with_mortality[
            labels_with_mortality["horizon_h"].astype("Int64").eq(int(horizon))
        ].copy()
        violations = semantic_violation_counts(subset)
        candidate_rows = int(subset.shape[0])
        labeled_rows = int(subset["proxy_horizon_labelable"].fillna(False).sum())
        positive_rows = int(subset["label_value"].fillna(-1).eq(1).sum())
        negative_rows = int(subset["label_value"].fillna(-1).eq(0).sum())
        unlabeled_rows = candidate_rows - labeled_rows
        rows.append(
            {
                "horizon_h": int(horizon),
                "candidate_rows": candidate_rows,
                "labeled_rows": labeled_rows,
                "positive_rows": positive_rows,
                "negative_rows": negative_rows,
                "unlabeled_rows": unlabeled_rows,
                "positive_rate_among_labeled": (
                    positive_rows / labeled_rows if labeled_rows else pd.NA
                ),
                "status": "pass" if candidate_rows and not any(violations.values()) else "fail",
                **violations,
            }
        )
    return pd.DataFrame(rows)


def build_unlabeled_reason_summary(
    labels_with_mortality: pd.DataFrame,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        subset = labels_with_mortality[
            labels_with_mortality["horizon_h"].astype("Int64").eq(int(horizon))
            & ~labels_with_mortality["proxy_horizon_labelable"].fillna(False)
        ].copy()
        if subset.empty:
            rows.append(
                {
                    "horizon_h": int(horizon),
                    "unlabeled_reason": "(none)",
                    "count": 0,
                }
            )
            continue
        reasons = subset["unlabeled_reason"].fillna("(missing_unlabeled_reason)").astype("string")
        for reason, count in reasons.value_counts(dropna=False).sort_index().items():
            rows.append(
                {
                    "horizon_h": int(horizon),
                    "unlabeled_reason": str(reason),
                    "count": int(count),
                }
            )
    return pd.DataFrame(rows)


def write_horizon_target_tables(
    labels: pd.DataFrame,
    config: MimicHorizonLabelConfig,
) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    extension = config.output_format
    for horizon in config.horizons_hours:
        subset = labels[labels["horizon_h"].astype("Int64").eq(int(horizon))].copy()
        subset["target_status"] = "unlabeled"
        subset.loc[subset["label_value"].fillna(-1).eq(1), "target_status"] = "positive"
        subset.loc[subset["label_value"].fillna(-1).eq(0), "target_status"] = "negative"
        path = config.target_output_root / f"ch1_mimic_proxy_horizon_targets_{int(horizon)}h.{extension}"
        _write_table(subset, path, config.output_format)
        paths[int(horizon)] = path
    return paths


def write_note(
    config: MimicHorizonLabelConfig,
    summary: pd.DataFrame,
    reason_summary: pd.DataFrame,
    target_paths: dict[int, Path],
) -> None:
    status = "pass" if summary["status"].eq("pass").all() else "fail"
    paths_text = "\n".join(
        f"- `{horizon}h`: `{path}`" for horizon, path in sorted(target_paths.items())
    )
    lines = [
        "# Chapter 1 MIMIC Horizon Label Generation Note",
        "",
        "## Purpose",
        "",
        "This report documents subtask 5.1.b4: verification/export of MIMIC horizon target tables using the frozen ASIC conservative proxy-label scheme.",
        "",
        "## Reuse Mode",
        "",
        "The existing ASIC label logic was reused directly through the b3 MIMIC-to-ASIC preprocessing adapter. This b4 step verifies the resulting labels and writes horizon-specific target tables; it does not implement a separate MIMIC-native event-time label.",
        "",
        "## Semantics",
        "",
        "- Positives require ICU mortality with proxy endpoint in `(prediction_time, prediction_time + H]`.",
        "- Negatives require non-ICU-mortality and full horizon observation through `prediction_time + H`.",
        "- Eventual non-survivors outside the current horizon remain unlabeled, not negative.",
        "- Early-discharged survivors without full horizon observation remain unlabeled, not negative.",
        "- Prediction time remains the completed 8h block end.",
        "",
        "MIMIC b4 uses the adapter-exposed `icu_end_time_proxy_hours` derived from retained-stay ICU LOS. Stronger MIMIC death timestamps are not substituted into this primary target because doing so would change the frozen ASIC proxy-label semantics.",
        "",
        "## Horizons",
        "",
        ", ".join(f"{horizon}h" for horizon in config.horizons_hours),
        "",
        "## Storage",
        "",
        f"- Preprocessing output root: `{config.preprocessing_output_root}`",
        f"- Horizon target output root: `{config.target_output_root}`",
        "- Full-MIMIC row-level target tables must remain outside the repo.",
        "",
        "## Target Tables",
        "",
        paths_text,
        "",
        "## Verification Status",
        "",
        f"`{status}`",
        "",
        "Unlabeled reasons were available from the reused ASIC label output and summarized in `reports/ch1_mimic_horizon_unlabeled_reasons.csv`.",
        "",
        "## Deferred Beyond b4",
        "",
        "- model fitting",
        "- standard event-time mortality targets",
        "- any redesign of the negative class",
    ]
    ensure_directory(config.reports_dir)
    (config.reports_dir / "ch1_mimic_horizon_label_note.md").write_text("\n".join(lines))


def run_horizon_label_verification(
    config: MimicHorizonLabelConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, Path]]:
    validate_inputs(config)
    ensure_directory(config.reports_dir)
    ensure_directory(config.target_output_root)
    labels, _, cohort = load_label_inputs(config)
    labels_with_mortality = _normalize_label_frame(labels, cohort)
    available_horizons = set(labels_with_mortality["horizon_h"].dropna().astype(int).unique())
    missing_horizons = sorted(set(config.horizons_hours) - available_horizons)
    if missing_horizons:
        raise ValueError(
            f"Preprocessing label output is missing required horizons: {missing_horizons}"
        )
    selected = labels_with_mortality[
        labels_with_mortality["horizon_h"].astype("Int64").isin(config.horizons_hours)
    ].copy()
    summary = build_label_summary(selected, config.horizons_hours)
    reason_summary = build_unlabeled_reason_summary(selected, config.horizons_hours)
    if not summary["status"].eq("pass").all():
        failed = summary.loc[~summary["status"].eq("pass"), "horizon_h"].tolist()
        raise ValueError(f"MIMIC horizon-label semantic verification failed for horizons: {failed}")

    target_paths = write_horizon_target_tables(selected.drop(columns=["icu_mortality"]), config)
    summary.to_csv(config.reports_dir / "ch1_mimic_horizon_label_summary.csv", index=False)
    reason_summary.to_csv(
        config.reports_dir / "ch1_mimic_horizon_unlabeled_reasons.csv",
        index=False,
    )
    write_note(config, summary, reason_summary, target_paths)
    return summary, reason_summary, target_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mimic-root", type=Path, default=None)
    parser.add_argument("--preprocessing-output-root", type=Path, default=None)
    parser.add_argument("--target-output-root", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=None)
    parser.add_argument("--output-format", choices=["csv", "parquet"], default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    replacements = {}
    for field_name, value in [
        ("mimic_root", args.mimic_root),
        ("preprocessing_output_root", args.preprocessing_output_root),
        ("target_output_root", args.target_output_root),
        ("reports_dir", args.reports_dir),
    ]:
        if value is not None:
            replacements[field_name] = _resolve_path(value)
    if args.horizons is not None:
        replacements["horizons_hours"] = tuple(int(horizon) for horizon in args.horizons)
    if args.output_format is not None:
        replacements["output_format"] = args.output_format
    if replacements:
        config = replace(config, **replacements)

    summary, _, target_paths = run_horizon_label_verification(config)
    print("MIMIC horizon label verification passed.")
    print(summary[["horizon_h", "candidate_rows", "labeled_rows", "positive_rows", "negative_rows", "unlabeled_rows"]].to_string(index=False))
    for horizon, path in sorted(target_paths.items()):
        print(f"{horizon}h target table: {path}")


if __name__ == "__main__":
    main()
