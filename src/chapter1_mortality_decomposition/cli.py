from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from chapter1_mortality_decomposition.artifacts import DEFAULT_CHAPTER1_OUTPUT_DIR
from chapter1_mortality_decomposition.config import default_chapter1_config, updated_chapter1_config
from chapter1_mortality_decomposition.pipeline import build_and_write_chapter1_dataset
from chapter1_mortality_decomposition.run_config import load_chapter1_run_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run standalone Chapter 1 mortality preprocessing from standardized ASIC artifacts."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing standardized ASIC artifacts.",
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        help="JSON run config with Chapter 1 input/output paths and runtime settings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where Chapter 1 outputs will be written.",
    )
    parser.add_argument(
        "--input-format",
        choices=("csv", "parquet"),
        help="Input artifact format.",
    )
    parser.add_argument(
        "--output-format",
        choices=("csv", "parquet"),
        help="Output artifact format.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        help="Override the default Chapter 1 prediction horizons in hours.",
    )
    parser.add_argument(
        "--min-required-core-groups",
        type=int,
        help="Minimum required core physiologic groups for site inclusion.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    run_config = load_chapter1_run_config(args.run_config) if args.run_config else None
    input_dir = args.input_dir or (run_config.input_dir if run_config else None)
    if input_dir is None:
        parser.error("Either --input-dir or --run-config must be provided.")

    output_dir = args.output_dir or (
        run_config.output_dir if run_config else DEFAULT_CHAPTER1_OUTPUT_DIR
    )
    input_format = args.input_format or (run_config.input_format if run_config else "csv")
    output_format = args.output_format or (run_config.output_format if run_config else "csv")
    horizons = args.horizons if args.horizons is not None else (
        run_config.horizons_hours if run_config else None
    )
    min_required_core_groups = (
        args.min_required_core_groups
        if args.min_required_core_groups is not None
        else (run_config.min_required_core_groups if run_config else 3)
    )
    feature_set_config_path = run_config.feature_set_config_path if run_config else None

    config = updated_chapter1_config(
        default_chapter1_config(),
        horizons_hours=horizons,
        min_required_core_groups=min_required_core_groups,
        feature_set_config_path=feature_set_config_path,
    )
    dataset, output_paths = build_and_write_chapter1_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        input_format=input_format,
        output_format=output_format,
        config=config,
    )

    print(f"Wrote {len(output_paths)} Chapter 1 tables to {output_dir}")
    print(f"Retained stays: {dataset.cohort.table.shape[0]}")
    for feature_set_name, feature_set_dataset in dataset.feature_sets.items():
        print(
            f"{feature_set_name} valid instances: "
            f"{feature_set_dataset.valid_instances.valid_instances.shape[0]}"
        )
        print(
            f"{feature_set_name} usable labels: "
            f"{feature_set_dataset.labels.usable_labels.shape[0]}"
        )
        print(
            f"{feature_set_name} model-ready rows: "
            f"{feature_set_dataset.model_ready.table.shape[0]}"
        )
    return 0
