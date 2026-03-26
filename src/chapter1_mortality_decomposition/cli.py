from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from chapter1_mortality_decomposition.artifacts import DEFAULT_CHAPTER1_OUTPUT_DIR
from chapter1_mortality_decomposition.config import default_chapter1_config, updated_chapter1_config
from chapter1_mortality_decomposition.pipeline import build_and_write_chapter1_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run standalone Chapter 1 mortality preprocessing from standardized ASIC artifacts."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing standardized ASIC artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CHAPTER1_OUTPUT_DIR,
        help="Directory where Chapter 1 outputs will be written.",
    )
    parser.add_argument(
        "--input-format",
        choices=("csv", "parquet"),
        default="csv",
        help="Input artifact format.",
    )
    parser.add_argument(
        "--output-format",
        choices=("csv", "parquet"),
        default="csv",
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
        default=3,
        help="Minimum required core physiologic groups for site inclusion.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = updated_chapter1_config(
        default_chapter1_config(),
        horizons_hours=args.horizons,
        min_required_core_groups=args.min_required_core_groups,
    )
    dataset, output_paths = build_and_write_chapter1_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        input_format=args.input_format,
        output_format=args.output_format,
        config=config,
    )

    print(f"Wrote {len(output_paths)} Chapter 1 tables to {args.output_dir}")
    print(f"Retained stays: {dataset.cohort.table.shape[0]}")
    print(f"Valid instances: {dataset.valid_instances.valid_instances.shape[0]}")
    print(f"Usable labels: {dataset.labels.usable_labels.shape[0]}")
    print(f"Model-ready rows: {dataset.model_ready.table.shape[0]}")
    return 0
