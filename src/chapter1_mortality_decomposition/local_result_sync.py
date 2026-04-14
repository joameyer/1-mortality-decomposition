from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from chapter1_mortality_decomposition.utils import ensure_directory


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGED_EXPORT_ROOT = PROJECT_ROOT / "export-staging" / "chapter1_true_results"
DEFAULT_LOCAL_RESULT_ROOT = PROJECT_ROOT / "cluster-results" / "chapter1_true_results"


@dataclass(frozen=True)
class ImportStagedExportsResult:
    source_root: Path
    destination_root: Path
    manifest_paths: tuple[Path, ...]
    copied_paths: tuple[Path, ...]


def _resolve_existing_dir(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Required directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Expected a directory, found: {resolved}")
    return resolved


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    copied_relative_paths = payload.get("copied_relative_paths")
    if not isinstance(copied_relative_paths, list) or not all(
        isinstance(item, str) and item for item in copied_relative_paths
    ):
        raise ValueError(
            f"Export manifest is missing a valid copied_relative_paths list: {path}"
        )
    return payload


def import_staged_chapter1_exports(
    *,
    source_root: Path = DEFAULT_STAGED_EXPORT_ROOT,
    destination_root: Path = DEFAULT_LOCAL_RESULT_ROOT,
    overwrite: bool = False,
) -> ImportStagedExportsResult:
    resolved_source_root = _resolve_existing_dir(source_root)
    resolved_destination_root = Path(destination_root).expanduser().resolve()

    manifest_paths = tuple(sorted(resolved_source_root.rglob("export_manifest.json")))
    if not manifest_paths:
        raise FileNotFoundError(
            "No export_manifest.json files were found under the staged export root: "
            f"{resolved_source_root}"
        )

    copied_paths: list[Path] = []
    for manifest_path in manifest_paths:
        manifest_payload = _load_manifest(manifest_path)
        staged_dir = manifest_path.parent
        relative_bundle_root = staged_dir.relative_to(resolved_source_root)

        relative_paths = [Path(path) for path in manifest_payload["copied_relative_paths"]]
        relative_paths.append(Path("export_manifest.json"))

        for relative_path in relative_paths:
            source_path = staged_dir / relative_path
            if not source_path.exists():
                raise FileNotFoundError(
                    f"Manifest-listed staged export file is missing: {source_path}"
                )

            destination_path = resolved_destination_root / relative_bundle_root / relative_path
            ensure_directory(destination_path.parent)
            if destination_path.exists() and not overwrite:
                raise FileExistsError(
                    "Refusing to overwrite imported local result without --overwrite: "
                    f"{destination_path}"
                )
            shutil.copy2(source_path, destination_path)
            copied_paths.append(destination_path)

    return ImportStagedExportsResult(
        source_root=resolved_source_root,
        destination_root=resolved_destination_root,
        manifest_paths=manifest_paths,
        copied_paths=tuple(copied_paths),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import manifest-listed Chapter 1 staged exports into "
            "cluster-results/chapter1_true_results."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_STAGED_EXPORT_ROOT,
        help="Local directory containing copied staged exports under chapter1_true_results/.",
    )
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=DEFAULT_LOCAL_RESULT_ROOT,
        help="Destination local mirror root, usually cluster-results/chapter1_true_results/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting already mirrored local result files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = import_staged_chapter1_exports(
        source_root=args.source_root,
        destination_root=args.destination_root,
        overwrite=args.overwrite,
    )
    print(f"Source root: {result.source_root}")
    print(f"Destination root: {result.destination_root}")
    print(f"Imported bundles: {len(result.manifest_paths)}")
    print(f"Copied files: {len(result.copied_paths)}")
    for manifest_path in result.manifest_paths:
        print(f"- {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
