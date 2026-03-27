from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported artifact extension for {path}")


def write_dataframe(
    df: pd.DataFrame,
    path: Path,
    *,
    output_format: str,
) -> Path:
    ensure_directory(path.parent)
    if output_format == "csv":
        df.to_csv(path, index=False)
        return path
    if output_format == "parquet":
        df.to_parquet(path, index=False)
        return path
    raise ValueError(f"Unsupported output format: {output_format}")


def require_columns(df: pd.DataFrame, required_columns: set[str], table_name: str) -> None:
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise KeyError(f"{table_name} is missing required Chapter 1 columns: {missing}")


def normalize_boolean_codes(series: pd.Series) -> pd.Series:
    string_values = series.astype("string").str.strip().str.lower()
    mapped = string_values.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        }
    )

    numeric_values = pd.to_numeric(series, errors="coerce")
    mapped = mapped.where(mapped.notna(), numeric_values.map({1.0: True, 0.0: False}))
    return pd.Series(pd.array(mapped, dtype="boolean"), index=series.index)


def normalize_binary_codes(series: pd.Series) -> tuple[pd.Series, list[str]]:
    numeric = pd.to_numeric(series, errors="coerce")
    unique_codes = (
        pd.Series(numeric.dropna().unique())
        .sort_values()
        .map(lambda value: str(int(value)) if float(value).is_integer() else str(float(value)))
        .tolist()
    )
    return numeric, unique_codes
