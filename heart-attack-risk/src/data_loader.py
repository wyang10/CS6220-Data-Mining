from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from . import config


def load_raw_csv(path: Optional[Path] = None) -> pd.DataFrame:
    """Load raw CSV and apply minimal cleaning.

    - Coerce CHARGES to numeric ('.' -> NaN)
    - Drop columns listed in config.DROP_COLS if present
    """
    csv_path = Path(path) if path is not None else config.RAW_CSV
    df = pd.read_csv(csv_path)

    # Basic type fixes
    if "CHARGES" in df.columns:
        df["CHARGES"] = pd.to_numeric(df["CHARGES"], errors="coerce")

    # Drop non-feature columns if present
    for col in config.DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=col)

    return df


def save_processed_csv(df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    config.ensure_dirs()
    out_path = Path(path) if path is not None else config.PROCESSED_CSV
    df.to_csv(out_path, index=False)
    return out_path


def load_processed_csv(path: Optional[Path] = None) -> pd.DataFrame:
    in_path = Path(path) if path is not None else config.PROCESSED_CSV
    return pd.read_csv(in_path)

