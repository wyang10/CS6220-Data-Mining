from __future__ import annotations

from pathlib import Path


# Project root = .../heart-attack-risk
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Default file names
RAW_CSV = RAW_DIR / "whole_table.csv"
PROCESSED_CSV = PROCESSED_DIR / "heart_attack_clean.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

# Model artifacts / metrics
MODEL_PATH = PROCESSED_DIR / "trained_model.joblib"
METRICS_PATH = PROCESSED_DIR / "metrics.json"


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# Column configuration
TARGET_COL = "DIED"

DROP_COLS = ["Patient"]

CATEGORICAL_COLS = [
    "SEX",        # 'M'/'F'
    "DIAGNOSIS",  # code, treat as categorical
    "DRG",        # code, treat as categorical
]

NUMERIC_COLS = [
    "AGE",
    "LOS",
    "CHARGES",  # has '.' in raw; converted to numeric
]

FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

