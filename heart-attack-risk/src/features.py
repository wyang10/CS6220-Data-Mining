from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor() -> ColumnTransformer:
    cat_pipe = PipelineForCategorical()
    num_pipe = PipelineForNumeric()

    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, config.CATEGORICAL_COLS),
            ("num", num_pipe, config.NUMERIC_COLS),
        ]
    )
    return pre


def PipelineForCategorical():
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )


def PipelineForNumeric():
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )


def get_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    X = df[config.FEATURE_COLS].copy()
    y = df[config.TARGET_COL].values
    return X, y

