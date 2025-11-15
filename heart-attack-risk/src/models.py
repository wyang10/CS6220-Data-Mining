from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from .features import build_preprocessor


SUPPORTED_MODELS = {
    "nb": GaussianNB,
    "knn": KNeighborsClassifier,
    "decision_tree": DecisionTreeClassifier,
    "logreg": LogisticRegression,
    "svm": SVC,
    "mlp": MLPClassifier,
}


def make_pipeline(model_name: str, smote: bool = False) -> ImbPipeline:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(SUPPORTED_MODELS)}")

    clf_cls = SUPPORTED_MODELS[model_name]

    # Reasonable defaults
    if model_name == "logreg":
        clf = clf_cls(max_iter=1000)
    elif model_name == "svm":
        clf = clf_cls(kernel="rbf", probability=True)
    elif model_name == "knn":
        clf = clf_cls(n_neighbors=5)
    elif model_name == "mlp":
        clf = clf_cls(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    elif model_name == "decision_tree":
        clf = clf_cls(random_state=42)
    else:
        clf = clf_cls()

    pre = build_preprocessor()

    steps = [("preprocess", pre)]
    if smote:
        steps.append(("smote", SMOTE(random_state=42)))
    steps.append(("clf", clf))

    pipe = ImbPipeline(steps=steps)
    return pipe


def train_and_evaluate(
    X,
    y,
    model_name: str,
    smote: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[ImbPipeline, Dict[str, float]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe = make_pipeline(model_name, smote=smote)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    report = metrics.classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    acc = metrics.accuracy_score(y_test, y_pred)
    cm = metrics.confusion_matrix(y_test, y_pred).tolist()

    results = {
        "accuracy": float(acc),
        "precision_0": float(report["0"]["precision"]) if "0" in report else 0.0,
        "recall_0": float(report["0"]["recall"]) if "0" in report else 0.0,
        "precision_1": float(report["1"]["precision"]) if "1" in report else 0.0,
        "recall_1": float(report["1"]["recall"]) if "1" in report else 0.0,
        "f1_1": float(report["1"]["f1-score"]) if "1" in report else 0.0,
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "confusion_matrix": cm,
    }

    return pipe, results
