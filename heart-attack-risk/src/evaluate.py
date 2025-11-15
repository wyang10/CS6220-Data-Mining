from __future__ import annotations

from typing import Dict

from sklearn import metrics


def evaluate_predictions(y_true, y_pred) -> Dict[str, float]:
    report = metrics.classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    acc = metrics.accuracy_score(y_true, y_pred)
    cm = metrics.confusion_matrix(y_true, y_pred).tolist()

    return {
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

