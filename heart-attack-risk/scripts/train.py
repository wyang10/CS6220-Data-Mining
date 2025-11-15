from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib

# Ensure project root (…/heart-attack-risk) is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data_loader import load_raw_csv, save_processed_csv
from src.features import get_xy
from src.models import train_and_evaluate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train model on heart attack dataset")
    p.add_argument("--model", default="decision_tree", choices=[
        "nb", "knn", "decision_tree", "logreg", "svm", "mlp"
    ], help="Model type")
    p.add_argument("--smote", action="store_true", help="Use SMOTE for class imbalance")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--raw-csv", type=str, default=None, help="Path to raw CSV (optional)")
    p.add_argument("--model-out", type=str, default=None, help="Path to save model (optional)")
    p.add_argument("--metrics-out", type=str, default=None, help="Path to save metrics (optional)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_dirs()

    # Load and minimally clean raw
    df = load_raw_csv(Path(args.raw_csv) if args.raw_csv else None)

    # Persist a cleaned processed CSV for reproducibility
    save_processed_csv(df)

    # Features/target
    X, y = get_xy(df)

    # Train & evaluate
    model, metrics_dict = train_and_evaluate(
        X, y,
        model_name=args.model,
        smote=args.smote,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # Outputs
    model_out = Path(args.model_out) if args.model_out else config.MODEL_PATH
    metrics_out = Path(args.metrics_out) if args.metrics_out else config.METRICS_PATH

    joblib.dump(model, model_out)
    metrics_out.write_text(json.dumps(metrics_dict, indent=2))

    print(f"Saved model: {model_out}")
    print(f"Saved metrics: {metrics_out}")
    print(json.dumps(metrics_dict, indent=2))


if __name__ == "__main__":
    main()
