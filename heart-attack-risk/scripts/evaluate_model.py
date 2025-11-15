from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
from sklearn.model_selection import train_test_split

# Ensure project root (…/heart-attack-risk) is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data_loader import load_raw_csv
from src.features import get_xy
from src.evaluate import evaluate_predictions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved model")
    p.add_argument("--model-path", type=str, default=None, help="Path to saved joblib model")
    p.add_argument("--raw-csv", type=str, default=None, help="Path to raw CSV (optional)")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--metrics-out", type=str, default=None, help="Path to save metrics (optional)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path) if args.model_path else config.MODEL_PATH
    metrics_out = Path(args.metrics_out) if args.metrics_out else config.METRICS_PATH

    model = joblib.load(model_path)

    # Load fresh data, split a test set, and evaluate
    df = load_raw_csv(Path(args.raw_csv) if args.raw_csv else None)
    X, y = get_xy(df)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    y_pred = model.predict(X_test)
    metrics_dict = evaluate_predictions(y_test, y_pred)
    metrics_out.write_text(json.dumps(metrics_dict, indent=2))

    print(f"Saved metrics: {metrics_out}")
    print(json.dumps(metrics_dict, indent=2))


if __name__ == "__main__":
    main()
