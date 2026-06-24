"""Evaluate a trained 4-class model: macro-F1, per-class report, confusion matrix.

Usage:
    python -m src.evaluate --data_dir <dir> --model_path runs/best_4class.keras
"""
from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from . import dataset as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _, val, _, va_df = D.four_class_datasets(args.data_dir,
                                             batch_size=args.batch_size,
                                             seed=args.seed)
    model = tf.keras.models.load_model(args.model_path)

    y_true, y_pred = [], []
    for x, y in val:
        p = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(p, axis=1))
        y_true.extend(np.argmax(y.numpy(), axis=1))

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\nMacro-F1: {macro_f1:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=D.LABELS, digits=3))
    print("Confusion matrix (rows = true, cols = pred):")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
