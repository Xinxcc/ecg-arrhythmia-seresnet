"""Two-stage transfer-learning training.

Stage 1: pre-train an SE-ResNet backbone on a binary task (Normal vs AF).
Stage 2: swap the head and fine-tune on all 4 classes with class weights.

Usage:
    python -m src.train --data_dir <dir> --epochs 40 --batch_size 64
"""
from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from . import dataset as D
from .seresnet import build_se_resnet, replace_head


def _callbacks(ckpt_path: str):
    return [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss",
                                           save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                             patience=4, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                         restore_best_weights=True),
    ]


def feature_shape(data_dir: str):
    df = D.load_reference(data_dir)
    sig = D.load_signal(data_dir, df.record.iloc[0])
    from .preprocessing import make_features
    return make_features(sig).shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tf.random.set_seed(args.seed)
    Path(args.runs).mkdir(parents=True, exist_ok=True)
    input_shape = feature_shape(args.data_dir)
    print("Feature shape:", input_shape)

    # ---- Stage 1: binary pre-training (Normal vs AF) --------------------- #
    print("\n=== Stage 1: binary pre-training (N vs A) ===")
    bin_train, bin_val = D.binary_datasets(args.data_dir, batch_size=args.batch_size,
                                           seed=args.seed)
    backbone = build_se_resnet(input_shape, n_classes=2)
    backbone.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                     loss="categorical_crossentropy", metrics=["accuracy"])
    backbone.fit(bin_train, validation_data=bin_val, epochs=args.epochs,
                 callbacks=_callbacks(f"{args.runs}/best_binary.keras"))

    # ---- Stage 2: 4-class fine-tuning ----------------------------------- #
    print("\n=== Stage 2: 4-class fine-tuning ===")
    train, val, tr_df, _ = D.four_class_datasets(args.data_dir,
                                                 batch_size=args.batch_size,
                                                 seed=args.seed)
    model = replace_head(backbone, n_classes=len(D.LABELS))
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr / 5),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train, validation_data=val, epochs=args.epochs,
              class_weight=D.class_weights(tr_df),
              callbacks=_callbacks(f"{args.runs}/best_4class.keras"))

    model.save(f"{args.runs}/final_4class.keras")
    print(f"\nSaved model to {args.runs}/best_4class.keras")


if __name__ == "__main__":
    main()
