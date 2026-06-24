"""Load the PhysioNet/CinC 2017 records and build tf.data pipelines.

Expected layout of `data_dir`:
    A00001.mat, A00002.mat, ...        # 1-lead ECG, 300 Hz, key 'val'
    REFERENCE.csv                      # "record,label" with label in {N, A, O, ~}
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .preprocessing import make_features

# 4-class task; "~" = noisy
LABELS = ["N", "A", "O", "~"]
LABEL_TO_IDX = {lab: i for i, lab in enumerate(LABELS)}


def load_reference(data_dir: str) -> pd.DataFrame:
    """Read REFERENCE.csv -> DataFrame with columns [record, label, y]."""
    path = Path(data_dir) / "REFERENCE.csv"
    df = pd.read_csv(path, header=None, names=["record", "label"])
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)
    df["y"] = df["label"].map(LABEL_TO_IDX)
    return df


def load_signal(data_dir: str, record: str) -> np.ndarray:
    """Read one .mat record as a 1-D float array."""
    mat = sio.loadmat(os.path.join(data_dir, f"{record}.mat"))
    return np.asarray(mat["val"], dtype=np.float32).squeeze()


def split(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42):
    """Stratified train/validation split."""
    return train_test_split(df, test_size=val_frac, stratify=df["y"],
                            random_state=seed)


def _make_tf_dataset(data_dir, records, ys, n_classes, *, batch_size,
                     augment, shuffle):
    """Build a tf.data pipeline that maps record ids -> (feature, one-hot)."""
    records = list(records)
    ys = np.asarray(ys, dtype=np.int32)

    def gen():
        order = np.random.permutation(len(records)) if shuffle else range(len(records))
        for i in order:
            sig = load_signal(data_dir, records[i])
            feat = make_features(sig, augment=augment)
            yield feat, ys[i]

    # infer feature shape from one sample
    sample = make_features(load_signal(data_dir, records[0]))
    out_sig = (
        tf.TensorSpec(shape=sample.shape, dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=out_sig)
    ds = ds.map(lambda x, y: (x, tf.one_hot(y, n_classes)),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def four_class_datasets(data_dir, *, batch_size=64, val_frac=0.2, seed=42):
    df = load_reference(data_dir)
    tr, va = split(df, val_frac, seed)
    train = _make_tf_dataset(data_dir, tr.record, tr.y, len(LABELS),
                             batch_size=batch_size, augment=True, shuffle=True)
    val = _make_tf_dataset(data_dir, va.record, va.y, len(LABELS),
                           batch_size=batch_size, augment=False, shuffle=False)
    return train, val, tr, va


def binary_datasets(data_dir, *, batch_size=64, val_frac=0.2, seed=42):
    """Normal (0) vs AF (1) subset for the pre-training stage."""
    df = load_reference(data_dir)
    df = df[df["label"].isin(["N", "A"])].copy()
    df["y"] = (df["label"] == "A").astype(int)
    tr, va = split(df, val_frac, seed)
    train = _make_tf_dataset(data_dir, tr.record, tr.y, 2,
                             batch_size=batch_size, augment=True, shuffle=True)
    val = _make_tf_dataset(data_dir, va.record, va.y, 2,
                           batch_size=batch_size, augment=False, shuffle=False)
    return train, val


def class_weights(df: pd.DataFrame) -> dict[int, float]:
    """Inverse-frequency class weights for the 4-class head."""
    counts = df["y"].value_counts().sort_index()
    total = counts.sum()
    return {int(i): float(total / (len(counts) * c)) for i, c in counts.items()}
