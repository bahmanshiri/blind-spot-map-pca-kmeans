# -*- coding: utf-8 -*-
"""
data_utils.py

Shared loading utility for the IEEE-CIS fraud experiments. All experiment
scripts subsample from the same cleaned feature table so that results are
comparable across scripts.
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # fraud_project/
CLEAN_PATH = BASE + "/data/fraud_clean.csv"

_cache = {}


def load_fraud_data():
    """Load the cleaned fraud feature table once per process (cached)."""
    if "df" not in _cache:
        df = pd.read_csv(CLEAN_PATH)
        # isFraud stored as float64 on read-back; cast back to int
        df["isFraud"] = df["isFraud"].astype(int)
        feat_cols = [c for c in df.columns if c != "isFraud"]
        _cache["df"] = df
        _cache["feat_cols"] = feat_cols
        _cache["fraud_idx"] = df.index[df["isFraud"] == 1].to_numpy()
        _cache["nonfraud_idx"] = df.index[df["isFraud"] == 0].to_numpy()
    return _cache["df"], _cache["feat_cols"], _cache["fraud_idx"], _cache["nonfraud_idx"]


def subsample(df, feat_cols, fraud_idx, nonfraud_idx, n_total, ratio, seed):
    """Draw a subsample with a target fraud ratio via subsampling only (no
    duplication, no SMOTE). Returns (X, y) or None if not enough real fraud
    examples exist for the requested ratio at this n_total."""
    rng = np.random.default_rng(seed)
    n_min = int(round(ratio * n_total))
    n_maj = n_total - n_min
    if n_min < 2 or n_min > len(fraud_idx) or n_maj > len(nonfraud_idx):
        return None, None
    sel_min = rng.choice(fraud_idx, size=n_min, replace=False)
    sel_maj = rng.choice(nonfraud_idx, size=n_maj, replace=False)
    sel = np.concatenate([sel_min, sel_maj])
    X = df.loc[sel, feat_cols].to_numpy(dtype=np.float64)
    y = df.loc[sel, "isFraud"].to_numpy()
    return X, y
