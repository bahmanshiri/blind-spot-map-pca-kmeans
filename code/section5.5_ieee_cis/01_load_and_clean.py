# -*- coding: utf-8 -*-
"""
01_load_and_clean.py

Loads the raw IEEE-CIS Fraud Detection transaction table (train_transaction.csv,
590,540 rows, 394 columns) and produces a clean, numeric-only feature matrix
for the Blind Spot Map extension experiments.

Honest notes on this dataset (read before trusting downstream numbers):
- This is NOT the popular 284,807-row "creditcard.csv" (V1-V28, 0.172% fraud)
  dataset. It is the IEEE-CIS Kaggle competition dataset: 590,540 transactions,
  394 raw columns, natural fraud rate ~3.499% (measured directly below, not
  assumed).
- Categorical columns (ProductCD, card4, card6, email domains, M1-M9) are
  dropped for this PCA+K-means pipeline, since one-hot encoding them would
  distort Euclidean distance in ways unrelated to the geometric question being
  studied. Only the 376 float64 + int-valued numeric columns are used.
- Columns with more than MAX_MISSING_FRAC missing values are dropped rather
  than heavily imputed, since imputing a mostly-missing column injects
  artificial structure that could bias the clustering geometry.
- Remaining missing values are median-imputed (fit on the full column, before
  any train/subsample split -- there is no target leakage here because the
  imputation does not use the isFraud label).

Note: the raw file is NOT included in this repository (Kaggle competition
terms of use do not permit redistribution). See data/raw/README.md for how
to obtain it.
"""
import os
import numpy as np
import pandas as pd

# repo/code/section5.5_ieee_cis/script.py -> repo root is 2 levels up
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_PATH = BASE + "/data/raw/train_transaction.csv"
OUT_PATH = BASE + "/data/fraud_clean.csv"
LOG_PATH = BASE + "/data/01_load_and_clean_log.txt"

MAX_MISSING_FRAC = 0.70  # drop columns missing more than this fraction of values

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(str(msg))


def main():
    import gc

    log("Reading header to determine dtypes (memory-efficient load)...")
    header = pd.read_csv(RAW_PATH, nrows=0)
    all_cols = header.columns.tolist()

    # Peek at a sample to classify column types without loading the full file at float64
    sample = pd.read_csv(RAW_PATH, nrows=2000)
    drop_always = {"TransactionID", "TransactionDT"}
    numeric_cols_all = [
        c for c in sample.select_dtypes(include=[np.number]).columns
        if c not in drop_always and c != "isFraud"
    ]
    non_numeric_or_dropped = [c for c in all_cols if c not in numeric_cols_all and c != "isFraud"]
    log(f"Numeric candidate columns: {len(numeric_cols_all)}")
    log(f"Non-numeric / dropped-at-source columns: {len(non_numeric_or_dropped)}")
    del sample

    usecols = ["isFraud"] + numeric_cols_all
    dtype_map = {c: np.float32 for c in numeric_cols_all}
    dtype_map["isFraud"] = np.int8

    log("Loading only the needed numeric columns as float32 (this may take a moment)...")
    df = pd.read_csv(RAW_PATH, usecols=usecols, dtype=dtype_map)
    log(f"Loaded shape: {df.shape}")

    n_total = len(df)
    n_fraud = int(df["isFraud"].sum())
    fraud_rate = n_fraud / n_total
    log(f"Total transactions: {n_total}")
    log(f"Fraud transactions: {n_fraud}")
    log(f"Natural fraud rate (measured): {fraud_rate:.6f} ({fraud_rate*100:.4f}%)")

    missing_frac = df[numeric_cols_all].isna().mean()
    keep_cols = missing_frac[missing_frac <= MAX_MISSING_FRAC].index.tolist()
    dropped_cols = sorted(set(numeric_cols_all) - set(keep_cols))
    log(f"Columns dropped for >{MAX_MISSING_FRAC:.0%} missingness: {len(dropped_cols)}")
    log(f"Columns retained: {len(keep_cols)}")

    labels = df["isFraud"].to_numpy(copy=True)
    df = df[keep_cols]
    gc.collect()

    # Median imputation column-by-column in place (label-independent, no leakage)
    for c in keep_cols:
        med = df[c].median()
        df[c] = df[c].fillna(med).astype(np.float32)

    remaining_na = int(df.isna().sum().sum())
    log(f"Remaining NaNs after imputation (should be 0): {remaining_na}")

    df.insert(0, "isFraud", labels)

    log("Writing cleaned CSV to disk (chunked)...")
    df.to_csv(OUT_PATH, index=False, chunksize=50000)
    log(f"Saved cleaned feature table to: {OUT_PATH}")
    log(f"Final shape: {df.shape} (1 label column + {df.shape[1]-1} numeric features)")

    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines))
    log(f"Log written to: {LOG_PATH}")


if __name__ == "__main__":
    main()
