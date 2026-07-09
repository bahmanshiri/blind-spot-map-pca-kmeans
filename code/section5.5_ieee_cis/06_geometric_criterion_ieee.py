# -*- coding: utf-8 -*-
"""
06_geometric_criterion_ieee.py

Blind Spot Map extension, Section 5.5.7: Geometric criterion (Section 4.4)
validation on real IEEE-CIS fraud data.

Uses the EXACT same formulas and operational definitions as the original
Dry Bean validation (05_theory_criterion.py in the primary project):

  BCSS(A) [isolate]        = (n_min * n_maj / N) * D^2
  BCSS(B) [absorb-into-majority-split] ~= (n1 * n2 / n_maj) * Delta^2
  criterion predicts "isolated" iff BCSS(A) > BCSS(B)
  observed "isolated" iff the cluster containing most of the minority
      points is >50% minority (purity > 0.5), from a crosstab of true
      label vs. K-means(2) cluster label.

Test 1 (natural ratio, whole-population): run once on the FULL cleaned
dataset (590,540 transactions, natural fraud ratio 3.499%), exactly
mirroring the "7 classes at natural ratio" table in Section 5.4.4.

Test 2 (ratio sweep, both directions): re-evaluate the criterion at the
same 1%-30% ratio grid used in Section 5.5.2 (5 seeds each), mirroring
the "60-cell grid, 88.3%" validation in Section 5.4.4.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from data_utils import load_fraud_data, subsample

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV_NATURAL = BASE + "/results/geometric_criterion_natural_results.csv"
OUT_CSV_GRID = BASE + "/results/geometric_criterion_grid_results.csv"
OUT_FIG = BASE + "/figures/fig14_geometric_criterion_ieee.png"

SEED = 42
SEEDS = [42, 43, 44, 45, 46]
RATIOS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
N_TOTAL = 4000

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)


def evaluate_criterion(X_min, X_maj, seed):
    """Apply the exact Section-4.4 formulas and operational definitions to
    one (minority, majority) point set already in 2D PCA space."""
    n_min, n_maj = len(X_min), len(X_maj)
    N = n_min + n_maj

    c_min, c_maj = X_min.mean(axis=0), X_maj.mean(axis=0)
    D2 = float(np.sum((c_min - c_maj) ** 2))
    bcss_isolate = (n_min * n_maj / N) * D2

    km_maj = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(X_maj)
    l = km_maj.labels_
    n1, n2 = int((l == 0).sum()), int((l == 1).sum())
    c1, c2 = X_maj[l == 0].mean(axis=0), X_maj[l == 1].mean(axis=0)
    delta2 = float(np.sum((c1 - c2) ** 2))
    bcss_majsplit = (n1 * n2 / n_maj) * delta2 if n_maj > 0 else 0.0

    predicted_isolated = bcss_isolate > bcss_majsplit
    predicted_ratio_threshold = bcss_majsplit / (n_maj * D2) if D2 > 0 else np.inf

    X_pool = np.vstack([X_min, X_maj])
    true_label = np.concatenate([np.ones(n_min, dtype=int), np.zeros(n_maj, dtype=int)])
    km_pool = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(X_pool)
    ari = adjusted_rand_score(true_label, km_pool.labels_)

    ct = pd.crosstab(true_label, km_pool.labels_)
    min_cluster = ct.loc[1].idxmax()
    purity = ct.loc[1, min_cluster] / ct[min_cluster].sum()
    actually_isolated = bool(purity > 0.5)

    return dict(
        n_min=n_min, n_maj=n_maj, minority_ratio=round(n_min / N, 5),
        D2=round(D2, 4), BCSS_isolate=round(bcss_isolate, 2),
        BCSS_majority_split=round(bcss_majsplit, 2),
        predicted_ratio_threshold=round(predicted_ratio_threshold, 5),
        criterion_predicts_isolated=predicted_isolated,
        observed_ARI=round(ari, 4),
        observed_isolated=actually_isolated,
        prediction_correct=(predicted_isolated == actually_isolated),
    )


def run_test1_natural():
    print("=== Test 1: natural ratio, full population (590,540 transactions) ===")
    df, feat_cols, fraud_idx, nonfraud_idx = load_fraud_data()
    Xs = StandardScaler().fit_transform(df[feat_cols].to_numpy(dtype=np.float64))
    Xp = PCA(n_components=2, random_state=SEED).fit_transform(Xs)

    X_min = Xp[fraud_idx]
    X_maj = Xp[nonfraud_idx]
    row = evaluate_criterion(X_min, X_maj, SEED)
    row["dataset"] = "IEEE-CIS (full, natural ratio)"
    result = pd.DataFrame([row])
    result.to_csv(OUT_CSV_NATURAL, index=False)
    print(result.to_string(index=False))
    print(f"Saved: {OUT_CSV_NATURAL}")
    return result


def run_test2_grid():
    print("\n=== Test 2: ratio sweep, 1%-30%, 5 seeds each ===")
    df, feat_cols, fraud_idx, nonfraud_idx = load_fraud_data()
    rows = []
    for ratio in RATIOS:
        for seed in SEEDS:
            X, y = subsample(df, feat_cols, fraud_idx, nonfraud_idx, N_TOTAL, ratio, seed)
            if X is None:
                continue
            Xs = StandardScaler().fit_transform(X)
            Xp = PCA(n_components=2, random_state=seed).fit_transform(Xs)
            X_min = Xp[y == 1]
            X_maj = Xp[y == 0]
            row = evaluate_criterion(X_min, X_maj, seed)
            row["target_ratio"] = ratio
            row["seed"] = seed
            rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(OUT_CSV_GRID, index=False)
    acc = result["prediction_correct"].mean()
    n_correct = int(result["prediction_correct"].sum())
    n_total = len(result)
    print(f"Aggregate prediction accuracy: {acc:.1%} ({n_correct}/{n_total})")
    print(f"Saved: {OUT_CSV_GRID}")
    return result, acc, n_correct, n_total


def make_figure(grid_result):
    agg = grid_result.groupby("target_ratio").agg(
        ari_mean=("observed_ARI", "mean"),
        pred_isolated_rate=("criterion_predicts_isolated", "mean"),
        acc=("prediction_correct", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    x = agg["target_ratio"] * 100
    ax.plot(x, agg["ari_mean"], marker="o", markersize=6, linewidth=2,
            color="#4393c3", label="Observed ARI (mean of 5 seeds)")
    ax.plot(x, agg["pred_isolated_rate"], marker="s", markersize=6, linewidth=2,
            color="#d6604d", label="Criterion: fraction predicting 'isolated'")
    ax.set_xlabel("Fraud ratio (%)", fontsize=11)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title("Predicted isolation vs. observed ARI", fontsize=12)
    ax.legend(fontsize=8.5)

    ax2 = axes[1]
    ax2.bar(agg["target_ratio"].astype(str), agg["acc"], color="#5aae61", edgecolor="black")
    ax2.axhline(0.883, color="gray", linestyle="--", linewidth=1.5,
                label="Dry Bean benchmark (88.3%)")
    ax2.set_xlabel("Fraud ratio", fontsize=11)
    ax2.set_ylabel("Criterion prediction accuracy", fontsize=11)
    ax2.set_title("Geometric criterion accuracy by ratio\n(real IEEE-CIS data)", fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8.5)

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=1200, bbox_inches="tight")
    fig.savefig(OUT_FIG.replace(".png", ".tiff"), dpi=1200, bbox_inches="tight", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"Saved: {OUT_FIG}")


if __name__ == "__main__":
    t1 = run_test1_natural()
    t2, acc, n_correct, n_total = run_test2_grid()
    make_figure(t2)
    print("\n=== SUMMARY ===")
    print(f"Test 1 (natural ratio): criterion predicts isolated = "
          f"{bool(t1.loc[0, 'criterion_predicts_isolated'])}, "
          f"observed isolated = {bool(t1.loc[0, 'observed_isolated'])}, "
          f"match = {bool(t1.loc[0, 'prediction_correct'])}")
    print(f"Test 2 (ratio sweep): overall accuracy = {acc:.1%} ({n_correct}/{n_total})")
