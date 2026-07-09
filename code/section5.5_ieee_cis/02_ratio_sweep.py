# -*- coding: utf-8 -*-
"""
02_ratio_sweep.py

Blind Spot Map extension, Section 5.5.1: Ratio Sweep on real IEEE-CIS fraud
data.

For each target minority (fraud) ratio, draw N_TOTAL-sized subsamples via
random subsampling only (no SMOTE, no duplication -- there are enough real
fraud cases, 20,663, to support every ratio tested here without synthetic
augmentation). Scale, run PCA(2)+K-means(2), score against the true isFraud
label with ARI and NMI. Repeat over 5 seeds and report mean +/- std.
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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from data_utils import load_fraud_data, subsample

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV = BASE + "/results/ratio_sweep_results.csv"
OUT_FIG = BASE + "/figures/fig1_ratio_sweep.png"

TARGET_RATIOS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
SEEDS = [42, 43, 44, 45, 46]
N_TOTAL = 4000

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)


def run():
    df, feat_cols, fraud_idx, nonfraud_idx = load_fraud_data()
    rows = []

    for ratio in TARGET_RATIOS:
        aris, nmis = [], []
        for seed in SEEDS:
            X, y = subsample(df, feat_cols, fraud_idx, nonfraud_idx, N_TOTAL, ratio, seed)
            if X is None:
                continue
            Xs = StandardScaler().fit_transform(X)
            Xp = PCA(n_components=2, random_state=seed).fit_transform(Xs)
            km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(Xp)
            aris.append(adjusted_rand_score(y, km.labels_))
            nmis.append(normalized_mutual_info_score(y, km.labels_))

        if not aris:
            print(f"ratio={ratio:.2f} -> SKIPPED (insufficient samples)")
            continue

        row = {
            "minority_ratio": ratio,
            "n_total": N_TOTAL,
            "n_seeds": len(aris),
            "ari_mean": float(np.mean(aris)),
            "ari_std": float(np.std(aris)),
            "nmi_mean": float(np.mean(nmis)),
            "nmi_std": float(np.std(nmis)),
        }
        rows.append(row)
        print(f"ratio={ratio:.2f}  ARI={row['ari_mean']:.3f}+/-{row['ari_std']:.3f}  "
              f"NMI={row['nmi_mean']:.3f}+/-{row['nmi_std']:.3f}")

    result = pd.DataFrame(rows)
    result.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")

    fig, ax = plt.subplots(figsize=(7, 5))
    x = result["minority_ratio"] * 100
    y = result["ari_mean"]
    std = result["ari_std"]
    ax.plot(x, y, marker="o", markersize=6, linewidth=2, color="#2166ac", label="ARI (mean of 5 seeds)")
    ax.fill_between(x, y - std, y + std, alpha=0.25, color="#2166ac", label="+/- 1 std")
    ax.set_xlabel("Minority (fraud) ratio (%)", fontsize=12)
    ax.set_ylabel("Adjusted Rand Index", fontsize=12)
    ax.set_title("PCA(2)+K-means(2) recovery of fraud label\nvs. minority ratio (real IEEE-CIS data)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=1200, bbox_inches="tight")
    fig.savefig(OUT_FIG.replace(".png", ".tiff"), dpi=1200, bbox_inches="tight", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"Saved: {OUT_FIG}")

    return result


if __name__ == "__main__":
    run()
