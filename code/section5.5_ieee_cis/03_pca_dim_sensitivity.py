# -*- coding: utf-8 -*-
"""
03_pca_dim_sensitivity.py

Blind Spot Map extension, Section 5.5.2: PCA dimension sensitivity.

Fix minority (fraud) ratio at 5%. Sweep PCA n_components from 2 to 10 and
measure how well K-means(2) recovers the fraud label as more components are
retained.
"""
import os
import sys
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import load_fraud_data, subsample

# repo/code/section5.5_ieee_cis/script.py -> repo root is 2 levels up
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = BASE + "/results/section5.5_ieee_cis"
FIG_DIR = BASE + "/figures/section5.5_ieee_cis"
OUT_CSV = OUT_DIR + "/pca_dim_sensitivity_results.csv"
OUT_FIG = FIG_DIR + "/fig2_pca_dim_sensitivity.png"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

FIXED_RATIO = 0.05
N_COMPONENTS_RANGE = list(range(2, 11))
SEEDS = [42, 43, 44, 45, 46]
N_TOTAL = 4000

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)


def run():
    df, feat_cols, fraud_idx, nonfraud_idx = load_fraud_data()
    rows = []

    for n_comp in N_COMPONENTS_RANGE:
        aris = []
        for seed in SEEDS:
            X, y = subsample(df, feat_cols, fraud_idx, nonfraud_idx, N_TOTAL, FIXED_RATIO, seed)
            if X is None:
                continue
            Xs = StandardScaler().fit_transform(X)
            Xp = PCA(n_components=n_comp, random_state=seed).fit_transform(Xs)
            km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(Xp)
            aris.append(adjusted_rand_score(y, km.labels_))

        row = {
            "n_components": n_comp,
            "minority_ratio": FIXED_RATIO,
            "n_seeds": len(aris),
            "ari_mean": float(np.mean(aris)),
            "ari_std": float(np.std(aris)),
        }
        rows.append(row)
        print(f"n_components={n_comp}  ARI={row['ari_mean']:.3f}+/-{row['ari_std']:.3f}")

    result = pd.DataFrame(rows)
    result.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(result["n_components"].astype(str), result["ari_mean"],
           yerr=result["ari_std"], capsize=4, color="#4393c3", edgecolor="black")
    ax.set_xlabel("Number of PCA components", fontsize=12)
    ax.set_ylabel("Adjusted Rand Index", fontsize=12)
    ax.set_title(f"K-means(2) recovery vs. PCA dimensionality\n"
                 f"(fraud ratio fixed at {FIXED_RATIO*100:.0f}%, real IEEE-CIS data)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=1200, bbox_inches="tight")
    fig.savefig(OUT_FIG.replace(".png", ".tiff"), dpi=1200, bbox_inches="tight", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"Saved: {OUT_FIG}")

    ari_2 = result.loc[result.n_components == 2, "ari_mean"].values[0]
    ari_10 = result.loc[result.n_components == 10, "ari_mean"].values[0]
    print(f"\nSummary: ARI at 2 components = {ari_2:.3f}, ARI at 10 components = {ari_10:.3f}")

    return result


if __name__ == "__main__":
    run()
