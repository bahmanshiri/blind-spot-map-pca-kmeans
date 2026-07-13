# -*- coding: utf-8 -*-
"""
05_noise_robustness.py

Blind Spot Map extension, Section 5.5.4: Noise robustness.

Fix minority (fraud) ratio at 5%. Add Gaussian noise scaled to a fraction of
each feature's standard deviation (added AFTER scaling, so noise_level is
directly in standardized units) and measure how K-means(2) and GMM(2)
degrade.
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
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import load_fraud_data, subsample

# repo/code/section5.5_ieee_cis/script.py -> repo root is 2 levels up
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = BASE + "/results/section5.5_ieee_cis"
FIG_DIR = BASE + "/figures/section5.5_ieee_cis"
OUT_CSV = OUT_DIR + "/noise_robustness_results.csv"
OUT_FIG = FIG_DIR + "/fig4_noise_robustness.png"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

FIXED_RATIO = 0.05
NOISE_LEVELS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
SEEDS = [42, 43, 44, 45, 46]
N_TOTAL = 4000

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)


def run():
    df, feat_cols, fraud_idx, nonfraud_idx = load_fraud_data()
    rows = []

    for noise in NOISE_LEVELS:
        km_aris, gmm_aris = [], []
        for seed in SEEDS:
            X, y = subsample(df, feat_cols, fraud_idx, nonfraud_idx, N_TOTAL, FIXED_RATIO, seed)
            if X is None:
                continue
            rng = np.random.default_rng(seed)
            Xs = StandardScaler().fit_transform(X)
            if noise > 0:
                Xs = Xs + rng.normal(loc=0.0, scale=noise, size=Xs.shape)
            Xp = PCA(n_components=2, random_state=seed).fit_transform(Xs)

            km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(Xp)
            km_aris.append(adjusted_rand_score(y, km.labels_))

            gmm = GaussianMixture(n_components=2, random_state=seed, n_init=3).fit(Xp)
            gmm_aris.append(adjusted_rand_score(y, gmm.predict(Xp)))

        row = {
            "noise_level": noise,
            "minority_ratio": FIXED_RATIO,
            "n_seeds": len(km_aris),
            "kmeans_ari_mean": float(np.mean(km_aris)),
            "kmeans_ari_std": float(np.std(km_aris)),
            "gmm_ari_mean": float(np.mean(gmm_aris)),
            "gmm_ari_std": float(np.std(gmm_aris)),
        }
        rows.append(row)
        print(f"noise={noise:.2f}  KMeans={row['kmeans_ari_mean']:.3f}+/-{row['kmeans_ari_std']:.3f}  "
              f"GMM={row['gmm_ari_mean']:.3f}+/-{row['gmm_ari_std']:.3f}")

    result = pd.DataFrame(rows)
    result.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")

    fig, ax = plt.subplots(figsize=(7, 5))
    x = result["noise_level"] * 100
    ax.plot(x, result["kmeans_ari_mean"], marker="o", markersize=6, linewidth=2,
            color="#4393c3", label="K-means(2)")
    ax.fill_between(x, result["kmeans_ari_mean"] - result["kmeans_ari_std"],
                     result["kmeans_ari_mean"] + result["kmeans_ari_std"],
                     alpha=0.2, color="#4393c3")
    ax.plot(x, result["gmm_ari_mean"], marker="s", markersize=6, linewidth=2,
            color="#d6604d", label="GMM(2)")
    ax.fill_between(x, result["gmm_ari_mean"] - result["gmm_ari_std"],
                     result["gmm_ari_mean"] + result["gmm_ari_std"],
                     alpha=0.2, color="#d6604d")
    ax.set_xlabel("Gaussian noise level (% of standardized std)", fontsize=12)
    ax.set_ylabel("Adjusted Rand Index", fontsize=12)
    ax.set_title(f"Noise robustness at fixed {FIXED_RATIO*100:.0f}% fraud ratio\n(real IEEE-CIS data)", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=1200, bbox_inches="tight")
    fig.savefig(OUT_FIG.replace(".png", ".tiff"), dpi=1200, bbox_inches="tight", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"Saved: {OUT_FIG}")

    return result


if __name__ == "__main__":
    run()
