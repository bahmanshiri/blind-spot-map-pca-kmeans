# -*- coding: utf-8 -*-
"""
04_algorithm_comparison.py

Blind Spot Map extension, Section 5.5.3: Algorithm comparison.

Compare K-means(2), GMM(2), and Isolation Forest at (a) the natural fraud
ratio measured from the data (~3.499%) and (b) an artificially balanced
ratio of 20%, via subsampling only.

Isolation Forest is unsupervised and scores each point as inlier/outlier;
contamination is set to the true minority ratio of the subsample (this is
optimistic for IF -- it is told the true ratio -- which is disclosed here as
a methodological caveat, not hidden).
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
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score

from data_utils import load_fraud_data, subsample

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV = BASE + "/results/algorithm_comparison_results.csv"
OUT_FIG = BASE + "/figures/fig3_algorithm_comparison.png"

SEEDS = [42, 43, 44, 45, 46]
N_TOTAL = 4000

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)


def run():
    df, feat_cols, fraud_idx, nonfraud_idx = load_fraud_data()
    natural_ratio = len(fraud_idx) / (len(fraud_idx) + len(nonfraud_idx))
    print(f"Measured natural fraud ratio in full dataset: {natural_ratio:.5f}")

    ratios_to_test = {"natural (~3.5%)": natural_ratio, "balanced (20%)": 0.20}
    rows = []

    for ratio_label, ratio in ratios_to_test.items():
        km_aris, gmm_aris, if_aris = [], [], []
        for seed in SEEDS:
            X, y = subsample(df, feat_cols, fraud_idx, nonfraud_idx, N_TOTAL, ratio, seed)
            if X is None:
                print(f"  {ratio_label}: SKIPPED seed={seed} (insufficient samples)")
                continue
            Xs = StandardScaler().fit_transform(X)
            Xp = PCA(n_components=2, random_state=seed).fit_transform(Xs)

            km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(Xp)
            km_aris.append(adjusted_rand_score(y, km.labels_))

            gmm = GaussianMixture(n_components=2, random_state=seed, n_init=3).fit(Xp)
            gmm_labels = gmm.predict(Xp)
            gmm_aris.append(adjusted_rand_score(y, gmm_labels))

            contamination = min(max(y.mean(), 1e-4), 0.5)
            iso = IsolationForest(contamination=contamination, random_state=seed, n_estimators=200)
            iso_pred = iso.fit_predict(Xp)  # -1 = outlier (predicted fraud), 1 = inlier
            iso_labels = (iso_pred == -1).astype(int)
            if_aris.append(adjusted_rand_score(y, iso_labels))

        row = {
            "ratio_label": ratio_label,
            "ratio_value": ratio,
            "n_seeds": len(km_aris),
            "kmeans_ari_mean": float(np.mean(km_aris)),
            "kmeans_ari_std": float(np.std(km_aris)),
            "gmm_ari_mean": float(np.mean(gmm_aris)),
            "gmm_ari_std": float(np.std(gmm_aris)),
            "isoforest_ari_mean": float(np.mean(if_aris)),
            "isoforest_ari_std": float(np.std(if_aris)),
        }
        rows.append(row)
        print(f"{ratio_label}: KMeans={row['kmeans_ari_mean']:.3f}+/-{row['kmeans_ari_std']:.3f}  "
              f"GMM={row['gmm_ari_mean']:.3f}+/-{row['gmm_ari_std']:.3f}  "
              f"IsoForest={row['isoforest_ari_mean']:.3f}+/-{row['isoforest_ari_std']:.3f}")

    result = pd.DataFrame(rows)
    result.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    labels = result["ratio_label"].tolist()
    x = np.arange(len(labels))
    width = 0.25
    algos = [("kmeans", "K-means(2)", "#4393c3"),
             ("gmm", "GMM(2)", "#d6604d"),
             ("isoforest", "Isolation Forest", "#5aae61")]
    for i, (key, name, color) in enumerate(algos):
        means = result[f"{key}_ari_mean"]
        stds = result[f"{key}_ari_std"]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=4,
               label=name, color=color, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Adjusted Rand Index", fontsize=12)
    ax.set_title("Algorithm comparison: natural vs. balanced fraud ratio\n(real IEEE-CIS data)", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=1200, bbox_inches="tight")
    fig.savefig(OUT_FIG.replace(".png", ".tiff"), dpi=1200, bbox_inches="tight", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"Saved: {OUT_FIG}")

    return result


if __name__ == "__main__":
    run()
