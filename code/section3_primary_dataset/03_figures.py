# -*- coding: utf-8 -*-
"""تولید نمودارهای بخش ۵.۳: Figure 3 (خوشه‌بندی نهایی k=3 در فضای PCA) و Figure 4 (منحنی‌های انتخاب k)."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
gen = import_module("01_generate_dataset")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = BASE + "/figures"
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams["figure.dpi"] = 600  # publication-quality raster resolution

SEED = 42
raw = gen.generate_primary_dataset(n_total=120_000, seed=SEED)
clean, _ = gen.clean_dataset(raw)

numeric_cols = ["age", "account_balance", "transaction_to_balance_ratio", "avg_monthly_transactions"]
Xnum = StandardScaler().fit_transform(clean[numeric_cols].values)
Xgender = clean[["gender"]].values.astype(float)
Xbranch = pd.get_dummies(clean["branch"], prefix="branch").values.astype(float)
X = np.hstack([Xnum, Xgender, Xbranch])

pca = PCA(n_components=2, random_state=SEED)
Xp = pca.fit_transform(X)

# ---------- Figure 3: خوشه‌بندی نهایی (k=3) در فضای PCA ----------
km3 = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(Xp)
fig, ax = plt.subplots(figsize=(7, 5.8))
sample_idx = np.random.RandomState(SEED).choice(len(Xp), size=15000, replace=False)
sc = ax.scatter(Xp[sample_idx, 0], Xp[sample_idx, 1], c=km3.labels_[sample_idx],
                 cmap="viridis", s=4, alpha=0.5)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Final K-means Clustering (k=3) in PCA Space")
plt.colorbar(sc, ax=ax, label="Cluster")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig3_final_clustering_pca.png", dpi=600, bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/fig3_final_clustering_pca.tiff", dpi=600, bbox_inches="tight", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.close()

# ---------- Figure 4: Elbow / Silhouette / Davies-Bouldin ----------
sil_sample_idx = np.random.RandomState(SEED).choice(len(Xp), size=20000, replace=False)
ks = [2, 3, 4, 5, 6]
inertias, sils, dbs = [], [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xp)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(Xp[sil_sample_idx], km.labels_[sil_sample_idx]))
    dbs.append(davies_bouldin_score(Xp[sil_sample_idx], km.labels_[sil_sample_idx]))

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
axes[0].plot(ks, inertias, marker="o")
axes[0].set_title("Elbow (Inertia)")
axes[0].set_xlabel("k")
axes[1].plot(ks, sils, marker="o", color="green")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("k")
axes[2].plot(ks, dbs, marker="o", color="red")
axes[2].set_title("Davies-Bouldin Index")
axes[2].set_xlabel("k")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig4_k_selection_curves.png", dpi=600, bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/fig4_k_selection_curves.tiff", dpi=600, bbox_inches="tight", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.close()

print("Silhouette by k:", dict(zip(ks, [round(s, 3) for s in sils])))
print(f"\nذخیره شد: {FIG_DIR}/fig3_final_clustering_pca.png")
print(f"ذخیره شد: {FIG_DIR}/fig4_k_selection_curves.png")
