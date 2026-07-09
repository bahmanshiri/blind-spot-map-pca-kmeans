# -*- coding: utf-8 -*-
"""تولید نمودارهای اعتبارسنجی خارجی (External Validity) روی داده واقعی Dry Bean."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # blindspot_real/

FIG_DIR = (BASE + "/figures")
plt.rcParams["figure.dpi"] = 600  # publication-quality raster resolution
plt.rcParams["font.size"] = 10

grid = pd.read_csv((BASE + "/data/real_blindspot_grid_results.csv"))
binary = pd.read_csv((BASE + "/data/algorithm_comparison_binary.csv"))
multi = pd.read_csv((BASE + "/data/algorithm_comparison_multiclass.csv"))

# ---------- Fig R1: Heatmap ARI (class ordered by separation) x (minority ratio) ----------
order = grid.groupby("class")["separation_multivar"].first().sort_values().index.tolist()
pivot = grid.pivot(index="class", columns="minority_ratio_target", values="ari_mean").reindex(order)

fig, ax = plt.subplots(figsize=(8, 4.5))
im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-0.05, vmax=0.35)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f"{c:.0%}" for c in pivot.columns], rotation=45)
ax.set_yticks(range(len(pivot.index)))
sep_map = grid.groupby("class")["separation_multivar"].first()
ax.set_yticklabels([f"{c} (sep={sep_map[c]:.1f})" for c in pivot.index])
ax.set_xlabel("Minority ratio (subsampled)")
ax.set_title("Real-Data Blind Spot Map (Dry Bean Dataset)\nPCA(2)+KMeans(2) — ARI vs. true class")
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        v = pivot.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, ax=ax, label="ARI")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/figR1_real_blindspot_heatmap.png", dpi=600, bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/figR1_real_blindspot_heatmap.tiff", dpi=600, bbox_inches="tight", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.close()

# ---------- Fig R2: ARI vs ratio curves for illustrative classes ----------
fig, ax = plt.subplots(figsize=(7, 5))
highlight = ["BOMBAY", "HOROZ", "CALI", "SEKER", "SIRA"]
for cls in highlight:
    sub = grid[grid["class"] == cls].sort_values("minority_ratio_target")
    sep = sub["separation_multivar"].iloc[0]
    ax.errorbar(sub["minority_ratio_target"], sub["ari_mean"], yerr=sub["ari_std"],
                marker="o", capsize=2, label=f"{cls} (sep={sep:.1f})")
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xlabel("Minority ratio")
ax.set_ylabel("ARI (mean ± std over 5 seeds)")
ax.set_title("Real-Data Recovery Curves — PCA(2)+KMeans(2)\n(Dry Bean, natural separation per class)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/figR2_real_recovery_curves.png", dpi=600, bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/figR2_real_recovery_curves.tiff", dpi=600, bbox_inches="tight", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.close()

# ---------- Fig R3: algorithm comparison per class (grouped bars, ARI) ----------
fig, ax = plt.subplots(figsize=(10, 5))
algos = ["KMeans", "GMM", "SpectralClustering", "DBSCAN", "HDBSCAN"]
classes = sorted(binary["class_"].unique())
x = np.arange(len(classes))
width = 0.15
for i, algo in enumerate(algos):
    vals = [binary[(binary["class_"] == c) & (binary["algorithm"] == algo)]["ARI"].values[0] for c in classes]
    ax.bar(x + i * width, vals, width, label=algo)
ax.set_xticks(x + width * (len(algos) - 1) / 2)
ax.set_xticklabels(classes, rotation=20)
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_ylabel("ARI")
ax.set_title("Algorithm Comparison per Class (natural ratio) — Real Data")
ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.15))
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/figR3_algorithm_comparison_binary.png", dpi=600, bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/figR3_algorithm_comparison_binary.tiff", dpi=600, bbox_inches="tight", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.close()

# ---------- Fig R4: multiclass algorithm comparison ----------
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.bar(multi["algorithm"], multi["ARI"], color="steelblue")
ax.set_ylabel("ARI (vs. true 7-class labels)")
ax.set_title("Full Multiclass Comparison (7 bean species) — Real Data")
ax.axhline(0, color="gray", linewidth=0.8)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/figR4_multiclass_algorithm_comparison.png", dpi=600, bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/figR4_multiclass_algorithm_comparison.tiff", dpi=600, bbox_inches="tight", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.close()

print("چهار نمودار ذخیره شد در:", FIG_DIR)
