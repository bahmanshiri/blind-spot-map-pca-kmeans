# -*- coding: utf-8 -*-
"""تولید نمودارهای بخش ۵.۱ (نقشه نقطه‌کوری) و ۵.۲ (تجزیه علّی)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = BASE + "/figures"
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams["figure.dpi"] = 600  # publication-quality raster resolution
plt.rcParams["font.size"] = 10

summary = pd.read_csv(BASE + "/results/blindspot_grid_summary.csv")

# ---------- Figure 1: نقشه ARI میانگین ----------
pivot = summary.pivot(index="minority_ratio", columns="separation", values="ari_mean")
fig, ax = plt.subplots(figsize=(7.5, 5.5))
im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, origin="lower")
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f"{c:g}x" for c in pivot.columns])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([f"{r:.0%}" for r in pivot.index])
ax.set_xlabel("Statistical separation (majority SD)")
ax.set_ylabel("Minority population ratio")
ax.set_title("Blind-Spot Map: ARI for Minority-Group Recovery by PCA+K-means (k=2, correct)")
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        v = pivot.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                     color="black" if 0.3 < v < 0.7 else "white")
plt.colorbar(im, ax=ax, label="Mean ARI (25 seeds)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig1_blindspot_map.png", dpi=600, bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/fig1_blindspot_map.tiff", dpi=600, bbox_inches="tight", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.close()

# ---------- Figure 2: حداقل جدایی لازم برای ARI>=0.5 ----------
ratios = sorted(summary["minority_ratio"].unique())
seps = sorted(summary["separation"].unique())
min_sep_needed = []
for r in ratios:
    sub = summary[summary.minority_ratio == r].sort_values("separation")
    reached = sub[sub.ari_mean >= 0.5]
    min_sep_needed.append(reached.separation.min() if len(reached) else np.nan)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot([r * 100 for r in ratios], min_sep_needed, marker="o", color="#1f77b4")
ax.set_xlabel("Minority population ratio (%)")
ax.set_ylabel("Min. separation needed for ARI >= 0.5 (majority SD)")
ax.set_title("Statistical Boundary: Minimum Statistical Separation for\nAcceptable Minority-Group Recovery")
ax.grid(alpha=0.3)
for x, y in zip(ratios, min_sep_needed):
    if not np.isnan(y):
        ax.annotate(f"{y:g}x", (x * 100, y), textcoords="offset points", xytext=(0, 6), fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig2_min_separation_curve.png", dpi=600, bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/fig2_min_separation_curve.tiff", dpi=600, bbox_inches="tight", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.close()

print("min separation needed per ratio:")
for r, y in zip(ratios, min_sep_needed):
    print(f"  ratio={r:.0%}: {y}")

print(f"\nذخیره شد: {FIG_DIR}/fig1_blindspot_map.png")
print(f"ذخیره شد: {FIG_DIR}/fig2_min_separation_curve.png")
