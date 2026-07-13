# -*- coding: utf-8 -*-
"""
Blind Spot Map sensitivity grid -- Sections 4.2 and 3.2 of the paper.

Design:
  - 9 minority-ratio levels: the same 9 values used in Section 5.4.2
    (Dry Bean validation) "to align with the main grid design":
    1%, 2%, 3%, 5%, 8%, 12%, 18%, 25%, 30%
  - 9 statistical-separation levels, from 0.5x to 12x the majority's
    standard deviation: 0.5, 1, 2, 3, 4, 6, 8, 10, 12
  - 25 independent seeds (42-66) per cell => 9x9x25 = 2,025 full pipeline runs.
  - Each dataset: 15,000 customers, k=2 (correct), no branch feature (per
    Section 4.2, branch only appears in the causal experiment of Section 4.3).

Output: per-run results for all 2,025 runs + mean/std summary over 81 cells.
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
utils = import_module("00_grid_data_utils")

# repo/code/section4_synthetic_grid/script.py -> repo root is 2 levels up
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_RUNS = BASE + "/results/section4/blindspot_grid_runs.csv"
OUT_SUMMARY = BASE + "/results/section4/blindspot_grid_summary.csv"
os.makedirs(BASE + "/results/section4", exist_ok=True)

N_TOTAL = 15000
RATIOS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.30]
SEPARATIONS = [0.5, 1, 2, 3, 4, 6, 8, 10, 12]
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]

t0 = time.time()
rows = []
n_done = 0
n_total_runs = len(RATIOS) * len(SEPARATIONS) * len(SEEDS)

for ratio in RATIOS:
    for sep in SEPARATIONS:
        for seed in SEEDS:
            df = utils.generate_two_group_dataset(
                n_total=N_TOTAL, minority_ratio=ratio, separation=sep,
                seed=seed, with_branch=False,
            )
            ari = utils.run_pipeline(df, k=2, with_branch=False, seed=seed)
            rows.append(dict(minority_ratio=ratio, separation=sep, seed=int(seed), ari=ari))
            n_done += 1

print(f"{n_done}/{n_total_runs} full pipeline runs completed in {time.time()-t0:.1f} seconds.")

runs = pd.DataFrame(rows)
runs.to_csv(OUT_RUNS, index=False)

summary = (
    runs.groupby(["minority_ratio", "separation"])["ari"]
    .agg(ari_mean="mean", ari_std="std")
    .reset_index()
)
summary.to_csv(OUT_SUMMARY, index=False)

print(f"\nSaved: {OUT_RUNS} ({len(runs)} rows)")
print(f"Saved: {OUT_SUMMARY} ({len(summary)} rows)")

print("\n--- Grid summary (mean ARI, row=minority ratio, column=separation) ---")
pivot = summary.pivot(index="minority_ratio", columns="separation", values="ari_mean")
pd.set_option("display.width", 200)
print(pivot.round(3).to_string())

# key points reported in Section 5.1 of the paper
print("\n--- Key points of the map ---")
for ratio in [0.01, 0.02]:
    row = summary[(summary.minority_ratio == ratio) & (summary.separation == 12)]
    if not row.empty:
        print(f"ratio={ratio:.0%}, separation=12x  -> mean ARI = {row.ari_mean.values[0]:.3f}")

top_right = summary[(summary.minority_ratio >= 0.12) & (summary.separation >= 8)]
print(f"\nTop-right corner (ratio>=12%, separation>=8x): mean ARI = {top_right.ari_mean.mean():.3f}")

bottom_left = summary[(summary.minority_ratio <= 0.02) & (summary.separation <= 1)]
print(f"Bottom-left corner (ratio<=2%, separation<=1x): mean ARI = {bottom_left.ari_mean.mean():.3f}")
