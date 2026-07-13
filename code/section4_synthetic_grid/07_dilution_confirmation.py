# -*- coding: utf-8 -*-
"""
Causal decomposition test -- Section 4.3 of the paper.

Goal: separate the effect of "k-mismatch" from "dimensional dilution due to
one-hot encoding branch", in the configuration equivalent to the primary
dataset: minority ratio=5%, separation~21x the majority's standard deviation.

Experimental design (per the exact text of Section 4.3): the same two-group
dataset from 3.2/4.2 is run in 4 conditions:
  {no branch, with branch (160-dim one-hot)} x {k=2 correct, k=3 incorrect}

Each condition is run over 25 independent seeds (n=15,000 per run). The
one-hot branch columns are added raw (no additional scaling), consistent
with the gender encoding in Section 4.1.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
utils = import_module("00_grid_data_utils")

# repo/code/section4_synthetic_grid/script.py -> repo root is 2 levels up
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_RUNS = BASE + "/results/section4/dilution_confirmation_runs.csv"
OUT_SUMMARY = BASE + "/results/section4/dilution_confirmation_summary.csv"
os.makedirs(BASE + "/results/section4", exist_ok=True)

N_TOTAL = 15000
RATIO = 0.05
SEPARATION = 21  # "~21x the majority's standard deviation" per the paper text
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]

CONDITIONS = [
    dict(name="Correct k, no branch", k=2, with_branch=False),
    dict(name="Correct k, with branch", k=2, with_branch=True),
    dict(name="Mismatched k, no branch", k=3, with_branch=False),
    dict(name="Mismatched k, with branch (primary dataset)", k=3, with_branch=True),
]

rows = []
for cond in CONDITIONS:
    for seed in SEEDS:
        df = utils.generate_two_group_dataset(
            n_total=N_TOTAL, minority_ratio=RATIO, separation=SEPARATION,
            seed=seed, with_branch=cond["with_branch"],
        )
        ari, var_explained = utils.run_pipeline(
            df, k=cond["k"], with_branch=cond["with_branch"], seed=seed,
            return_variance=True, scale_branch=False,
        )
        rows.append(dict(
            condition=cond["name"], k=cond["k"], branch=cond["with_branch"],
            seed=int(seed), ari=ari, pc1_2_variance_explained=var_explained,
        ))

runs = pd.DataFrame(rows)
runs.to_csv(OUT_RUNS, index=False)

summary = (
    runs.groupby(["condition", "k", "branch"], sort=False)
    .agg(ari_mean=("ari", "mean"), ari_std=("ari", "std"),
         pc1_2_variance_mean=("pc1_2_variance_explained", "mean"))
    .reset_index()
)
order = [c["name"] for c in CONDITIONS]
summary["condition"] = pd.Categorical(summary["condition"], categories=order, ordered=True)
summary = summary.sort_values("condition").reset_index(drop=True)
summary.to_csv(OUT_SUMMARY, index=False)

pd.set_option("display.width", 160)
print("--- Table 1 (final, 25 seeds): causal decomposition of k-mismatch and one-hot dilution ---\n")
print(summary.round(3).to_string(index=False))
print(f"\nSaved: {OUT_RUNS} ({len(runs)} rows)")
print(f"Saved: {OUT_SUMMARY} ({len(summary)} rows)")
