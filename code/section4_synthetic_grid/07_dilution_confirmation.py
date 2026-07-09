# -*- coding: utf-8 -*-
"""
آزمون تجزیه علّی (Causal Decomposition) — بخش ۴.۳ مقاله.

هدف: جدا کردن اثر «عدم‌تطابق k» (k-mismatch) از «رقیق‌شدن ابعادی به‌خاطر
one-hot کردن branch» (dimensional dilution)، در پیکربندیِ معادلِ دیتاست
اصلی: نسبت اقلیت=۵٪، جدایی≈۲۱ برابر انحراف‌معیار اکثریت.

طرح آزمایش (طبق متن دقیق بخش ۴.۳): همان دیتاست دوگروهیِ ۳.۲/۴.۲، در
۴ حالت اجرا می‌شود:
  {بدون branch, با branch (۱۶۰ بعد one-hot)} × {k=۲ درست, k=۳ نادرست}

هر حالت روی ۲۵ سید مستقل اجرا می‌شود (n=15,000 هر اجرا). ستون‌های
one-hot شده‌ی branch به‌صورت خام (بدون استانداردسازی اضافی) اضافه
می‌شوند، هم‌راستا با انکودینگ gender در بخش ۴.۱.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
utils = import_module("00_grid_data_utils")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_RUNS = BASE + "/results/dilution_confirmation_runs.csv"
OUT_SUMMARY = BASE + "/results/dilution_confirmation_summary.csv"

N_TOTAL = 15000
RATIO = 0.05
SEPARATION = 21  # "≈21 برابر انحراف‌معیار اکثریت" طبق متن مقاله
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
print(f"\nذخیره شد: {OUT_RUNS} ({len(runs)} ردیف)")
print(f"ذخیره شد: {OUT_SUMMARY} ({len(summary)} ردیف)")
