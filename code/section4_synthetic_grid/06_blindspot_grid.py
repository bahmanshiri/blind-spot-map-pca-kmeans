# -*- coding: utf-8 -*-
"""
شبکه حساسیتِ نقشه نقطه‌کوری (Blind Spot Map) — بخش ۴.۲ و ۳.۲ مقاله.

طرح:
  - ۹ سطح نسبت اقلیت: همان ۹ مقداری که در بخش ۵.۴.۲ (اعتبارسنجی Dry Bean)
    برای «هم‌راستایی با طرح شبکه اصلی» استفاده شده‌اند:
    1%, 2%, 3%, 5%, 8%, 12%, 18%, 25%, 30%
  - ۹ سطح جدایی آماری، از ۰.۵ تا ۱۲ برابر انحراف‌معیار اکثریت:
    0.5, 1, 2, 3, 4, 6, 8, 10, 12
  - ۲۵ سید مستقل (42-66) در هر خانه => ۹×۹×۲۵ = ۲۰۲۵ اجرای کامل پایپ‌لاین.
  - هر دیتاست: ۱۵٬۰۰۰ مشتری، k=2 (درست)، بدون branch (طبق بخش ۴.۲،
    branch فقط در آزمایش علّی بخش ۴.۳ ظاهر می‌شود).

خروجی: نتیجه تک‌تک ۲۰۲۵ اجرا + خلاصه‌ی میانگین/انحراف‌معیار روی ۸۱ خانه.
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
utils = import_module("00_grid_data_utils")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_RUNS = BASE + "/results/blindspot_grid_runs.csv"
OUT_SUMMARY = BASE + "/results/blindspot_grid_summary.csv"

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

print(f"{n_done}/{n_total_runs} اجرای کامل پایپ‌لاین در {time.time()-t0:.1f} ثانیه انجام شد.")

runs = pd.DataFrame(rows)
runs.to_csv(OUT_RUNS, index=False)

summary = (
    runs.groupby(["minority_ratio", "separation"])["ari"]
    .agg(ari_mean="mean", ari_std="std")
    .reset_index()
)
summary.to_csv(OUT_SUMMARY, index=False)

print(f"\nذخیره شد: {OUT_RUNS} ({len(runs)} ردیف)")
print(f"ذخیره شد: {OUT_SUMMARY} ({len(summary)} ردیف)")

print("\n--- خلاصه شبکه (ARI میانگین، ردیف=نسبت اقلیت، ستون=جدایی) ---")
pivot = summary.pivot(index="minority_ratio", columns="separation", values="ari_mean")
pd.set_option("display.width", 200)
print(pivot.round(3).to_string())

# نقاط کلیدیِ گزارش‌شده در بخش ۵.۱ مقاله
print("\n--- نقاط کلیدی نقشه ---")
for ratio in [0.01, 0.02]:
    row = summary[(summary.minority_ratio == ratio) & (summary.separation == 12)]
    if not row.empty:
        print(f"نسبت={ratio:.0%}, جدایی=12x  -> ARI میانگین = {row.ari_mean.values[0]:.3f}")

top_right = summary[(summary.minority_ratio >= 0.12) & (summary.separation >= 8)]
print(f"\nگوشه بالا-راست (نسبت>=12%, جدایی>=8x): ARI میانگین = {top_right.ari_mean.mean():.3f}")

bottom_left = summary[(summary.minority_ratio <= 0.02) & (summary.separation <= 1)]
print(f"گوشه پایین-چپ (نسبت<=2%, جدایی<=1x): ARI میانگین = {bottom_left.ari_mean.mean():.3f}")
