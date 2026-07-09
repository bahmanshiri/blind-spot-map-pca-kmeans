# -*- coding: utf-8 -*-
"""
مولد دیتاست اصلی (Primary Dataset) — بخش ۳.۱ مقاله.

۱۲۰,۰۰۰ مشتری سینتتیک با ۳ سگمنت پنهان (ground truth):
  - normal (۶۲٪): مشتریان عادی
  - anomalous (۵٪): گروه اقلیت با گردش حساب غیرعادی (account turnover)
  - low_activity (۳۳٪): مشتریان کم‌تراکنش

ویژگی‌ها:
  - عددی (۴ ویژگی): age, account_balance, transaction_to_balance_ratio,
    avg_monthly_transactions
  - دسته‌ای: gender (باینری، مستقل از سگمنت)، branch (۱۶۰ شعبه، مستقل از سگمنت)

طراحی سگمنت‌ها:
  - normal: مقادیر پایه برای همه ویژگی‌ها
  - anomalous: transaction_to_balance_ratio به‌شدت جابه‌جا شده (~21 برابر
    انحراف‌معیار گروه normal روی همین ویژگی) — این همان «گردش حساب
    غیرعادی» است که در بخش ۴.۳ به‌عنوان «پیکربندی معادل» با نسبت=۵٪ و
    جدایی≈۲۱ برابر مورد استفاده قرار می‌گیرد.
  - low_activity: avg_monthly_transactions بسیار پایین‌تر از normal
"""
import numpy as np
import pandas as pd


def generate_primary_dataset(n_total=120_000, seed=42, n_branches=160,
                              duplicate_frac=0.045, outlier_frac=0.015):
    rng = np.random.default_rng(seed)

    n_normal = int(round(0.62 * n_total))
    n_anomalous = int(round(0.05 * n_total))
    n_low_activity = n_total - n_normal - n_anomalous

    segment = np.array(
        ["normal"] * n_normal + ["anomalous"] * n_anomalous + ["low_activity"] * n_low_activity
    )

    n = n_total
    age = rng.normal(40, 12, n)
    account_balance = rng.lognormal(mean=9.0, sigma=0.8, size=n)
    gender = rng.integers(0, 2, n)
    branch = rng.integers(0, n_branches, n)

    # transaction_to_balance_ratio: normal ~ N(0,1); anomalous shifted ~21 SD
    transaction_to_balance_ratio = rng.normal(0.0, 1.0, n)
    # avg_monthly_transactions: normal ~ N(20,5); low_activity much lower
    avg_monthly_transactions = rng.normal(20.0, 5.0, n)

    is_anomalous = segment == "anomalous"
    is_low_activity = segment == "low_activity"

    transaction_to_balance_ratio[is_anomalous] = rng.normal(21.0, 1.0, is_anomalous.sum())
    avg_monthly_transactions[is_low_activity] = rng.normal(2.0, 1.0, is_low_activity.sum())
    avg_monthly_transactions = np.clip(avg_monthly_transactions, 0, None)

    df = pd.DataFrame({
        "age": age,
        "account_balance": account_balance,
        "transaction_to_balance_ratio": transaction_to_balance_ratio,
        "avg_monthly_transactions": avg_monthly_transactions,
        "gender": gender,
        "branch": branch,
        "segment": segment,
    })

    # --- تزریق چند رکورد تکراریِ کامل، جایگزین چند ردیف تصادفی (بدون تغییر
    #     تعداد کل ردیف‌ها) — برای شبیه‌سازی داده‌ی خام واقعی ---
    n_dup = int(round(duplicate_frac * n_total))
    source_idx = rng.choice(df.index, size=n_dup, replace=True)
    target_idx = rng.choice(df.index, size=n_dup, replace=False)
    df.loc[target_idx, df.columns] = df.loc[source_idx, df.columns].values

    # --- تزریق چند مقدار پرت شدید در account_balance (خطای ثبت داده) ---
    n_outliers = int(round(outlier_frac * n_total))
    outlier_idx = rng.choice(df.index, size=n_outliers, replace=False)
    df.loc[outlier_idx, "account_balance"] *= rng.uniform(50, 200, n_outliers)

    df = df.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
    return df


def clean_dataset(df):
    """حذف رکوردهای تکراری کامل + مقادیر پرت شدید (IQR روی account_balance)."""
    before = len(df)
    df = df.drop_duplicates(
        subset=["age", "account_balance", "transaction_to_balance_ratio",
                 "avg_monthly_transactions", "gender", "branch"]
    ).copy()

    q1, q3 = df["account_balance"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 8.0 * iqr
    df = df[df["account_balance"] <= upper].copy()

    after = len(df)
    removed_frac = 1 - after / before
    return df.reset_index(drop=True), removed_frac
