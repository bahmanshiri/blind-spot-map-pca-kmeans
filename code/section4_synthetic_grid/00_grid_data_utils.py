# -*- coding: utf-8 -*-
"""
مولد دیتاست شبکه حساسیت (Sensitivity Grid Dataset) — بخش ۳.۲ مقاله.

طرح دیتاست (طبق بخش ۳.۲):
  - دقیقاً دو گروه: اکثریت و اقلیت.
  - دو پارامتر کنترل‌شده:
      * minority_ratio: نسبت جمعیتی اقلیت (۹ سطح، ۱٪ تا ۳۰٪)
      * separation: جدایی آماریِ اقلیت از اکثریت روی ویژگیِ
        "نسبت تراکنش‌به‌موجودی" (transaction_to_balance_ratio)، بر حسب
        انحراف‌معیار اکثریت (۹ سطح، ۰.۵× تا ۱۲×).
  - ویژگی‌های عددی: age, account_balance, transaction_to_balance_ratio
    (فقط ویژگیِ سوم بین دو گروه جابه‌جا می‌شود؛ دو ویژگیِ دیگر نویزِ
    یکسان‌توزیع هستند و صرفاً برای معنادار کردن PCA(2) روی بیش از یک بعد
    وجود دارند).
  - ویژگی دسته‌ای اختیاری: branch (۱۶۰ شعبه، مستقل از گروه — یعنی هیچ
    سیگنالی درباره‌ی گروه اقلیت ندارد) — با one-hot، فقط در آزمایش‌های
    مرتبط (بخش ۴.۳) فعال می‌شود، دقیقاً طبق بخش ۴.۱.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def generate_two_group_dataset(n_total, minority_ratio, separation, seed,
                                with_branch=False, n_branches=160):
    """
    تولید دیتاست دوگروهیِ پارامتریک.

    minority_ratio : نسبت جمعیتی اقلیت (مثلاً 0.05 برای ۵٪)
    separation     : جدایی میانگین اقلیت از اکثریت روی
                      transaction_to_balance_ratio، بر حسب انحراف‌معیار
                      اکثریت (که ۱٫۰ نرمال‌سازی شده است)
    seed           : سید مستقل برای این اجرا
    with_branch    : اگر True، ستون دسته‌ایِ ۱۶۰-سطحیِ branch هم اضافه می‌شود
                      (مستقل از برچسب گروه — صرفاً رقیق‌کننده‌ی بعدی)
    """
    rng = np.random.default_rng(seed)

    n_min = int(round(minority_ratio * n_total))
    n_maj = n_total - n_min
    assert n_min >= 1 and n_maj >= 1

    label = np.array([0] * n_maj + [1] * n_min)  # 0=اکثریت, 1=اقلیت

    # ویژگی‌های نویز (یکسان‌توزیع بین دو گروه، بدون سیگنال گروه)
    age = rng.normal(40, 12, n_total)
    account_balance = rng.lognormal(mean=9.0, sigma=0.8, size=n_total)

    # ویژگیِ سیگنال‌دار: transaction_to_balance_ratio
    # اکثریت: میانگین=0، انحراف‌معیار=1 (این انحراف‌معیار، مبنای "برابر جدایی" است)
    # اقلیت: میانگین=separation (بر حسب انحراف‌معیار اکثریت)، همان انحراف‌معیار
    ratio_maj = rng.normal(0.0, 1.0, n_maj)
    ratio_min = rng.normal(separation, 1.0, n_min)
    transaction_to_balance_ratio = np.concatenate([ratio_maj, ratio_min])

    df = pd.DataFrame({
        "age": age,
        "account_balance": account_balance,
        "transaction_to_balance_ratio": transaction_to_balance_ratio,
        "label": label,
    })

    if with_branch:
        df["branch"] = rng.integers(0, n_branches, n_total)  # مستقل از گروه

    # به‌هم‌ریختن ترتیب ردیف‌ها (بی‌اثر روی نتیجه، ولی جلوگیری از توهم ترتیب)
    df = df.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
    return df


def run_pipeline(df, k, with_branch, seed, return_variance=False, scale_branch=False):
    """
    پایپ‌لاین پایه، دقیقاً طبق بخش ۴.۱:
    StandardScaler (ویژگی‌های عددی) -> encoding (one-hot برای branch)
    -> PCA(2) -> K-means(n_init=10) -> ARI در برابر برچسب پنهان.

    branch (وقتی حاضر باشد) به‌صورت one-hot خام اضافه می‌شود (بدون
    استانداردسازی اضافی)، هم‌راستا با انکودینگ gender در بخش ۴.۱؛
    ویژگی‌های عددی با StandardScaler نرمال می‌شوند.
    """
    numeric_cols = ["age", "account_balance", "transaction_to_balance_ratio"]
    Xnum = df[numeric_cols].values.astype(float)
    Xnum_scaled = StandardScaler().fit_transform(Xnum)

    blocks = [Xnum_scaled]
    if with_branch and "branch" in df.columns:
        Xbranch = pd.get_dummies(df["branch"], prefix="branch").values.astype(float)
        if scale_branch:
            Xbranch = StandardScaler().fit_transform(Xbranch)
        blocks.append(Xbranch)

    X = np.hstack(blocks)

    pca = PCA(n_components=2, random_state=int(seed))
    Xp = pca.fit_transform(X)

    km = KMeans(n_clusters=k, n_init=10, random_state=int(seed))
    pred = km.fit_predict(Xp)

    ari = adjusted_rand_score(df["label"].values, pred)

    if return_variance:
        return ari, float(np.sum(pca.explained_variance_ratio_))
    return ari
