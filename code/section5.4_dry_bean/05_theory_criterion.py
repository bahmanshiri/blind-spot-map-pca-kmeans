# -*- coding: utf-8 -*-
"""
معیار هندسی نقطه‌کوری (Geometric Blind-Spot Criterion) — پاسخ به ایراد #۸ داور
("بحث تئوری ضعیف است... چرا از نظر هندسی K-means گروه اقلیت را جذب اکثریت می‌کند؟")

مبنای ریاضی: هدف K-means کمینه‌کردن WCSS (Within-Cluster Sum of Squares) است.
چون مجموع کل مربعات (TSS) نسبت به میانگین کلی برای یک دیتاست ثابت است:
    TSS = WCSS + BCSS
کمینه‌کردن WCSS دقیقاً معادل بیشینه‌کردن BCSS (Between-Cluster Sum of Squares)
است. بنابراین بین هر دو افراز رقیب، K-means (به‌صورت محلی-بهینه، از طریق
الگوریتم Lloyd) افرازی را ترجیح می‌دهد که BCSS بزرگ‌تری دارد.

برای k=2 روی مجموعه (اقلیت ∪ اکثریت)، دو افراز رقیب طبیعی وجود دارد:

  افراز A ("جداسازی درست"): یک خوشه = اقلیت، خوشه دیگر = اکثریت.
      BCSS(A) = (n_min * n_maj / N) * D^2
      که D فاصله اقلیدسی بین میانگین اقلیت و میانگین اکثریت است. این یک
      اتحاد دقیق (exact identity) برای BCSS دوگروهی است، نه تقریب.

  افراز B ("جذب در اکثریت"): اکثریت با بهترین افراز درونیِ خودش (K-means(2)
      روی فقط نقاط اکثریت) به دو زیرخوشه تقسیم می‌شود؛ اقلیت (چون تعدادش کم
      است) عملاً به نزدیک‌ترین زیرخوشه اکثریت می‌چسبد و اثر چندانی روی مراکز
      ندارد.
      BCSS(B) ≈ (n1*n2/n_maj) * Δ^2
      که Δ فاصله بین دو مرکز بهینه‌ی همین افراز-درونیِ اکثریت است.

معیار پیش‌بینی: اقلیت وقتی توسط K-means(2) به‌عنوان خوشه مستقل «دیده می‌شود»
که:
      (n_min * n_maj / N) * D^2   >   BCSS(B)
و چون معمولاً n_min << n_maj (یعنی N ≈ n_maj)، این تقریباً معادل است با:
      minority_ratio = n_min / n_maj   ≳   BCSS(B) / (n_maj * D^2)

⚠️ محدودیت صادقانه: این یک استدلال هندسی/اکتشافی (heuristic) است، نه اثبات
ریاضی کامل. الگوریتم Lloyd فقط به بهینه محلی می‌رسد (K-means به‌طور کلی
NP-hard برای بهینه سراسری است)، و در عمل ممکن است افراز واقعی الگوریتم نه
دقیقاً A و نه دقیقاً B بلکه ترکیبی از این دو باشد. این معیار یک شرط لازم
تقریبی است، نه یک قانون قطعی.
"""
import numpy as np
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

IN = BASE + "/data/dry_bean_clean.csv"
OUT = BASE + "/data/theory_criterion_validation.csv"
SEED = 42

df = pd.read_csv(IN)
feat_cols = [c for c in df.columns if c != "Class"]
Xs = StandardScaler().fit_transform(df[feat_cols].values)
pca = PCA(n_components=2, random_state=SEED)
Xp = pca.fit_transform(Xs)  # همان فضایی که K-means واقعاً می‌بیند

rows = []
for cls in sorted(df["Class"].unique()):
    mask = (df["Class"] == cls).values
    X_min, X_maj = Xp[mask], Xp[~mask]
    n_min, n_maj = len(X_min), len(X_maj)
    N = n_min + n_maj

    c_min, c_maj = X_min.mean(axis=0), X_maj.mean(axis=0)
    D2 = float(np.sum((c_min - c_maj) ** 2))
    bcss_isolate = (n_min * n_maj / N) * D2

    km_maj = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit(X_maj)
    l = km_maj.labels_
    n1, n2 = int((l == 0).sum()), int((l == 1).sum())
    c1, c2 = X_maj[l == 0].mean(axis=0), X_maj[l == 1].mean(axis=0)
    delta2 = float(np.sum((c1 - c2) ** 2))
    bcss_majsplit = (n1 * n2 / n_maj) * delta2

    predicted_isolated = bcss_isolate > bcss_majsplit
    predicted_min_ratio_threshold = bcss_majsplit / (n_maj * D2) if D2 > 0 else np.inf

    km_pool = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit(Xp)
    true_label = mask.astype(int)
    ari = adjusted_rand_score(true_label, km_pool.labels_)
    ct = pd.crosstab(true_label, km_pool.labels_)
    min_cluster = ct.loc[1].idxmax()
    purity_of_min_in_its_cluster = ct.loc[1, min_cluster] / ct[min_cluster].sum()
    actually_isolated = purity_of_min_in_its_cluster > 0.5

    rows.append(dict(
        cls=cls, n_min=n_min, n_maj=n_maj, minority_ratio=round(n_min / N, 4),
        D2=round(D2, 3), BCSS_isolate=round(bcss_isolate, 1),
        BCSS_majority_split=round(bcss_majsplit, 1),
        predicted_ratio_threshold=round(predicted_min_ratio_threshold, 4),
        criterion_predicts_isolated=predicted_isolated,
        observed_ARI=round(ari, 3),
        observed_isolated=actually_isolated,
        prediction_correct=(predicted_isolated == actually_isolated),
    ))

result = pd.DataFrame(rows)
result.to_csv(OUT, index=False)
pd.set_option("display.width", 160)
print(result.to_string(index=False))
acc = result["prediction_correct"].mean()
print(f"\nدقت پیش‌بینی معیار هندسی روی ۷ کلاس واقعی: {acc:.0%} ({result['prediction_correct'].sum()}/{len(result)})")
print(f"ذخیره شد: {OUT}")
