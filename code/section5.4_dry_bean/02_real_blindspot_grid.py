# -*- coding: utf-8 -*-
"""
نسخه «داده واقعی» شبکه نقشه نقطه‌کوری (Blind Spot Map).

هدف: پاسخ به ایراد شماره ۲ داور (تمام آزمایش‌های مقاله روی داده مصنوعی است).

روش: به‌جای تولید داده مصنوعی، از دیتاست واقعی Dry Bean استفاده می‌کنیم.
هر یک از ۷ نوع لوبیا به‌نوبت به‌عنوان «گروه اقلیت واقعی» در نظر گرفته می‌شود
(جدایی آماری‌اش نسبت به بقیه، ثابت و طبیعی است چون از داده واقعی می‌آید و
قابل کنترل نیست). سپس با زیرنمونه‌گیری (subsampling) از همان کلاس، نسبت
جمعیتی آن گروه در بازه‌ای مشابه شبکه اصلی مقاله (۱٪ تا ۳۰٪) تغییر داده می‌شود.

این یک "کنترل جزئی" است، نه کنترل کامل مثل شبیه‌سازی: فقط minority_ratio
دستکاری می‌شود، در حالی که separation ثابت و طبیعیِ همان کلاس واقعی باقی
می‌ماند. این دقیقاً هم‌راستا با محدودیتی است که باید در مقاله ذکر شود.

خروجی: برای هر (کلاس, نسبت هدف) با ۵ زیرنمونه‌ی تصادفی مستقل، PCA(2)+KMeans(2)
اجرا و ARI با برچسب واقعی محاسبه می‌شود.
"""
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # blindspot_real/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

IN = BASE + "/data/dry_bean_clean.csv"
OUT = BASE + "/data/real_blindspot_grid_results.csv"

df = pd.read_csv(IN)
feat_cols = [c for c in df.columns if c != "Class"]
classes = sorted(df["Class"].unique())

TARGET_RATIOS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.30]
SEEDS = [42, 43, 44, 45, 46]
N_TOTAL = 4000  # اندازه ثابت هر نمونه‌گیری، برای سرعت و یکنواختی شبکه

# جدایی چندمتغیره طبیعی هر کلاس (Cohen's d چندمتغیره، ثابت و غیرقابل‌دستکاری)
Xs_full = StandardScaler().fit_transform(df[feat_cols].values)


def multivar_separation(class_name):
    mask = (df["Class"] == class_name).values
    Xc, Xr = Xs_full[mask], Xs_full[~mask]
    pooled_std = np.sqrt((Xc.std(axis=0) ** 2 + Xr.std(axis=0) ** 2) / 2)
    pooled_std[pooled_std == 0] = 1e-9
    d = (Xc.mean(axis=0) - Xr.mean(axis=0)) / pooled_std
    return float(np.linalg.norm(d))


rows = []
skipped = []
total_cells = len(classes) * len(TARGET_RATIOS)
done = 0

for cls in classes:
    sep = multivar_separation(cls)
    cls_idx = df.index[df["Class"] == cls].to_numpy()
    other_idx = df.index[df["Class"] != cls].to_numpy()
    n_cls_available = len(cls_idx)

    for ratio in TARGET_RATIOS:
        n_min = int(round(ratio * N_TOTAL))
        n_maj = N_TOTAL - n_min
        done += 1

        if n_min > n_cls_available or n_min < 2:
            skipped.append((cls, ratio, "تعداد نمونه واقعی کلاس کافی نیست"))
            print(f"[{done}/{total_cells}] {cls} ratio={ratio:.2f} -> SKIP (n_min={n_min} > موجود={n_cls_available})")
            continue

        aris = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            sel_min = rng.choice(cls_idx, size=n_min, replace=False)
            sel_maj = rng.choice(other_idx, size=n_maj, replace=False)
            sel = np.concatenate([sel_min, sel_maj])

            X_sub = df.loc[sel, feat_cols].values
            y_sub = (df.loc[sel, "Class"] == cls).astype(int).values
            Xs_sub = StandardScaler().fit_transform(X_sub)

            pca = PCA(n_components=2, random_state=seed)
            Xp = pca.fit_transform(Xs_sub)
            km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(Xp)
            ari = adjusted_rand_score(y_sub, km.labels_)
            aris.append(ari)

        rows.append({
            "class": cls,
            "minority_ratio_target": ratio,
            "n_minority": n_min,
            "n_total": N_TOTAL,
            "separation_multivar": sep,
            "ari_mean": float(np.mean(aris)),
            "ari_std": float(np.std(aris)),
            "n_seeds": len(SEEDS),
        })
        print(f"[{done}/{total_cells}] {cls} ratio={ratio:.2f} sep={sep:.2f} "
              f"ARI={np.mean(aris):.3f}±{np.std(aris):.3f}")

result = pd.DataFrame(rows)
result.to_csv(OUT, index=False)
print(f"\nذخیره شد: {OUT}")
print(f"تعداد خانه‌های اجراشده: {len(result)} از {total_cells}")
if skipped:
    print(f"تعداد خانه‌های رد‌شده (به دلیل کمبود نمونه واقعی): {len(skipped)}")
    for s in skipped:
        print("  -", s)
