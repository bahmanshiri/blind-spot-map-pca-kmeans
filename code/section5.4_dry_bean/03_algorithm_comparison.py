# -*- coding: utf-8 -*-
"""
مقایسه واقعیِ الگوریتم‌های جایگزین (نه فقط پیشنهاد) + متریک‌های چندگانه.
پاسخ به ایراد شماره ۴ (فقط ARI) و شماره ۵ (GMM/HDBSCAN فقط پیشنهاد شده‌اند) داور.

بخش الف) سناریوی دوتایی (یک کلاس = اقلیت، بقیه = اکثریت) با نسبت طبیعی واقعی هرکلاس.
بخش ب) سناریوی کامل ۷‌کلاسه (تمام گونه‌های لوبیا هم‌زمان).

الگوریتم‌های تست‌شده: KMeans (خط پایه مقاله)، Gaussian Mixture Model،
Spectral Clustering، DBSCAN، HDBSCAN.
متریک‌ها: ARI، NMI، Fowlkes-Mallows، Silhouette، Davies-Bouldin، Calinski-Harabasz.

هشدار صادقانه: Silhouette/DB/CH فقط با دسترسی به برچسب واقعی «بهترین» تفسیر
می‌شوند؛ در عمل این سه معیارِ داخلی (بدون برچسب) هستند و ممکن است با ARI/NMI
(که بیرونی و برچسب‌محورند) هم‌جهت نباشند. این ناهم‌سویی خودش یک یافته است،
نه یک باگ.
"""
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # blindspot_real/
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)

IN = BASE + "/data/dry_bean_clean.csv"
OUT_BINARY = BASE + "/data/algorithm_comparison_binary.csv"
OUT_MULTI = BASE + "/data/algorithm_comparison_multiclass.csv"

df = pd.read_csv(IN)
feat_cols = [c for c in df.columns if c != "Class"]
SEED = 42
SAMPLE_N = 3000  # برای امکان‌پذیر بودن Spectral/HDBSCAN از نظر زمان اجرا

rng = np.random.default_rng(SEED)


def safe_internal_metrics(Xp, labels):
    """Silhouette/DB/CH فقط وقتی حداقل ۲ خوشه غیرنویز وجود دارد قابل‌محاسبه‌اند."""
    valid = labels != -1
    uniq = np.unique(labels[valid])
    if len(uniq) < 2 or valid.sum() < 10:
        return np.nan, np.nan, np.nan
    try:
        sil = silhouette_score(Xp[valid], labels[valid])
        db = davies_bouldin_score(Xp[valid], labels[valid])
        ch = calinski_harabasz_score(Xp[valid], labels[valid])
    except Exception:
        sil, db, ch = np.nan, np.nan, np.nan
    return sil, db, ch


def run_algorithms(Xp, y_true, n_clusters_guess):
    algos = {}

    km = KMeans(n_clusters=n_clusters_guess, n_init=10, random_state=SEED).fit(Xp)
    algos["KMeans"] = km.labels_

    gmm = GaussianMixture(n_components=n_clusters_guess, random_state=SEED, n_init=3).fit(Xp)
    algos["GMM"] = gmm.predict(Xp)

    try:
        sc = SpectralClustering(n_clusters=n_clusters_guess, random_state=SEED,
                                 affinity="nearest_neighbors", n_neighbors=10,
                                 assign_labels="kmeans").fit(Xp)
        algos["SpectralClustering"] = sc.labels_
    except Exception as e:
        algos["SpectralClustering"] = np.full(len(Xp), -1)
        print("  SpectralClustering failed:", e)

    # DBSCAN: eps تقریبی از فاصله k-نزدیک‌ترین‌همسایه (heuristic ساده)، مینیمم نمونه ثابت
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=10).fit(Xp)
    dists, _ = nn.kneighbors(Xp)
    eps_guess = float(np.median(dists[:, -1]))
    db_ = DBSCAN(eps=eps_guess, min_samples=10).fit(Xp)
    algos["DBSCAN"] = db_.labels_

    hdb = HDBSCAN(min_cluster_size=max(15, len(Xp) // 100)).fit(Xp)
    algos["HDBSCAN"] = hdb.labels_

    return algos


def evaluate(y_true, labels, Xp):
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    fmi = fowlkes_mallows_score(y_true, labels)
    sil, db, ch = safe_internal_metrics(Xp, labels)
    n_found = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    return dict(ARI=ari, NMI=nmi, FMI=fmi, Silhouette=sil, Davies_Bouldin=db,
                Calinski_Harabasz=ch, n_clusters_found=n_found, n_noise_points=n_noise)


# ---------------- بخش الف: سناریوهای دوتایی (اقلیت طبیعی هر کلاس) ----------------
print("=== بخش الف: هر کلاس در برابر بقیه (نسبت طبیعی واقعی) ===")
binary_rows = []
classes = sorted(df["Class"].unique())

for cls in classes:
    cls_idx = df.index[df["Class"] == cls].to_numpy()
    other_idx = df.index[df["Class"] != cls].to_numpy()
    n_min_available = len(cls_idx)
    natural_ratio = n_min_available / len(df)

    n_min = min(n_min_available, max(30, int(round(natural_ratio * SAMPLE_N))))
    n_maj = SAMPLE_N - n_min
    n_maj = min(n_maj, len(other_idx))

    sel_min = rng.choice(cls_idx, size=n_min, replace=False)
    sel_maj = rng.choice(other_idx, size=n_maj, replace=False)
    sel = np.concatenate([sel_min, sel_maj])

    X_sub = df.loc[sel, feat_cols].values
    y_sub = (df.loc[sel, "Class"] == cls).astype(int).values
    Xs = StandardScaler().fit_transform(X_sub)
    Xp = PCA(n_components=2, random_state=SEED).fit_transform(Xs)

    print(f"\n-- {cls} (نسبت واقعی≈{natural_ratio:.3f}, n_min={n_min}, n_maj={n_maj}) --")
    algos = run_algorithms(Xp, y_sub, n_clusters_guess=2)
    for algo_name, labels in algos.items():
        m = evaluate(y_sub, labels, Xp)
        m.update(class_=cls, algorithm=algo_name, natural_ratio=natural_ratio)
        binary_rows.append(m)
        print(f"   {algo_name:20s} ARI={m['ARI']:.3f} NMI={m['NMI']:.3f} FMI={m['FMI']:.3f} "
              f"Sil={m['Silhouette']}, clusters_found={m['n_clusters_found']}, noise={m['n_noise_points']}")

binary_df = pd.DataFrame(binary_rows)
binary_df.to_csv(OUT_BINARY, index=False)
print(f"\nذخیره شد: {OUT_BINARY}")

# ---------------- بخش ب: سناریوی کامل ۷ کلاسه ----------------
print("\n\n=== بخش ب: تمام ۷ گونه هم‌زمان ===")
sel_all = rng.choice(df.index.to_numpy(), size=min(SAMPLE_N, len(df)), replace=False)
X_all = df.loc[sel_all, feat_cols].values
y_all = LabelEncoder().fit_transform(df.loc[sel_all, "Class"].values)
Xs_all = StandardScaler().fit_transform(X_all)
Xp_all = PCA(n_components=2, random_state=SEED).fit_transform(Xs_all)

multi_rows = []
algos_all = run_algorithms(Xp_all, y_all, n_clusters_guess=7)
for algo_name, labels in algos_all.items():
    m = evaluate(y_all, labels, Xp_all)
    m.update(algorithm=algo_name)
    multi_rows.append(m)
    print(f"{algo_name:20s} ARI={m['ARI']:.3f} NMI={m['NMI']:.3f} FMI={m['FMI']:.3f} "
          f"clusters_found={m['n_clusters_found']}, noise={m['n_noise_points']}")

multi_df = pd.DataFrame(multi_rows)
multi_df.to_csv(OUT_MULTI, index=False)
print(f"\nذخیره شد: {OUT_MULTI}")
