# -*- coding: utf-8 -*-
"""
Geometric Blind-Spot Criterion -- addresses reviewer point #8 ("the theoretical
discussion is weak... why does K-means geometrically absorb the minority group
into the majority?").

Mathematical basis: K-means minimizes WCSS (Within-Cluster Sum of Squares).
Because the total sum of squares (TSS) around the grand mean is fixed for a
given dataset:
    TSS = WCSS + BCSS
minimizing WCSS is exactly equivalent to maximizing BCSS (Between-Cluster Sum
of Squares). So between any two competing partitions, K-means (as a locally
optimal algorithm, via Lloyd's algorithm) prefers the partition with the
larger BCSS.

For k=2 on the set (minority U majority), there are two natural competing
partitions:

  Partition A ("correct separation"): one cluster = minority, the other =
      majority.
      BCSS(A) = (n_min * n_maj / N) * D^2
      where D is the Euclidean distance between the minority mean and the
      majority mean. This is an exact identity for two-group BCSS, not an
      approximation.

  Partition B ("absorbed into the majority"): the majority is split into two
      sub-clusters by its own best internal partition (K-means(2) run on the
      majority points only); the minority (being small in number) effectively
      attaches to the nearest majority sub-cluster with little effect on the
      centroids.
      BCSS(B) ~ (n1*n2/n_maj) * Delta^2
      where Delta is the distance between the two optimal centroids of this
      majority-internal partition.

Prediction rule: the minority is "seen" by K-means(2) as an independent
cluster when:
      (n_min * n_maj / N) * D^2   >   BCSS(B)
and since typically n_min << n_maj (i.e. N ~ n_maj), this is approximately
equivalent to:
      minority_ratio = n_min / n_maj   >=   BCSS(B) / (n_maj * D^2)

Honest limitation: this is a geometric/heuristic argument, not a complete
mathematical proof. Lloyd's algorithm only reaches a local optimum (K-means is
in general NP-hard to solve globally), and in practice the algorithm's actual
partition may be neither exactly A nor exactly B but some mixture of the two.
This criterion is an approximate necessary condition, not a deterministic law.
"""
import numpy as np
import os
# repo/code/section5.4_dry_bean/script.py -> repo root is 2 levels up
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

IN = BASE + "/data/dry_bean_clean.csv"
OUT_DIR = BASE + "/results/section5.4_dry_bean"
OUT = OUT_DIR + "/theory_criterion_validation.csv"
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42

df = pd.read_csv(IN)
feat_cols = [c for c in df.columns if c != "Class"]
Xs = StandardScaler().fit_transform(df[feat_cols].values)
pca = PCA(n_components=2, random_state=SEED)
Xp = pca.fit_transform(Xs)  # the same space K-means actually sees

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
print(f"\nGeometric criterion prediction accuracy on the 7 real classes: {acc:.0%} ({result['prediction_correct'].sum()}/{len(result)})")
print(f"Saved: {OUT}")
