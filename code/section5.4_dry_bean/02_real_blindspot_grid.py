# -*- coding: utf-8 -*-
"""
"Real-data" version of the Blind Spot Map sensitivity grid.

Goal: address reviewer point #2 (all experiments in the paper are on
synthetic data).

Method: instead of generating synthetic data, we use the real Dry Bean
dataset. Each of the 7 bean species is treated in turn as the "real minority
group" (its statistical separation from the rest is fixed and natural, since
it comes from real data and cannot be controlled). We then subsample from
that same class to vary its population ratio over a range similar to the
paper's main grid (1% to 30%).

This is a "partial control", not a full control as in the simulation: only
minority_ratio is manipulated, while separation remains fixed and natural to
that real class. This is exactly the limitation that should be (and is)
noted in the paper.

Output: for each (class, target ratio), over 5 independent random
subsamples, PCA(2)+KMeans(2) is run and ARI against the true label is
computed.
"""
import os
# repo/code/section5.4_dry_bean/script.py -> repo root is 2 levels up
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

IN = BASE + "/data/dry_bean_clean.csv"
OUT_DIR = BASE + "/results/section5.4_dry_bean"
OUT = OUT_DIR + "/real_blindspot_grid_results.csv"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN)
feat_cols = [c for c in df.columns if c != "Class"]
classes = sorted(df["Class"].unique())

TARGET_RATIOS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.30]
SEEDS = [42, 43, 44, 45, 46]
N_TOTAL = 4000  # fixed sample size per draw, for speed and grid consistency

# natural multivariate separation of each class (multivariate Cohen's d, fixed and not manipulable)
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
            skipped.append((cls, ratio, "not enough real samples available for this class"))
            print(f"[{done}/{total_cells}] {cls} ratio={ratio:.2f} -> SKIP (n_min={n_min} > available={n_cls_available})")
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
              f"ARI={np.mean(aris):.3f}\u00b1{np.std(aris):.3f}")

result = pd.DataFrame(rows)
result.to_csv(OUT, index=False)
print(f"\nSaved: {OUT}")
print(f"Cells completed: {len(result)} of {total_cells}")
if skipped:
    print(f"Cells skipped (insufficient real samples): {len(skipped)}")
    for s in skipped:
        print("  -", s)
