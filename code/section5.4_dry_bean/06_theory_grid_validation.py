# -*- coding: utf-8 -*-
"""
Stronger test of the geometric criterion: instead of testing only a single
point (the natural ratio) per class, we recompute the criterion across every
cell of the ratio-based grid (the same subsamples as
02_real_blindspot_grid.py) to see whether the predicted threshold actually
coincides with the point where ARI starts rising. This test is considerably
more stringent than 05_theory_criterion.py because it tests both
"recoverable" and "unrecoverable" cases across the full range of ratios.
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
OUT = OUT_DIR + "/theory_criterion_grid_validation.csv"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN)
feat_cols = [c for c in df.columns if c != "Class"]
classes = sorted(df["Class"].unique())

TARGET_RATIOS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.30]
SEEDS = [42, 43, 44, 45, 46]
N_TOTAL = 4000

rows = []
for cls in classes:
    cls_idx = df.index[df["Class"] == cls].to_numpy()
    other_idx = df.index[df["Class"] != cls].to_numpy()
    n_cls_available = len(cls_idx)

    for ratio in TARGET_RATIOS:
        n_min = int(round(ratio * N_TOTAL))
        n_maj = N_TOTAL - n_min
        if n_min > n_cls_available or n_min < 2:
            continue

        aris, correct_flags, pred_flags = [], [], []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            sel_min = rng.choice(cls_idx, size=n_min, replace=False)
            sel_maj = rng.choice(other_idx, size=n_maj, replace=False)
            sel = np.concatenate([sel_min, sel_maj])

            X_sub = df.loc[sel, feat_cols].values
            y_sub = (df.loc[sel, "Class"] == cls).astype(int).values
            Xs_sub = StandardScaler().fit_transform(X_sub)
            Xp_sub = PCA(n_components=2, random_state=seed).fit_transform(Xs_sub)

            X_min, X_maj = Xp_sub[y_sub == 1], Xp_sub[y_sub == 0]
            c_min, c_maj = X_min.mean(axis=0), X_maj.mean(axis=0)
            D2 = float(np.sum((c_min - c_maj) ** 2))
            N = len(Xp_sub)
            bcss_isolate = (n_min * n_maj / N) * D2

            km_maj = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(X_maj)
            l = km_maj.labels_
            n1, n2 = int((l == 0).sum()), int((l == 1).sum())
            c1, c2 = X_maj[l == 0].mean(axis=0), X_maj[l == 1].mean(axis=0)
            delta2 = float(np.sum((c1 - c2) ** 2))
            bcss_majsplit = (n1 * n2 / n_maj) * delta2 if n_maj > 0 else 0.0

            predicted_isolated = bcss_isolate > bcss_majsplit

            km_pool = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(Xp_sub)
            ari = adjusted_rand_score(y_sub, km_pool.labels_)
            ct = pd.crosstab(y_sub, km_pool.labels_)
            if 1 in ct.index:
                min_cluster = ct.loc[1].idxmax()
                purity = ct.loc[1, min_cluster] / ct[min_cluster].sum()
                actually_isolated = purity > 0.5
            else:
                actually_isolated = False

            aris.append(ari)
            pred_flags.append(predicted_isolated)
            correct_flags.append(predicted_isolated == actually_isolated)

        rows.append(dict(
            cls=cls, minority_ratio_target=ratio, n_min=n_min,
            ari_mean=float(np.mean(aris)),
            criterion_predicts_isolated_rate=float(np.mean(pred_flags)),
            prediction_accuracy=float(np.mean(correct_flags)),
        ))
        print(f"{cls:10s} ratio={ratio:.2f} ARI={np.mean(aris):.3f} "
              f"criterion_predicts_isolated={np.mean(pred_flags):.0%} "
              f"match_rate={np.mean(correct_flags):.0%}")

result = pd.DataFrame(rows)
result.to_csv(OUT, index=False)
overall_acc = result["prediction_accuracy"].mean()
print(f"\nOverall geometric criterion accuracy across the full grid ({len(result)} cells): {overall_acc:.1%}")
print(f"Saved: {OUT}")

print("\nNote: this raw accuracy figure should be read together with the")
print("confusion-matrix analysis in 07_confusion_matrix_analysis.py, which")
print("shows that a large share of this accuracy reflects class imbalance")
print("(few cells are truly 'isolated') rather than discriminative power.")
