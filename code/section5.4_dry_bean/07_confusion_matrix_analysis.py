# -*- coding: utf-8 -*-
"""
Exact confusion-matrix analysis for the geometric criterion on the 60-cell
Dry Bean grid (the same grid as 06_theory_grid_validation.py).

Goal: answer whether the raw 88.3% accuracy reported in Section 5.4.4 of the
paper actually reflects the criterion's discriminative power, or simply
reflects strong class imbalance (only ~7% of cells are truly "isolated").
To do this, beyond overall accuracy, this script computes Recall and
Precision specifically for the "isolated" class, and compares against a
simple baseline ("always predict not-isolated").

This script is an extended version of 06_theory_grid_validation.py; the only
difference is that the true "observed_isolated" outcome of each cell (not
just whether the prediction was right or wrong) is also stored, which is
what makes the confusion-matrix computation possible.
"""
import numpy as np
import os
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

IN = BASE + "/data/dry_bean_clean.csv"
OUT = BASE + "/results/section5.4_dry_bean/grid_validation_with_confusion_matrix.csv"

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

        aris, correct_flags, pred_flags, truth_flags = [], [], [], []
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
            truth_flags.append(actually_isolated)

        rows.append(dict(
            cls=cls, minority_ratio_target=ratio, n_min=n_min,
            ari_mean=float(np.mean(aris)),
            criterion_predicts_isolated_rate=float(np.mean(pred_flags)),
            prediction_accuracy=float(np.mean(correct_flags)),
            observed_isolated_rate=float(np.mean(truth_flags)),
        ))

result = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
result.to_csv(OUT, index=False)

# --- confusion matrix over all 300 seed-cell instances ---
TP = FN = FP = TN = 0
for _, r in result.iterrows():
    n = 5
    n_obs_isolated = round(r["observed_isolated_rate"] * n)
    n_pred_isolated = round(r["criterion_predicts_isolated_rate"] * n)
    n_correct = round(r["prediction_accuracy"] * n)
    if n_pred_isolated == 0 or n_pred_isolated == n:
        # deterministic direction for this cell: no ambiguity in the split
        if n_obs_isolated == 0 or n_obs_isolated == n:
            # both deterministic -> simple case
            if n_pred_isolated == n_obs_isolated:
                if n_pred_isolated == n:
                    TP += n
                else:
                    TN += n
            else:
                if n_pred_isolated == n:
                    FP += n
                else:
                    FN += n
        else:
            # predicted deterministic, observed mixed
            if n_pred_isolated == n:
                TP += n_obs_isolated
                FP += (n - n_obs_isolated)
            else:
                FN += n_obs_isolated
                TN += (n - n_obs_isolated)
    else:
        # mixed prediction (only occurs for SEKER 0.30 in this grid); observed here is
        # deterministic (0), so all predicted-isolated instances are false positives
        # and all predicted-not-isolated instances are true negatives.
        FP += n_pred_isolated
        TN += (n - n_pred_isolated)

total = TP + FN + FP + TN
accuracy = (TP + TN) / total
recall = TP / (TP + FN) if (TP + FN) else float("nan")
precision = TP / (TP + FP) if (TP + FP) else float("nan")
baseline_acc = (TP + FN and (FN + TN) / total) or (TN / total)
trivial_baseline_acc = (total - (TP + FN)) / total  # always predict "not isolated"

print(f"Confusion matrix over {total} seed-cell instances:")
print(f"  TP={TP}  FN={FN}  FP={FP}  TN={TN}")
print(f"  Accuracy:  {accuracy:.1%}")
print(f"  Recall (isolated class):    {recall:.1%}")
print(f"  Precision (isolated class): {precision:.1%}")
print(f"  Trivial 'always not-isolated' baseline accuracy: {trivial_baseline_acc:.1%}")
print(f"\nSaved per-cell results to: {OUT}")
