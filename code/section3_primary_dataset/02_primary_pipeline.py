# -*- coding: utf-8 -*-
"""
Primary dataset pipeline -- Section 5.3 of the paper (Table 2).

Flow: load/clean -> encode -> PCA(2) -> select k (Silhouette) ->
validate clusters with a decision tree -> recover the minority group (ARI).
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
gen = import_module("01_generate_dataset")

# repo/code/section3_primary_dataset/script.py -> repo root is 2 levels up
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE + "/data"
RESULTS_DIR = BASE + "/results/section3"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
N_TOTAL = 120_000

# 1) generate and clean
raw = gen.generate_primary_dataset(n_total=N_TOTAL, seed=SEED)
clean, removed_frac = gen.clean_dataset(raw)
clean.to_csv(f"{DATA_DIR}/primary_dataset_clean.csv", index=False)

# 2) encoding: numeric -> StandardScaler, gender -> ordinal, branch -> one-hot
numeric_cols = ["age", "account_balance", "transaction_to_balance_ratio", "avg_monthly_transactions"]
Xnum = StandardScaler().fit_transform(clean[numeric_cols].values)
Xgender = clean[["gender"]].values.astype(float)
Xbranch = pd.get_dummies(clean["branch"], prefix="branch").values.astype(float)
X = np.hstack([Xnum, Xgender, Xbranch])
n_dims = X.shape[1]

# 3) PCA(2)
pca = PCA(n_components=2, random_state=SEED)
Xp = pca.fit_transform(X)
var_pc1, var_pc2 = pca.explained_variance_ratio_[:2]
var_total = var_pc1 + var_pc2

# 4) select k by maximum Silhouette (on a subsample, for speed)
sil_sample_idx = np.random.RandomState(SEED).choice(len(Xp), size=min(20000, len(Xp)), replace=False)
sil_scores = {}
for k in [2, 3, 4]:
    km_k = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xp)
    sil_scores[k] = silhouette_score(Xp[sil_sample_idx], km_k.labels_[sil_sample_idx])

k_statistical = max(sil_scores, key=sil_scores.get)
k_business = 3  # per Section 5.3: final choice for business interpretability

# 5) final clustering with the business k (k=3), used for Fig. 3
km_business = KMeans(n_clusters=k_business, n_init=10, random_state=SEED).fit(Xp)
sil_business = silhouette_score(Xp[sil_sample_idx], km_business.labels_[sil_sample_idx])

# 6) validate clusters with a decision tree (predict cluster label from raw features)
X_train, X_test, y_train, y_test = train_test_split(
    X, km_business.labels_, test_size=0.3, random_state=SEED, stratify=km_business.labels_
)
tree = DecisionTreeClassifier(random_state=SEED, max_depth=8)
tree.fit(X_train, y_train)
tree_test_acc = tree.score(X_test, y_test)
cv_scores = cross_val_score(tree, X, km_business.labels_, cv=5)
cv_mean, cv_std = cv_scores.mean(), cv_scores.std()

# 7) recover the minority (anomalous) group with k=2
km2 = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit(Xp)
true_minority = (clean["segment"] == "anomalous").astype(int).values
ari_k2 = adjusted_rand_score(true_minority, km2.labels_)

# --- output: Table 2 ---
table2 = pd.DataFrame([
    {"Metric": "Dimensions after encoding", "Value": f"{n_dims} ({len(clean):,} samples)"},
    {"Metric": "Explained variance PC1+PC2", "Value": f"{var_total:.3f} (PC1={var_pc1:.3f}, PC2={var_pc2:.3f})"},
    {"Metric": "Purely statistical suggested k (max Silhouette)", "Value": f"{k_statistical} (Silhouette={sil_scores[k_statistical]:.3f})"},
    {"Metric": "Final selected k (business interpretability)", "Value": f"{k_business} (Silhouette={sil_business:.3f})"},
    {"Metric": "Decision-tree accuracy (cluster validation, test set)", "Value": f"{tree_test_acc:.3f}"},
    {"Metric": "5-fold cross-validation accuracy", "Value": f"{cv_mean:.3f} \u00b1 {cv_std:.3f}"},
    {"Metric": "Minority-group recovery ARI (k=2)", "Value": f"{ari_k2:.3f}"},
    {"Metric": "Records removed (duplicates/outliers)", "Value": f"{removed_frac:.4f} ({N_TOTAL - len(clean):,} of {N_TOTAL:,})"},
])
table2.to_csv(f"{RESULTS_DIR}/table2_primary_dataset_metrics.csv", index=False)

pd.set_option("display.width", 160)
print("--- Table 2: Primary dataset pipeline metrics ---\n")
print(table2.to_string(index=False))
print(f"\nSaved: {DATA_DIR}/primary_dataset_clean.csv ({len(clean):,} rows)")
print(f"Saved: {RESULTS_DIR}/table2_primary_dataset_metrics.csv")
