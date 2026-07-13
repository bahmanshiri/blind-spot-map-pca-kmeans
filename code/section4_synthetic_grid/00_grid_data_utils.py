# -*- coding: utf-8 -*-
"""
Sensitivity grid dataset generator -- Section 3.2 of the paper.

Dataset design (per Section 3.2):
  - Exactly two groups: majority and minority.
  - Two controlled parameters:
      * minority_ratio: minority population ratio (9 levels, 1% to 30%)
      * separation: statistical separation of the minority from the majority
        on the "transaction-to-balance ratio" feature
        (transaction_to_balance_ratio), in units of the majority's standard
        deviation (9 levels, 0.5x to 12x).
  - Numeric features: age, account_balance, transaction_to_balance_ratio
    (only the third feature is shifted between the two groups; the other two
    are identically-distributed noise, included only to make PCA(2) meaningful
    across more than one dimension).
  - Optional categorical feature: branch (160 branches, independent of group
    -- i.e. carries no signal about the minority group) -- added via one-hot
    only in the relevant experiments (Section 4.3), exactly per Section 4.1.
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
    Generate a parametric two-group dataset.

    minority_ratio : minority population ratio (e.g. 0.05 for 5%)
    separation     : mean separation of the minority from the majority on
                      transaction_to_balance_ratio, in units of the
                      majority's standard deviation (which is normalized to 1.0)
    seed           : independent seed for this run
    with_branch    : if True, also adds the 160-level categorical "branch"
                      column (independent of the group label -- purely a
                      dimensionality diluter)
    """
    rng = np.random.default_rng(seed)

    n_min = int(round(minority_ratio * n_total))
    n_maj = n_total - n_min
    assert n_min >= 1 and n_maj >= 1

    label = np.array([0] * n_maj + [1] * n_min)  # 0=majority, 1=minority

    # noise features (identically distributed between groups, no group signal)
    age = rng.normal(40, 12, n_total)
    account_balance = rng.lognormal(mean=9.0, sigma=0.8, size=n_total)

    # signal-carrying feature: transaction_to_balance_ratio
    # majority: mean=0, std=1 (this std is the basis for "separation in units of")
    # minority: mean=separation (in units of majority std), same std
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
        df["branch"] = rng.integers(0, n_branches, n_total)  # independent of group

    # shuffle row order (does not affect the result, just avoids any ordering artifact)
    df = df.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
    return df


def run_pipeline(df, k, with_branch, seed, return_variance=False, scale_branch=False):
    """
    Base pipeline, exactly per Section 4.1:
    StandardScaler (numeric features) -> encoding (one-hot for branch)
    -> PCA(2) -> K-means(n_init=10) -> ARI against the hidden label.

    branch (when present) is added as raw one-hot (no additional scaling),
    consistent with the gender encoding in Section 4.1; numeric features are
    normalized with StandardScaler.
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
