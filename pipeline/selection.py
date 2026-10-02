"""Feature selection by XGBoost importance over repeated subject subsamples.

Paper: 50 trials; each fits XGBoost on the sections of a random 90% of subjects and
ranks all 855 features. The 50 rankings are stacked (855 x 50) and aggregated by
walking the ranks from best to worst: the earlier a feature appears in any trial,
the higher its final rank. The top 34 are kept.

select_features only sees the arrays it is given, so in-fold selection cannot leak.
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from xgboost import XGBClassifier


def _as_rng(rng):
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def rank_once(X, y, cfg, seed) -> np.ndarray:
    """Rank of every feature in one XGBoost fit, 0 = most important.

    Ties (typically the many zero-importance features) are broken at random so
    that low column indices are not favoured.
    """
    sel = cfg["selection"]
    model = XGBClassifier(**sel.get("xgb_params", {}), importance_type=sel["importance_type"],
                          random_state=int(seed), verbosity=0)
    model.fit(X, y)
    imp = model.feature_importances_
    tiebreak = np.random.default_rng(seed).permutation(len(imp))
    order = np.lexsort((tiebreak, -imp))          # best first
    ranks = np.empty(len(imp), dtype=int)
    ranks[order] = np.arange(len(imp))
    return ranks


def rank_matrix(X, y, groups, cfg, rng) -> np.ndarray:
    """(n_features, n_trials) ranks, one column per random subject subsample."""
    rng = _as_rng(rng)
    sel = cfg["selection"]
    subjects = np.unique(groups)
    n_take = max(2, int(round(sel["subject_frac"] * len(subjects))))
    # draw every subsample and seed first, so results do not depend on n_jobs
    trials = []
    for _ in range(sel["n_trials"]):
        for _attempt in range(100):
            chosen = rng.choice(subjects, size=n_take, replace=False)
            mask = np.isin(groups, chosen)
            if len(np.unique(y[mask])) == 2:
                break
        else:
            raise ValueError("could not draw a subject subsample containing both classes")
        trials.append((mask, int(rng.integers(2**31 - 1))))
    cols = Parallel(n_jobs=sel.get("n_jobs", 1))(
        delayed(rank_once)(X[mask], y[mask], cfg, seed) for mask, seed in trials)
    return np.stack(cols, axis=1)


def aggregate_first_appearance(R: np.ndarray) -> np.ndarray:
    """Full feature ordering from a rank matrix, literal reading of the paper.

    Walk rank positions r = 0, 1, ...; a feature enters the final list the first
    time it appears at position r in any trial. Features entering at the same r
    are ordered by how many trials rank them at r or better, then by mean rank,
    then by index.
    """
    n_feat = R.shape[0]
    first = R.min(axis=1)
    count_at_first = (R <= first[:, None]).sum(axis=1)
    mean_rank = R.mean(axis=1)
    return np.lexsort((np.arange(n_feat), mean_rank, -count_at_first, first))


def aggregate_mean_rank(R: np.ndarray) -> np.ndarray:
    """Full feature ordering by mean rank across trials (comparison variant)."""
    return np.lexsort((np.arange(R.shape[0]), R.mean(axis=1)))


AGGREGATORS = {"first_appearance": aggregate_first_appearance, "mean_rank": aggregate_mean_rank}


def select_features(X_train, y_train, groups_train, cfg, rng, return_ranks=False, aggregation=None):
    """Indices of the n_keep selected features, from training-fold arrays only.

    Returns idx, or (idx, R) with R the (n_features, n_trials) rank matrix.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    groups_train = np.asarray(groups_train)
    R = rank_matrix(X_train, y_train, groups_train, cfg, rng)
    agg = AGGREGATORS[aggregation or cfg["selection"]["aggregation"]]
    idx = agg(R)[: cfg["selection"]["n_keep"]]
    return (idx, R) if return_ranks else idx
