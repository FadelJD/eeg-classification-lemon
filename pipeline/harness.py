"""Subject-level cross-validation, with feature selection pooled or inside each fold."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from pipeline.confound import residualise
from pipeline.models import make_model
from pipeline.selection import select_features

METRICS = ("bal_acc", "auc", "acc")
LEVELS = ("section", "subject")


def check_disjoint(train_groups, test_groups):
    overlap = set(np.unique(train_groups)) & set(np.unique(test_groups))
    assert not overlap, f"subjects in both train and test: {sorted(overlap)[:5]}"


def default_splitter(cfg):
    return lambda seed: StratifiedGroupKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True,
                                             random_state=seed)


def metrics(y_true, prob, threshold) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(prob) > threshold).astype(int)  # paper: c = 1 if y > theta
    both = len(np.unique(y_true)) == 2
    return {"bal_acc": balanced_accuracy_score(y_true, pred),
            "auc": roc_auc_score(y_true, prob) if both else np.nan,
            "acc": accuracy_score(y_true, pred)}


def subject_level(y, prob, groups) -> pd.DataFrame:
    """One row per subject: label and mean probability over that subject's rows."""
    df = pd.DataFrame({"g": np.asarray(groups), "y": np.asarray(y), "p": np.asarray(prob)})
    return df.groupby("g", sort=True).agg(y=("y", "first"), p=("p", "mean"))


def _subject_labels(y, groups):
    return pd.Series(np.asarray(y)).groupby(np.asarray(groups)).first()


def evaluate(X, y, groups, model, cfg, selection=None, *, ages=None, pooled_idx=None,
             X_long=None, long_groups=None, splitter_factory=None, selection_cache=None,
             return_predictions=False):
    """One row per (seed, fold) with section- and subject-level metrics.

    model: a name from pipeline.models or an sklearn estimator.
    selection: None (all features), "pooled" (selected once on all rows before
    splitting, as the paper does) or "in_fold" (selected on each training fold).
    selection_cache: optional dict shared across models so in-fold selection runs
    once per (seed, fold).
    """
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    cv = cfg["cv"]
    thr = cv["threshold"]
    est = make_model(model, cfg) if isinstance(model, str) else model
    splitter_factory = splitter_factory or default_splitter(cfg)
    selection_cache = {} if selection_cache is None else selection_cache
    residual = cv.get("residualise_age", False)
    use_long = cv.get("test_representation", "sections") == "long"
    if residual and ages is None:
        raise ValueError("cv.residualise_age needs ages")
    if use_long:
        if X_long is None:
            raise ValueError("cv.test_representation=long needs X_long and long_groups")
        long_groups = np.asarray(long_groups)
        subj_y = _subject_labels(y, groups)
        subj_age = pd.Series(np.asarray(ages)).groupby(groups).first() if ages is not None else None

    if selection == "pooled" and pooled_idx is None:
        pooled_idx = select_features(X, y, groups, cfg, cfg["selection"]["seed"])
    elif selection not in (None, "pooled", "in_fold"):
        raise ValueError(f"selection must be None, 'pooled' or 'in_fold', got {selection!r}")

    rows, preds = [], []
    for seed in cv["seeds"]:
        for fold, (tr, te) in enumerate(splitter_factory(seed).split(X, y, groups)):
            check_disjoint(groups[tr], groups[te])
            X_tr, y_tr, g_tr = X[tr], y[tr], groups[tr]
            if use_long:
                keep = np.isin(long_groups, np.unique(groups[te]))
                g_te = long_groups[keep]
                X_te, y_te = X_long[keep], subj_y.loc[g_te].to_numpy()
                age_te = subj_age.loc[g_te].to_numpy() if subj_age is not None else None
            else:
                X_te, y_te, g_te = X[te], y[te], groups[te]
                age_te = ages[te] if ages is not None else None
            if residual:
                X_tr, X_te = residualise(X_tr, np.asarray(ages)[tr], X_te, age_te)

            if selection == "in_fold":
                key = (seed, fold)
                if key not in selection_cache:
                    rng = np.random.default_rng([cfg["selection"]["seed"], seed, fold])
                    selection_cache[key] = select_features(X_tr, y_tr, g_tr, cfg, rng)
                idx = selection_cache[key]
            elif selection == "pooled":
                idx = pooled_idx
            else:
                idx = np.arange(X.shape[1])

            m = clone(est)
            if "clf__random_state" in m.get_params():
                m.set_params(clf__random_state=seed)
            m.fit(X_tr[:, idx], y_tr)
            prob = m.predict_proba(X_te[:, idx])[:, 1]

            subj = subject_level(y_te, prob, g_te)
            row = {"seed": seed, "fold": fold, "n_train_subjects": len(np.unique(g_tr)),
                   "n_test_subjects": len(subj), "n_features": len(idx)}
            row.update({f"section_{k}": v for k, v in metrics(y_te, prob, thr).items()})
            row.update({f"subject_{k}": v for k, v in metrics(subj["y"], subj["p"], thr).items()})
            rows.append(row)
            if return_predictions:
                preds.append(pd.DataFrame({"seed": seed, "fold": fold, "subject": g_te,
                                           "y": y_te, "p": prob}))
    out = pd.DataFrame(rows)
    return (out, pd.concat(preds, ignore_index=True)) if return_predictions else out


def permute_subject_labels(y, groups, rng):
    """Shuffle labels between subjects; every row of a subject gets its subject's new label."""
    labels = _subject_labels(y, groups)
    shuffled = pd.Series(rng.permutation(labels.to_numpy()), index=labels.index)
    return shuffled.loc[np.asarray(groups)].to_numpy()


def permutation_test(X, y, groups, model, cfg, selection=None, n_permutations=None,
                     rng=None, **kwargs) -> dict:
    """p = (#{null >= real} + 1) / (n + 1), labels shuffled at subject level.

    Pooled selection is recomputed for every permutation, otherwise the null
    would use features chosen with the real labels.
    """
    perm = cfg["permutation"]
    n = perm["n_permutations"] if n_permutations is None else n_permutations
    rng = np.random.default_rng(rng)
    col = f"{perm['level']}_{perm['metric']}"
    cfg = copy.deepcopy(cfg)
    cfg["cv"]["seeds"] = cfg["cv"]["seeds"][: perm.get("n_seeds", len(cfg["cv"]["seeds"]))]
    real = evaluate(X, y, groups, model, cfg, selection, **kwargs)[col].mean()
    kwargs = {k: v for k, v in kwargs.items() if k not in ("pooled_idx", "selection_cache")}
    null = np.array([
        evaluate(X, permute_subject_labels(y, groups, rng), groups, model, cfg, selection,
                 **kwargs)[col].mean()
        for _ in range(n)])
    return {"metric": col, "real": float(real), "null": null,
            "p": float((np.sum(null >= real) + 1) / (n + 1)), "n_permutations": n}
