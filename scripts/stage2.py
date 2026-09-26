"""Stage 2: runs on the real cache. One step per call, in order:

    python scripts/stage2.py summary       # 0: integrity + results/cache_summary.md
    python scripts/stage2.py baselines     # 1: logreg, rf on all 855
    python scripts/stage2.py in_fold       # 2: in-fold selection, all models, sections + long
    python scripts/stage2.py pooled        # 3: pooled selection (leaks, mirrors the paper)
    python scripts/stage2.py permutation   # 4: best pooled pair, subject-level label shuffles
    python scripts/stage2.py table         # 5: results/table.csv (add --check to diff at 1e-3)
    python scripts/stage2.py age_clusters  # 6.1
    python scripts/stage2.py residualised  # 6.2
    python scripts/stage2.py stability     # 6.3
    python scripts/stage2.py aggregation   # 6.4
    python scripts/stage2.py sectioning    # 6.5 (needs cache/subjects/)
    python scripts/stage2.py interp_sensitivity  # 6.6 (added): subjects without interpolation

Per-fold results go to results/stage2/<step>.csv; every step appends to results/timings.csv.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
import warnings
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import features as F  # noqa: E402
from pipeline.harness import evaluate, permute_subject_labels  # noqa: E402
from pipeline.models import FACTORIES  # noqa: E402
from pipeline.selection import (aggregate_first_appearance, aggregate_mean_rank,  # noqa: E402
                                select_features)

log = logging.getLogger("stage2")
RES = ROOT / "results"
S2 = RES / "stage2"
CACHE = ROOT / "cache"
MODELS = list(FACTORIES)            # 'paper' is xgb with the same parameters, not rerun
REPS = ("sections", "long")
YOUNG_MAX_MIDPOINT = 45.0           # bins 20-25..35-40 vs 55-60..75-80
MEASURE = "coh, 5 bands (paper 3.3), EC, 19 ch, interpolated missing"
LEAK = "selection on all subjects before CV: leaks test labels (mirrors paper)"
BUDGET_MIN = 45.0


# ---------------------------------------------------------------- helpers

def cfg_real():
    cfg = F.load_config(ROOT / "config.yaml")
    cfg["selection"]["n_jobs"] = -1
    return cfg


def load(cache_dir=CACHE):
    return F.load_cache(cache_dir)


class timed:
    """Append (step, seconds, started_utc, notes) to results/timings.csv."""

    def __init__(self, step, notes=""):
        self.step, self.notes = step, notes

    def __enter__(self):
        self.t0 = time.time()
        self.started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        log.info("step %s: start", self.step)
        return self

    def __exit__(self, *exc):
        dt = time.time() - self.t0
        path = RES / "timings.csv"
        row = pd.DataFrame([{"step": self.step, "seconds": round(dt, 1),
                             "started_utc": self.started,
                             "status": "failed" if exc[0] else "ok", "notes": self.notes}])
        row.to_csv(path, mode="a", header=not path.exists(), index=False)
        log.info("step %s: %.1f s", self.step, dt)
        return False


def run_eval(c, model, cfg, selection, rep, **kw):
    cfg = copy.deepcopy(cfg)
    cfg["cv"]["test_representation"] = rep
    res = evaluate(c["X"], c["y"], c["subjects"], model, cfg, selection, ages=c["ages"],
                   X_long=c["X_long"], long_groups=c["long_subjects"], **kw)
    res.insert(0, "representation", rep)
    res.insert(0, "selection", selection or "none")
    res.insert(0, "model", model)
    return res


def save(df, name):
    S2.mkdir(parents=True, exist_ok=True)
    df.to_csv(S2 / f"{name}.csv", index=False)


def best_pair(df, level="subject", metric="auc"):
    """(model, representation) with the highest mean over seeds of the per-seed fold mean."""
    col = f"{level}_{metric}"
    per_seed = df.groupby(["model", "representation", "seed"])[col].mean()
    return per_seed.groupby(["model", "representation"]).mean().idxmax()


def age_cluster(ages):
    return np.where(np.asarray(ages) < YOUNG_MAX_MIDPOINT, "young", "older")


def subset(c, keep_subjects):
    rows = np.isin(c["subjects"], keep_subjects)
    lrows = np.isin(c["long_subjects"], keep_subjects)
    return {"X": c["X"][rows], "y": c["y"][rows], "subjects": c["subjects"][rows],
            "ages": c["ages"][rows], "X_long": c["X_long"][lrows],
            "long_subjects": c["long_subjects"][lrows]}


def md(df, index=True):
    """Markdown table without the optional tabulate dependency."""
    d = df.reset_index() if index else df
    cols = [str(x) for x in d.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(out)


def subject_table(c):
    return (pd.DataFrame({"subject": c["subjects"], "y": c["y"], "age": c["ages"]})
            .groupby("subject").agg(y=("y", "first"), age=("age", "first"),
                                    sections=("y", "size")))


# ---------------------------------------------------------------- 0

def check_cache(c, cfg):
    """Integrity assertions; raises AssertionError with the reason (hard stop 1)."""
    X, y, s, a = c["X"], c["y"], c["subjects"], c["ages"]
    assert X.shape[1] == F.n_features(cfg) == 855, f"X has {X.shape[1]} columns"
    assert not np.isnan(X).any(), "NaN in X"
    assert not np.isnan(c["X_long"]).any(), "NaN in X_long"
    assert (X.std(axis=0) > 0).all(), f"{int((X.std(axis=0) == 0).sum())} constant columns"
    assert len(y) == len(s) == len(a) == X.shape[0], "row counts disagree"
    subj = subject_table(c)
    assert set(np.unique(y)) <= {0, 1}
    assert (pd.Series(y).groupby(s).nunique() == 1).all(), "a subject has two labels"
    assert (pd.Series(a).groupby(s).nunique() == 1).all(), "a subject has two ages"
    ls = list(c["long_subjects"])
    assert len(ls) == len(set(ls)) == c["X_long"].shape[0], "X_long rows not one per subject"
    assert set(subj.index) == set(ls), "subjects without exactly one X_long row"
    pos = cfg["participants"]["positive_class"]
    neg = next(v for v in cfg["participants"]["sex_coding"].values() if v != pos)
    want = {1: cfg["sections"]["per_class"][pos], 0: cfg["sections"]["per_class"][neg]}
    off = subj[(subj["sections"] - subj["y"].map(want)).abs() > 1]
    assert off.empty, f"section counts off config for {list(off.index)[:5]}"
    return subj


def bridged(c, names, thr=0.999):
    """Subjects with sections at coherence >= thr, and the pairs involved (all bands)."""
    X, subs, y = c["X"], c["subjects"], c["y"]
    rows = []
    for s in pd.unique(subs[(X >= thr).any(axis=1)]):
        m = subs == s
        cols = np.where((X[m] >= thr).any(axis=0))[0]
        pairs = sorted({names[i].split(":")[1] for i in cols})
        rows.append({"subject": s, "sex": "female" if y[m][0] == 1 else "male",
                     "sections_hit": int((X[m] >= thr).any(axis=1).sum()),
                     "sections": int(m.sum()), "pairs": " ".join(pairs)})
    return pd.DataFrame(rows)


def step_summary(args):
    cfg = cfg_real()
    c = load()
    with timed("0_summary"):
        subj = check_cache(c, cfg)
        labels = F.load_labels(cfg)
        bins = {v: k for k, v in cfg["age_midpoints"].items()}
        subj["sex"] = subj["y"].map({1: "female", 0: "male"})
        subj["age_bin"] = subj["age"].map(bins)
        subj["cluster"] = age_cluster(subj["age"])
        skipped = pd.read_csv(CACHE / "skipped.csv")
        X = c["X"]
        hi = np.where(X.max(axis=0) >= 0.999)[0]
        names = F.feature_names(cfg)
        interp = []
        sdir = CACHE / "subjects" / F.feature_hash(cfg)
        if sdir.is_dir():
            for f in sorted(sdir.glob("*.npz")):
                ch = [str(x) for x in np.load(f)["interpolated"]]
                if ch:
                    interp.append((f.stem, " ".join(ch)))
        # labels in the cache agree with the participants CSV
        lab = labels.loc[subj.index]
        assert (lab["y"].values == subj["y"].values).all(), "cache labels disagree with CSV"
        assert np.allclose(lab["age"].values, subj["age"].values), "cache ages disagree with CSV"

        L = ["# Cache summary", "",
             f"Cache: `cache/` as uploaded, feature hash `{F.feature_hash(cfg)}`. "
             f"Measure: {MEASURE}. All integrity assertions passed "
             "(855 columns, no NaN, no constant column, aligned lengths, one `X_long` row per "
             "subject, labels and ages constant within subject and equal to the participants CSV, "
             "section counts equal to config).", "",
             "## Subjects and sections per class", "",
             md(pd.DataFrame({"subjects": subj.groupby("sex").size(),
                              "sections": subj.groupby("sex")["sections"].sum(),
                              "sections per subject (config)":
                                  pd.Series(cfg["sections"]["per_class"])})), "",
             f"Total: {len(subj)} subjects, {len(c['y'])} sections.", "",
             "## Sections per subject", "",
             md(subj.groupby(["sex", "sections"]).size().rename("subjects").to_frame()),
             "", "Available complete sections per recording in the per-subject cache: "
             + (", ".join(f"{k}: {v}" for k, v in pd.Series(
                 [np.load(f)["X"].shape[0] for f in sorted(sdir.glob('*.npz'))]
             ).value_counts().sort_index().items()) if sdir.is_dir() else "cache/subjects absent")
             + " (sections: subjects).", "",
             "## Age bins per class", "",
             md(pd.crosstab(subj["age_bin"], subj["sex"], margins=True)), "",
             "## Sex ratio within age clusters", "",
             f"Split at bin midpoint {YOUNG_MAX_MIDPOINT:g}: young = bins 20–25 to 35–40, "
             "older = 55–60 to 75–80 (LEMON has no 40–55 bins).", "",
             md(pd.crosstab(subj["cluster"], subj["sex"]).assign(
                 female_share=lambda d: (d["female"] / d.sum(axis=1)).round(3))), "",
             "## skipped.csv", "", md(skipped, index=False), "",
             f"## Interpolated channels ({len(interp)} subjects)", "",
             "Subjects whose missing channels were rebuilt by spherical-spline interpolation:", "",
             ", ".join(f"{s} ({ch})" for s, ch in interp) or "none", "",
             "## Feature range", "",
             f"min {X.min():.4f}, max {X.max():.4f}. Columns reaching ≥ 0.999 in some section: "
             f"{len(hi)}.", "",
             md(bridged(c, names), index=False), "",
             "Each of these subjects has one neighbouring channel pair (or a small cluster) at "
             "coherence ≈ 1 in most bands and sections: the signature of electrode "
             "bridging (two electrodes shorted by gel), not of interpolation (only one of them "
             "had an interpolated channel). They are kept; see notes/DECISIONS.md D5.", ""]
        (RES / "cache_summary.md").write_text("\n".join(L))
    print((RES / "cache_summary.md").read_text())


# ---------------------------------------------------------------- 1-3

def step_baselines(args):
    cfg, c = cfg_real(), load()
    with timed("1_baselines", "logreg, rf on 855; sections"):
        save(pd.concat([run_eval(c, m, cfg, None, "sections") for m in ("logreg", "rf")]),
             "baselines")


def step_in_fold(args):
    cfg, c = cfg_real(), load()
    shared = {}  # one selection per (seed, fold), reused by every model and both representations
    frames = []
    with timed("2_in_fold", f"{len(cfg['cv']['seeds'])} seeds x {cfg['cv']['n_splits']} folds, "
                            f"{cfg['selection']['n_trials']} trials, models {MODELS}"):
        for rep in REPS:
            for m in MODELS:
                log.info("in_fold %s %s", m, rep)
                frames.append(run_eval(c, m, cfg, "in_fold", rep, selection_cache=shared))
                save(pd.concat(frames), "in_fold")
        np.savez(RES / "selected_in_fold.npz",
                 **{f"seed{s}_fold{f}": idx for (s, f), idx in sorted(shared.items())})


def step_pooled(args):
    cfg, c = cfg_real(), load()
    with timed("3_pooled", f"one selection on all subjects, {cfg['selection']['n_trials']} trials"):
        idx, R = select_features(c["X"], c["y"], c["subjects"], cfg, cfg["selection"]["seed"],
                                 return_ranks=True)
        np.save(RES / "rank_matrix_pooled.npy", R)
        chans = cfg["signal"]["channels"]
        rows, cols = F.triu_pairs(len(chans))
        bands = list(cfg["connectivity"]["bands"])
        n_pairs = len(rows)
        pd.DataFrame({"rank": np.arange(1, len(idx) + 1), "index": idx,
                      "band": [bands[i // n_pairs] for i in idx],
                      "channel_a": [chans[rows[i % n_pairs]] for i in idx],
                      "channel_b": [chans[cols[i % n_pairs]] for i in idx]}
                     ).to_csv(RES / "selected_features_pooled.csv", index=False)
        frames = [run_eval(c, m, cfg, "pooled", rep, pooled_idx=idx)
                  for rep in REPS for m in MODELS]
        save(pd.concat(frames), "pooled")


# ---------------------------------------------------------------- 4

def step_permutation(args):
    cfg, c = cfg_real(), load()
    pooled = pd.read_csv(S2 / "pooled.csv")
    model, rep = best_pair(pooled)
    perm = cfg["permutation"]
    cfg["cv"]["seeds"] = cfg["cv"]["seeds"][: perm["n_seeds"]]
    rng = np.random.default_rng(cfg["selection"]["seed"])
    reductions = []

    def score(y):
        # pooled selection recomputed from these labels, same settings for real and null
        return run_eval({**c, "y": y}, model, cfg, "pooled", rep)["subject_auc"].mean()

    with timed("4_permutation") as t:
        t0 = time.time()
        real = score(c["y"])
        first_null = [score(permute_subject_labels(c["y"], c["subjects"], rng))]
        per = (time.time() - t0) / 2
        n = perm["n_permutations"]
        if per * n / 60 > BUDGET_MIN:          # brief: seeds (already 1) -> permutations -> trials
            n = 50
            reductions.append("permutations 100 -> 50")
        if per * n / 60 > BUDGET_MIN:
            cfg["selection"]["n_trials"] = 30
            reductions.append("trials 50 -> 30 (real score recomputed with 30)")
            real = score(c["y"])
            rng = np.random.default_rng(cfg["selection"]["seed"])
            first_null = []
        proj = per * n / 60 * (0.65 if cfg["selection"]["n_trials"] == 30 else 1)
        log.info("permutation: %.1f s each, n=%d, projected %.0f min, reductions %s",
                 per, n, proj, reductions)
        if proj > 90:
            raise SystemExit("HARD STOP 3: permutation projected past 90 min after reductions")
        null = first_null
        while len(null) < n:
            null.append(score(permute_subject_labels(c["y"], c["subjects"], rng)))
            if len(null) % 10 == 0:
                log.info("permutation %d/%d", len(null), n)
        null = np.array(null)
        p = float((np.sum(null >= real) + 1) / (n + 1))
        t.notes = f"{model}/{rep}, n={n}, {per:.1f} s per permutation" + (
            "; reduced: " + ", ".join(reductions) if reductions else "")
    np.save(RES / "perm_null.npy", null)
    (S2 / "permutation.json").write_text(json.dumps(
        {"model": model, "representation": rep, "selection": "pooled", "level": "subject",
         "metric": "auc", "real": float(real), "p": p, "n_permutations": int(n),
         "n_seeds": len(cfg["cv"]["seeds"]), "n_trials": cfg["selection"]["n_trials"],
         "seconds_per_permutation": per, "reductions": reductions}, indent=1))
    print(f"p = {p:.4f}  real subject AUC (1 seed) = {real:.3f}  null mean = {null.mean():.3f}")


# ---------------------------------------------------------------- 6

def step_age_clusters(args):
    cfg, c = cfg_real(), load()
    model, rep = best_pair(pd.read_csv(S2 / "in_fold.csv"))
    subj = subject_table(c)
    frames = []
    with timed("6.1_age_clusters", f"{model}/{rep} in_fold") as t:
        for cl in ("young", "older"):
            keep = subj.index[age_cluster(subj["age"]) == cl]
            sc = subset(c, keep)
            ccfg = copy.deepcopy(cfg)
            n_min = int(subj.loc[keep, "y"].value_counts().min())
            if n_min < ccfg["cv"]["n_splits"]:
                ccfg["cv"]["n_splits"] = n_min
                t.notes += f"; {cl}: n_splits -> {n_min} (smallest class)"
            res = run_eval(sc, model, ccfg, "in_fold", rep)
            res.insert(2, "cluster", cl)
            res["n_splits"] = ccfg["cv"]["n_splits"]
            frames.append(res)
            save(pd.concat(frames), "age_clusters")


def step_residualised(args):
    cfg, c = cfg_real(), load()
    model, rep = best_pair(pd.read_csv(S2 / "in_fold.csv"))
    cfg["cv"]["residualise_age"] = True
    with timed("6.2_residualised", f"{model}/{rep} in_fold, age regressed out of 855 per fold"):
        save(run_eval(c, model, cfg, "in_fold", rep), "residualised")


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def step_stability(args):
    with timed("6.3_stability"):
        z = np.load(RES / "selected_in_fold.npz")
        sets = [z[k] for k in z.files]
        j = np.array([jaccard(a, b) for a, b in combinations(sets, 2)])
        freq = np.bincount(np.concatenate(sets), minlength=855) / len(sets)
        out = {"n_sets": len(sets), "jaccard_mean": float(j.mean()), "jaccard_std": float(j.std()),
               "jaccard_min": float(j.min()), "jaccard_max": float(j.max()),
               "features_ever_selected": int((freq > 0).sum()),
               "features_in_all_sets": int((freq == 1).sum()),
               "features_in_half_or_more": int((freq >= 0.5).sum())}
        (S2 / "stability.json").write_text(json.dumps(out, indent=1))
    print(out)


def step_aggregation(args):
    cfg = cfg_real()
    with timed("6.4_aggregation"):
        R = np.load(RES / "rank_matrix_pooled.npy")
        k = cfg["selection"]["n_keep"]
        fa, mr = aggregate_first_appearance(R)[:k], aggregate_mean_rank(R)[:k]
        out = {"overlap": len(set(fa) & set(mr)), "n_keep": k, "jaccard": jaccard(fa, mr)}
        (S2 / "aggregation.json").write_text(json.dumps(out, indent=1))
    print(out)


def step_sectioning(args):
    cfg = cfg_real()
    sdir = CACHE / "subjects" / F.feature_hash(cfg)
    if not sdir.is_dir():
        log.warning("cache/subjects absent: sectioning ablations require local reassembly")
        return
    model, rep = best_pair(pd.read_csv(S2 / "in_fold.csv"))
    variants = {"sections_5_5": {"female": 5, "male": 5},       # equal per subject
                "sections_5_8": {"female": 5, "male": 8}}       # paper-literal
    seeds = cfg["cv"]["seeds"][: args.seeds] if args.seeds else cfg["cv"]["seeds"]
    frames = []
    with timed("6.5_sectioning", f"{model}/{rep} in_fold, seeds {len(seeds)}"
               + (" (reduced 5 -> 3: two ablations projected > 45 min)" if len(seeds) < 5 else "")):
        for name, per_class in variants.items():
            vcfg = copy.deepcopy(cfg)
            vcfg["sections"]["per_class"] = per_class
            vcfg["paths"].update(cache_dir=str(CACHE / f"ablation_{name}"),
                                 subject_cache_dir=str(CACHE / "subjects"))
            logging.getLogger("pipeline.features").setLevel(logging.ERROR)
            F.extract_all(vcfg)                   # reassembly only, no connectivity recompute
            vc = load(CACHE / f"ablation_{name}")
            vcfg["cv"]["seeds"] = seeds
            res = run_eval(vc, model, vcfg, "in_fold", rep)
            res.insert(2, "variant", name)
            res["n_sections"] = len(vc["y"])
            frames.append(res)
            save(pd.concat(frames), "sectioning")


def step_interp_sensitivity(args):
    """6.6 (added): baseline on subjects with no interpolated channel only.

    Interpolation is more common in females (44 % vs 30 %; T7 22 % vs 8 %), so the
    all-subject numbers could partly reflect interpolation rather than sex.
    """
    cfg, c = cfg_real(), load()
    sdir = CACHE / "subjects" / F.feature_hash(cfg)
    clean = [f.stem for f in sorted(sdir.glob("*.npz")) if len(np.load(f)["interpolated"]) == 0]
    with timed("6.6_interp_sensitivity", f"logreg on 855, {len(clean)} subjects without "
                                         "interpolated channels"):
        res = run_eval(subset(c, clean), "logreg", cfg, None, "sections")
        res["n_subjects"] = len(clean)
        save(res, "interp_sensitivity")


# ---------------------------------------------------------------- 5 table

TABLE_COLS = ["model", "selection", "representation", "level", "metric", "mean", "std",
              "n_seeds", "notes"]


def _agg(df, level, metric):
    per_seed = df.groupby("seed")[f"{level}_{metric}"].mean()
    return per_seed.mean(), (per_seed.std(ddof=1) if len(per_seed) > 1 else np.nan), len(per_seed)


def _rows(df, selection_label, notes, levels=("section", "subject"), metrics=("bal_acc", "auc")):
    rows = []
    for (m, rep), g in df.groupby(["model", "representation"], sort=False):
        for level in levels:
            if rep == "long" and level == "section":
                continue  # one test row per subject: identical to subject level
            for metric in metrics:
                mean, std, n = _agg(g, level, metric)
                rows.append({"model": m, "selection": selection_label, "representation": rep,
                             "level": level, "metric": metric, "mean": mean, "std": std,
                             "n_seeds": n, "notes": notes})
    return rows


def build_table():
    rows = []
    if (S2 / "baselines.csv").exists():
        rows += _rows(pd.read_csv(S2 / "baselines.csv"), "none", f"all 855 features; {MEASURE}")
    if (S2 / "in_fold.csv").exists():
        rows += _rows(pd.read_csv(S2 / "in_fold.csv"), "in_fold",
                      f"top 34 selected inside each training fold; not permuted, budget; {MEASURE}")
    if (S2 / "pooled.csv").exists():
        rows += _rows(pd.read_csv(S2 / "pooled.csv"), "pooled", f"{LEAK}; {MEASURE}")
    if (S2 / "permutation.json").exists():
        p = json.loads((S2 / "permutation.json").read_text())
        red = ("; reduced: " + ", ".join(p["reductions"])) if p["reductions"] else ""
        rows.append({"model": p["model"], "selection": "pooled", "representation":
                     p["representation"], "level": "subject", "metric": "p_value", "mean": p["p"],
                     "std": np.nan, "n_seeds": p["n_seeds"],
                     "notes": f"{p['n_permutations']} subject-level label permutations, pooled "
                              f"selection recomputed each time ({p['n_trials']} trials); real AUC "
                              f"{p['real']:.3f} on {p['n_seeds']} seed{red}"})
    # extensions: one row each (subject-level AUC), balanced accuracy in notes
    if (S2 / "age_clusters.csv").exists():
        df = pd.read_csv(S2 / "age_clusters.csv")
        for cl, g in df.groupby("cluster", sort=False):
            mean, std, n = _agg(g, "subject", "auc")
            ba = _agg(g, "subject", "bal_acc")[0]
            rows.append({"model": g["model"].iloc[0], "selection": f"in_fold/age_{cl}",
                         "representation": g["representation"].iloc[0], "level": "subject",
                         "metric": "auc", "mean": mean, "std": std, "n_seeds": n,
                         "notes": f"6.1 trained and tested within the {cl} cluster "
                                  f"({g['n_test_subjects'].groupby(g['seed']).sum().iloc[0]} "
                                  f"subjects, {g['n_splits'].iloc[0]} splits"
                                  + ("; 6-7 test subjects per fold, so fold AUCs are noisy"
                                     if g["n_test_subjects"].max() < 10 else "")
                                  + f"); subject bal_acc {ba:.3f}"})
    if (S2 / "residualised.csv").exists():
        g = pd.read_csv(S2 / "residualised.csv")
        mean, std, n = _agg(g, "subject", "auc")
        rows.append({"model": g["model"].iloc[0], "selection": "in_fold/age_residualised",
                     "representation": g["representation"].iloc[0], "level": "subject",
                     "metric": "auc", "mean": mean, "std": std, "n_seeds": n,
                     "notes": "6.2 age (bin midpoint) regressed out of all 855 inside each fold, "
                              f"then in-fold selection; subject bal_acc "
                              f"{_agg(g, 'subject', 'bal_acc')[0]:.3f}"})
    if (S2 / "stability.json").exists():
        s = json.loads((S2 / "stability.json").read_text())
        rows.append({"model": "xgb_selector", "selection": "in_fold", "representation": "-",
                     "level": "selection", "metric": "jaccard", "mean": s["jaccard_mean"],
                     "std": s["jaccard_std"], "n_seeds": 5,
                     "notes": f"6.3 mean pairwise Jaccard of the {s['n_sets']} in-fold 34-sets; "
                              f"{s['features_ever_selected']} features ever selected, "
                              f"{s['features_in_half_or_more']} in at least half"})
    if (S2 / "aggregation.json").exists():
        a = json.loads((S2 / "aggregation.json").read_text())
        rows.append({"model": "xgb_selector", "selection": "pooled", "representation": "-",
                     "level": "selection", "metric": "overlap", "mean": a["overlap"],
                     "std": np.nan, "n_seeds": 1,
                     "notes": f"6.4 first-appearance vs mean-rank top {a['n_keep']}: "
                              f"{a['overlap']} shared"})
    if (S2 / "sectioning.csv").exists():
        df = pd.read_csv(S2 / "sectioning.csv")
        ref = ""
        if (S2 / "in_fold.csv").exists():   # main 8/5 run on the same seeds, for a fair comparison
            inf = pd.read_csv(S2 / "in_fold.csv")
            r = inf[(inf.model == df.model.iloc[0]) & (inf.representation == df.representation.iloc[0])
                    & inf.seed.isin(df.seed.unique())]
            rm, rs, rn = _agg(r, "subject", "auc")
            ref = f"; main 8/5 on the same {rn} seeds: {rm:.3f} ± {rs:.3f}"
        for v, g in df.groupby("variant", sort=False):
            mean, std, n = _agg(g, "subject", "auc")
            rows.append({"model": g["model"].iloc[0], "selection": f"in_fold/{v}",
                         "representation": g["representation"].iloc[0], "level": "subject",
                         "metric": "auc", "mean": mean, "std": std, "n_seeds": n,
                         "notes": f"6.5 sections per subject female/male = "
                                  f"{v.split('_')[1]}/{v.split('_')[2]} "
                                  f"({g['n_sections'].iloc[0]} sections); seeds reduced 5 -> {n}"
                                  f" (budget); subject bal_acc {_agg(g, 'subject', 'bal_acc')[0]:.3f}{ref}"
                                  if n < 5 else
                                  f"6.5 sections per subject female/male = "
                                  f"{v.split('_')[1]}/{v.split('_')[2]}; subject bal_acc "
                                  f"{_agg(g, 'subject', 'bal_acc')[0]:.3f}{ref}"})
    if (S2 / "interp_sensitivity.csv").exists():
        g = pd.read_csv(S2 / "interp_sensitivity.csv")
        mean, std, n = _agg(g, "subject", "auc")
        rows.append({"model": "logreg", "selection": "none/no_interpolated",
                     "representation": "sections", "level": "subject", "metric": "auc",
                     "mean": mean, "std": std, "n_seeds": n,
                     "notes": f"6.6 (added) all 855 features, only the {g['n_subjects'].iloc[0]} "
                              "subjects with no interpolated channel (interpolation is more "
                              "common in females); subject bal_acc "
                              f"{_agg(g, 'subject', 'bal_acc')[0]:.3f}"})
    t = pd.DataFrame(rows, columns=TABLE_COLS)
    t[["mean", "std"]] = t[["mean", "std"]].round(6)
    return t


def step_table(args):
    new = build_table()
    path = RES / "table.csv"
    if args.check:
        ref = pd.read_csv(path)
        keys = ["model", "selection", "representation", "level", "metric"]
        m = ref.merge(new, on=keys, how="outer", suffixes=("_ref", "_new"), indicator=True)
        bad = m[(m["_merge"] != "both")
                | ((m["mean_ref"] - m["mean_new"]).abs() > 1e-3)
                | ((m["std_ref"] - m["std_new"]).abs() > 1e-3)]
        if len(bad):
            print(bad.to_string(), file=sys.stderr)
            raise SystemExit(1)
        print(f"table check passed: {len(new)} rows within 1e-3")
        return
    with timed("5_table"):
        new.to_csv(path, index=False)
    print(new.drop(columns="notes").to_string())


STEPS = {"summary": step_summary, "baselines": step_baselines, "in_fold": step_in_fold,
         "pooled": step_pooled, "permutation": step_permutation, "table": step_table,
         "age_clusters": step_age_clusters, "residualised": step_residualised,
         "stability": step_stability, "aggregation": step_aggregation,
         "sectioning": step_sectioning, "interp_sensitivity": step_interp_sensitivity}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=STEPS)
    ap.add_argument("--check", action="store_true", help="table: diff against results/table.csv")
    ap.add_argument("--seeds", type=int, help="sectioning: use the first N seeds")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        datefmt="%H:%M:%S")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    RES.mkdir(exist_ok=True)
    STEPS[args.step](args)


if __name__ == "__main__":
    main()
