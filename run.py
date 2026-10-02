"""Pipeline driver: extract -> select -> evaluate -> report.

    python run.py                    # every stage, each skipped if its output exists
    python run.py --stage extract    # one stage
    python run.py --force            # recompute even if outputs exist
    python run.py --smoke            # synthetic cache, tiny settings, writes results/smoke/
    python run.py --check            # rerun evaluate + report, diff table.csv (tol 1e-3)
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import subprocess
import sys
import tempfile
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import features as F
from pipeline.harness import LEVELS, METRICS, evaluate, permutation_test
from pipeline.selection import aggregate_first_appearance, aggregate_mean_rank, select_features

log = logging.getLogger("run")
STAGES = ("extract", "select", "evaluate", "report")
KEYS = ["model", "selection", "level", "metric"]


def smoke_config(cfg):
    cfg = copy.deepcopy(cfg)
    s = cfg["smoke"]
    cfg["paths"]["cache_dir"] = str(Path(cfg["paths"]["cache_dir"]) / "smoke")
    cfg["paths"]["results_dir"] = str(Path(cfg["paths"]["results_dir"]) / "smoke")
    cfg["selection"]["n_trials"] = s["n_trials"]
    cfg["cv"]["seeds"] = s["seeds"]
    cfg["cv"]["n_splits"] = s["n_splits"]
    cfg["permutation"]["n_permutations"] = s["n_permutations"]
    cfg["models"] = s["models"]
    cfg["selection"]["xgb_params"].update(s.get("selection_xgb_params") or {})
    cfg["selection"]["n_jobs"] = s.get("selection_n_jobs", 1)
    for name, params in (s.get("model_params") or {}).items():
        cfg["model_params"].setdefault(name, {}).update(params)
    return cfg


def outputs(cfg):
    c, r = Path(cfg["paths"]["cache_dir"]), Path(cfg["paths"]["results_dir"])
    return {"extract": c / "X.npy", "select": c / "selected.npy",
            "evaluate": r / "evaluate.csv", "report": r / "table.csv"}


# ---------------------------------------------------------------- stages

def stage_extract(cfg, smoke):
    if smoke:
        from scripts.make_synthetic_cache import make_cache
        make_cache(cfg["paths"]["cache_dir"])
    else:
        F.extract_all(cfg)


def stage_select(cfg, smoke):
    c = F.load_cache(cfg["paths"]["cache_dir"])
    idx, R = select_features(c["X"], c["y"], c["subjects"], cfg, cfg["selection"]["seed"],
                             return_ranks=True)
    cache = Path(cfg["paths"]["cache_dir"])
    np.save(cache / "rank_matrix.npy", R)
    np.save(cache / "selected.npy", idx)
    names = F.feature_names(cfg) if c["X"].shape[1] == F.n_features(cfg) else None
    first, mean = aggregate_first_appearance(R), aggregate_mean_rank(R)
    n = cfg["selection"]["n_keep"]
    table = pd.DataFrame({
        "rank": np.arange(1, n + 1),
        "first_appearance": first[:n], "mean_rank": mean[:n],
        "first_appearance_name": [names[i] for i in first[:n]] if names else "",
        "mean_rank_name": [names[i] for i in mean[:n]] if names else ""})
    out = Path(cfg["paths"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "selected_features.csv", index=False)
    log.info("selected %d features (%s); overlap with mean_rank: %d", n,
             cfg["selection"]["aggregation"], len(set(first[:n]) & set(mean[:n])))


def _mode(name):
    return None if name in (None, "none") else name


def stage_evaluate(cfg, smoke):
    c = F.load_cache(cfg["paths"]["cache_dir"])
    pooled = np.load(Path(cfg["paths"]["cache_dir"]) / "selected.npy")
    kw = dict(ages=c["ages"], X_long=c["X_long"], long_groups=c["long_subjects"])
    frames = []
    for mode in cfg["cv"]["selection_modes"]:
        shared = {}  # in-fold selections reused across models
        for model in cfg["models"]:
            log.info("evaluate %s / %s", model, mode)
            res = evaluate(c["X"], c["y"], c["subjects"], model, cfg, _mode(mode),
                           pooled_idx=pooled, selection_cache=shared, **kw)
            res.insert(0, "selection", mode)
            res.insert(0, "model", model)
            frames.append(res)
    res = pd.concat(frames, ignore_index=True)
    out = Path(cfg["paths"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "evaluate.csv", index=False)

    perm = cfg["permutation"]
    col = f"{perm['level']}_{perm['metric']}"
    best = res.groupby(["model", "selection"])[col].mean().idxmax()
    log.info("permutation test on %s / %s (%d permutations)", *best, perm["n_permutations"])
    pt = permutation_test(c["X"], c["y"], c["subjects"], best[0], cfg, _mode(best[1]),
                          rng=cfg["selection"]["seed"], pooled_idx=pooled, **kw)
    n_seeds = len(cfg["cv"]["seeds"][: perm.get("n_seeds", len(cfg["cv"]["seeds"]))])
    with open(out / "permutation.json", "w") as f:
        json.dump({"model": best[0], "selection": best[1], "level": perm["level"],
                   "metric": perm["metric"], "real": pt["real"], "p": pt["p"],
                   "n_permutations": pt["n_permutations"], "n_seeds": n_seeds,
                   "null": pt["null"].tolist()}, f, indent=1)


def build_table(res: pd.DataFrame, perm: dict) -> pd.DataFrame:
    rows = []
    for (model, sel), g in res.groupby(["model", "selection"], sort=False):
        per_seed = g.groupby("seed").mean(numeric_only=True)
        for level in LEVELS:
            for metric in METRICS:
                v = per_seed[f"{level}_{metric}"]
                rows.append({"model": model, "selection": sel, "level": level, "metric": metric,
                             "mean": v.mean(), "std": v.std(ddof=1) if len(v) > 1 else np.nan,
                             "n_seeds": len(v)})
    rows.append({"model": perm["model"], "selection": perm["selection"], "level": perm["level"],
                 "metric": "p_value", "mean": perm["p"], "std": np.nan,
                 "n_seeds": perm["n_seeds"]})
    return pd.DataFrame(rows).round(6)


def stage_report(cfg, smoke):
    out = Path(cfg["paths"]["results_dir"])
    res = pd.read_csv(out / "evaluate.csv")
    with open(out / "permutation.json") as f:
        perm = json.load(f)
    table = build_table(res, perm)
    table.to_csv(out / "table.csv", index=False)
    subj = table[(table.level == "subject") & (table.metric.isin(["auc", "bal_acc"]))]
    print(subj.pivot_table(index=["model", "selection"], columns="metric", values="mean")
          .round(3).to_string())
    print(f"p = {perm['p']:.4f} ({perm['model']}/{perm['selection']}, "
          f"{perm['n_permutations']} permutations)")


RUNNERS = {"extract": stage_extract, "select": stage_select,
           "evaluate": stage_evaluate, "report": stage_report}


# ---------------------------------------------------------------- check

def reference_table(path: Path):
    """The committed table if tracked by git, else the one on disk, else None."""
    try:
        rel = path.resolve().relative_to(Path.cwd().resolve())
        txt = subprocess.run(["git", "show", f"HEAD:{rel.as_posix()}"], capture_output=True,
                             text=True, check=True).stdout
        return pd.read_csv(StringIO(txt))
    except (ValueError, subprocess.CalledProcessError, FileNotFoundError):
        pass
    return pd.read_csv(path) if path.exists() else None


def compare_tables(ref: pd.DataFrame, new: pd.DataFrame, tol: float) -> list[str]:
    problems = []
    a, b = ref.set_index(KEYS), new.set_index(KEYS)
    for k in sorted(set(a.index) ^ set(b.index)):
        problems.append(f"row only in {'reference' if k in a.index else 'new'}: {k}")
    for k in a.index.intersection(b.index):
        for col in ("mean", "std", "n_seeds"):
            x, y = a.loc[k, col], b.loc[k, col]
            if (pd.isna(x) and pd.isna(y)) or (not pd.isna(x) and not pd.isna(y) and abs(x - y) <= tol):
                continue
            problems.append(f"{k} {col}: reference {x} vs new {y}")
    return problems


def run_check(cfg, smoke) -> int:
    ref_path = Path(cfg["paths"]["results_dir"]) / "table.csv"
    ref = reference_table(ref_path)
    if ref is None:
        print(f"--check: no reference table at {ref_path}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        cfg = copy.deepcopy(cfg)
        cfg["paths"]["results_dir"] = tmp
        stage_evaluate(cfg, smoke)
        stage_report(cfg, smoke)
        new = pd.read_csv(Path(tmp) / "table.csv")
    problems = compare_tables(ref, new, cfg["check"]["tolerance"])
    if problems:
        print(f"--check FAILED against {ref_path}:", *problems, sep="\n  ", file=sys.stderr)
        return 1
    print(f"--check passed: {len(new)} rows match {ref_path} within {cfg['check']['tolerance']}")
    return 0


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stage", choices=STAGES)
    ap.add_argument("--force", action="store_true", help="recompute even if outputs exist")
    ap.add_argument("--smoke", action="store_true", help="synthetic cache, results/smoke/")
    ap.add_argument("--check", action="store_true", help="rerun evaluate+report, diff table.csv")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    cfg = F.load_config(args.config)
    if args.smoke:
        cfg = smoke_config(cfg)
    if args.check:
        return run_check(cfg, args.smoke)
    outs = outputs(cfg)
    for stage in ([args.stage] if args.stage else STAGES):
        if outs[stage].exists() and not args.force:
            log.info("%s: %s exists, skipped (use --force)", stage, outs[stage])
            continue
        log.info("%s: running", stage)
        RUNNERS[stage](cfg, args.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
