"""Regenerate every report figure from cache/ and results/:  python scripts/make_figures.py

Writes PNGs to report/figures/. A figure whose inputs are missing is skipped with a note.
Colours: validated categorical slots (blue, orange, aqua, yellow, magenta), a one-hue blue
ramp for magnitude, blue <-> red with a grey midpoint for signed differences.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import features as F  # noqa: E402

RES, OUT = ROOT / "results", ROOT / "report" / "figures"
S2 = RES / "stage2"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # validated, fixed order
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red",
                                        ["#184f95", "#6da7ec", "#f0efec", "#ee8a89", "#a82f2f"])
MODEL_ORDER = ["xgb", "mlp", "svm_rbf", "rf", "logreg"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8,
})


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def _skip(name, why):
    print(f"skipped {name}: {why}")


def _matrix(vec, n):
    m = np.full((n, n), np.nan)
    r, c = F.triu_pairs(n)
    m[r, c] = vec
    m[c, r] = vec
    return m


def _per_seed(df, col):
    """Mean over seeds of per-seed fold means, and std across seeds."""
    s = df.groupby("seed")[col].mean()
    return s.mean(), (s.std(ddof=1) if len(s) > 1 else 0.0)


# ---------------------------------------------------------------- figures

def fig_connectivity(cfg):
    """Mean coherence per band for female and male subjects, and the difference."""
    if not (ROOT / "cache" / "X.npy").exists():
        return _skip("connectivity_matrices.png", "cache/X.npy absent")
    c = F.load_cache(ROOT / "cache")
    chans, bands = cfg["signal"]["channels"], list(cfg["connectivity"]["bands"])
    n, npair = len(chans), len(chans) * (len(chans) - 1) // 2
    # subject means first, so subjects with more sections do not weigh more
    df = pd.DataFrame(c["X"]).groupby(c["subjects"]).mean()
    ysub = pd.Series(c["y"]).groupby(c["subjects"]).first().loc[df.index]
    fem, mal = df[ysub == 1].mean().to_numpy(), df[ysub == 0].mean().to_numpy()
    fig, axes = plt.subplots(len(bands), 3, figsize=(9.2, 3.0 * len(bands)))
    dmax = np.abs(fem - mal).max()
    for b, band in enumerate(bands):
        sl = slice(b * npair, (b + 1) * npair)
        vmax = max(fem[sl].max(), mal[sl].max())
        for j, (vec, title, cmap, lim) in enumerate([
                (fem[sl], f"{band}: female (n={int((ysub == 1).sum())})", SEQ, (0, vmax)),
                (mal[sl], f"{band}: male (n={int((ysub == 0).sum())})", SEQ, (0, vmax)),
                (fem[sl] - mal[sl], f"{band}: female − male", DIV, (-dmax, dmax))]):
            ax = axes[b, j]
            im = ax.imshow(_matrix(vec, n), cmap=cmap, vmin=lim[0], vmax=lim[1])
            ax.set_title(title, loc="left")
            ax.set_xticks(range(n), chans, rotation=90, fontsize=6)
            ax.set_yticks(range(n), chans, fontsize=6)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=6, color=MUTED, length=2)
    fig.suptitle("Mean coherence per band (subject means, then class means)", x=0.01,
                 ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, "connectivity_matrices.png")


def _scalp_positions(chans):
    import mne
    info = mne.create_info(chans, 250.0, "eeg")
    names = mne.channels.get_builtin_montages()
    info.set_montage(mne.channels.make_standard_montage(
        "standard_1020" if "standard_1020" in names else names[0]), match_case=False)
    pos = mne.channels.make_eeg_layout(info).pos[:, :2]
    pos = pos + mne.channels.make_eeg_layout(info).pos[:, 2:4] / 2 - 0.5
    return pos / np.abs(pos).max() * 0.9


def fig_pooled_scalp(cfg):
    """The pooled 34, one scalp per band (edges coloured by band)."""
    path = RES / "selected_features_pooled.csv"
    if not path.exists():
        return _skip("pooled_features_scalp.png", "selected_features_pooled.csv absent")
    sel = pd.read_csv(path)
    chans, bands = cfg["signal"]["channels"], list(cfg["connectivity"]["bands"])
    pos = dict(zip(chans, _scalp_positions(chans)))
    fig, axes = plt.subplots(1, len(bands), figsize=(2.3 * len(bands), 2.9))
    for b, (band, ax) in enumerate(zip(bands, axes)):
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color=AXIS, lw=1))
        ax.plot([-0.1, 0, 0.1], [0.99, 1.1, 0.99], color=AXIS, lw=1)        # nose
        rows = sel[sel["band"] == band]
        for _, r in rows.iterrows():
            (x1, y1), (x2, y2) = pos[r["channel_a"]], pos[r["channel_b"]]
            w = 1 + 3.0 * (len(sel) - r["rank"] + 1) / len(sel)                # thicker = higher rank
            ax.plot([x1, x2], [y1, y2], color=SLOTS[b], lw=w, solid_capstyle="round", zorder=2)
        for ch, (x, y) in pos.items():
            ax.scatter(x, y, s=46, color=SURFACE, edgecolor=MUTED, lw=0.8, zorder=3)
            ax.text(x, y - 0.13, ch, ha="center", va="top", fontsize=5.5, color=INK2, zorder=4)
        ax.set_title(f"{band} ({len(rows)})", loc="center")
        ax.set_xlim(-1.2, 1.2), ax.set_ylim(-1.2, 1.25), ax.set_aspect("equal"), ax.axis("off")
    fig.suptitle(f"Pooled selection: top {len(sel)} coherence features by band "
                 "(line width = rank)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, "pooled_features_scalp.png")


def fig_pooled_vs_in_fold():
    """Subject-level AUC per model, pooled vs in-fold, per test representation."""
    if not ((S2 / "pooled.csv").exists() and (S2 / "in_fold.csv").exists()):
        return _skip("auc_pooled_vs_in_fold.png", "pooled.csv or in_fold.csv absent")
    df = pd.concat([pd.read_csv(S2 / "in_fold.csv"), pd.read_csv(S2 / "pooled.csv")])
    reps = [r for r in ("sections", "long") if r in set(df["representation"])]
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    fig, axes = plt.subplots(1, len(reps), figsize=(4.4 * len(reps), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, rep in zip(axes, reps):
        for k, (sel, color, dx) in enumerate([("in_fold", SLOTS[0], -0.12),
                                              ("pooled", SLOTS[1], 0.12)]):
            for i, m in enumerate(models):
                g = df[(df.model == m) & (df.selection == sel) & (df.representation == rep)]
                mu, sd = _per_seed(g, "subject_auc")
                ax.errorbar(i + dx, mu, yerr=sd, fmt="o", ms=6, color=color, mec=SURFACE,
                            mew=1.5, elinewidth=1.5, capsize=0,
                            label=("in-fold selection" if sel == "in_fold"
                                   else "pooled selection (leaks)") if i == 0 else None)
        ax.axhline(0.5, color=AXIS, lw=0.8)
        ax.text(len(models) - 0.5, 0.5, "chance", ha="right", va="bottom", fontsize=7, color=MUTED)
        ax.set_xticks(range(len(models)), models)
        ax.set_title(f"test rows: {rep}", loc="left")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("subject-level AUC (mean ± sd over seeds)")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Pooled vs in-fold feature selection", x=0.01, ha="left", fontsize=11,
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "auc_pooled_vs_in_fold.png")


def fig_permutation():
    if not ((RES / "perm_null.npy").exists() and (S2 / "permutation.json").exists()):
        return _skip("permutation_null.png", "perm_null.npy absent")
    null = np.load(RES / "perm_null.npy")
    p = json.loads((S2 / "permutation.json").read_text())
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.hist(null, bins=20, color=SLOTS[0], alpha=0.85, edgecolor=SURFACE, linewidth=2)
    ax.axvline(p["real"], color=INK, lw=2)
    ax.text(p["real"], ax.get_ylim()[1] * 0.95, f"  observed {p['real']:.3f}\n  p = {p['p']:.3f}",
            va="top", ha="left", fontsize=8, color=INK)
    ax.set_xlabel("subject-level AUC")
    ax.set_ylabel("permutations")
    ax.set_title(f"Label-permutation null: {p['model']}, pooled, {p['representation']} "
                 f"({p['n_permutations']} permutations)", loc="left")
    fig.tight_layout()
    _save(fig, "permutation_null.png")


def fig_age_clusters():
    if not (S2 / "age_clusters.csv").exists():
        return _skip("age_clusters.png", "age_clusters.csv absent")
    df = pd.read_csv(S2 / "age_clusters.csv")
    rows = []
    for cl in ("young", "older"):
        g = df[df.cluster == cl]
        rows.append((cl, *_per_seed(g, "subject_auc"), g.groupby("seed")["n_test_subjects"].sum().iloc[0]))
    if (S2 / "in_fold.csv").exists():
        inf = pd.read_csv(S2 / "in_fold.csv")
        g = inf[(inf.model == df.model.iloc[0]) & (inf.representation == df.representation.iloc[0])]
        rows.insert(0, ("all ages", *_per_seed(g, "subject_auc"),
                        g.groupby("seed")["n_test_subjects"].sum().iloc[0]))
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for i, (lab, mu, sd, n) in enumerate(rows):
        ax.bar(i, mu - 0.5, bottom=0.5, width=0.45, color=SLOTS[0])
        ax.errorbar(i, mu, yerr=sd, color=INK2, elinewidth=1.5, capsize=0)
        ax.text(i, max(mu + sd, 0.5) + 0.01, f"{mu:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color=AXIS, lw=0.8)
    ax.set_xticks(range(len(rows)), [f"{lab}\n(n={n})" for lab, _, _, n in rows])
    ax.set_ylabel("subject-level AUC")
    ax.set_title(f"In-fold {df.model.iloc[0]} ({df.representation.iloc[0]}) within age clusters",
                 loc="left")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "age_clusters.png")


def fig_stability(cfg):
    path = RES / "selected_in_fold.npz"
    if not path.exists():
        return _skip("selection_stability.png", "selected_in_fold.npz absent")
    z = np.load(path)
    sets = [z[k] for k in z.files]
    freq = np.bincount(np.concatenate(sets), minlength=F.n_features(cfg)) / len(sets)
    chans, bands = cfg["signal"]["channels"], list(cfg["connectivity"]["bands"])
    n, npair = len(chans), len(chans) * (len(chans) - 1) // 2
    fig, axes = plt.subplots(1, len(bands), figsize=(3.0 * len(bands), 3.4))
    for b, (band, ax) in enumerate(zip(bands, axes)):
        im = ax.imshow(_matrix(freq[b * npair:(b + 1) * npair], n), cmap=SEQ, vmin=0, vmax=1)
        ax.set_title(band, loc="left")
        ax.set_xticks(range(n), chans, rotation=90, fontsize=5.5)
        ax.set_yticks(range(n), chans, fontsize=5.5)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
    cb.outline.set_visible(False)
    cb.set_label(f"share of the {len(sets)} in-fold selections", color=INK2)
    fig.suptitle("Selection stability: how often each feature is in the in-fold top 34",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    _save(fig, "selection_stability.png")


def main():
    cfg = F.load_config(ROOT / "config.yaml")
    fig_connectivity(cfg)
    fig_pooled_scalp(cfg)
    fig_pooled_vs_in_fold()
    fig_permutation()
    fig_age_clusters()
    fig_stability(cfg)


if __name__ == "__main__":
    main()
