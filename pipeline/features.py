"""Connectivity features: raw EEG -> (n_sections, 855), and the on-disk cache.

Feature order is band-major: for band b and pair p (from np.triu_indices(19, k=1)),
the column is b * 171 + p.
"""
from __future__ import annotations

import hashlib
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from mne_connectivity import spectral_connectivity_epochs
from scipy.signal import butter, sosfiltfilt

log = logging.getLogger("pipeline.features")

SPECTRAL_METHODS = ("coh", "plv", "wpli")
METHODS = SPECTRAL_METHODS + ("corr",)

# Config keys that change the per-subject features. Anything else (section counts,
# missing-channel policy, labels) is applied after the per-subject cache.
FEATURE_KEYS = {
    "signal": ["channels", "sfreq", "condition", "reference", "discard_start_s", "epoch_s"],
    "sections": ["section_epochs", "layout", "aggregation", "long_epochs", "long_start_epoch"],
    "connectivity": ["method", "mode", "bands", "band_edges", "corr_filter_order"],
}


class MissingChannelsError(Exception):
    def __init__(self, missing):
        super().__init__(f"missing channels: {' '.join(missing)}")
        self.missing = list(missing)


def load_config(path="config.yaml") -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def triu_pairs(n_ch: int = 19):
    return np.triu_indices(n_ch, k=1)


def feature_names(cfg) -> list[str]:
    chans = cfg["signal"]["channels"]
    rows, cols = triu_pairs(len(chans))
    return [f"{band}:{chans[i]}-{chans[j]}"
            for band in cfg["connectivity"]["bands"]
            for i, j in zip(rows, cols)]


def n_features(cfg) -> int:
    n = len(cfg["signal"]["channels"])
    return n * (n - 1) // 2 * len(cfg["connectivity"]["bands"])


def feature_hash(cfg) -> str:
    sub = {sec: {k: cfg[sec].get(k) for k in keys} for sec, keys in FEATURE_KEYS.items()}
    return hashlib.sha1(json.dumps(sub, sort_keys=True).encode()).hexdigest()[:10]


# ---------------------------------------------------------------- raw -> epochs

def read_raw(path) -> mne.io.BaseRaw:
    path = Path(path)
    if path.suffix == ".set":
        return mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    if path.suffix == ".fif":
        return mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    raise ValueError(f"unsupported file type: {path}")


def pick_channels(raw, cfg) -> mne.io.BaseRaw:
    """Return a copy with exactly cfg channels, in cfg order, names as in cfg.

    Matching is case-insensitive. Missing channels raise MissingChannelsError
    under policy 'skip', or are spherical-spline interpolated under 'interpolate'.
    """
    wanted = cfg["signal"]["channels"]
    policy = cfg["signal"].get("missing_channel_policy", "skip")
    by_lower = {ch.lower(): ch for ch in raw.ch_names}
    missing = [ch for ch in wanted if ch.lower() not in by_lower]
    raw = raw.copy()
    raw.rename_channels({by_lower[ch.lower()]: ch for ch in wanted if ch.lower() in by_lower})
    if missing:
        if policy != "interpolate":
            raise MissingChannelsError(missing)
        raw = _interpolate_missing(raw, missing)
    raw.pick(wanted)
    raw.reorder_channels(wanted)
    return raw


def _interpolate_missing(raw, missing):
    raw.pick("eeg")
    fill = mne.io.RawArray(np.zeros((len(missing), raw.n_times)),
                           mne.create_info(missing, raw.info["sfreq"], "eeg"),
                           verbose="ERROR")
    raw.add_channels([fill], force_update_info=True)
    names = mne.channels.get_builtin_montages()
    montage = mne.channels.make_standard_montage(  # renamed in MNE 1.13
        "colin27_1005" if "colin27_1005" in names else "standard_1005")
    has_pos = all(np.isfinite(ch["loc"][:3]).all() and np.any(ch["loc"][:3])
                  for ch in raw.info["chs"] if ch["ch_name"] not in missing)
    if has_pos:
        # keep the recorded positions, take standard ones only for the new channels
        pos = montage.get_positions()["ch_pos"]
        lookup = {k.lower(): v for k, v in pos.items()}
        for ch in raw.info["chs"]:
            if ch["ch_name"] in missing:
                ch["loc"][:3] = lookup[ch["ch_name"].lower()]
    else:
        raw.set_montage(montage, match_case=False, on_missing="ignore")
    raw.info["bads"] = list(missing)
    raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    return raw


def epoch_data(raw, cfg) -> np.ndarray:
    """(n_epochs, n_channels, n_times) after the start discard."""
    sig = cfg["signal"]
    if not np.isclose(raw.info["sfreq"], sig["sfreq"]):
        raise ValueError(f"sfreq {raw.info['sfreq']} != config {sig['sfreq']}")
    raw = raw.copy().crop(tmin=sig["discard_start_s"])
    if sig.get("reference", "keep") == "average":
        raw.set_eeg_reference("average", verbose="ERROR")
    epochs = mne.make_fixed_length_epochs(raw, duration=sig["epoch_s"], preload=True,
                                          reject_by_annotation=True, verbose="ERROR")
    return epochs.get_data(copy=False)


# ---------------------------------------------------------------- epochs -> 855

def _band_masks(freqs, cfg):
    bands = list(cfg["connectivity"]["bands"].values())
    masks = []
    for k, (lo, hi) in enumerate(bands):
        last = k == len(bands) - 1
        m = (freqs >= lo) & ((freqs <= hi) if last else (freqs < hi))
        if not m.any():
            raise ValueError(f"no frequency bins in band {lo}-{hi}")
        masks.append(m)
    return masks


def connectivity_vector(data: np.ndarray, cfg) -> np.ndarray:
    """One connectivity feature vector from a stack of epochs (n_epochs, n_ch, n_times)."""
    con = cfg["connectivity"]
    sfreq = cfg["signal"]["sfreq"]
    n_ch = data.shape[1]
    rows, cols = triu_pairs(n_ch)
    method = con["method"]
    if method in SPECTRAL_METHODS:
        fmax = max(hi for _, hi in con["bands"].values())
        with warnings.catch_warnings():
            # delta starts at 0 Hz, below the 5-cycle floor for 2-s epochs (QUESTIONS.md #5)
            warnings.filterwarnings("ignore", message=".*5 cycles.*")
            warnings.filterwarnings("ignore", message="divide by zero")
            res = spectral_connectivity_epochs(
                data, method=method, indices=(rows, cols), sfreq=sfreq,
                mode=con.get("mode", "multitaper"), fmin=0.0, fmax=fmax, verbose=False)
        vals = res.get_data()  # (n_pairs, n_freqs), triu order
        freqs = np.asarray(res.freqs)
        return np.concatenate([vals[:, m].mean(axis=1) for m in _band_masks(freqs, cfg)])
    if method == "corr":
        return _corr_vector(data, cfg, rows, cols)
    raise ValueError(f"connectivity.method must be one of {METHODS}, got {method!r}")


def _corr_vector(data, cfg, rows, cols):
    sfreq = cfg["signal"]["sfreq"]
    order = cfg["connectivity"].get("corr_filter_order", 4)
    out = []
    for lo, hi in cfg["connectivity"]["bands"].values():
        if lo <= 0:
            sos = butter(order, hi, btype="lowpass", fs=sfreq, output="sos")
        else:
            sos = butter(order, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
        filt = sosfiltfilt(sos, data, axis=-1)
        per_epoch = np.stack([np.corrcoef(ep)[rows, cols] for ep in filt])
        out.append(per_epoch.mean(axis=0))  # GUESS: mean over the section's epochs
    return np.concatenate(out)


# ---------------------------------------------------------------- subject

def _extract(path, cfg):
    """Return (sections (n, F), epochs_used list, long row (F,) or None, interpolated channels)."""
    raw = read_raw(path)
    wanted = {ch.lower() for ch in raw.ch_names}
    interpolated = [ch for ch in cfg["signal"]["channels"] if ch.lower() not in wanted]
    raw = pick_channels(raw, cfg)
    data = epoch_data(raw, cfg)
    sec = cfg["sections"]
    n_ep = sec["section_epochs"]
    n_sections = data.shape[0] // n_ep
    X = np.stack([connectivity_vector(data[i * n_ep:(i + 1) * n_ep], cfg)
                  for i in range(n_sections)]) if n_sections else np.empty((0, n_features(cfg)))
    start, n_long = sec["long_start_epoch"], sec["long_epochs"]
    long_row = None
    if data.shape[0] >= start + n_long:
        long_row = connectivity_vector(data[start:start + n_long], cfg)
    return X, [n_ep] * n_sections, long_row, interpolated


def extract_subject(set_path, cfg) -> np.ndarray:
    """All complete, consecutive, non-overlapping sections of one recording: (n_sections, 855)."""
    return _extract(set_path, cfg)[0]


# ---------------------------------------------------------------- labels

def load_labels(cfg) -> pd.DataFrame:
    """DataFrame indexed by subject id with columns sex, y, age."""
    p = cfg["participants"]
    df = pd.read_csv(cfg["paths"]["participants_csv"], sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    ids = df[p["id_col"]].astype(str).str.strip()
    if p.get("use_name_match"):
        nm = pd.read_csv(cfg["paths"]["name_match_csv"], sep=None, engine="python")
        nm.columns = [c.strip() for c in nm.columns]
        mapping = dict(zip(nm[p["name_match_from"]].astype(str).str.strip(),
                           nm[p["name_match_to"]].astype(str).str.strip()))
        ids = ids.map(lambda s: mapping.get(s, s))
    coding = {int(k): v for k, v in p["sex_coding"].items()}
    sex = pd.to_numeric(df[p["sex_col"]], errors="coerce").map(coding)
    ages = df[p["age_col"]].astype(str).str.strip().map(cfg["age_midpoints"]).astype(float)
    out = pd.DataFrame({"sex": sex.values, "age": ages.values}, index=ids.values)
    out = out[~out.index.duplicated()].dropna(subset=["sex"])
    out["y"] = (out["sex"] == p["positive_class"]).astype(int)
    return out


# ---------------------------------------------------------------- cache io

CACHE_FILES = ("X.npy", "y.npy", "subjects.npy", "ages.npy", "sections.csv",
               "X_long.npy", "long_subjects.npy", "skipped.csv")


def save_cache(cache_dir, X, y, subjects, ages, sections, X_long, long_subjects, skipped):
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "X.npy", np.asarray(X, dtype=float))
    np.save(d / "y.npy", np.asarray(y, dtype=int))
    np.save(d / "subjects.npy", np.asarray(subjects, dtype=str))
    np.save(d / "ages.npy", np.asarray(ages, dtype=float))
    X_long = np.asarray(X_long, dtype=float)
    if len(long_subjects) == 0:
        X_long = np.empty((0, np.shape(X)[1]))
    np.save(d / "X_long.npy", X_long)
    np.save(d / "long_subjects.npy", np.asarray(long_subjects, dtype=str))
    pd.DataFrame(sections, columns=["subject", "section", "n_epochs_used"]).to_csv(
        d / "sections.csv", index=False)
    pd.DataFrame(skipped, columns=["subject", "reason"]).to_csv(d / "skipped.csv", index=False)


def load_cache(cache_dir) -> dict:
    d = Path(cache_dir)
    return {
        "X": np.load(d / "X.npy"),
        "y": np.load(d / "y.npy"),
        "subjects": np.load(d / "subjects.npy"),
        "ages": np.load(d / "ages.npy"),
        "X_long": np.load(d / "X_long.npy"),
        "long_subjects": np.load(d / "long_subjects.npy"),
    }


# ---------------------------------------------------------------- all subjects

def _subject_cache_path(cfg, sub):
    return Path(cfg["paths"]["cache_dir"]) / "subjects" / feature_hash(cfg) / f"{sub}.npz"


def _subject_features(path, cfg, sub):
    """Per-subject features, cached under a hash of the feature-defining config."""
    cp = _subject_cache_path(cfg, sub)
    if cp.exists():
        z = np.load(cp)
        interpolated = [str(c) for c in z["interpolated"]]
        # a cache written under 'interpolate' must not leak into a 'skip' run
        if interpolated and cfg["signal"].get("missing_channel_policy", "skip") != "interpolate":
            raise MissingChannelsError(interpolated)
        return z["X"], list(z["epochs_used"]), (z["long"] if z["has_long"] else None)
    X, used, long_row, interpolated = _extract(path, cfg)
    if interpolated:
        log.warning("%s: interpolated %s", sub, " ".join(interpolated))
    cp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cp, X=X, epochs_used=np.asarray(used), has_long=long_row is not None,
             long=long_row if long_row is not None else np.zeros(n_features(cfg)),
             interpolated=np.asarray(interpolated, dtype=str))
    return X, used, long_row


def discover_subjects(cfg) -> list[str]:
    root = Path(cfg["paths"]["preproc_dir"])
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def extract_all(cfg) -> None:
    """Loop subjects, attach labels, apply per-class section counts, write the cache."""
    cond = cfg["signal"]["condition"]
    per_class = cfg["sections"]["per_class"]
    labels = load_labels(cfg)
    root = Path(cfg["paths"]["preproc_dir"])
    X, y, subjects, ages, sections, skipped = [], [], [], [], [], []
    X_long, long_subjects = [], []
    for sub in discover_subjects(cfg):
        path = root / cfg["paths"]["file_pattern"].format(sub=sub, cond=cond)
        if sub not in labels.index:
            skipped.append((sub, "no label in participants csv"))
            log.warning("%s: no label, skipped", sub)
            continue
        if not path.exists():
            skipped.append((sub, f"missing {cond} file"))
            log.warning("%s: %s not found, skipped", sub, path.name)
            continue
        try:
            Xs, used, long_row = _subject_features(path, cfg, sub)
        except MissingChannelsError as e:
            skipped.append((sub, str(e)))
            log.warning("%s: %s, skipped", sub, e)
            continue
        except Exception as e:  # keep a long real-data run going; the reason is logged
            skipped.append((sub, f"error: {type(e).__name__}: {e}"))
            log.exception("%s: extraction failed, skipped", sub)
            continue
        lab = labels.loc[sub]
        want = per_class[lab["sex"]]
        n = min(want, len(Xs))
        if n == 0:
            skipped.append((sub, "no complete section"))
            log.warning("%s: no complete section, skipped", sub)
            continue
        if n < want:
            log.warning("%s: only %d of %d sections available", sub, n, want)
        X.append(Xs[:n])
        y += [lab["y"]] * n
        subjects += [sub] * n
        ages += [lab["age"]] * n
        sections += [(sub, i, used[i]) for i in range(n)]
        if long_row is not None:
            X_long.append(long_row)
            long_subjects.append(sub)
        else:
            log.warning("%s: fewer than %d epochs, no long row", sub, cfg["sections"]["long_epochs"])
    if not X:
        raise RuntimeError("no subject produced features; see skipped list")
    save_cache(cfg["paths"]["cache_dir"], np.concatenate(X), y, subjects, ages, sections,
               np.array(X_long), long_subjects, skipped)
    log.info("extracted %d sections from %d subjects, skipped %d",
             len(y), len(set(subjects)), len(skipped))
