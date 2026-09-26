"""Synthetic data for testing mechanics only. Nothing here goes in results/table.csv.

- A fake feature cache (same files as pipeline.features.save_cache): 20 subjects,
  two sexes, two age clusters, a few sections each, 855 features with a weak
  planted sex signal in SIGNAL_COLS.
- A fake recording: mne RawArray, 19 channels, 250 Hz, 200 s, pink-ish noise plus a
  shared band-limited alpha component, saved as .fif for the extraction test.

Usage: python scripts/make_synthetic_cache.py [--out cache/smoke] [--fixture PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.features import save_cache  # noqa: E402

N_FEATURES = 855
SIGNAL_COLS = np.array([5, 171 + 40, 342 + 77, 513 + 120, 684 + 160])  # one per band
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_raw.fif"
CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4",
            "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]


def make_cache(out_dir, n_subjects=20, seed=0, effect=0.8):
    rng = np.random.default_rng(seed)
    subjects = [f"synth-{i:02d}" for i in range(n_subjects)]
    y_subj = np.array([i % 2 for i in range(n_subjects)])          # 1 = female (positive)
    old = np.array([(i // 2) % 2 for i in range(n_subjects)])       # age cluster, crossed with sex
    ages_subj = np.where(old == 1, rng.choice([67.5, 72.5], n_subjects),
                         rng.choice([22.5, 27.5], n_subjects))
    X, y, subs, ages, sections, X_long = [], [], [], [], [], []
    for s, sub in enumerate(subjects):
        centre = rng.normal(0.0, 0.5, N_FEATURES)
        centre[SIGNAL_COLS] += effect * (2 * y_subj[s] - 1)
        n_sec = int(rng.integers(3, 7))
        X.append(centre + rng.normal(0.0, 1.0, (n_sec, N_FEATURES)))
        y += [y_subj[s]] * n_sec
        subs += [sub] * n_sec
        ages += [ages_subj[s]] * n_sec
        sections += [(sub, k, 15) for k in range(n_sec)]
        X_long.append(centre + rng.normal(0.0, 0.3, N_FEATURES))
    save_cache(out_dir, np.concatenate(X), y, subs, ages, sections,
               np.array(X_long), subjects, [])
    return Path(out_dir)


def make_raw(sfreq=250.0, duration=200.0, seed=0, channels=CHANNELS):
    import mne
    from scipy.signal import butter, sosfiltfilt

    rng = np.random.default_rng(seed)
    n = int(sfreq * duration)
    freqs = np.fft.rfftfreq(n, 1 / sfreq)
    scale = 1 / np.sqrt(np.maximum(freqs, 1.0))                     # ~1/f power
    spec = rng.normal(size=(len(channels), len(freqs))) + 1j * rng.normal(size=(len(channels), len(freqs)))
    pink = np.fft.irfft(spec * scale, n=n, axis=1)
    pink /= pink.std(axis=1, keepdims=True)
    sos = butter(4, [8, 12], btype="bandpass", fs=sfreq, output="sos")
    alpha = sosfiltfilt(sos, rng.normal(size=n))
    alpha /= alpha.std()
    t = np.arange(n) / sfreq
    gains = np.linspace(0.2, 1.5, len(channels))                   # stronger towards occipital
    data = pink + gains[:, None] * (alpha + 0.5 * np.sin(2 * np.pi * 10 * t))[None, :]
    info = mne.create_info(list(channels), sfreq, "eeg")
    return mne.io.RawArray(data * 1e-5, info, verbose="ERROR")


def make_fixture(path=FIXTURE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_raw().save(path, overwrite=True, verbose="ERROR")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "cache" / "smoke"))
    ap.add_argument("--fixture", default=str(FIXTURE))
    args = ap.parse_args(argv)
    print("cache:", make_cache(args.out))
    print("fixture:", make_fixture(args.fixture))


if __name__ == "__main__":
    main()
