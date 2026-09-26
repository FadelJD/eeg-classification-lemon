# Questions for the human

Every one of these is a config value. The code supports all the options, so answering means editing `config.yaml`, not code. Current defaults are marked. Items 1 to 4 change the cached features, so please answer them before running `run.py --stage extract` on real data.

## 1. Missing T7 (`signal.missing_channel_policy`)

`check_headers.py` on `sub-032301_EC.set` reports 59 channels with **T7 missing**. The brief says to skip any subject missing one of the 19 channels. If T7 (or another of the 19) was dropped as a bad channel in many subjects, skipping could remove a large share of the 202.

- **(a) `skip`** (current default, as the brief says): log it in `cache/skipped.csv` and drop the subject.
- **(b) `interpolate`**: spherical-spline interpolation of the missing channels from the remaining ~59, using the positions stored in the `.set`. Keeps every subject; interpolated channels have inflated coherence with their neighbours.
- **(c)** Drop T7 for everyone: 18 channels, 153 pairs, 765 features. Not implemented; it would break the paper's 855.

Cheap way to decide: run `python run.py --stage extract` with `skip`, read the size of `cache/skipped.csv`, then decide. Per-subject features are cached, so switching to `interpolate` and rerunning `python run.py --stage extract --force` only computes the subjects that were skipped.

## 2. Sections per class (`sections.per_class`)

The paper had 150 F / 91 M and used 5 sections per female and 8 per male, which gives more sections to the **minority** class. LEMON is the other way round: 82 F / 146 M in the CSV. Keeping 5/8 makes the imbalance worse (about 410 F vs 1168 M sections).

- **(a) `{female: 5, male: 8}`** (current default, the brief's instruction): the paper's literal numbers.
- **(b) `{female: 8, male: 5}`**: the paper's intent (rebalance), about 656 F vs 730 M.
- **(c)** Something else.

Cost: extraction caches every available section for every subject, so changing this only reassembles `X.npy` (`run.py --stage extract --force`) and does not recompute connectivity.

## 3. Condition (`signal.condition`)

The paper says "resting-state" and does not say eyes open or closed. LEMON has both. Current default: **EC** (the file you ran check_headers on). Options: EC, EO. Switching means a full re-extraction.

## 4. Reference and spectral estimator (`signal.reference`, `connectivity.mode`)

Neither is in 3.2 or 3.3. Current defaults: **keep LEMON's reference as delivered**, and **multitaper** (the MNE default). Alternatives: re-reference the 19-channel subset to its own average, or use `fourier`. Both change coherence values, so both mean a full re-extraction if changed.

## 5. Delta band below 2.5 Hz (no action needed unless you disagree)

3.3's delta band is 0–4 Hz. With 2-s epochs, `mne_connectivity` warns that anything below 2.5 Hz has fewer than 5 cycles per epoch. The code passes `fmin=0` explicitly so the band follows the paper, and suppresses the warning. LEMON's preprocessing high-passes at about 1 Hz, so the 0–1 Hz bins carry little signal anyway.

## 6. Validation protocol (no re-extraction needed)

3.5 says validation subjects are not sectioned: one 3-minute (90-epoch) connectivity per subject, 50 random 90/10 subject splits, and a tuned probability threshold (best θ = 0.39). The brief asks for `StratifiedGroupKFold` on sections. Both are wired: extraction also caches the 90-epoch "long" row per subject (`X_long.npy`), and `cv.test_representation: long` evaluates on it. Current default: `sections`. Tell me if you want `long` as the default.
