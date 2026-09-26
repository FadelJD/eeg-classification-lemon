# Handoff

Everything here was built and tested on synthetic data only. No LEMON file has been read, downloaded or processed in the cloud session. Open questions that affect the cached features are in `notes/QUESTIONS.md`: **answer items 1–4 before running `extract`.**

## Stage 2 results (real LEMON cache, `results/table.csv`)

Without leakage, coherence predicts sex above chance but well below the paper: the best in-fold model (XGB on 34 features selected inside each training fold) reaches subject-level AUC 0.72 ± 0.02, and logistic regression on all 855 features does as well or better (0.74 ± 0.01), so the 34-feature selection adds nothing. Selecting the 34 on all subjects first, as the paper does, lifts every model by 0.03–0.09 AUC (best SVM 0.80 ± 0.02, still short of the paper's 0.89), and the label-permutation null of that pooled pipeline centres on 0.59 rather than 0.5, which measures the leak on its own. The pooled SVM beats all 50 permutations (p = 0.020, the smallest attainable with 50), so there is real signal, but the honest estimate is the in-fold one.

Budget reductions: permutation test 100 → 50 permutations (32.5 s each, 100 projected at 54 min > 45). Recorded in the table's `notes` column. Timings in `results/timings.csv`.

## Stage 2 extensions

- **6.1 Age clusters** (best in-fold configuration, xgb / sections): young (138 subjects) AUC 0.74 ± 0.03, the same as the full sample. Older (63) is 0.50 ± 0.05, i.e. chance, but with only 6–7 test subjects per fold that estimate is very noisy. The whole-sample signal comes from the young cluster.
- **6.2 Age residualised** inside each fold: 0.73 ± 0.03, against 0.72 unresidualised. Age (which differs by sex here: 49 % female among older subjects, 30 % among young) does not explain the result.
- **6.4 Aggregation:** the pooled first-appearance and mean-rank top-34 sets share only 8 of 34 features.
- **6.5 Sectioning** (3 seeds, reduced from 5 for budget): 5/5 sections gives 0.69 ± 0.04 and the paper-literal 5/8 gives 0.69 ± 0.03, against 0.72 ± 0.02 for the committed 8/5 on the same 3 seeds. That is a small edge for giving the minority class more sections, within about one seed-sd.
- **6.6 (added)** Interpolation sensitivity: interpolation is more common in females (44 % vs 30 %; T7 alone 22 % vs 8 %). Logistic regression on all 855 features using only the 131 subjects with no interpolated channel reaches 0.75 ± 0.02, so interpolation is not driving the result.

**Selection stability (6.3).** The 50 in-fold 34-feature sets overlap little: mean pairwise Jaccard is 0.17 ± 0.05 (range 0.06–0.33). 303 of the 855 features are selected at least once, and only 11 in at least half of the folds. The most stable are delta F7–T7 (88 % of folds), beta Cz–O2 (86 %), delta Fz–T7 (84 %), alpha P3–O2 (84 %) and gamma C4–P4 (72 %); none is one of the bridged pairs flagged in step 0. So a few fronto-temporal delta and parieto-occipital alpha/beta couplings carry a reproducible sex difference, while the rest of any particular 34-set is noise-driven. This fits the finding that all 855 features do as well as any selected 34, and it means the paper's single list of "chosen features" should not be read as a stable biomarker. Only 7 of the pooled 34 appear in at least half of the in-fold sets (`report/figures/selection_stability.png`).

Budget reductions in stage 2: permutation test 100 → 50 permutations; 6.5 ablations 5 → 3 seeds. Both are recorded in the table's `notes` column. Total compute: 2 h 29 min (`results/timings.csv`).

## Commands to run locally, in order

```bash
git pull origin develop
pip install -r requirements.txt       # requirements.lock is a Linux freeze (Python 3.11), for reference
python -m pytest -q                   # 27 tests, about 30 s
python run.py --smoke                 # about 7 s, writes results/smoke/ (ignored)

# data: either you already have it in data/, or
python download.py --i-am-local --dry-run   # read the commands first
python download.py --i-am-local             # several GB; resumable (aws s3 sync) and idempotent untar

# time one subject before the full extract
python -c "import time; from pipeline.features import load_config, extract_subject as e; c=load_config(); t=time.time(); print(e('data/lemon_preproc/sub-032302/sub-032302_EC.set', c).shape, time.time()-t)"
python run.py --stage extract         # expect a few seconds per subject
#   then read cache/skipped.csv and cache/sections.csv, answer QUESTIONS.md #1 and #2
python run.py --stage select          # 50 XGB fits, about 2.3 s each on one core, parallel over cores
python run.py --stage evaluate        # the long one, see "Runtime" below
python run.py --stage report          # prints the subject-level table, writes results/table.csv
git add results/table.csv results/selected_features.csv && git commit -m "results: first LEMON run"
python run.py --check                 # must pass on an unchanged tree
```

## Runtime (estimated from one timed fit: 1500 × 855 XGBoost, 100 trees, 2.3 s on one core)

- `extract`: coherence takes about 0.1 s per 15-epoch section. With ~15 sections plus the 90-epoch row and `.set` loading, expect 2–4 s per subject, so roughly 10 minutes for 202 subjects. Results are cached per subject in `cache/subjects/<hash>/`, so an interrupted run resumes.
- `select` (pooled): 50 fits ≈ 2 minutes on one core, less in parallel (`selection.n_jobs: -1`).
- `evaluate`: `in_fold` runs 50 trials per fold: 5 seeds × 10 folds × 50 = 2,500 fits ≈ 1.6 h on one core, divided by your core count. The selections are shared across models. `pooled` is minutes.
- Permutation test (inside `evaluate`): it runs on the best (model, selection) pair by subject AUC. If that pair is `in_fold`, each permutation costs 10 folds × 50 fits, so 100 permutations would take about 30 h on one core. If it is `pooled`, about 3 h. **Try `permutation.n_permutations: 20` first** and scale up. `--check` reruns this too.

## Assumptions (all in `config.yaml`)

From the paper (pasted 3.2, 3.3, 3.5):
- First 5 s discarded, 2-s epochs, sections of 15 epochs, one sample per section.
- Coherence as in Eq. 1 (`mne_connectivity` `coh`, magnitude, not squared), computed **across the epochs of a section**. So "section aggregation" is not a mean of per-epoch values: coherence of a single epoch is identically 1.
- Bands 0–4, 4–8, 8–13, 13–30, 30–45 Hz. The band value is the mean over frequency bins in the band.
- Four classifiers XGB, MLP, SVM, RF. Positive class = female (the paper predicts P(female)). Threshold `y > θ`.
- The `paper` model is XGB (the best in 3.5) with guessed hyperparameters. 3.5 gives none.

From LEMON (pasted blocks):
- Participant IDs are already `sub-0323xx`, matching folder names, so `name_match.csv` is not used (`participants.use_name_match: false`; the code path exists).
- Sex column `Gender_ 1=female_2=male`, 1 = female, 2 = male. Age is a 5-year bin mapped to its midpoint.
- The CSV is read with delimiter sniffing, because your paste showed tabs.
- sfreq 250 Hz is asserted per file. A subject whose rate differs is skipped with the reason logged.

Every GUESS:
- `signal.condition: EC`, `signal.reference: keep`, `signal.missing_channel_policy: skip`, `connectivity.mode: multitaper`: see QUESTIONS.md #1, #3, #4.
- `sections.per_class: {female: 5, male: 8}`: paper's literal numbers, which worsen LEMON's imbalance. See QUESTIONS.md #2.
- `sections.layout: consecutive`: non-overlapping, from the first epoch after the discard. LEMON's EC file concatenates eight 1-min blocks, so some sections straddle a block boundary. `make_fixed_length_epochs` drops only epochs overlapping `BAD_*` annotations, not EEGLAB `boundary` events.
- `sections.long_start_epoch: 0`: the paper's 3-minute validation recording is taken as the first 90 kept epochs.
- `connectivity.band_edges`: bins in `[lo, hi)`, the last band `[30, 45]`. Delta uses `fmin=0` below mne-connectivity's 5-cycle floor (2.5 Hz for 2-s epochs). The warning is silenced; see QUESTIONS.md #5.
- `corr` (not the paper's measure): zero-phase 4th-order Butterworth per epoch, Pearson per epoch, mean over the section's epochs.
- `selection.importance_type: gain`, `selection.xgb_params`. Zero-importance ties are broken at random, seeded, so low column indices are not favoured. First-appearance ties are broken by the number of trials at that rank, then mean rank, then index.
- `cv`: StratifiedGroupKFold, 10 splits × 5 seeds, threshold 0.5. The paper used 50 random 90/10 subject splits and a tuned θ (best 0.39). `cv.test_representation: long` reproduces its unsectioned 3-minute validation rows.
- All model hyperparameters, permutation settings (100 permutations, 1 seed, subject-level AUC), and the smoke overrides.

## Design notes

- **Per-subject cache.** `extract` stores every complete section per subject under a hash of the feature-defining config keys (`pipeline/features.py:FEATURE_KEYS`). Changing `per_class` or `missing_channel_policy` only reassembles `X.npy` with `--force`; changing any hashed key recomputes. A subject cached under `interpolate` is re-skipped under `skip`.
- **Pooled vs in_fold.** `pooled` mirrors the paper: 34 features chosen on all subjects, then cross-validated. That leaks the test labels into selection. On the synthetic smoke cache, pooled gives AUC ≈ 1.0 where in-fold gives 0.5–0.7. Report both; the in-fold number is the honest one.
- **Leak guard.** `harness.evaluate` asserts disjoint train/test subjects in every fold. `select_features` receives only training-fold arrays.
- **Cache files.** `cache/X.npy`, `y.npy`, `subjects.npy`, `ages.npy`, `sections.csv`, `skipped.csv`, `X_long.npy` + `long_subjects.npy` (the 90-epoch row per subject), `selected.npy`, `rank_matrix.npy` (855 × 50).
- **Missing condition files** (e.g. `sub-032483`, `sub-032484`) are logged as `missing EC file` in `skipped.csv`. `download.py` warns when an archive holds only one condition.
- `--check` compares against `git show HEAD:results/table.csv` when the table is tracked, otherwise the file on disk.

## Unresolved

- QUESTIONS.md #1–#4. They decide what `extract` computes.
- The paper's section 3.4 (how training recordings are split into a "reasonably balanced" set) was not pasted. The per-class section counts come from the brief, not from the paper text.
- θ tuning (paper Fig. 4) is not implemented. The threshold is fixed per run (`cv.threshold`).
- The band-power baseline (31 features) the paper compares against is not implemented.
- `requirements.txt` is UTF-16 with CRLF line endings. pip reads it, but grep and diff do not. Consider re-saving it as UTF-8.
