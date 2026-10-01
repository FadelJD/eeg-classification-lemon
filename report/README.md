# Sex classification from resting-state EEG coherence: a LEMON reproduction

**What was reproduced.** *Resting-state EEG sex classification using selected brain connectivity representation*. Its dataset is unavailable, so this runs on **MPI-Leipzig LEMON**: preprocessed, eyes-closed, 201 subjects (73 female, 128 male, ages 20–80 in two clusters). The paper's pipeline is followed where it is specified:

- 19 channels at 250 Hz, first 5 s discarded, 2-s epochs, 30-s sections.
- Coherence across a section's epochs in five bands, giving 855 features.
- 50-trial XGBoost importance ranking with first-appearance aggregation, keeping the top 34.
- XGB, MLP, SVM and RF classifiers.

Every step and its config keys are in [`methods.md`](methods.md). All numbers are in [`tables/table.csv`](tables/table.csv), mean ± sd over CV seeds, subject-level unless stated.

**Headline.** Coherence carries a real but moderate sex signal.

| | subject-level AUC |
|---|---|
| Without leakage (34 features selected inside each training fold) | best 0.72 ± 0.02 (XGB) |
| Logistic regression on all 855 features, no selection | 0.74 ± 0.01 |
| Paper's protocol (features selected on all subjects before CV) | 0.80 ± 0.02 (SVM) |
| Paper | 0.89 |

The paper's protocol leaks: the permutation null of that pipeline centres on 0.59, not 0.5. The pooled SVM still beats all 50 label permutations (p = 0.020). The honest estimate is the in-fold one.

## Figures (all from `python scripts/make_figures.py`)

1. [`figures/connectivity_matrices.png`](figures/connectivity_matrices.png): mean coherence per band for female and male subjects, and female − male. Differences are small (|Δ| ≤ 0.07); females are higher in fronto-temporal delta and beta. *`fig_connectivity`*, from `cache/X.npy`.
2. [`figures/pooled_features_scalp.png`](figures/pooled_features_scalp.png): the pooled top 34 on a 10–20 layout, one head per band, line width = rank. Alpha contributes most (11). *`fig_pooled_scalp`*, from `results/selected_features_pooled.csv` (`scripts/stage2.py pooled`).
3. [`figures/auc_pooled_vs_in_fold.png`](figures/auc_pooled_vs_in_fold.png): every model, pooled vs in-fold, for section and 3-minute ("long") test rows, with the no-selection baseline marked. Pooled is higher for every model; in-fold never beats the baseline. *`fig_pooled_vs_in_fold`*, from `results/stage2/{in_fold,pooled,baselines}.csv` (`scripts/stage2.py in_fold|pooled|baselines`).
4. [`figures/permutation_null.png`](figures/permutation_null.png): 50 subject-level label permutations of the pooled SVM pipeline, with selection recomputed each time. *`fig_permutation`*, from `results/perm_null.npy` (`scripts/stage2.py permutation`).
5. [`figures/age_clusters.png`](figures/age_clusters.png): in-fold XGB within each age cluster. Young 0.74 ± 0.03; older 0.50 ± 0.05, but with only 6–7 test subjects per fold. *`fig_age_clusters`*, from `results/stage2/age_clusters.csv` (`scripts/stage2.py age_clusters`).
6. [`figures/selection_stability.png`](figures/selection_stability.png): how often each feature enters the in-fold top 34 across 50 folds. Mean pairwise Jaccard is 0.17. Only 11 features appear in at least half the folds (delta F7–T7, beta Cz–O2, delta Fz–T7, alpha P3–O2, …). *`fig_stability`*, from `results/selected_in_fold.npz` (`scripts/stage2.py stability`).

## Checks on the result

- **Age** differs by sex in LEMON (49 % female among older subjects, 30 % among young). Regressing age out inside each fold leaves AUC at 0.73.
- **Channel interpolation** is more common in females (44 % vs 30 %). Restricting to the 131 subjects with no interpolated channel gives 0.75.
- **Electrode bridging** (a neighbouring pair at coherence ≈ 1) affects 11 subjects. They are kept and listed in `results/cache_summary.md`; its effect was not tested.
- **Section counts:** 5/5 and the paper-literal 5/8 are within about one seed-sd of the committed 8/5.

## Substitution and every GUESS, stated plainly

- **Data:** LEMON, not the paper's dataset. LEMON's own preprocessing is used as delivered: its filtering, ICA and bad-channel removal.
- **Eyes-closed** recording (the paper says only "resting state").
- **Reference** kept as delivered.
- **Missing channels** interpolated (spherical spline), in 70 subjects.
- **Sections per subject:** 8 for females, 5 for males. This reverses the paper's 5/8, because LEMON's imbalance runs the other way.
- **Section layout:** consecutive, non-overlapping sections from the start. The 3-minute row is the first 90 epochs.
- **Multitaper** spectral estimator.
- **Band edges:** ranges from the paper; which band a bin exactly on a boundary joins is guessed.
- **XGBoost selector:** gain importance, 100 trees, depth 3. Ties broken at random.
- **Classifier hyperparameters:** all guessed (the paper gives none). SVM probabilities are Platt-calibrated.
- **CV:** stratified group 10-fold × 5 seeds, instead of the paper's 50 random 90/10 splits. The threshold is fixed at 0.5, not tuned.
- **Permutation test:** 50 permutations on 1 seed.

Details: `config.yaml` (every value tagged `paper`, `LEMON` or `GUESS`), [`../notes/DECISIONS.md`](../notes/DECISIONS.md), [`../notes/handoff.md`](../notes/handoff.md).
