# Decisions (stage 2)

Choices made during the stage-2 run, each with the alternative not taken. The most conservative option for the reported numbers is the default.

## Before compute

- **D1. Band edges are not a hard stop** (asked; the human chose to proceed). The band ranges 0–4, 4–8, 8–13, 13–30, 30–45 Hz come from paper §3.3. Only the assignment of a frequency bin lying exactly on a boundary is a convention the paper doesn't state (`[lo, hi)`, last band `[lo, hi]`). The `config.yaml` comment was relabelled accordingly. *Alternative:* stop under hard stop #2 and report nothing.
- **D2. Inputs verified.** The upload arrived in two parts: the 8 cache files, then `data/` (participants CSV, name_match, EEG_Info) and `cache/subjects/647a1e76b0/`. The subject-cache hash equals `feature_hash(config.yaml)` on `develop`, and reassembling from it reproduces the uploaded `X`, `y`, `subjects`, `ages`, `X_long` and `long_subjects` exactly.
- **D3. Reassembly without raw files.** `extract_all` now discovers subjects from the per-subject cache too, and `paths.subject_cache_dir` lets a reassembly write to a different `cache_dir`. Step 6.5 ablations therefore write to `cache/ablation_*` and never overwrite the uploaded `cache/X.npy`. *Alternative:* run `run.py --stage extract --force` in place, which would need raw `.set` files and would overwrite the uploaded cache.
- **D4. Test fixture pinned to `skip`.** The `develop` commit setting `missing_channel_policy: interpolate` made `test_extract_all_skips_and_counts` fail, because it read the policy from `config.yaml`. The test now sets its own policy.

## Step 0

- **D5. Bridged electrodes kept.** 11 subjects (6 F, 5 M) have one neighbouring channel pair, or a small cluster, at coherence ≈ 1 across bands and sections. That is the signature of electrode bridging; only one of the 11 had an interpolated channel. They are kept, and the uploaded cache is used as is, so the reported numbers are not shaped by post-hoc curation. The list is in `results/cache_summary.md`. *Alternative:* exclude the 11, or set the bridged pairs to NaN and drop those columns. Either would be a separate sensitivity run, not done here.
- **D6. Age clusters split at bin midpoint 45.** Young = 20–25 … 35–40 (138 subjects, 30 % female); older = 55–60 … 75–80 (63 subjects, 49 % female). LEMON has no 40–55 bins, so the split is unambiguous.
- **D7. `results/*.npy` tracked.** A `.gitignore` exception (`!/results/*.npy`) lets `perm_null.npy` and `rank_matrix_pooled.npy` be committed. `*.npy` stays ignored everywhere else, including `results/smoke/`.

## Steps 1–5

- **D8. `paper` model not run separately.** In `models.py` it is XGB with the same parameters as `xgb`, so its rows would duplicate `xgb`. "All models" means `logreg`, `rf`, `svm_rbf`, `mlp`, `xgb`.
- **D9. One in-fold selection per (seed, fold).** It is shared by every model and by both test representations. Selection sees only the training fold's sections, and the splits do not depend on the representation, so sharing changes no number and saves roughly 10× the compute.
- **D10. `long` rows reported at subject level only.** With `test_representation: long`, each test subject contributes one 90-epoch row, so section-level metrics equal subject-level ones. The duplicates are left out of the table.
- **D11. Permutation test uses 1 seed** (`permutation.n_seeds: 1`) and recomputes pooled selection from the permuted labels every time. The observed score in the p-value is recomputed the same way, so it can differ slightly from the 5-seed table mean. *Alternative:* 5 seeds per permutation, which is 5× the budget.
- **D12. Stage-2 table has its own check.** `results/table.csv` now has the stage-2 columns (including `representation` and `notes`), so the old `run.py --check` format does not apply to it. `python scripts/stage2.py table --check` rebuilds the table from the committed per-fold CSVs in `results/stage2/` and diffs it at 1e-3.
- **D13. Figures are static PNGs** (matplotlib), using the validated categorical slots, a one-hue blue ramp for magnitude, and blue↔red with a grey midpoint for female − male differences. The pooled-feature scalp plot is faceted by band (one colour per panel) rather than overlaying five colours on one head, which keeps the lines legible.

## Step 6

- **D14. "Best in-fold configuration" = xgb / sections** (subject AUC 0.72), the highest in-fold (model, representation) pair from step 2. It is used for 6.1, 6.2 and 6.5.
- **D15. "Equal sections per class" = 5/5.** Every subject contributes 5 sections, the smaller of the committed counts, so no class gets extra samples. *Alternative:* 8/8, equally balanced but with more sections per subject (every subject has at least 13). Not run.
- **D16. 6.5 run on 3 seeds.** Two ablations at 5 seeds were projected at about 55 min (> 45). The table notes add the committed 8/5 configuration on the same 3 seeds (0.723 ± 0.022) for a like-for-like comparison.
- **D17. Older age cluster kept at 10 splits.** Both classes have ≥ 31 subjects, so 10 splits is allowed, but each test fold holds only 6–7 subjects. The AUC of 0.500 is an average of noisy fold AUCs (per-seed 0.41–0.56), and exactly 0.500 is a coincidence of the fold-size denominators. *Alternative:* 5 splits, for larger folds.
- **D18. 6.6 added: interpolation sensitivity** (not in the brief, about 1 min). Step 0 showed interpolation is more common in females, a possible confound. Logistic regression on 855 features restricted to the 131 subjects with no interpolated channel is reported as one extra row.
- **D19. Duplicate `5_table` timing rows** from rebuilding the table (to add extension rows) were collapsed to the first; the table build itself takes < 1 s.
