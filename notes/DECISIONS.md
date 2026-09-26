# Decisions (stage 2)

Choices made during the stage-2 run, each with the alternative not taken. The most conservative option for the reported numbers is the default.

## Before compute

- **D1. Band edges are not a hard stop** (asked; the human chose to proceed). The band ranges 0–4, 4–8, 8–13, 13–30, 30–45 Hz come from paper §3.3. Only the assignment of a frequency bin lying exactly on a boundary is a convention the paper doesn't state (`[lo, hi)`, last band `[lo, hi]`). The `config.yaml` comment was relabelled accordingly. *Alternative:* stop under hard stop #2 and report nothing.
- **D2. Inputs verified.** The upload arrived in two parts: the 8 cache files, then `data/` (participants CSV, name_match, EEG_Info) and `cache/subjects/647a1e76b0/`. The subject-cache hash equals `feature_hash(config.yaml)` on `develop`, and reassembling from it reproduces the uploaded `X`, `y`, `subjects`, `ages`, `X_long` and `long_subjects` exactly.
- **D3. Reassembly without raw files.** `extract_all` now discovers subjects from the per-subject cache too, and `paths.subject_cache_dir` lets a reassembly write to a different `cache_dir`. Step 6.5 ablations therefore write to `cache/ablation_*` and never overwrite the uploaded `cache/X.npy`. *Alternative:* run `run.py --stage extract --force` in place, which would need raw `.set` files and would overwrite the uploaded cache.
- **D4. Test fixture pinned to `skip`.** The `develop` commit setting `missing_channel_policy: interpolate` made `test_extract_all_skips_and_counts` fail, because it read the policy from `config.yaml`. The test now sets its own policy.
