# EEG sex classification from connectivity: LEMON reproduction

A conceptual replication of *Resting-state EEG sex classification using selected brain connectivity representation*. The paper's data is not available, so this runs on the MPI-Leipzig **LEMON** preprocessed resting-state EEG set instead.

Pipeline: 19 channels, 2-s epochs, 30-s sections → coherence for 171 channel pairs × 5 bands (855 features) → XGBoost-importance selection of 34 features over 50 subject subsamples → subject-level cross-validation (XGB, MLP, SVM, RF, plus a logistic-regression baseline), with selection either pooled (as the paper does) or inside each fold (leak-free).

Every parameter lives in `config.yaml`, marked `paper`, `LEMON` or `GUESS`. Open questions are in `notes/open-issues.md`. Resolved all but 2. Assumptions and the first-run checklist are in `notes/assumptions.md`.

## What this shows

Coherence between 19 scalp channels carries a moderate, real sex signal on LEMON (n = 200, eyes-closed). With the paper's 34-feature selection done inside each training fold, the best model (XGBoost) reaches a subject-level AUC of 0.72 ± 0.02 over five seeds; using all 855 features with no selection gives 0.74 ± 0.01. Running selection the paper's way, on all subjects before cross-validation, raises the best model to 0.80 ± 0.02 (SVM). That gap is attributable to the selection step seeing test-subject labels; a subject-level permutation test on the pooled configuration gives p = 0.02 over 50 permutations, so the signal itself is real. The paper reports 0.89 on its own, unavailable data.

The selected feature sets are unstable: mean Jaccard overlap between folds is 0.17, and only 11 of 855 features appear in at least half of the in-fold selections. Age is not driving the result: residualising age inside each fold leaves 0.73, and the young cohort alone (n = 138) gives 0.74. The older cohort (n = 63) is too small for a stable estimate. Full table in `results/table.csv`; figures and a one-page summary in `report/`.

## How this was built

Study design, dataset substitution, configuration choices and local data extraction are mine. Pipeline code, tests and figures were written with an AI coding assistant (Claude) against the written briefs; every decision it made is logged in `notes/decisions.md`.

## Run

```bash
pip install -r requirements.txt
python -m pytest -q                  # synthetic data only, under a minute
python run.py --smoke                # whole pipeline on a synthetic cache -> results/smoke/

python download.py --i-am-local      # AWS CLI: LEMON preprocessed archives + CSVs into data/, then untar
python run.py --stage extract        # real data -> cache/; then read cache/skipped.csv
python run.py                        # select, evaluate, report -> results/table.csv
python run.py --check                # rerun evaluate + report, diff table.csv at 1e-3
```

Each stage is skipped when its output exists. Pass `--force` to recompute. `data/`, `cache/` and `results/smoke/` are gitignored. Only `results/table.csv` (and `selected_features.csv`) are meant to be committed.
