# EEG sex classification from connectivity: LEMON reproduction

A reproduction of *Resting-state EEG sex classification using selected brain connectivity representation*. The paper's data is not available, so this runs on the MPI-Leipzig **LEMON** preprocessed resting-state EEG set instead.

Pipeline: 19 channels, 2-s epochs, 30-s sections → coherence for 171 channel pairs × 5 bands (855 features) → XGBoost-importance selection of 34 features over 50 subject subsamples → subject-level cross-validation (XGB, MLP, SVM, RF, plus a logistic-regression baseline), with selection either pooled (as the paper does) or inside each fold (leak-free).

Every parameter lives in `config.yaml`, marked `paper`, `LEMON` or `GUESS`. Open questions are in `notes/QUESTIONS.md`. Assumptions and the first-run checklist are in `notes/handoff.md`.

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
