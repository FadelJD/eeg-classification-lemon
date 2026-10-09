# Questions for the human

Every one of these is a config value. The code supports all the options, so answering means editing `config.yaml`, not code. Current defaults are marked. Items 1 to 4 change the cached features, so please answer them before running `run.py --stage extract` on real data.

## 1. Condition (`signal.condition`)

The paper says "resting-state" and does not say eyes open or closed. LEMON has both. Current default: **EC** (the file you ran check_headers on). Options: EC, EO. Switching means a full re-extraction.

## 2. Reference and spectral estimator (`signal.reference`, `connectivity.mode`)

Neither is in 3.2 or 3.3. Current defaults: **keep LEMON's reference as delivered**, and **multitaper** (the MNE default). Alternatives: re-reference the 19-channel subset to its own average, or use `fourier`. Both change coherence values, so both mean a full re-extraction if changed.
