import numpy as np

from pipeline.confound import residualise


def test_residualised_features_are_uncorrelated_with_age():
    rng = np.random.default_rng(0)
    age = np.r_[rng.choice([22.5, 27.5], 1000), rng.choice([67.5, 72.5], 1000)]
    slopes, icpts = rng.normal(0, 1, 20), rng.normal(0, 5, 20)
    X = icpts + age[:, None] * slopes + rng.normal(0, 0.5, (2000, 20))
    tr, te = np.arange(0, 2000, 2), np.arange(1, 2000, 2)
    before = [abs(np.corrcoef(age, X[:, j])[0, 1]) for j in range(20)]
    R_tr, R_te = residualise(X[tr], age[tr], X[te], age[te])
    assert max(before) > 0.9
    # held-out rows: only noise and train-fit slope error remain (sd ~0.045 at n=1000)
    after = np.array([abs(np.corrcoef(age[te], R_te[:, j])[0, 1]) for j in range(20)])
    assert after.mean() < 0.05 and after.max() < 0.15
    # fitted on train only: train residuals have exactly zero correlation
    assert max(abs(np.corrcoef(age[tr], R_tr[:, j])[0, 1]) for j in range(20)) < 1e-8
