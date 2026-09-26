"""Age confound: regress age out of every feature, fit on training rows only."""
from __future__ import annotations

import numpy as np


def residualise(X_train, age_train, X_other, age_other):
    """Per-column least squares feature ~ 1 + age on train; residuals for train and other."""
    A_tr = np.column_stack([np.ones(len(age_train)), np.asarray(age_train, dtype=float)])
    A_ot = np.column_stack([np.ones(len(age_other)), np.asarray(age_other, dtype=float)])
    beta, *_ = np.linalg.lstsq(A_tr, np.asarray(X_train, dtype=float), rcond=None)
    return X_train - A_tr @ beta, X_other - A_ot @ beta
