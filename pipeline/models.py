"""Model zoo: name -> sklearn Pipeline(StandardScaler, model). Hyperparameters from cfg."""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


def _params(cfg, name):
    p = dict((cfg.get("model_params") or {}).get(name) or {})
    if "hidden_layer_sizes" in p:
        p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
    return p


FACTORIES = {
    "logreg": lambda p: LogisticRegression(**p),
    "rf": lambda p: RandomForestClassifier(**p),
    # Platt-scaled probabilities; SVC(probability=True) is deprecated from sklearn 1.9
    "svm_rbf": lambda p: CalibratedClassifierCV(SVC(kernel="rbf", **p), ensemble=False),
    "mlp": lambda p: MLPClassifier(**p),
    "xgb": lambda p: XGBClassifier(verbosity=0, **p),
}


def make_model(name, cfg) -> Pipeline:
    if name == "paper":
        # 3.5 names XGB as the best of its four classifiers but gives no hyperparameters,
        # so "paper" is XGB with the configured (GUESS) xgb parameters.
        return Pipeline([("scale", StandardScaler()), ("clf", FACTORIES["xgb"](_params(cfg, "xgb")))])
    if name not in FACTORIES:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(FACTORIES) + ['paper']}")
    return Pipeline([("scale", StandardScaler()), ("clf", FACTORIES[name](_params(cfg, name)))])


def get_models(cfg) -> dict:
    """Every known model, name -> Pipeline."""
    return {name: make_model(name, cfg) for name in list(FACTORIES) + ["paper"]}
