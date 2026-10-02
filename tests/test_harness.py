import numpy as np
import pytest
from sklearn.model_selection import KFold, StratifiedGroupKFold

from pipeline import harness as H
from pipeline.features import load_cache
from pipeline.models import get_models, make_model
from scripts.make_synthetic_cache import make_cache


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return load_cache(make_cache(tmp_path_factory.mktemp("synth"), effect=1.5))


@pytest.fixture
def fast_cfg(cfg):
    cfg["cv"].update(n_splits=3, seeds=[0, 1])
    cfg["selection"]["n_trials"] = 2
    return cfg


def test_leaky_split_is_caught(synth, fast_cfg):
    leaky = lambda seed: KFold(n_splits=3, shuffle=True, random_state=seed)  # rows, not subjects
    with pytest.raises(AssertionError, match="both train and test"):
        H.evaluate(synth["X"], synth["y"], synth["subjects"], "logreg", fast_cfg,
                   splitter_factory=leaky)


def test_sgkf_keeps_subjects_on_one_side(synth):
    g = synth["subjects"]
    for seed in range(3):
        seen = []
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
                synth["X"], synth["y"], g):
            assert not set(g[tr]) & set(g[te])
            seen += list(np.unique(g[te]))
        assert sorted(seen) == sorted(np.unique(g))  # every subject tested exactly once


def test_subject_level_aggregation():
    groups = np.array(["a", "a", "b", "b", "b", "c"])
    y = np.array([1, 1, 0, 0, 0, 1])
    p = np.array([0.2, 0.8, 0.1, 0.4, 0.7, 0.9])
    s = H.subject_level(y, p, groups)
    np.testing.assert_allclose(s.loc["a", "p"], 0.5)
    np.testing.assert_allclose(s.loc["b", "p"], 0.4)
    np.testing.assert_allclose(s.loc["c", "p"], 0.9)
    assert list(s["y"]) == [1, 0, 1]
    m = H.metrics(s["y"], s["p"], 0.5)
    # a: 0.5 is not > 0.5 -> predicted 0 (wrong); b, c right
    assert m["acc"] == pytest.approx(2 / 3)
    assert m["bal_acc"] == pytest.approx(0.75)
    assert m["auc"] == pytest.approx(1.0)


def test_in_fold_selection_sees_only_training_rows(synth, fast_cfg, monkeypatch):
    calls = []

    def fake_select(X_tr, y_tr, g_tr, cfg, rng):
        calls.append(set(np.unique(g_tr)))
        return np.arange(34)

    monkeypatch.setattr(H, "select_features", fake_select)
    g = synth["subjects"]
    cache = {}
    res = H.evaluate(synth["X"], synth["y"], g, "logreg", fast_cfg, "in_fold",
                     selection_cache=cache)
    H.evaluate(synth["X"], synth["y"], g, "rf", fast_cfg, "in_fold", selection_cache=cache)
    assert len(calls) == len(res) == 6              # once per (seed, fold), reused by rf
    for (seed, fold), train_subjects in zip(
            [(s, f) for s in (0, 1) for f in range(3)], calls):
        tr, te = list(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(
            synth["X"], synth["y"], g))[fold]
        assert train_subjects == set(g[tr])
        assert not train_subjects & set(g[te])


def test_evaluate_rows_and_signal(synth, fast_cfg):
    res = H.evaluate(synth["X"], synth["y"], synth["subjects"], "logreg", fast_cfg, "pooled")
    assert len(res) == 6
    assert (res["n_features"] == 34).all()
    for lvl in H.LEVELS:
        for m in H.METRICS:
            assert f"{lvl}_{m}" in res
    assert res["subject_auc"].mean() > 0.7     # planted signal is found


def test_long_test_representation(synth, fast_cfg):
    fast_cfg["cv"]["test_representation"] = "long"
    res = H.evaluate(synth["X"], synth["y"], synth["subjects"], "logreg", fast_cfg, None,
                     X_long=synth["X_long"], long_groups=synth["long_subjects"])
    # one test row per subject, so section and subject level agree
    np.testing.assert_allclose(res["section_auc"], res["subject_auc"])


def test_permutation_test_p_value(synth, fast_cfg):
    fast_cfg["permutation"]["n_seeds"] = 1
    out = H.permutation_test(synth["X"], synth["y"], synth["subjects"], "logreg", fast_cfg,
                             None, n_permutations=4, rng=0)
    assert out["null"].shape == (4,)
    assert out["p"] == pytest.approx((np.sum(out["null"] >= out["real"]) + 1) / 5)
    assert out["p"] == pytest.approx(0.2)        # strong planted signal beats every shuffle


def test_permuted_labels_constant_within_subject(synth):
    y = H.permute_subject_labels(synth["y"], synth["subjects"], np.random.default_rng(0))
    for s in np.unique(synth["subjects"]):
        assert len(np.unique(y[synth["subjects"] == s])) == 1
    assert y.sum() != 0


def test_models(cfg):
    models = get_models(cfg)
    assert {"logreg", "rf", "svm_rbf", "mlp", "xgb", "paper"} <= set(models)
    for m in models.values():
        assert m.steps[0][0] == "scale"
    with pytest.raises(KeyError):
        make_model("nope", cfg)
