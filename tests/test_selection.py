import builtins
import copy

import numpy as np
import pytest

from pipeline import selection as S
from scripts.make_synthetic_cache import SIGNAL_COLS, make_cache
from pipeline.features import load_cache


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = make_cache(tmp_path_factory.mktemp("synth"), effect=1.5)
    return load_cache(d)


@pytest.fixture
def small_cfg(cfg):
    cfg["selection"]["n_trials"] = 5
    return cfg


def test_returns_34_unique_in_range(synth, small_cfg):
    idx, R = S.select_features(synth["X"], synth["y"], synth["subjects"], small_cfg, 0,
                               return_ranks=True)
    assert len(idx) == 34 == len(np.unique(idx))
    assert idx.min() >= 0 and idx.max() < 855
    assert R.shape == (855, 5)
    for t in range(5):  # every column is a permutation of ranks
        assert sorted(R[:, t]) == list(range(855))


def test_planted_columns_rank_highly(synth, small_cfg):
    for agg in ("first_appearance", "mean_rank"):
        idx = S.select_features(synth["X"], synth["y"], synth["subjects"], small_cfg, 0,
                                aggregation=agg)
        assert len(set(SIGNAL_COLS) & set(idx[:10])) >= 3, agg


def test_first_appearance_aggregation_literal():
    # feature 2 is best in trial 1 only; feature 0 is 2nd everywhere; feature 1 is 1st in trial 0
    R = np.array([[1, 1, 1],
                  [0, 2, 2],
                  [2, 0, 0],
                  [3, 3, 3]])
    order = S.aggregate_first_appearance(R)
    # rank 0 appears for features 1 (once) and 2 (twice) -> 2 before 1, then 0, then 3
    assert list(order) == [2, 1, 0, 3]
    assert list(S.aggregate_mean_rank(R)) == [2, 0, 1, 3]  # means 1, 4/3, 2/3, 3


def test_uses_only_its_arguments(synth, small_cfg, monkeypatch):
    X, y, g = synth["X"].copy(), synth["y"].copy(), synth["subjects"].copy()
    cfg_before = copy.deepcopy(small_cfg)

    def no_io(*a, **k):
        raise AssertionError("select_features touched the filesystem")

    monkeypatch.setattr(builtins, "open", no_io)
    monkeypatch.setattr(np, "load", no_io)
    a = S.select_features(X, y, g, small_cfg, 7)
    b = S.select_features(X, y, g, small_cfg, np.random.default_rng(7))
    monkeypatch.undo()
    np.testing.assert_array_equal(a, b)                 # deterministic from its rng alone
    np.testing.assert_array_equal(X, synth["X"])        # inputs not modified
    np.testing.assert_array_equal(y, synth["y"])
    assert small_cfg == cfg_before
    # training rows only: dropping subjects changes nothing it could have seen otherwise
    keep = np.isin(g, np.unique(g)[:14])
    idx = S.select_features(X[keep], y[keep], g[keep], small_cfg, 7)
    assert len(idx) == 34
