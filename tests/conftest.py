import copy
from pathlib import Path

import pytest

from pipeline.features import load_config
from scripts.make_synthetic_cache import FIXTURE, make_fixture

ROOT = Path(__file__).resolve().parents[1]
_CFG = load_config(ROOT / "config.yaml")
_CFG["selection"]["n_jobs"] = 1  # tiny fits; test_selection covers the parallel path


@pytest.fixture
def cfg():
    return copy.deepcopy(_CFG)


@pytest.fixture(scope="session")
def raw_fixture():
    if not FIXTURE.exists():
        make_fixture(FIXTURE)
    return FIXTURE
