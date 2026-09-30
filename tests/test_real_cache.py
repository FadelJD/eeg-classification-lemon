"""Integrity of the real cache (stage 2). Skipped when cache/X.npy is absent."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "cache" / "X.npy").exists(),
                                reason="real cache not present")


def test_real_cache_integrity(cfg):
    from pipeline.features import load_cache
    from scripts.stage2 import check_cache

    c = load_cache(ROOT / "cache")
    subj = check_cache(c, cfg)   # 855 cols, no NaN/constant, aligned, one X_long row, counts
    assert len(subj) == len(set(c["long_subjects"]))
