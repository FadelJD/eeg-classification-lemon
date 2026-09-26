import logging

import numpy as np
import pandas as pd
import pytest

from pipeline import features as F
from scripts.make_synthetic_cache import CHANNELS, make_raw


def test_extract_subject_shape(raw_fixture, cfg):
    X = F.extract_subject(raw_fixture, cfg)
    # 200 s - 5 s discard = 97 two-second epochs -> 6 complete 15-epoch sections
    assert X.shape == (6, 855)
    assert np.isfinite(X).all()
    alpha = X.reshape(6, 5, 171)[:, 2].mean()
    assert alpha > X.reshape(6, 5, 171)[:, [0, 1, 3, 4]].mean()  # planted alpha coupling


def test_triu_ordering_is_band_major_and_stable(cfg):
    names = F.feature_names(cfg)
    assert len(names) == 855 == F.n_features(cfg)
    assert names[0] == "delta:Fp1-Fp2"
    assert names[1] == "delta:Fp1-F7"
    assert names[170] == "delta:O1-O2"
    assert names[171] == "theta:Fp1-Fp2"
    assert names[-1] == "gamma:O1-O2"
    r, c = F.triu_pairs(19)
    np.testing.assert_array_equal(r, np.triu_indices(19, k=1)[0])
    np.testing.assert_array_equal(c, np.triu_indices(19, k=1)[1])


def test_channel_order_in_file_does_not_change_features(tmp_path, cfg):
    raw = make_raw(duration=40.0, seed=3)
    a = tmp_path / "a_raw.fif"
    raw.save(a, verbose="ERROR")
    shuffled = raw.copy().reorder_channels(list(reversed(CHANNELS)))
    shuffled.rename_channels({"Fz": "FZ", "O1": "o1"})  # case must not matter
    b = tmp_path / "b_raw.fif"
    shuffled.save(b, verbose="ERROR")
    np.testing.assert_allclose(F.extract_subject(a, cfg), F.extract_subject(b, cfg))


def test_corr_method_shape(raw_fixture, cfg):
    cfg["connectivity"]["method"] = "corr"
    X = F.extract_subject(raw_fixture, cfg)
    assert X.shape == (6, 855)
    assert np.abs(X).max() <= 1.0


# ---------------------------------------------------------------- extract_all

@pytest.fixture(scope="module")
def fake_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("lemon")
    pre = root / "preproc"
    specs = {  # subject: (sex code, age bin, channels dropped, has EC file)
        "synth-f1": (1, "20-25", [], True),
        "synth-m1": (2, "65-70", [], True),
        "synth-m2": (2, "25-30", ["T7"], True),
        "synth-f2": (1, "70-75", [], False),
    }
    for i, (sub, (_, _, drop, has_ec)) in enumerate(specs.items()):
        (pre / sub).mkdir(parents=True)
        raw = make_raw(duration=100.0, seed=10 + i)  # 47 epochs -> 3 sections
        if drop:
            raw.drop_channels(drop)
        raw.save(pre / sub / f"{sub}_{'EC' if has_ec else 'EO'}_raw.fif", verbose="ERROR")
    rows = [f"{s}\t{code}\t{age}" for s, (code, age, _, _) in specs.items()]
    (root / "participants.csv").write_text(
        "ID\tGender_ 1=female_2=male\tAge\n" + "\n".join(rows) + "\n")
    return root


def _cfg_for(cfg, root, cache):
    cfg["paths"].update(preproc_dir=str(root / "preproc"),
                        participants_csv=str(root / "participants.csv"),
                        file_pattern="{sub}/{sub}_{cond}_raw.fif",
                        cache_dir=str(cache))
    cfg["sections"]["per_class"] = {"female": 2, "male": 3}
    cfg["sections"]["long_epochs"] = 20
    cfg["signal"]["missing_channel_policy"] = "skip"  # independent of the committed choice
    return cfg


def test_extract_all_skips_and_counts(fake_dataset, tmp_path, cfg, caplog):
    cfg = _cfg_for(cfg, fake_dataset, tmp_path / "cache")
    with caplog.at_level(logging.WARNING, logger="pipeline.features"):
        F.extract_all(cfg)
    cache = tmp_path / "cache"
    skipped = pd.read_csv(cache / "skipped.csv").set_index("subject")["reason"]
    assert set(skipped.index) == {"synth-m2", "synth-f2"}
    assert "T7" in skipped["synth-m2"]
    assert skipped["synth-f2"] == "missing EC file"
    assert "synth-m2" in caplog.text and "T7" in caplog.text

    c = F.load_cache(cache)
    sections = pd.read_csv(cache / "sections.csv")
    counts = sections.groupby("subject").size().to_dict()
    assert counts == {"synth-f1": 2, "synth-m1": 3}               # per-class counts
    assert c["X"].shape == (5, 855)
    assert list(c["y"]) == [1, 1, 0, 0, 0]                          # female positive
    assert list(c["ages"]) == [22.5, 22.5, 67.5, 67.5, 67.5]
    assert (sections["n_epochs_used"] == 15).all()
    assert list(c["long_subjects"]) == ["synth-f1", "synth-m1"]
    assert c["X_long"].shape == (2, 855)


def test_interpolate_policy_keeps_subject_and_cache_does_not_leak(fake_dataset, tmp_path, cfg):
    cfg = _cfg_for(cfg, fake_dataset, tmp_path / "cache")
    cfg["signal"]["missing_channel_policy"] = "interpolate"
    F.extract_all(cfg)
    assert "synth-m2" in set(F.load_cache(tmp_path / "cache")["subjects"])
    # same per-subject cache, back to 'skip': the interpolated subject must be skipped again
    cfg["signal"]["missing_channel_policy"] = "skip"
    F.extract_all(cfg)
    assert "synth-m2" not in set(F.load_cache(tmp_path / "cache")["subjects"])


def test_reassembly_from_subject_cache_without_raw_files(fake_dataset, tmp_path, cfg):
    cfg = _cfg_for(cfg, fake_dataset, tmp_path / "cache")
    F.extract_all(cfg)
    first = F.load_cache(tmp_path / "cache")
    # new section counts, raw files gone, per-subject cache kept elsewhere
    cfg["paths"].update(preproc_dir=str(tmp_path / "no_raw"), cache_dir=str(tmp_path / "re"),
                        subject_cache_dir=str(tmp_path / "cache" / "subjects"))
    cfg["sections"]["per_class"] = {"female": 1, "male": 1}
    F.extract_all(cfg)
    re = F.load_cache(tmp_path / "re")
    assert list(re["subjects"]) == ["synth-f1", "synth-m1"]
    np.testing.assert_array_equal(re["X"], first["X"][[0, 2]])
