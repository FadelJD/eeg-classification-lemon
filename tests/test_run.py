import subprocess
import sys

import pandas as pd
import pytest
import yaml

from tests.conftest import ROOT, _CFG


def _run(args, cwd):
    return subprocess.run([sys.executable, str(ROOT / "run.py"), *args], cwd=cwd,
                          capture_output=True, text=True, timeout=120)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    cfg = dict(_CFG)
    cfg["paths"] = dict(_CFG["paths"], cache_dir=str(tmp / "cache"), results_dir=str(tmp / "results"))
    (tmp / "config.yaml").write_text(yaml.safe_dump(cfg))
    r = _run(["--config", "config.yaml", "--smoke"], tmp)
    assert r.returncode == 0, r.stderr[-3000:]
    return tmp


def test_smoke_completes(smoke):
    out = smoke / "results" / "smoke"
    table = pd.read_csv(out / "table.csv")
    assert list(table.columns) == ["model", "selection", "level", "metric", "mean", "std", "n_seeds"]
    assert set(table["level"]) == {"section", "subject"}
    assert (table["metric"] == "p_value").sum() == 1
    assert set(table["model"]) == set(_CFG["smoke"]["models"])
    assert (table.loc[table.metric != "p_value", "n_seeds"] == 2).all()
    assert (out / "selected_features.csv").exists()
    assert (smoke / "cache" / "smoke" / "selected.npy").exists()


def test_stages_skip_when_outputs_exist(smoke):
    r = _run(["--config", "config.yaml", "--smoke", "--stage", "select"], smoke)
    assert r.returncode == 0
    assert "exists, skipped" in r.stderr


def test_check_passes_then_fails_on_perturbed_table(smoke):
    r = _run(["--config", "config.yaml", "--smoke", "--check"], smoke)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "--check passed" in r.stdout

    path = smoke / "results" / "smoke" / "table.csv"
    table = pd.read_csv(path)
    table.loc[0, "mean"] += 0.01
    table.to_csv(path, index=False)
    r = _run(["--config", "config.yaml", "--smoke", "--check"], smoke)
    assert r.returncode == 1
    assert "--check FAILED" in r.stderr
