import io
import subprocess
import tarfile

import download


def test_refuses_without_flag(monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    assert download.main([]) == 2
    assert "refusing" in capsys.readouterr().err


def test_dry_run_prints_commands_only(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    assert download.main(["--i-am-local", "--dry-run", "--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "aws s3 sync " + download.S3_PREPROC in out and "--no-sign-request" in out
    assert out.count("aws s3 cp") == 2
    assert not (tmp_path / "lemon_preproc").exists()


def test_extraction_is_idempotent(tmp_path, capsys):
    tars = tmp_path / "lemon_tars"
    tars.mkdir()
    with tarfile.open(tars / "sub-fake01.tar.gz", "w:gz") as tar:
        for name in ("sub-fake01_EC.set", "sub-fake01_EC.fdt", "README.txt"):
            info = tarfile.TarInfo(f"nested/dir/{name}")
            info.size = 3
            tar.addfile(info, io.BytesIO(b"abc"))
    download.extract_all(tmp_path)
    sub = tmp_path / "lemon_preproc" / "sub-fake01"
    assert sorted(p.name for p in sub.iterdir()) == ["sub-fake01_EC.fdt", "sub-fake01_EC.set"]
    assert "only ['EC']" in capsys.readouterr().err
    download.extract_all(tmp_path)
    assert "already extracted" in capsys.readouterr().out
