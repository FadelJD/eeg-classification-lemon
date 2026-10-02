"""Fetch LEMON preprocessed EEG and the participant CSVs, then unpack the archives.

Run this on your own machine only:

    python download.py --i-am-local            # sync archives + CSVs, then extract
    python download.py --i-am-local --dry-run  # print the commands, do nothing
    python download.py --i-am-local --extract-only

Needs the AWS CLI. The bucket is public, so no credentials (--no-sign-request).
Extraction is idempotent: an archive is unpacked once and marked with
<archive>.extracted next to it.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

S3_ROOT = "s3://fcp-indi/data/Projects/INDI/MPI-LEMON/"
S3_PREPROC = (S3_ROOT + "Compressed_tar/EEG_MPILMBB_LEMON/EEG_Preprocessed_BIDS_ID/"
              "EEG_Preprocessed/")
CSVS = ("Participants_MPILMBB_LEMON.csv", "name_match.csv")
EEG_FILE = re.compile(r"^(sub-[A-Za-z0-9]+)_(EC|EO)\.(set|fdt)$")


def commands(data_dir: Path) -> list[list[str]]:
    cmds = [["aws", "s3", "sync", S3_PREPROC, str(data_dir / "lemon_tars"), "--no-sign-request"]]
    cmds += [["aws", "s3", "cp", S3_ROOT + name, str(data_dir / name), "--no-sign-request"]
             for name in CSVS]
    return cmds


def extract_archive(archive: Path, dest: Path) -> list[str]:
    """Unpack one archive; .set/.fdt files land in dest/<sub>/. Returns files placed."""
    marker = archive.with_name(archive.name + ".extracted")
    if marker.exists():
        return []
    placed = []
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        with tarfile.open(archive) as tar:
            try:
                tar.extractall(tmp, filter="data")
            except TypeError:  # Python without extraction filters
                tar.extractall(tmp)
        for f in sorted(Path(tmp).rglob("*")):
            m = EEG_FILE.match(f.name)
            if not (f.is_file() and m):
                continue
            target = dest / m.group(1) / f.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.move(str(f), target)
                placed.append(str(target.relative_to(dest)))
    marker.write_text("\n".join(placed) + "\n")
    return placed


def extract_all(data_dir: Path) -> None:
    tars = sorted((data_dir / "lemon_tars").glob("*.tar.gz"))
    dest = data_dir / "lemon_preproc"
    dest.mkdir(parents=True, exist_ok=True)
    for archive in tars:
        placed = extract_archive(archive, dest)
        conds = sorted({p.split("_")[-1].split(".")[0] for p in placed})
        if placed and conds != ["EC", "EO"]:
            print(f"warning: {archive.name} holds only {conds}", file=sys.stderr)
        print(f"{archive.name}: {len(placed)} files" if placed else f"{archive.name}: already extracted")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--i-am-local", action="store_true",
                    help="required: confirms this is your own machine, not a cloud sandbox")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extract-only", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_am_local:
        print("refusing to download: pass --i-am-local on your own machine", file=sys.stderr)
        return 2
    data_dir = Path(args.data_dir)
    if not args.extract_only:
        for cmd in commands(data_dir):
            print("+", " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)
    if not args.dry_run:
        extract_all(data_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
