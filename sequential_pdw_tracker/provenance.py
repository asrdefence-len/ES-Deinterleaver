"""Compact reproducibility metadata for local experiments."""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
import subprocess


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_provenance(dataset: str | Path) -> dict:
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(("git", *args), text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    path = Path(dataset)
    packages = {}
    for name in ("numpy", "h5py"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "processing_path": "raw PDWs -> causal probabilistic association -> per-emitter RF-mode HMM + online AOA/PW/PRI models",
        "dataset": {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)},
        "git": {
            "revision": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "dirty": bool(git("status", "--porcelain")),
        },
        "packages": packages,
    }
