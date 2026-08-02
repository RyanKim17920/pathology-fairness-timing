#!/usr/bin/env python3
"""Write a path-free, machine-readable software and accelerator receipt."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "pathology-fairness-environment/v1"


def _git_value(arguments: list[str], repository: Path) -> str | None:
    result = subprocess.run(
        ["git", *arguments], cwd=repository, text=True,
        capture_output=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_receipt(repository: Path) -> dict:
    packages = sorted(
        {
            distribution.metadata.get("Name", distribution.name):
                distribution.version
            for distribution in importlib.metadata.distributions()
        }.items(),
        key=lambda item: item[0].lower(),
    )
    receipt = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": dict(packages),
        "source": {
            "git_commit": _git_value(["rev-parse", "HEAD"], repository),
            "git_dirty": bool(_git_value(["status", "--porcelain"], repository)),
        },
        "torch": None,
    }
    try:
        import torch

        receipt["torch"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version() if torch.backends.cudnn.is_available()
                else None
            ),
            "cuda_available": torch.cuda.is_available(),
            "accelerators": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        pass
    return receipt


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    atomic_json(args.out, build_receipt(repository))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
