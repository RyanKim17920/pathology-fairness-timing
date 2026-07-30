#!/usr/bin/env python3
"""One-time builder for a diagnosis-free cancer/race population manifest."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile


FIELDS = ("patient_barcode", "cancer_type", "race")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    destination = args.destination.resolve()
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}")
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(FIELDS).issubset(reader.fieldnames or ()):
            raise ValueError("source lacks required population fields")
        rows = [
            {field: row[field].strip() for field in FIELDS}
            for row in reader
        ]
    if not rows or any(not row[field] for row in rows for field in FIELDS):
        raise ValueError("population fields must be complete")
    if len({row["patient_barcode"] for row in rows}) != len(rows):
        raise ValueError("patient barcodes must be unique")
    rows.sort(key=lambda row: row["patient_barcode"])

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    main()
