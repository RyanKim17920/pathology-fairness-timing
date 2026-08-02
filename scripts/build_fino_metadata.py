#!/usr/bin/env python3
"""Build the patient-level ``fino_meta.json`` consumed by pretraining."""

import argparse
import csv
import json
import math
from pathlib import Path


def build(rows, id_column, discrete, continuous):
    by_id = {}
    for row in rows:
        patient_id = str(row.get(id_column, "")).strip()
        if patient_id:
            by_id[patient_id] = row

    output = {
        "discrete": {},
        "continuous": {},
        "n": {},
        "cont_dim": {},
        "categories": {},
    }
    for factor in discrete:
        values = sorted(
            {str(row.get(factor, "")).strip() for row in by_id.values()}
            - {"", "nan", "None"}
        )
        if len(values) < 2:
            raise ValueError(f"discrete factor {factor!r} has fewer than two classes")
        category_id = {value: index for index, value in enumerate(values)}
        output["categories"][factor] = values
        output["n"][factor] = len(values)
        output["discrete"][factor] = {
            patient_id: category_id[value]
            for patient_id, row in by_id.items()
            if (value := str(row.get(factor, "")).strip()) in category_id
        }

    for factor in continuous:
        numeric = {}
        for patient_id, row in by_id.items():
            try:
                value = float(row.get(factor, ""))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                numeric[patient_id] = value
        if len(numeric) < 2:
            raise ValueError(f"continuous factor {factor!r} has fewer than two values")
        mean = sum(numeric.values()) / len(numeric)
        variance = sum((value - mean) ** 2 for value in numeric.values()) / (len(numeric) - 1)
        std = math.sqrt(variance)
        if std == 0:
            raise ValueError(f"continuous factor {factor!r} has zero variance")
        output["continuous"][factor] = {
            patient_id: round((value - mean) / std, 8)
            for patient_id, value in numeric.items()
        }
        output["cont_dim"][factor] = 1
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="patient-level metadata CSV")
    parser.add_argument("--out", required=True, help="output fino_meta.json")
    parser.add_argument("--id-column", default="patient_barcode")
    parser.add_argument("--discrete", nargs="*", default=[])
    parser.add_argument("--continuous", nargs="*", default=[])
    args = parser.parse_args()
    if not args.discrete and not args.continuous:
        parser.error("select at least one --discrete or --continuous factor")
    with open(args.csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = build(rows, args.id_column, args.discrete, args.continuous)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output_path} for {len(rows)} metadata rows")


if __name__ == "__main__":
    main()
