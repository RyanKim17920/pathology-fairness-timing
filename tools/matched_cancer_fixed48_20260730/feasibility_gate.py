#!/usr/bin/env python3
"""Pass/fail-only denominator feasibility gate for the fixed diagnostic.

The gate opens the already authorized outcome sources only to prove that every
locked nested-threshold and equalized-odds denominator is nonempty.  It never
persists patient IDs, labels, counts, or subgroup summaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold

from tools import fairness_eval, post_hoc_debias
from tools.matched_cancer_diagnostic_20260730 import vetted_loader as legacy
from tools.matched_cancer_diagnostic_20260730.deployment import (
    load_authorization_manifest as load_legacy_authorization,
)
from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    file_identity,
    verify_receipt,
)

from .diag_authorization import verify_authorization
from .diag_contract import (
    CANCERS,
    LEGACY_SCENARIO,
    LEGACY_STUDY_ID,
    STUDY_ID,
    cohort_contracts,
)


SCHEMA = "matched-cancer-fixed48-feasibility-gate/v1"
SCENARIO = "fixed48_denominator_feasibility"
SPLIT_SEED = 288_850_999


def _authorized_cohort(
    *,
    cancer: str,
    declaration: Mapping[str, str],
    contract: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    source_path = Path(declaration["patient_records"]).resolve(strict=True)
    source = legacy._source_bundle(
        source_path,
        cancer=cancer,
        task=contract["task"],
        cohort_contract=contract,
        expected_study_id=LEGACY_STUDY_ID,
        expected_scenario=LEGACY_SCENARIO,
    )
    identities = source["identities"]
    demo = fairness_eval.load_demographics(
        identities["demographics_csv"]["canonical_path"], "patient_barcode"
    )
    molecular = fairness_eval.load_demographics(
        identities["molecular_csv"]["canonical_path"], "patient_barcode"
    )
    label_of, _, task_cohort = post_hoc_debias.build_task_cohort(
        contract["task"],
        demo,
        molecular,
        None,
        None,
        lambda *_: None,
    )
    frozen = legacy._target_rows(
        Path(identities["frozen_folds_csv"]["canonical_path"])
    )
    raw_target = sorted(set(frozen) & set(task_cohort))
    if len(raw_target) != contract["expected_target_rows"]:
        raise ValueError("raw target cardinality differs")
    target = [
        patient
        for patient in raw_target
        if legacy._canonical_race(frozen[patient].get("race"))
        in {"Black", "White"}
    ]
    if len(target) != contract["expected_eligible_patients"]:
        raise ValueError("eligible target cardinality differs")
    labels = np.asarray([int(label_of[patient]) for patient in target], dtype=int)
    races = np.asarray([
        legacy._canonical_race(frozen[patient].get("race"))
        for patient in target
    ], dtype=object)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("outcome is not binary")
    folds = np.full(len(target), -1, dtype=int)
    splitter = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=SPLIT_SEED
    )
    for fold, (_, heldout) in enumerate(splitter.split(target, labels)):
        folds[heldout] = fold
    if set(folds.tolist()) != set(range(5)):
        raise ValueError("outer folds are incomplete")

    # The final endpoint pools both race/outcome denominators, and each nested
    # heldout fold needs both heldout and calibration White negatives.
    for race in ("Black", "White"):
        for outcome in (0, 1):
            if not np.any((races == race) & (labels == outcome)):
                raise ValueError("required pooled denominator is empty")
    for heldout in range(5):
        white_negative = (races == "White") & (labels == 0)
        if (
            not np.any(white_negative & (folds == heldout))
            or not np.any(white_negative & (folds != heldout))
        ):
            raise ValueError("required nested White-negative denominator is empty")
    return labels, folds


def create(
    *,
    authorization_manifest: str | Path,
    destination: str | Path,
) -> Path:
    authorization_path = Path(authorization_manifest).resolve(strict=True)
    authorization = verify_authorization(authorization_path)
    legacy_path = Path(
        authorization["identities"]["legacy_authorization_manifest"][
            "canonical_path"
        ]
    ).resolve(strict=True)
    legacy_authorization = load_legacy_authorization(legacy_path)
    contracts = cohort_contracts()
    try:
        for cancer in CANCERS:
            _authorized_cohort(
                cancer=cancer,
                declaration=legacy_authorization["cohorts"][cancer],
                contract=contracts[cancer],
            )
    except Exception as error:
        # Do not expose a subgroup, count, outcome, or patient through a durable
        # failure artifact or CLI message.
        raise RuntimeError("fixed48 denominator feasibility: FAIL") from None

    receipt = build_receipt(
        schema=SCHEMA,
        study_id=STUDY_ID,
        scenario=SCENARIO,
        identities={
            "authorization_manifest": file_identity(authorization_path),
            "legacy_authorization_manifest": file_identity(legacy_path),
            "source_bundles": {
                cancer: file_identity(
                    legacy_authorization["cohorts"][cancer][
                        "patient_records"
                    ]
                )
                for cancer in CANCERS
            },
            "gate_source": file_identity(Path(__file__)),
        },
        fields={
            "status": "pass",
            "all_required_denominators_nonempty": True,
            "outcomes_opened_for_feasibility": True,
            "outcome_values_persisted": False,
            "counts_or_labels_exposed": False,
        },
    )
    output = atomic_write_receipt(destination, receipt)
    verify(output, authorization_manifest=authorization_path)
    return output


def verify(
    path: str | Path,
    *,
    authorization_manifest: str | Path | None = None,
) -> dict[str, Any]:
    receipt = verify_receipt(
        path,
        expected_schema=SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=SCENARIO,
    )
    if authorization_manifest is None:
        authorization_manifest = receipt.get("identities", {}).get(
            "authorization_manifest", {}
        ).get("canonical_path")
        if not authorization_manifest:
            raise ValueError("feasibility gate lacks authorization ancestry")
    authorization_path = Path(authorization_manifest).resolve(strict=True)
    authorization = verify_authorization(authorization_path)
    expected_fields = {
        "status": "pass",
        "all_required_denominators_nonempty": True,
        "outcomes_opened_for_feasibility": True,
        "outcome_values_persisted": False,
        "counts_or_labels_exposed": False,
    }
    for key, value in expected_fields.items():
        if receipt.get(key) != value:
            raise ValueError(f"feasibility gate {key} differs")
    identities = receipt.get("identities", {})
    if set(identities) != {
        "authorization_manifest",
        "legacy_authorization_manifest",
        "source_bundles",
        "gate_source",
    } or set(identities["source_bundles"]) != set(CANCERS):
        raise ValueError("feasibility gate identity topology differs")
    if identities["authorization_manifest"] != file_identity(
        authorization_path
    ):
        raise ValueError("feasibility authorization differs")
    legacy_path = authorization["identities"][
        "legacy_authorization_manifest"
    ]["canonical_path"]
    if identities["legacy_authorization_manifest"] != file_identity(
        legacy_path
    ):
        raise ValueError("feasibility legacy authorization differs")
    authorized_bundles = authorization["identities"]["legacy_cohorts"]
    for cancer in CANCERS:
        if identities["source_bundles"][cancer] != authorized_bundles[
            cancer
        ]["source_bundle"]:
            raise ValueError(
                f"feasibility {cancer} source bundle differs from authorization"
            )
    if identities["gate_source"] != file_identity(Path(__file__)):
        raise ValueError("feasibility source differs")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--authorization-manifest", required=True)
    create_parser.add_argument("--destination", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--authorization-manifest", required=True)
    verify_parser.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    if args.command == "create":
        create(
            authorization_manifest=args.authorization_manifest,
            destination=args.destination,
        )
    else:
        verify(
            args.receipt,
            authorization_manifest=args.authorization_manifest,
        )
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
