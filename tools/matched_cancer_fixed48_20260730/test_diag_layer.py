from __future__ import annotations

import ast
from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    file_identity,
    verify_receipt as verify_real_receipt,
)
from tools.matched_cancer_fixed48_20260730 import diag_authorization
from tools.matched_cancer_fixed48_20260730 import diag_contract
from tools.matched_cancer_fixed48_20260730 import diag_deployment
from tools.matched_cancer_fixed48_20260730 import diag_exporter
from tools.matched_cancer_fixed48_20260730 import diag_loader
from tools.matched_cancer_fixed48_20260730 import diag_structural_auditor
from tools.matched_cancer_fixed48_20260730 import diag_worker


class ContractTests(unittest.TestCase):
    def test_exact_48_seed_namespace_and_scenario(self):
        scenarios = {
            diag_contract.build_contract(seed)["scenario"]
            for seed in diag_contract.SEEDS
        }
        self.assertEqual(len(scenarios), 48)
        self.assertEqual(
            diag_contract.scenario_for(32001),
            "brca_luad_black_white_calibration_seed32001",
        )
        self.assertEqual(
            diag_contract.scenario_for(32048),
            "brca_luad_black_white_calibration_seed32048",
        )
        for invalid in (32000, 32049, True, "32001"):
            with self.assertRaises(ValueError):
                diag_contract.validate_seed(invalid)

    def test_contract_rejects_seed_scenario_redirect(self):
        value = diag_contract.build_contract(32002)
        value["scenario"] = diag_contract.scenario_for(32003)
        with self.assertRaisesRegex(ValueError, "contract differs"):
            diag_contract.validate_contract(value)

    def test_contract_freezes_cohorts_heads_and_task(self):
        value = diag_contract.build_contract(32017)
        self.assertEqual(value["arms"], ["B", "P", "H"])
        self.assertEqual(value["head_seeds"], [42001, 42002, 42003, 42004])
        self.assertEqual(value["cohorts"]["BRCA"]["task"], "brca_tp53")
        self.assertEqual(value["cohorts"]["LUAD"]["task"], "luad_tp53")
        self.assertEqual(
            value["cohorts"]["BRCA"]["expected_eligible_patients"], 328
        )
        self.assertEqual(
            value["cohorts"]["LUAD"]["expected_eligible_patients"], 281
        )


class AuthorizationAndGateTests(unittest.TestCase):
    def _legacy_declaration(self, root: Path):
        paths = {}
        for name in (
            "legacy_manifest", "legacy_loader", "tile_view",
            "brca_source", "brca_ledger", "luad_source", "luad_ledger",
        ):
            path = root / name
            path.write_text(name)
            paths[name] = str(path.resolve())
        legacy = {
            "loader_source": paths["legacy_loader"],
            "tile_view_receipt": paths["tile_view"],
            "cohorts": {
                "BRCA": {
                    "patient_records": paths["brca_source"],
                    "tile_source": str(root.resolve()),
                    "cohort_ledger": paths["brca_ledger"],
                },
                "LUAD": {
                    "patient_records": paths["luad_source"],
                    "tile_source": str(root.resolve()),
                    "cohort_ledger": paths["luad_ledger"],
                },
            },
        }
        for cancer, name in (("BRCA", "brca_source"), ("LUAD", "luad_source")):
            atomic_write_receipt(
                paths[name],
                build_receipt(
                    schema=diag_authorization.SOURCE_SCHEMA,
                    study_id=diag_contract.LEGACY_STUDY_ID,
                    scenario=diag_contract.LEGACY_SCENARIO,
                    identities={
                        "estimand_amendment": file_identity(
                            diag_authorization.AMENDMENT
                        )
                    },
                    fields={"cancer": cancer},
                ),
            )
        return paths, legacy

    def test_authorization_binds_legacy_sources_and_rejects_redirect(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, legacy = self._legacy_declaration(root)
            destination = root / "AUTH.json"
            with mock.patch.object(
                diag_authorization, "load_legacy_authorization",
                return_value=legacy,
            ):
                diag_authorization.build_authorization(
                    paths["legacy_manifest"], destination
                )
                diag_authorization.verify_authorization(destination)
                Path(paths["brca_source"]).write_text("redirected")
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    diag_authorization.verify_authorization(destination)

    def test_authorization_rejects_wrong_estimand_amendment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, legacy = self._legacy_declaration(root)
            wrong = root / "wrong_amendment.md"
            wrong.write_text("wrong")
            atomic_write_receipt(
                paths["brca_source"],
                build_receipt(
                    schema=diag_authorization.SOURCE_SCHEMA,
                    study_id=diag_contract.LEGACY_STUDY_ID,
                    scenario=diag_contract.LEGACY_SCENARIO,
                    identities={"estimand_amendment": file_identity(wrong)},
                    fields={"cancer": "BRCA"},
                ),
            )
            with mock.patch.object(
                diag_authorization, "load_legacy_authorization",
                return_value=legacy,
            ):
                with self.assertRaisesRegex(ValueError, "amendment ancestry"):
                    diag_authorization.build_authorization(
                        paths["legacy_manifest"], root / "AUTH.json"
                    )

    def test_gate_source_tamper_fails_before_loader_resolution(self):
        seed = 32008
        common = {"canonical_path": "/x", "bytes": 1, "sha256": "0" * 64}
        identities = {
            "deployment_contract": common,
            "calibration_root_receipt": common,
            "calibration_audit_receipt": common,
            "completion_receipts": {
                arm: common for arm in diag_contract.ARMS
            },
            "authorization_manifest": common,
            "legacy_authorization_manifest": common,
            "legacy_tile_view_receipt": common,
            "legacy_cohorts": {},
            "loader_source": common,
            "estimand_amendment": common,
            "sources": {
                name: (
                    {"canonical_path": "/bad", "bytes": 2,
                     "sha256": "f" * 64}
                    if name == "diag_contract" else common
                )
                for name, path in diag_deployment.RUNTIME_SOURCE_PATHS.items()
            },
        }
        fake = {
            "schema": diag_contract.GATE_SCHEMA,
            "study_id": diag_contract.STUDY_ID,
            "scenario": diag_contract.scenario_for(seed),
            "representation_seed": seed,
            "identities": identities,
        }
        with mock.patch.object(
            diag_deployment, "verify_receipt", return_value=fake
        ), mock.patch.object(
            diag_deployment, "file_identity",
            return_value=common,
        ):
            with self.assertRaisesRegex(ValueError, "runtime source"):
                diag_deployment.verify_gate("/gate")

    def test_loader_resolution_is_explicit_and_source_checkable(self):
        loader, source = diag_deployment.resolve_loader(
            diag_authorization.LOADER_ENTRYPOINT
        )
        self.assertIs(loader, diag_loader.load)
        self.assertEqual(source, Path(diag_loader.__file__).resolve())

    def test_runtime_import_redirection_fails_closed(self):
        gate = {
            "identities": {
                "sources": {
                    name: file_identity(path)
                    for name, path in (
                        ("legacy_runner", Path(
                            diag_loader.legacy_runner.__file__
                        )),
                        ("legacy_vetted_loader", Path(
                            diag_loader.legacy_loader_module.__file__
                        )),
                        ("fairness_eval", diag_deployment.RUNTIME_SOURCE_PATHS[
                            "fairness_eval"
                        ]),
                        ("post_hoc_debias", diag_deployment.RUNTIME_SOURCE_PATHS[
                            "post_hoc_debias"
                        ]),
                        ("stage_objectives", diag_deployment.RUNTIME_SOURCE_PATHS[
                            "stage_objectives"
                        ]),
                    )
                }
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            redirected = Path(temporary) / "runner.py"
            redirected.write_text("redirect")
            with mock.patch.object(
                diag_loader.legacy_runner, "__file__", str(redirected)
            ):
                with self.assertRaisesRegex(ValueError, "redirected"):
                    diag_loader._verify_runtime_callables(gate)

    def test_calibration_audit_exact_root_and_arm_ancestry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root_receipt = root / "root.json"
            root_receipt.write_text("root")
            paths = {}
            for arm in diag_contract.ARMS:
                path = root / f"{arm}.json"
                path.write_text(arm)
                paths[arm] = path
            calibration = {"root_path": root_receipt, "paths": paths}
            audit = {
                "status": "fixed48_calibration_independent_audit_pass",
                "representation_seed": 32001,
                "values_or_outcomes_accessed": False,
                "identities": {
                    "root_completion_receipt": file_identity(root_receipt),
                    "runs": {
                        arm: {"completion_receipt": file_identity(paths[arm])}
                        for arm in diag_contract.ARMS
                    },
                },
            }
            with mock.patch.object(
                diag_deployment, "verify_receipt", return_value=audit
            ):
                diag_deployment.verify_calibration_audit(
                    root / "audit.json", calibration, seed=32001
                )
                audit["identities"]["root_completion_receipt"] = (
                    file_identity(paths["B"])
                )
                with self.assertRaisesRegex(ValueError, "root ancestry"):
                    diag_deployment.verify_calibration_audit(
                        root / "audit.json", calibration, seed=32001
                    )


class LegacyAdapterTests(unittest.TestCase):
    def test_current_wrapper_cannot_be_mistaken_for_legacy_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = root / "cohort.jsonl"
            source = root / "source"
            ledger = root / "ledger"
            old_loader = root / "loader"
            gate_path = root / "gate"
            for path in (records, source, ledger, old_loader, gate_path):
                path.write_text(path.name)
            legacy_receipt = atomic_write_receipt(
                root / "legacy_receipt.json",
                build_receipt(
                    schema=diag_loader.LEGACY_COHORT_SCHEMA,
                    study_id=diag_contract.LEGACY_STUDY_ID,
                    scenario=diag_contract.LEGACY_SCENARIO,
                    identities={
                        "source_bundle": file_identity(source),
                        "tile_ledger": file_identity(ledger),
                        "cohort_records": file_identity(records),
                        "loader": file_identity(old_loader),
                    },
                    fields={
                        "cancer": "BRCA", "task": "brca_tp53",
                        "patient_count": 328, "raw_target_count": 334,
                        "eligible_patient_count": 328, "tile_count": 400,
                        "split_seed": 288_850_999,
                        "fold_counts": {str(i): 1 for i in range(5)},
                        "exclusions_by_race": {
                            "Asian": 5,
                            "American Indian or Alaska Native": 1,
                        },
                        "race_counts": {"Black": 118, "White": 210},
                        "eligible_patient_ids_sha256": "a" * 64,
                        "fold_sha256": "b" * 64,
                    },
                ),
            )
            gate = {
                "study_id": diag_contract.STUDY_ID,
                "scenario": diag_contract.scenario_for(32011),
                "representation_seed": 32011,
            }
            destination = root / "current" / "COHORT_RECEIPT.json"
            with mock.patch.object(
                diag_loader, "verify_gate", return_value=gate
            ):
                diag_loader._wrap_cohort(
                    cancer="BRCA",
                    legacy_receipt_path=legacy_receipt,
                    cohort_records=records,
                    gate_receipt=gate_path,
                    destination=destination,
                )
            value = json.loads(destination.read_text())
            self.assertEqual(value["schema"], diag_contract.COHORT_SCHEMA)
            self.assertEqual(value["scenario"], gate["scenario"])
            self.assertEqual(
                value["identities"]["legacy_cohort_receipt"],
                file_identity(legacy_receipt),
            )
            self.assertEqual(
                value["identities"]["deployment_gate"],
                file_identity(gate_path),
            )
            self.assertNotEqual(
                value["schema"], diag_loader.LEGACY_COHORT_SCHEMA
            )


class RepresentationFirewallTests(unittest.TestCase):
    def _cache(self, root: Path, metadata: dict) -> Path:
        path = root / "cache.npz"
        embeddings = np.zeros((2, 128), dtype=np.float32)
        embeddings[:, 0] = 1
        source = metadata["source_identity"]
        metadata["cache_key"] = hashlib.sha256(
            (
                diag_structural_auditor.CACHE_SCHEMA + "\0"
                + json.dumps(
                    source, sort_keys=True, separators=(",", ":")
                )
            ).encode()
        ).hexdigest()
        barcodes = np.asarray(["p1", "p2"])
        keep = np.asarray([True, True])
        payload_sha = np.asarray(["a" * 64, "b" * 64])
        payload_bytes = np.asarray([1, 1], dtype=np.int64)
        source["ordered_tiles_sha256"] = (
            diag_structural_auditor.reliable._ordered_digest_from_evidence(
                barcodes, payload_sha, payload_bytes
            )
        )
        # cache_key includes ordered evidence, so recompute after setting it.
        metadata["cache_key"] = hashlib.sha256(
            (
                diag_structural_auditor.CACHE_SCHEMA + "\0"
                + json.dumps(
                    source, sort_keys=True, separators=(",", ":")
                )
            ).encode()
        ).hexdigest()
        entry = diag_structural_auditor.reliable._entry_sha256(
            metadata, embeddings, barcodes, keep, barcodes,
            payload_sha, payload_bytes,
        )
        np.savez(
            path,
            emb=embeddings,
            barcodes=barcodes,
            keep_mask=keep,
            input_barcodes=barcodes,
            payload_sha256=payload_sha,
            payload_bytes=payload_bytes,
            metadata_json=np.asarray(json.dumps(metadata)),
            entry_sha256=np.asarray(entry),
        )
        return path

    def test_cache_accepts_pixels_checkpoint_control_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = {
                "schema": diag_structural_auditor.CACHE_SCHEMA,
                "cache_key": "d" * 64,
                "source_identity": {
                    "completion_receipt": {"canonical_path": "/a", "bytes": 1,
                                           "sha256": "a" * 64},
                    "checkpoint": {"canonical_path": "/b", "bytes": 1,
                                   "sha256": "b" * 64},
                    "encoder_state_sha256": "e" * 64,
                    "adapter_state_sha256": "f" * 64,
                    "sources": {"runner": {"canonical_path": "/r", "bytes": 1,
                                            "sha256": "1" * 64}},
                    "ordered_tiles_sha256": "c" * 64,
                    "tile_count": 2,
                    "tag_sha256": "2" * 64,
                    "normalization": "per_tile_l2",
                    "embedding_dim": 128,
                },
            }
            path = self._cache(root, metadata)
            summary = diag_structural_auditor._verify_cache(
                path, file_identity(path)
            )
            self.assertEqual(summary["columns"], 128)

    def test_cache_rejects_downstream_outcome_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = {
                "schema": diag_structural_auditor.CACHE_SCHEMA,
                "cache_key": "d" * 64,
                "source_identity": {
                    "y_true": [0, 1],
                    "tile_count": 2,
                    "normalization": "per_tile_l2",
                    "embedding_dim": 128,
                },
            }
            path = self._cache(root, metadata)
            with self.assertRaisesRegex(ValueError, "outcome key"):
                diag_structural_auditor._verify_cache(
                    path, file_identity(path)
                )

    def test_cache_rejects_self_consistent_file_with_wrong_entry_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = {
                "schema": diag_structural_auditor.CACHE_SCHEMA,
                "source_identity": {
                    "completion_receipt": {
                        "canonical_path": "/a", "bytes": 1,
                        "sha256": "a" * 64,
                    },
                    "checkpoint": {
                        "canonical_path": "/b", "bytes": 1,
                        "sha256": "b" * 64,
                    },
                    "encoder_state_sha256": "e" * 64,
                    "adapter_state_sha256": "f" * 64,
                    "sources": {"runner": {
                        "canonical_path": "/r", "bytes": 1,
                        "sha256": "1" * 64,
                    }},
                    "ordered_tiles_sha256": "c" * 64,
                    "tile_count": 2,
                    "tag_sha256": "2" * 64,
                    "normalization": "per_tile_l2",
                    "embedding_dim": 128,
                },
            }
            path = self._cache(root, metadata)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {
                    name: np.asarray(archive[name])
                    for name in archive.files
                }
            arrays["entry_sha256"] = np.asarray("0" * 64)
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "hash differs"):
                diag_structural_auditor._verify_cache(
                    path, file_identity(path)
                )

    def test_cache_match_rejects_arm_specific_tile_input(self):
        summaries = {
            f"{cancer}|{arm}": {
                "ordered_tiles_sha256": f"{cancer}-tiles",
                "input_tile_count": 10,
                "rows": 10,
                "kept_barcodes_sha256": f"{cancer}-barcodes",
                "keep_mask_sha256": f"{cancer}-keep",
            }
            for cancer in diag_contract.CANCERS
            for arm in diag_contract.ARMS
        }
        diag_structural_auditor._verify_matched_cache_summaries(summaries)
        summaries["BRCA|H"]["ordered_tiles_sha256"] = "redirected"
        with self.assertRaisesRegex(ValueError, "B/P/H"):
            diag_structural_auditor._verify_matched_cache_summaries(summaries)

    def test_production_path_has_no_analyzer_or_verifier_import(self):
        self.assertEqual(
            diag_structural_auditor._assert_no_analysis_imports(), 5
        )
        for filename in (
            "diag_deployment.py", "diag_loader.py", "diag_exporter.py",
        ):
            tree = ast.parse(
                Path(diag_structural_auditor.__file__).with_name(
                    filename
                ).read_text()
            )
            imported = " ".join(
                (node.module or "")
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            )
            self.assertNotIn("analyzer", imported)
            self.assertNotIn("verifier", imported)


class StructuralAuditorTopologyTests(unittest.TestCase):
    @staticmethod
    def _fold_counts(n):
        return {
            str(fold): n // 5 + (1 if fold < n % 5 else 0)
            for fold in range(5)
        }

    @classmethod
    def _training_audit(cls, n):
        counts = cls._fold_counts(n)
        return [
            {
                "calibration_outer_fold": outer,
                "outer_test": {
                    "excluded_folds": [outer],
                    "train_count": n - counts[str(outer)],
                    "eval_count": counts[str(outer)],
                },
                "inner_fits": [
                    {
                        "inner_fold": inner,
                        "excluded_folds": sorted((outer, inner)),
                        "train_count": (
                            n - counts[str(outer)] - counts[str(inner)]
                        ),
                        "eval_count": counts[str(inner)],
                    }
                    for inner in range(5) if inner != outer
                ],
            }
            for outer in range(5)
        ]

    @staticmethod
    def _write_cache(
        path: Path,
        *,
        completion_identity,
        checkpoint_identity,
        encoder_sha,
        adapter_sha,
        sources,
    ):
        embeddings = np.zeros((2, 128), dtype=np.float32)
        embeddings[:, 0] = 1
        barcodes = np.asarray(["p1", "p2"])
        keep = np.asarray([True, True])
        payload_sha = np.asarray(["2" * 64, "3" * 64])
        payload_bytes = np.asarray([1, 1], dtype=np.int64)
        ordered = (
            diag_structural_auditor.reliable._ordered_digest_from_evidence(
                barcodes, payload_sha, payload_bytes
            )
        )
        metadata = {
            "schema": diag_structural_auditor.CACHE_SCHEMA,
            "source_identity": {
                "completion_receipt": completion_identity,
                "checkpoint": checkpoint_identity,
                "encoder_state_sha256": encoder_sha,
                "adapter_state_sha256": adapter_sha,
                "sources": sources,
                "ordered_tiles_sha256": ordered,
                "tile_count": 2,
                "tag_sha256": "1" * 64,
                "normalization": "per_tile_l2",
                "embedding_dim": 128,
            },
        }
        metadata["cache_key"] = hashlib.sha256(
            (
                diag_structural_auditor.CACHE_SCHEMA + "\0"
                + json.dumps(
                    metadata["source_identity"],
                    sort_keys=True, separators=(",", ":"),
                )
            ).encode()
        ).hexdigest()
        entry = diag_structural_auditor.reliable._entry_sha256(
            metadata, embeddings, barcodes, keep, barcodes,
            payload_sha, payload_bytes,
        )
        np.savez(
            path,
            emb=embeddings,
            barcodes=barcodes,
            keep_mask=keep,
            input_barcodes=barcodes,
            payload_sha256=payload_sha,
            payload_bytes=payload_bytes,
            metadata_json=np.asarray(json.dumps(metadata)),
            entry_sha256=np.asarray(entry),
        )
        return metadata["source_identity"], entry

    def test_auditor_requires_24_cells_six_caches_and_no_analysis(self):
        seed = 32007
        scenario = diag_contract.scenario_for(seed)
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            output_root = base / "run"
            output_root.mkdir()
            gate_path = base / "gate.json"
            loader_path = output_root / "LOADER_ROOT_RECEIPT.json"
            collection_path = base / "collection.jsonl"
            for path in (gate_path, loader_path, collection_path):
                path.write_text(path.name)
            collection_path.with_suffix(
                collection_path.suffix + ".receipt.json"
            ).write_text("collection receipt")
            receipt_by_path = {}
            completions = {}
            completion_values = {}
            checkpoints = {}
            for arm in diag_contract.ARMS:
                path = base / f"completion_{arm}.json"
                path.write_text(arm)
                completions[arm] = file_identity(path)
                checkpoint = base / f"checkpoint_{arm}.pt"
                checkpoint.write_text(f"checkpoint {arm}")
                checkpoints[arm] = file_identity(checkpoint)
                completion_values[arm] = {
                    "identities": {"latest_checkpoint": checkpoints[arm]},
                    "encoder_post_sha256": "5" * 64,
                    "adapter_post_sha256": "6" * 64,
                }
                receipt_by_path[path.resolve()] = completion_values[arm]
            common_source_path = base / "runtime_source.py"
            common_source_path.write_text("runtime")
            common_source = file_identity(common_source_path)
            gate_sources = {
                "legacy_runner": common_source,
                "legacy_cache": common_source,
                "stage_objectives": common_source,
                "receipts": common_source,
                "completion_receipt": common_source,
                "encoder_model": common_source,
                "reliable_fairness_head": file_identity(
                    Path(diag_structural_auditor.reliable.__file__)
                ),
            }
            expected_cell_sources = {
                "runner": gate_sources["legacy_runner"],
                "cache": gate_sources["legacy_cache"],
                "stage_objectives": gate_sources["stage_objectives"],
                "receipts": gate_sources["receipts"],
                "completion_receipt": gate_sources["completion_receipt"],
                "reliable_cache_rows": gate_sources[
                    "reliable_fairness_head"
                ],
                "encoder_model": gate_sources["encoder_model"],
            }
            gate = {
                "study_id": diag_contract.STUDY_ID,
                "scenario": scenario,
                "representation_seed": seed,
                "identities": {
                    "completion_receipts": completions,
                    "sources": gate_sources,
                },
            }
            loader = {"identities": {"diagnostics": {}, "cohorts": {}}}
            collection_exports = {}
            for cancer in diag_contract.CANCERS:
                n = diag_contract.COHORT_SIZES[cancer]
                cohort_path = base / f"{cancer}_cohort.json"
                cohort_path.write_text(cancer)
                loader["identities"]["cohorts"][cancer] = file_identity(
                    cohort_path
                )
                receipt_by_path[cohort_path.resolve()] = {
                    "fold_counts": self._fold_counts(n)
                }
                root_path = base / f"{cancer}_root.json"
                root_path.write_text(cancer)
                loader["identities"]["diagnostics"][cancer] = file_identity(
                    root_path
                )
                cells = {arm: {} for arm in diag_contract.ARMS}
                for arm in diag_contract.ARMS:
                    cache_path = base / f"cache_{cancer}_{arm}.npz"
                    sources = expected_cell_sources
                    _, cache_entry = self._write_cache(
                        cache_path,
                        completion_identity=completions[arm],
                        checkpoint_identity=checkpoints[arm],
                        encoder_sha="5" * 64,
                        adapter_sha="6" * 64,
                        sources=sources,
                    )
                    for head in diag_contract.HEAD_SEEDS:
                        coordinate = f"{seed}|{arm}|{cancer}|{head}"
                        cell_path = base / f"cell_{cancer}_{arm}_{head}.json"
                        cell_path.write_text(coordinate)
                        training_path = (
                            base / f"audit_{cancer}_{arm}_{head}.jsonl"
                        )
                        with training_path.open("w") as handle:
                            for row in self._training_audit(n):
                                handle.write(json.dumps(row) + "\n")
                        cell = {
                            "status": "complete",
                            "arm": arm,
                            "head_seed": head,
                            "task_id": diag_contract.TASK_IDS[cancer],
                            "prediction_rows": (
                                5 * diag_contract.COHORT_SIZES[cancer]
                            ),
                            "outer_rows": diag_contract.COHORT_SIZES[cancer],
                            "inner_rows": (
                                4 * diag_contract.COHORT_SIZES[cancer]
                            ),
                            "fit_topology": {"outer": 5, "inner": 20},
                            "optimizer_objective": "BCEWithLogits_task_only",
                            "encoder_pre_sha256": "5" * 64,
                            "encoder_post_sha256": "5" * 64,
                            "adapter_pre_sha256": "6" * 64,
                            "adapter_post_sha256": "6" * 64,
                            "cache_entry_sha256": cache_entry,
                            "identities": {
                                "completion_receipt": completions[arm],
                                "checkpoint": checkpoints[arm],
                                "training_audit": file_identity(training_path),
                                "adapter_cache": file_identity(cache_path),
                                "predictions": completions[arm],
                                "cohort_source": completions[arm],
                                "sources": sources,
                            },
                        }
                        receipt_by_path[cell_path.resolve()] = cell
                        cells[arm][str(head)] = file_identity(cell_path)
                        export_path = base / f"export_{coordinate}.jsonl"
                        export_receipt_path = base / f"export_{coordinate}.json"
                        export_path.write_text("sealed")
                        export_receipt_path.write_text("receipt")
                        collection_exports[coordinate] = {
                            "predictions": file_identity(export_path),
                            "receipt": file_identity(export_receipt_path),
                        }
                        receipt_by_path[export_receipt_path.resolve()] = {
                            "identities": {
                                "diagnostic_receipt": file_identity(cell_path)
                            }
                        }
                receipt_by_path[root_path.resolve()] = {
                    "identities": {"cells": cells}
                }
            collection_receipt = {
                "identities": {"exports": collection_exports}
            }
            audit_path = base / "STRUCTURAL_AUDIT_RECEIPT.json"

            def receipt_dispatch(path, *args, **kwargs):
                resolved = Path(path).resolve()
                if resolved == audit_path.resolve():
                    return verify_real_receipt(path, *args, **kwargs)
                return receipt_by_path[resolved]

            with mock.patch.object(
                diag_structural_auditor, "verify_gate", return_value=gate
            ), mock.patch.object(
                diag_structural_auditor, "verify_loader_result",
                return_value=loader,
            ), mock.patch.object(
                diag_structural_auditor, "verify_collection",
                return_value=collection_receipt,
            ), mock.patch.object(
                diag_structural_auditor, "verify_receipt",
                side_effect=receipt_dispatch,
            ):
                diag_structural_auditor.audit(
                    output_root=output_root,
                    deployment_gate_receipt=gate_path,
                    loader_root_receipt=loader_path,
                    collection=collection_path,
                    expected_fm_seed=seed,
                    destination=audit_path,
                )
            value = json.loads(audit_path.read_text())
            self.assertEqual(value["cell_count"], 24)
            self.assertEqual(value["cache_count"], 6)
            self.assertEqual(value["export_count"], 24)
            self.assertIs(value["outcome_used_in_representation"], False)
            self.assertIs(value["analysis_imported_or_invoked"], False)

    def test_training_audit_rejects_wrong_fit_counts(self):
        n = 328
        rows = self._training_audit(n)
        counts = self._fold_counts(n)
        diag_structural_auditor._verify_training_audit(
            rows, fold_counts=counts
        )
        rows[0]["outer_test"]["train_count"] += 1
        with self.assertRaisesRegex(ValueError, "outer counts"):
            diag_structural_auditor._verify_training_audit(
                rows, fold_counts=counts
            )


class CollectionTests(unittest.TestCase):
    @staticmethod
    def _prediction_rows(seed: int, arm: str, cancer: str, head: int):
        n = diag_contract.COHORT_SIZES[cancer]
        for index in range(n):
            patient = f"{cancer}-P{index:04d}"
            fold = index % 5
            common = {
                "schema": diag_contract.ROW_SCHEMA,
                "fm_seed": seed,
                "arm": arm,
                "cancer": cancer,
                "head_seed": head,
                "patient_id": patient,
                "y_true": index % 2,
                "race": "Black" if index % 3 == 0 else "White",
                "fold": fold,
                "probability": 0.5,
            }
            yield {
                **common, "role": "outer_test", "outer_fold": fold,
                "inner_fold": None,
            }
            for outer in range(5):
                if outer != fold:
                    yield {
                        **common, "role": "inner_calibration",
                        "outer_fold": outer, "inner_fold": fold,
                    }

    def test_exact_24_cells_and_36540_rows_without_analysis(self):
        seed = 32001
        scenario = diag_contract.scenario_for(seed)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate_path = root / "gate.json"
            loader_path = root / "loader.json"
            gate_path.write_text("gate")
            loader_path.write_text("loader")
            loader_value = {"identities": {"diagnostics": {}}}
            cell_paths = {}
            nested_paths = {}
            for cancer in diag_contract.CANCERS:
                cells = {arm: {} for arm in diag_contract.ARMS}
                for arm in diag_contract.ARMS:
                    for head in diag_contract.HEAD_SEEDS:
                        coordinate = (seed, arm, cancer, head)
                        nested = root / f"nested_{arm}_{cancer}_{head}.jsonl"
                        nested.write_text("nested")
                        cell_path = root / f"cell_{arm}_{cancer}_{head}.json"
                        atomic_write_receipt(
                            cell_path,
                            build_receipt(
                                schema=diag_exporter.DIAGNOSTIC_CELL_SCHEMA,
                                study_id=diag_contract.STUDY_ID,
                                scenario=scenario,
                                identities={
                                    "predictions": file_identity(nested)
                                },
                                fields={"arm": arm, "head_seed": head},
                            ),
                        )
                        cells[arm][str(head)] = file_identity(cell_path)
                        cell_paths[coordinate] = cell_path
                        nested_paths[coordinate] = nested
                diagnostic_path = root / f"diagnostic_{cancer}.json"
                atomic_write_receipt(
                    diagnostic_path,
                    build_receipt(
                        schema=diag_exporter.DIAGNOSTIC_ROOT_SCHEMA,
                        study_id=diag_contract.STUDY_ID,
                        scenario=scenario,
                        identities={"cells": cells},
                    ),
                )
                loader_value["identities"]["diagnostics"][cancer] = (
                    file_identity(diagnostic_path)
                )
            exports = []
            for arm in diag_contract.ARMS:
                for cancer in diag_contract.CANCERS:
                    for head in diag_contract.HEAD_SEEDS:
                        path = root / f"{arm}_{cancer}_{head}.jsonl"
                        rows = list(self._prediction_rows(
                            seed, arm, cancer, head
                        ))
                        with path.open("w") as handle:
                            for row in rows:
                                handle.write(json.dumps(row) + "\n")
                        receipt = build_receipt(
                            schema=diag_contract.EXPORT_SCHEMA,
                            study_id=diag_contract.STUDY_ID,
                            scenario=scenario,
                            identities={
                                "deployment_gate_receipt": file_identity(
                                    gate_path
                                ),
                                "loader_root_receipt": file_identity(
                                    loader_path
                                ),
                                "exported_predictions": file_identity(path),
                                "exporter": file_identity(
                                    Path(diag_exporter.__file__)
                                ),
                                "diagnostic_receipt": file_identity(
                                    cell_paths[(seed, arm, cancer, head)]
                                ),
                                "nested_predictions": file_identity(
                                    nested_paths[(seed, arm, cancer, head)]
                                ),
                            },
                            fields={
                                "fm_seed": seed, "arm": arm,
                                "cancer": cancer, "head_seed": head,
                            },
                        )
                        atomic_write_receipt(
                            path.with_suffix(".jsonl.receipt.json"), receipt
                        )
                        exports.append(path)
            gate = {
                "representation_seed": seed,
                "scenario": scenario,
                "study_id": diag_contract.STUDY_ID,
                "identities": {
                    "sources": {
                        "legacy_exporter": file_identity(
                            Path(diag_exporter.legacy.__file__)
                        ),
                        "reliable_fairness_head": file_identity(
                            Path(diag_exporter.reliable.__file__)
                        ),
                    }
                },
            }
            destination = root / "seed_collection.jsonl"
            with mock.patch.object(
                diag_exporter, "verify_gate", return_value=gate
            ), mock.patch.object(
                diag_exporter, "verify_loader_result",
                return_value=loader_value,
            ):
                diag_exporter.collect_exports(
                    exports,
                    destination=destination,
                    expected_fm_seed=seed,
                    deployment_gate_receipt=gate_path,
                    loader_root_receipt=loader_path,
                )
                receipt = diag_exporter.verify_collection(
                    destination, expected_fm_seed=seed,
                    deployment_gate_receipt=gate_path,
                    loader_root_receipt=loader_path,
                )
            self.assertEqual(receipt["row_count"], 36_540)
            self.assertEqual(receipt["combination_count"], 24)
            self.assertIs(receipt["analysis_performed"], False)
            with mock.patch.object(
                diag_exporter, "verify_gate", return_value=gate
            ), mock.patch.object(
                diag_exporter, "verify_loader_result",
                return_value=loader_value,
            ):
                with self.assertRaises(ValueError):
                    diag_exporter.verify_collection(
                        destination, expected_fm_seed=32002,
                        deployment_gate_receipt=gate_path,
                        loader_root_receipt=loader_path,
                    )


class WorkerContractTests(unittest.TestCase):
    def test_worker_requires_explicit_authorization_and_paths(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                diag_worker.parser().parse_args(["32001", "attempt_01"])
        args = diag_worker.parser().parse_args([
            "32001", "attempt_01",
            "--calibration-attempt-root", "/x/seed_32001/attempt_01",
            "--diagnostic-attempt-root", "/y/seed_32001/attempt_01",
            "--authorization-manifest", "/z/AUTH.json",
        ])
        self.assertEqual(args.seed, 32001)

    def test_worker_attempt_seed_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "mismatch"):
            diag_worker._validate_attempt(
                Path("/x/seed_32002/attempt_01"),
                seed=32001,
                attempt_name="attempt_01",
            )
        with self.assertRaisesRegex(ValueError, "attempt_NN"):
            diag_worker._validate_attempt(
                Path("/x/seed_32001/latest"),
                seed=32001,
                attempt_name="latest",
            )

    def test_worker_has_no_scheduler_or_analysis_imports(self):
        source = Path(diag_worker.__file__).read_text()
        tree = ast.parse(source)
        imported = []
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
            elif isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.Call):
                calls.append(ast.unparse(node.func))
        self.assertFalse(any("analyzer" in value for value in imported))
        self.assertFalse(any("verifier" in value for value in imported))
        self.assertFalse(any(value in {"sbatch", "srun"} for value in calls))


if __name__ == "__main__":
    unittest.main()
