from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import weakref

import numpy as np

from . import analyzer, contract, extractor, pipeline


class PipelinePrimitiveTests(unittest.TestCase):
    def test_full_cache_stream_retains_reference_and_subsets_not_candidates(self) -> None:
        class FakeCache:
            def __init__(self, path: Path) -> None:
                self.path = path

        reference = FakeCache(Path("/tmp/reference.npz"))
        candidate_refs: list[weakref.ReferenceType[FakeCache]] = []
        reads = 0

        def read_cache(path: Path) -> FakeCache:
            nonlocal reads
            reads += 1
            if reads == 1:
                return reference
            candidate = FakeCache(Path(path))
            candidate_refs.append(weakref.ref(candidate))
            return candidate

        def discover(seed: int, cancer: str, layer: str) -> Path:
            return Path(f"/tmp/{seed}_{cancer}_{layer}.npz")

        selection = ({"selected": True},)
        with (
            mock.patch.object(contract, "FM_SEEDS", (1,)),
            mock.patch.object(extractor, "discover_final_cache", new=discover),
            mock.patch.object(extractor, "read_full_cache", new=read_cache),
            mock.patch.object(
                extractor,
                "validate_final_cache_provenance",
                new=lambda *args, **kwargs: None,
            ),
            mock.patch.object(
                extractor,
                "assert_same_tile_evidence",
                new=lambda *args, **kwargs: None,
            ),
            mock.patch.object(
                extractor,
                "build_selection_rows",
                new=lambda *args, **kwargs: selection,
            ),
            mock.patch.object(
                extractor,
                "subset_final_embeddings",
                new=lambda *args, **kwargs: np.ones((1, 128), dtype=np.float32),
            ),
            mock.patch.object(
                pipeline,
                "file_identity",
                new=lambda path: {
                    "canonical_path": str(path), "bytes": 1, "sha256": "0" * 64
                },
            ),
        ):
            observed_reference, observed_selection, paths, subsets, identities = (
                pipeline._stream_cancer_final_caches(
                    (), cancer="BRCA", retain_subsets=True
                )
            )
        self.assertIs(observed_reference, reference)
        self.assertEqual(observed_selection, selection)
        self.assertEqual(len(paths), 3)
        self.assertEqual(len(subsets), 3)
        self.assertEqual(len(identities), 3)
        self.assertEqual(reads, 3)
        self.assertTrue(candidate_refs)
        self.assertTrue(all(item() is None for item in candidate_refs))
        self.assertNotIn(
            "full_caches",
            pipeline.ValidatedInputs.__dataclass_fields__,
        )

    def test_faircon_support_reconstructs_two_global_views(self) -> None:
        batch = [
            {"cancer": 2, "race": 2},
            {"cancer": 2, "race": 4},
            {"cancer": 15, "race": 2},
            {"cancer": 15, "race": 2},
        ]
        result = pipeline._faircon_batch_support(batch)
        self.assertEqual(result["anchor_count"], 8)
        self.assertEqual(result["denominator_candidates_per_anchor"], 7)
        self.assertEqual(result["denominator_ordered_pair_count"], 56)
        # Only the two cancer=2 source rows have an opposite-race positive.
        self.assertEqual(result["eligible_anchor_count"], 4)
        self.assertEqual(result["positive_ordered_pair_count"], 8)
        self.assertEqual(result["omitted_anchor_without_positive_count"], 4)

    def test_analyzer_records_use_payload_and_occurrence_identity(self) -> None:
        rows = []
        embeddings = []
        for rank in range(contract.TILES_PER_VIEW):
            rows.append(
                {
                    "patient_id": "P1",
                    "cancer": "BRCA",
                    "race": "Black",
                    "tss": "A2",
                    "view": "A",
                    "view_rank": rank,
                    "occurrence_index": rank * 2,
                    "global_index": rank,
                    "payload_sha256": f"{rank:064x}",
                }
            )
            embeddings.append([1.0, float(rank + 1)])
        with mock.patch.object(
            contract,
            "EXPECTED_POPULATION",
            {"BRCA": {"patients": 1}, "LUAD": {"patients": 0}},
        ):
            records = pipeline._records_for(
                np.asarray(embeddings), rows, view="A", cancer="BRCA"
            )
        self.assertEqual(len(records), 16)
        self.assertEqual(records[3]["tile_id"], f"{3:064x}:6")
        self.assertEqual(set(records[0]["metadata"]), set(contract.METADATA_ALLOWLIST))

    def test_semantic_contrasts_have_exact_40_and_10_raw_cells(self) -> None:
        race = []
        cancer = []
        for layer in contract.LAYER_DIMENSIONS:
            for seed in contract.FM_SEEDS:
                for view in contract.TILE_VIEWS:
                    cancer.append(
                        {
                            "fm_seed": seed,
                            "layer": layer,
                            "view": view,
                            "result": {"pooled_heldout_patient_auroc": 0.9},
                        }
                    )
                    for cancer_name in contract.CANCERS:
                        for level in contract.PROBE_LEVELS:
                            race.append(
                                {
                                    "fm_seed": seed,
                                    "layer": layer,
                                    "cancer": cancer_name,
                                    "view": view,
                                    "probe_level": level,
                                    "result": {"oriented_leakage": 0.2},
                                }
                            )
        result = pipeline._semantic_contrast_inputs(
            {"race_probes": race, "cancer_probes": cancer}
        )
        self.assertEqual(
            [(row["candidate"], row["baseline"]) for row in result],
            list(contract.GATE_ELIGIBLE_CONTRASTS),
        )
        self.assertTrue(all(len(row["race_cells"]) == 40 for row in result))
        self.assertTrue(all(len(row["cancer_cells"]) == 10 for row in result))

    def test_training_stream_selects_781_rows_and_one_exact_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "slot1_plain"
            run_root.mkdir()
            replay = root / "replay.json"
            replay.write_text("{}\n")
            training = []
            for step in range(1, 782):
                row = {
                    field: (False if field == "h_dose_grad_conflict" else 1.0)
                    for field in pipeline.TRAINING_LOG_FIELDS
                }
                row["step"] = step
                if step == 1:
                    row.update(
                        {
                            "encoder_cancer_grad_norm": 1.0,
                            "encoder_fair_raw_grad_norm": 1.0,
                            "encoder_fair_weighted_grad_norm": 0.0,
                            "encoder_stage_grad_finite": True,
                            "encoder_probe_parameter_names_sha256": "0" * 64,
                            "encoder_probe_parameter_count": 2,
                        }
                    )
                training.append(row)
            validation = {
                "step": 781,
                "val_dino": 1.0,
                "val_jepa": 1.0,
                "val_kde": 0.0,
                "val_cancer": 1.0,
                "val_fair": 1.0,
                "val_total": 3.0,
            }
            metrics = run_root / "metrics.jsonl"
            metrics.write_text(
                "\n".join(json.dumps(row) for row in [*training, validation])
                + "\n"
            )
            paths = {"root": root, "replay_manifest": replay}
            with (
                mock.patch.object(contract, "FM_SEEDS", (1,)),
                mock.patch.object(contract, "RUNS", ("slot1_plain",)),
                mock.patch.object(contract, "production_paths", return_value=paths),
                mock.patch.object(
                    pipeline,
                    "_replay_support",
                    return_value=[{"step": step} for step in range(1, 782)],
                ),
            ):
                result = pipeline._training_streams()
                self.assertEqual(len(result), 1)
                self.assertEqual(len(result[0]["steps"]), 781)
                self.assertEqual(result[0]["validation_row"], validation)
                self.assertIsNotNone(
                    result[0]["encoder_reachability_first_batch_only"]
                )
                with metrics.open("a") as output:
                    output.write(json.dumps(validation) + "\n")
                with self.assertRaisesRegex(
                    pipeline.PipelineError, "one validation"
                ):
                    pipeline._training_streams()

    def test_cuda_runtime_contract_is_one_named_gpu_and_nonce(self) -> None:
        environment = {
            "SLURM_JOB_NAME": "main_1gpu",
            "SLURM_JOB_ID": "123",
            "SLURM_NTASKS": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "REP_AUDIT_LAUNCH_NONCE": "a" * 32,
            "REP_AUDIT_SLURM_COMMENT": "matched_cancer_rep_audit_20260801",
        }
        with (
            mock.patch.dict("os.environ", environment, clear=True),
            mock.patch("torch.cuda.device_count", return_value=1),
        ):
            pipeline._validate_cuda_allocation({"launch_nonce": "a" * 32})
        environment["CUDA_VISIBLE_DEVICES"] = "0,1"
        with mock.patch.dict("os.environ", environment, clear=True):
            with self.assertRaisesRegex(pipeline.PipelineError, "exactly one"):
                pipeline._validate_cuda_allocation({"launch_nonce": "a" * 32})


class PipelineStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.txt"
        self.source.write_text("bound\n")
        identity = pipeline.file_identity(self.source)
        self.inputs = pipeline.ValidatedInputs(
            population=tuple(),
            references={},
            selections={},
            final_cache_paths={},
            selected_final={},
            source_identities={"one": identity},
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_preflight_is_exclusive_and_binds_topology(self) -> None:
        with (
            mock.patch.object(pipeline, "_validate_production_inputs", return_value=self.inputs),
            mock.patch.object(pipeline, "_selection_rows", return_value=tuple()),
        ):
            path = pipeline.preflight(
                self.root / "attempt", launch_nonce="a" * 32
            )
            value = json.loads(path.read_text())
            self.assertEqual(value["schema"], pipeline.PREFLIGHT_SCHEMA)
            self.assertEqual(value["expected"]["full_cache_count"], 30)
            with self.assertRaisesRegex(pipeline.PipelineError, "overwrite"):
                pipeline.preflight(
                    self.root / "attempt", launch_nonce="a" * 32
                )

    def test_bound_identity_change_is_detected(self) -> None:
        pipeline._revalidate_identity_tree(self.inputs.source_identities)
        self.source.write_text("changed\n")
        with self.assertRaisesRegex(pipeline.PipelineError, "identity changed"):
            pipeline._revalidate_identity_tree(self.inputs.source_identities)

    def test_run_orchestrates_report_receipt_and_independent_verification(self) -> None:
        attempt = self.root / "attempt"
        attempt.mkdir()
        preflight_value = {
            "schema": pipeline.PREFLIGHT_SCHEMA,
            "study_id": contract.STUDY_ID,
            "status": "pass",
            "diagnosis_free": True,
            "launch_nonce": "a" * 32,
            "expected": pipeline.EXPECTED_PREFLIGHT_TOPOLOGY,
            "population_sha256": pipeline._sha256_json(tuple()),
            "selection_sha256": pipeline._sha256_json(tuple()),
            "sources": self.inputs.source_identities,
        }
        pipeline._exclusive_json(attempt / "PREFLIGHT_RECEIPT.json", preflight_value)
        metric = {"race_probes": [], "cancer_probes": []}
        contrast_rows = []
        for candidate, baseline in contract.GATE_ELIGIBLE_CONTRASTS:
            race_cells = []
            cancer_cells = []
            for seed in contract.FM_SEEDS:
                for cancer in contract.CANCERS:
                    for view in contract.TILE_VIEWS:
                        for level in contract.PROBE_LEVELS:
                            race_cells.append(
                                {
                                    "fm_seed": seed, "cancer": cancer, "view": view,
                                    "probe_level": level,
                                    "baseline_oriented_leakage": 0.2,
                                    "candidate_oriented_leakage": 0.14,
                                }
                            )
                for view in contract.TILE_VIEWS:
                    cancer_cells.append(
                        {
                            "fm_seed": seed, "view": view,
                            "baseline_auroc": 0.9, "candidate_auroc": 0.89,
                        }
                    )
            contrast_rows.append(
                {"candidate": candidate, "baseline": baseline,
                 "race_cells": race_cells, "cancer_cells": cancer_cells}
            )
        verification = {
            "schema": "mock-verification/v1",
            "study_id": contract.STUDY_ID,
            "status": "pass",
        }
        with (
            mock.patch.object(pipeline, "_validate_production_inputs", return_value=self.inputs),
            mock.patch.object(pipeline, "_selection_rows", return_value=tuple()),
            mock.patch.object(pipeline, "_extract_all", return_value={}),
            mock.patch.object(pipeline, "_metric_input", return_value=metric),
            mock.patch.object(pipeline, "_semantic_contrast_inputs", return_value=contrast_rows),
            mock.patch.object(pipeline.verifier, "verify_analysis_files", return_value=verification) as verify,
        ):
            result = pipeline.run(attempt, device="cpu")
            with self.assertRaises(pipeline.PipelineError):
                pipeline.run(attempt, device="cpu")
        self.assertEqual(json.loads(result["verification"].read_text()), verification)
        self.assertEqual(json.loads(result["analysis_receipt"].read_text())["status"], "complete")
        verify.assert_called_once()


if __name__ == "__main__":
    unittest.main()
