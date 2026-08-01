#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools import reliable_fairness_head as reliable
from tools.matched_cancer_representation_audit_20260801 import contract
from tools.matched_cancer_representation_audit_20260801 import extractor
from tools.matched_cancer_stage_20260730.receipts import file_identity


def _normalized(rows: int, columns: int) -> np.ndarray:
    rng = np.random.default_rng(17)
    values = rng.normal(size=(rows, columns)).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return values


def _write_full_cache(path: Path) -> None:
    patients = ("P1", "P2")
    input_barcodes = np.asarray(
        [patient for patient in patients for _ in range(40)], dtype=np.str_
    )
    payload_sha256 = np.asarray(
        [hashlib.sha256(f"tile-{index}".encode()).hexdigest() for index in range(80)],
        dtype=np.str_,
    )
    payload_bytes = np.arange(100, 180, dtype=np.int64)
    keep_mask = np.ones(80, dtype=np.bool_)
    embeddings = _normalized(80, 128)
    barcodes = input_barcodes.copy()
    metadata = {
        "schema": extractor.FINAL_CACHE_SCHEMA,
        "source_identity": {"normalization": "per_tile_l2", "embedding_dim": 128},
    }
    entry = reliable._entry_sha256(
        metadata,
        embeddings,
        barcodes,
        keep_mask,
        input_barcodes,
        payload_sha256,
        payload_bytes,
    )
    np.savez(
        path,
        emb=embeddings,
        barcodes=barcodes,
        keep_mask=keep_mask,
        input_barcodes=input_barcodes,
        payload_sha256=payload_sha256,
        payload_bytes=payload_bytes,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
        entry_sha256=np.asarray(entry),
    )


class FullCacheAndSelectionTests(unittest.TestCase):
    def test_cache_digest_selection_and_subset_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            _write_full_cache(path)
            cache = extractor.read_full_cache(path)
            population = (
                {"patient_id": "P1", "cancer": "BRCA", "race": "Black", "tss": "A2"},
                {"patient_id": "P2", "cancer": "BRCA", "race": "White", "tss": "A7"},
            )
            selection = extractor.build_selection_rows(cache, population, cancer="BRCA")
            self.assertEqual(len(selection), 64)
            self.assertEqual({row["view"] for row in selection}, {"A", "B"})
            self.assertEqual(
                {(row["patient_id"], row["view"]): 16 for row in selection},
                {(patient, view): 16 for patient in ("P1", "P2") for view in ("A", "B")},
            )
            subset = extractor.subset_final_embeddings(cache, selection)
            self.assertEqual(subset.shape, (64, 128))
            np.testing.assert_allclose(np.linalg.norm(subset, axis=1), 1.0, atol=2e-4)

    def test_cache_digest_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            _write_full_cache(path)
            with np.load(path, allow_pickle=False) as stored:
                arrays = {name: stored[name] for name in stored.files}
            arrays["emb"] = arrays["emb"].copy()
            arrays["emb"][0, 0] += 0.1
            np.savez(path, **arrays)
            with self.assertRaisesRegex(extractor.ExtractionError, "digest mismatch"):
                extractor.read_full_cache(path)

    def test_shared_evidence_rejects_one_payload_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            _write_full_cache(path)
            cache = extractor.read_full_cache(path)
            payloads = cache.payload_sha256.copy()
            payloads[0] = "f" * 64
            altered = extractor.FullCache(
                path=cache.path,
                metadata=cache.metadata,
                embeddings=cache.embeddings,
                barcodes=cache.barcodes,
                keep_mask=cache.keep_mask,
                input_barcodes=cache.input_barcodes,
                payload_sha256=payloads,
                payload_bytes=cache.payload_bytes,
                entry_sha256=cache.entry_sha256,
            )
            with self.assertRaisesRegex(extractor.ExtractionError, "payload_sha256"):
                extractor.assert_same_tile_evidence(cache, altered)


class CompactCacheTests(unittest.TestCase):
    def test_round_trip_and_entry_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "compact.npz"
            rows = tuple(
                {
                    "patient_id": "P1",
                    "cancer": "BRCA",
                    "race": "Black",
                    "tss": "A2",
                    "view": "A",
                    "view_rank": index,
                    "occurrence_index": index,
                    "global_index": index,
                    "payload_sha256": hashlib.sha256(f"p-{index}".encode()).hexdigest(),
                }
                for index in range(16)
            )
            embeddings = _normalized(16, 384)
            extractor.write_compact_cache(
                path,
                seed=32001,
                layer="E_plain",
                embeddings=embeddings,
                rows=rows,
                source_identity={"kind": "synthetic"},
            )
            metadata, observed, observed_rows = extractor.read_compact_cache(path)
            self.assertEqual(metadata["normalization"], "per_tile_l2")
            np.testing.assert_array_equal(observed, embeddings)
            self.assertEqual(observed_rows, rows)
            with np.load(path, allow_pickle=False) as stored:
                arrays = {name: stored[name] for name in stored.files}
            arrays["race"] = arrays["race"].copy()
            arrays["race"][0] = "White"
            np.savez(path, **arrays)
            with self.assertRaisesRegex(extractor.ExtractionError, "digest mismatch"):
                extractor.read_compact_cache(path)


class TileBundleTests(unittest.TestCase):
    def test_receipt_bound_bundle_round_trip_and_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = (b"payload-a", b"payload-b")
            data = root / "tiles.bin"
            data.write_bytes(b"".join(payloads))
            rows = []
            offset = 0
            for index, payload in enumerate(payloads):
                rows.append(
                    {
                        "patient_id": "P1",
                        "cancer": "BRCA",
                        "race": "Black",
                        "tss": "A2",
                        "view": "A",
                        "view_rank": index,
                        "occurrence_index": index,
                        "global_index": index,
                        "payload_sha256": hashlib.sha256(payload).hexdigest(),
                        "offset": offset,
                        "payload_bytes": len(payload),
                    }
                )
                offset += len(payload)
            manifest = root / "tiles.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
            tile_view = root / "tile_view.json"
            tile_view.write_text("{}\n")
            receipt = root / "TILE_BUNDLE_RECEIPT.json"
            receipt.write_text(
                json.dumps(
                    {
                        "schema": extractor.TILE_BUNDLE_RECEIPT_SCHEMA,
                        "study_id": contract.STUDY_ID,
                        "row_count": 2,
                        "identities": {
                            "data": file_identity(data),
                            "manifest": file_identity(manifest),
                            "tile_view_receipt": file_identity(tile_view),
                        },
                    }
                )
            )
            observed_rows, observed_tiles = extractor.load_tile_bundle(receipt)
            self.assertEqual([payload for _, payload in observed_tiles], list(payloads))
            self.assertEqual([row["patient_id"] for row in observed_rows], ["P1", "P1"])
            data.write_bytes(b"X" + data.read_bytes()[1:])
            with self.assertRaisesRegex(extractor.ExtractionError, "identity drift"):
                extractor.load_tile_bundle(receipt)


if __name__ == "__main__":
    unittest.main()
