#!/usr/bin/env python3

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tools.matched_cancer_stage_20260730.replay import (
    BalancedReplayBatchSampler,
    MANIFEST_SCHEMA,
    ReplayContract,
)
from vendor.matched_stage_train_20260730.dataloader import TCGATileDataset


class FakeDataset:
    def __init__(self) -> None:
        self.tissue_thresh = 0.0
        self._tile_barcodes = [
            "B0", "B1", "W0", "W1", "LB0", "LB1", "LW0", "LW1"
        ]
        self.meta_disc = {
            "cancer": {
                "B0": 2, "B1": 2, "W0": 2, "W1": 2,
                "LB0": 15, "LB1": 15, "LW0": 15, "LW1": 15,
            },
            "race": {
                "B0": 2, "B1": 2, "W0": 4, "W1": 4,
                "LB0": 2, "LB1": 2, "LW0": 4, "LW1": 4,
            },
        }
        self._identities = []
        for index, patient in enumerate(self._tile_barcodes):
            shard_name = f"shard-{index // 2:05d}.parquet"
            self._identities.append(
                {
                    "dataset_index": index,
                    "shard_path": f"/frozen/tiles/{shard_name}",
                    "shard_sha256": hashlib.sha256(
                        shard_name.encode()
                    ).hexdigest(),
                    "row": index % 2,
                    "tile_path": f"{patient}-01A-SLIDE/tile-{index}.jpg",
                    "tile_jpeg_sha256": hashlib.sha256(
                        f"jpeg-{index}".encode()
                    ).hexdigest(),
                    "patient": patient,
                    "cancer": self.meta_disc["cancer"][patient],
                    "race": self.meta_disc["race"][patient],
                }
            )

    def replay_identity(self, dataset_index):
        return deepcopy(self._identities[dataset_index])

    def clear_replay_identity_cache(self):
        pass


def canonical_json(value):
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


class ReplayTest(unittest.TestCase):
    def contract(self):
        return ReplayContract(
            cancer_ids=(2, 15),
            race_ids=(2, 4),
            batch_size=8,
            steps=4,
            seed=701,
        )

    def test_exact_balance_and_determinism(self) -> None:
        dataset = FakeDataset()
        contract = self.contract()
        first = BalancedReplayBatchSampler(dataset, contract)
        second = BalancedReplayBatchSampler(dataset, contract)
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first.patient_sha256, second.patient_sha256)
        self.assertEqual(first.tile_sha256, second.tile_sha256)
        self.assertEqual(
            first.augmentation_seed_sha256,
            second.augmentation_seed_sha256,
        )
        self.assertEqual(list(first), list(second))
        self.assertEqual(first.manifest["schema"], MANIFEST_SCHEMA)
        self.assertEqual(len(first.manifest["occurrences"]), 32)
        self.assertEqual(
            first.manifest["traces"]["augmentation_seed_sha256"],
            first.augmentation_seed_sha256,
        )
        for batch in first:
            counts = {}
            for index, augmentation_seed in batch:
                barcode = dataset._tile_barcodes[index]
                key = (
                    dataset.meta_disc["cancer"][barcode],
                    dataset.meta_disc["race"][barcode],
                )
                counts[key] = counts.get(key, 0) + 1
                self.assertGreaterEqual(augmentation_seed, 0)
            self.assertEqual(set(counts.values()), {2})

    def test_manifest_round_trip_consumes_recorded_batches(self) -> None:
        dataset = FakeDataset()
        sampler = BalancedReplayBatchSampler(dataset, self.contract())
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "replay.json"
            sampler.write_manifest(manifest_path)
            loaded = BalancedReplayBatchSampler.from_manifest(
                dataset,
                manifest_path,
                expected_contract=self.contract(),
            )
            self.assertEqual(list(loaded), list(sampler))
            self.assertEqual(loaded.sha256, sampler.sha256)
            self.assertEqual(
                loaded.augmentation_seed_sha256,
                sampler.augmentation_seed_sha256,
            )
            self.assertEqual(loaded.manifest_path, str(manifest_path.resolve()))
            self.assertEqual(
                loaded.manifest_file_sha256,
                hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                manifest_path.read_bytes(),
                canonical_json(json.loads(manifest_path.read_text())),
            )
            with self.assertRaises(FileExistsError):
                sampler.write_manifest(manifest_path)

    def test_manifest_rejects_runtime_tile_identity_drift(self) -> None:
        source = FakeDataset()
        sampler = BalancedReplayBatchSampler(source, self.contract())
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "replay.json"
            sampler.write_manifest(manifest_path)
            selected = sampler.manifest["occurrences"][0]["dataset_index"]
            mutations = {
                "shard path": ("shard_path", "/different/shard.parquet"),
                "shard content": ("shard_sha256", "0" * 64),
                "row": ("row", 999),
                "tile path": ("tile_path", "different/tile.jpg"),
                "JPEG content": ("tile_jpeg_sha256", "1" * 64),
                "patient": ("patient", "DIFFERENT"),
            }
            for label, (field, value) in mutations.items():
                with self.subTest(label=label):
                    runtime = FakeDataset()
                    runtime._identities[selected][field] = value
                    with self.assertRaisesRegex(
                        ValueError, "runtime tile identity mismatch"
                    ):
                        BalancedReplayBatchSampler.from_manifest(
                            runtime, manifest_path
                        )

    def test_manifest_rejects_seed_or_payload_tamper(self) -> None:
        dataset = FakeDataset()
        sampler = BalancedReplayBatchSampler(dataset, self.contract())
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "replay.json"
            sampler.write_manifest(manifest_path)
            tampered = json.loads(manifest_path.read_text())
            tampered["occurrences"][0]["augmentation_seed"] += 1
            tampered_path = Path(directory) / "tampered.json"
            tampered_path.write_bytes(canonical_json(tampered))
            with self.assertRaisesRegex(ValueError, "payload SHA-256 mismatch"):
                BalancedReplayBatchSampler.from_manifest(
                    dataset, tampered_path
                )

    def test_manifest_rejects_noncanonical_or_wrong_contract(self) -> None:
        dataset = FakeDataset()
        sampler = BalancedReplayBatchSampler(dataset, self.contract())
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "replay.json"
            sampler.write_manifest(manifest_path)
            pretty_path = Path(directory) / "pretty.json"
            pretty_path.write_text(
                json.dumps(sampler.manifest, indent=2, sort_keys=True)
            )
            with self.assertRaisesRegex(ValueError, "not canonical JSON"):
                BalancedReplayBatchSampler.from_manifest(dataset, pretty_path)
            wrong_contract = ReplayContract(
                cancer_ids=(2, 15),
                race_ids=(2, 4),
                batch_size=8,
                steps=4,
                seed=702,
            )
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                BalancedReplayBatchSampler.from_manifest(
                    dataset,
                    manifest_path,
                    expected_contract=wrong_contract,
                )

    def test_replay_rejects_tissue_fallback(self) -> None:
        dataset = FakeDataset()
        dataset.tissue_thresh = 0.01
        with self.assertRaisesRegex(ValueError, "tissue_thresh"):
            BalancedReplayBatchSampler(dataset, self.contract())

    def test_real_dataset_identity_binds_parquet_and_jpeg_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shard_path = Path(directory) / "shard-00000.parquet"
            tile_path = "TCGA-AA-0001-01A-SLIDE/tile-0.jpg"
            jpeg_bytes = b"synthetic-jpeg-payload"
            pq.write_table(
                pa.table({"path": [tile_path], "jpeg": [jpeg_bytes]}),
                shard_path,
                row_group_size=1,
            )
            dataset = TCGATileDataset.__new__(TCGATileDataset)
            dataset.shards = [shard_path]
            dataset.shard_of = np.asarray([0], dtype=np.int32)
            dataset.row_of = np.asarray([0], dtype=np.int32)
            dataset._tile_barcodes = ["TCGA-AA-0001"]
            dataset.meta_disc = {
                "cancer": {"TCGA-AA-0001": 2},
                "race": {"TCGA-AA-0001": 4},
            }
            identity = dataset.replay_identity(0)
            self.assertEqual(identity["dataset_index"], 0)
            self.assertEqual(identity["shard_path"], str(shard_path.resolve()))
            self.assertEqual(
                identity["shard_sha256"],
                hashlib.sha256(shard_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(identity["row"], 0)
            self.assertEqual(identity["tile_path"], tile_path)
            self.assertEqual(
                identity["tile_jpeg_sha256"],
                hashlib.sha256(jpeg_bytes).hexdigest(),
            )
            self.assertEqual(identity["patient"], "TCGA-AA-0001")
            self.assertEqual(identity["cancer"], 2)
            self.assertEqual(identity["race"], 4)
            dataset.clear_replay_identity_cache()

    def test_invalid_or_empty_contract_fails(self) -> None:
        dataset = FakeDataset()
        with self.assertRaisesRegex(ValueError, "divisible"):
            BalancedReplayBatchSampler(
                dataset,
                ReplayContract((2, 15), (2, 4), 7, 1, 1),
            )
        with self.assertRaisesRegex(ValueError, "empty replay strata"):
            BalancedReplayBatchSampler(
                dataset,
                ReplayContract((2, 99), (2, 4), 8, 1, 1),
            )


if __name__ == "__main__":
    unittest.main()
