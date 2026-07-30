#!/usr/bin/env python3

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

from tools.matched_cancer_fixed48_20260730 import auditor, contract, runner


class Fixed48ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = yaml.safe_load(contract.PLAN.read_text())
        cls.seed_plan = yaml.safe_load(contract.SEED_PLAN.read_text())

    def test_exact_seed_range_and_mappings(self) -> None:
        lookup, legacy = contract.validate_plan(self.plan, self.seed_plan)
        self.assertEqual(tuple(lookup), tuple(range(32001, 32049)))
        self.assertEqual(len(lookup), 48)
        for seed, row in lookup.items():
            self.assertEqual(row["replay_seed"], seed + 20_000)
            self.assertEqual(row["data_order_seed"], seed + 30_000)
            self.assertEqual(row["adapter_init_seed"], seed + 40_000)
        self.assertEqual(
            legacy,
            {
                "disposition": "systems_only_excluded_from_inference",
                "reusable": False,
                "rerun_in_fixed48_namespace": True,
            },
        )

    def test_seed32001_is_rerun_but_legacy_output_is_not_reusable(self) -> None:
        lookup, legacy = contract.validate_plan(self.plan, self.seed_plan)
        self.assertIn(32001, lookup)
        self.assertTrue(legacy["rerun_in_fixed48_namespace"])
        self.assertFalse(legacy["reusable"])
        self.assertIn("excluded_from_inference", legacy["disposition"])

    def test_exact_arm_timing_and_exposure(self) -> None:
        contract.validate_plan(self.plan, self.seed_plan)
        self.assertEqual(
            self.plan["arms"],
            {
                "B": {
                    "slot1_fair_weight": 0.0,
                    "slot2_fair_weight": 0.0,
                },
                "P": {
                    "slot1_fair_weight": 0.1,
                    "slot2_fair_weight": 0.0,
                },
                "H": {
                    "slot1_fair_weight": 0.0,
                    "slot2_fair_weight": 0.1,
                },
            },
        )
        self.assertEqual(
            self.plan["replay"]["steps"]
            * self.plan["replay"]["batch_size"],
            99_968,
        )
        self.assertEqual(runner.RUN_SPECS["B"]["parent"], "slot1_plain")
        self.assertEqual(runner.RUN_SPECS["H"]["parent"], "slot1_plain")
        self.assertEqual(runner.RUN_SPECS["P"]["parent"], "slot1_fair")
        self.assertEqual(runner.RUN_SPECS["H"]["fair_weight"], 0.1)
        self.assertEqual(
            runner.PRETRAINED_ENCODER_STATE_SHA256,
            "ba9418ed2138e42250085b04e0502d621"
            "b072c4bb60240f2845a27fbf3184bd6",
        )
        self.assertEqual(runner.RUN_SPECS["P"]["fair_weight"], 0.0)

    def test_plan_tamper_and_unknown_keys_fail_closed(self) -> None:
        tampered = copy.deepcopy(self.seed_plan)
        tampered["seeds"][12]["replay_seed"] += 1
        with self.assertRaisesRegex(ValueError, "frozen mapping"):
            contract.validate_plan(self.plan, tampered)
        tampered = copy.deepcopy(self.plan)
        tampered["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "keys differ"):
            contract.validate_plan(tampered, self.seed_plan)
        tampered = copy.deepcopy(self.plan)
        tampered["arms"]["P"]["slot2_fair_weight"] = 0.1
        with self.assertRaisesRegex(ValueError, "frozen fixed48"):
            contract.validate_plan(tampered, self.seed_plan)

    def test_complete_base_template_semantics_are_frozen(self) -> None:
        base = yaml.safe_load(contract.BASE_TEMPLATE.read_text())
        contract._validate_base_template(base)
        mutations = (
            ("model", "type", "different_model"),
            ("dino", "lr", 0.5),
            ("data", "split_seed", 9999),
            ("train", "global_views", 3),
        )
        for section, key, value in mutations:
            with self.subTest(section=section, key=key):
                tampered = copy.deepcopy(base)
                tampered[section][key] = value
                with self.assertRaisesRegex(
                    ValueError, "complete frozen semantic mapping"
                ):
                    contract._validate_base_template(tampered)

    def test_attempt_root_is_exact_and_seed_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            namespace = Path(directory) / "fixed48"
            valid = namespace / "seed_32001" / "attempt_01"
            self.assertEqual(
                contract.validate_attempt_root(
                    valid, seed=32001, output_namespace=namespace
                ),
                valid.resolve(),
            )
            for invalid in (
                namespace / "seed_32002" / "attempt_01",
                namespace / "seed_32001" / "job_1",
                namespace.parent / "legacy" / "seed_32001" / "attempt_01",
            ):
                with self.subTest(invalid=invalid):
                    with self.assertRaises(ValueError):
                        contract.validate_attempt_root(
                            invalid, seed=32001, output_namespace=namespace
                        )
            with self.assertRaisesRegex(ValueError, "32001..32048"):
                contract.validate_attempt_root(
                    namespace / "seed_32049" / "attempt_01",
                    seed=32049,
                    output_namespace=namespace,
                )

    def test_representation_metadata_allowlist_rejects_egfr_and_kras(self) -> None:
        effective = yaml.safe_load(contract.BASE_TEMPLATE.read_text())
        effective["data"]["include_discrete"] = {
            "cancer": [2, 15],
            "race": [2, 4],
        }
        effective["fino"] = copy.deepcopy(contract.FROZEN_EFFECTIVE_FINO)
        effective["matched_stage"] = {
            "enabled": True,
            "mode": "joint",
            "study_id": contract.STUDY_ID,
            "scenario": "brca_luad_black_white_calibration_seed32001",
            "contract_receipt": "/fixed/contract",
            "effective_config_receipt": "/fixed/effective",
            "replay_manifest": "/fixed/replay",
            "adapter_init_seed": 72001,
            "fair_weight": 0.0,
            "adapter_lr": 0.001,
            "adapter_weight_decay": 0.0001,
            "data_order_seed": 62001,
            "encoder_checkpoint": None,
            "encoder_checkpoint_sha256": None,
            "expected_encoder_state_sha256": None,
            "parent_completion_receipt": None,
            "replay": {
                "cancer_ids": [2, 15],
                "race_ids": [2, 4],
                "steps": 781,
                "seed": 52001,
            },
        }
        contract.validate_effective_representation_metadata(effective)
        egfr = copy.deepcopy(effective)
        egfr["data"]["include_discrete"]["EGFR"] = [0, 1]
        with self.assertRaisesRegex(ValueError, "pixel inputs plus"):
            contract.validate_effective_representation_metadata(egfr)
        kras = copy.deepcopy(effective)
        kras["fino"]["contrastive_condition_on"] = "KRAS"
        with self.assertRaisesRegex(ValueError, "exactly cancer/race"):
            contract.validate_effective_representation_metadata(kras)
        extra_condition = copy.deepcopy(effective)
        extra_condition["matched_stage"]["outcome_condition"] = "EGFR"
        with self.assertRaisesRegex(ValueError, "matched_stage keys differ"):
            contract.validate_effective_representation_metadata(extra_condition)

    def test_auditor_runtime_topology_and_namespace_are_exact(self) -> None:
        self.assertEqual(
            set(auditor.RUNTIME_SOURCE_PATHS), set(contract.RUNTIME_SOURCES)
        )
        for role, path in auditor.RUNTIME_SOURCE_PATHS.items():
            self.assertEqual(path, contract.RUNTIME_SOURCES[role])
        with tempfile.TemporaryDirectory() as directory:
            outside = Path(directory) / "seed_32001" / "attempt_01"
            outside.mkdir(parents=True)
            with self.assertRaisesRegex(
                auditor.AuditError, "exact fixed48 calibration namespace"
            ):
                auditor.audit(outside, seed=32001)

    def test_runtime_batch_traces_are_order_and_batch_sensitive(self) -> None:
        values = list(range(256))
        expected = auditor._runtime_batch_trace(values)
        self.assertEqual(expected, auditor._runtime_batch_trace(values))
        swapped = list(values)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        self.assertNotEqual(expected, auditor._runtime_batch_trace(swapped))
        with self.assertRaisesRegex(auditor.AuditError, "batch-divisible"):
            auditor._runtime_batch_trace(values[:-1])

    def test_materialized_seed_contract_is_exact_and_immutable(self) -> None:
        lookup, legacy = contract.validate_plan(self.plan, self.seed_plan)
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            namespace = temporary / "calibration"
            root = namespace / "seed_32017" / "attempt_01"
            root.mkdir(parents=True)
            plan = copy.deepcopy(self.plan)
            plan["output_namespace"] = str(namespace)
            source = temporary / "runtime.py"
            source.write_text("pass\n")
            fino = temporary / "fino_meta.json"
            fino.write_text("{}\n")
            counts = {
                "BRCA/black or african american": 56,
                "BRCA/white": 521,
                "LUAD/black or african american": 11,
                "LUAD/white": 111,
            }
            with (
                mock.patch.object(
                    contract,
                    "load_and_validate_plan",
                    return_value=(plan, lookup, legacy),
                ),
                mock.patch.object(
                    contract, "RUNTIME_SOURCES", {"runtime": source}
                ),
                mock.patch.object(contract, "FINO_META", fino),
                mock.patch.object(
                    contract, "_validate_population", return_value=counts
                ),
            ):
                base, seed_contract, receipt = contract.materialize(
                    seed=32017, root=root
                )
                generated = yaml.safe_load(seed_contract.read_text())
                self.assertEqual(generated["representation_seed"], 32017)
                self.assertEqual(generated["replay"]["seed"], 52017)
                self.assertEqual(generated["data_order_seed"], 62017)
                self.assertEqual(
                    generated["adapter"]["init_seed"], 72017
                )
                self.assertEqual(
                    generated["scenario"],
                    "brca_luad_black_white_calibration_seed32017",
                )
                self.assertEqual(
                    yaml.safe_load(base.read_text())["train"]["seed"], 32017
                )
                persisted = json.loads(receipt.read_text())
                self.assertEqual(persisted["representation_seed"], 32017)
                self.assertEqual(
                    persisted["schema"], contract.RECEIPT_SCHEMA
                )
                replay = root / "CALIBRATION_REPLAY_MANIFEST.json"
                replay.write_text("{}\n")
                with mock.patch.object(
                    runner, "_paths", return_value=(plan, root)
                ):
                    effective_path = runner.build_run_config(
                        32017, root, "slot1_plain"
                    )
                    effective = yaml.safe_load(effective_path.read_text())
                    self.assertEqual(effective["train"]["seed"], 32017)
                    self.assertEqual(
                        effective["train"]["max_train_samples"], 99_968
                    )
                    self.assertEqual(
                        effective["matched_stage"]["mode"], "joint"
                    )
                    self.assertEqual(
                        effective["matched_stage"]["fair_weight"], 0.0
                    )
                    with self.assertRaises(FileExistsError):
                        runner.build_run_config(
                            32017, root, "slot1_plain"
                        )
                with self.assertRaises(FileExistsError):
                    contract.materialize(seed=32017, root=root)

    def test_independent_canonical_and_bound_file_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.write_text("version one\n")
            identities = {"source": auditor.identity(source)}
            receipt = {
                "schema": "test/v1",
                "study_id": auditor.STUDY_ID,
                "scenario": "scenario",
                "identities": identities,
                "topology_sha256": auditor.topology(identities),
            }
            receipt_path = root / "receipt.json"
            receipt_path.write_bytes(auditor.canonical_json(receipt) + b"\n")
            checks = auditor.Checks()
            auditor.verify_receipt(
                receipt_path,
                checks,
                schema="test/v1",
                scenario="scenario",
            )
            self.assertGreater(checks.count, 0)
            source.write_text("version two\n")
            with self.assertRaisesRegex(auditor.AuditError, "bound file changed"):
                auditor.verify_receipt(
                    receipt_path,
                    auditor.Checks(),
                    schema="test/v1",
                    scenario="scenario",
                )
            pretty = root / "pretty.json"
            pretty.write_text(json.dumps(receipt, indent=2))
            with self.assertRaisesRegex(auditor.AuditError, "not canonical"):
                auditor.load_canonical(pretty)

    def test_worker_is_one_seed_one_gpu_and_never_submits(self) -> None:
        text = (
            Path(__file__).parent / "calibration_one_seed.sbatch"
        ).read_text()
        self.assertIn("#SBATCH --gpus-per-task=1", text)
        self.assertIn("#SBATCH --time=08:00:00", text)
        self.assertIn("torch.cuda.device_count() != 1", text)
        self.assertNotIn("SLURM_GPUS_PER_TASK", text)
        self.assertIn("SLURM_GPUS_ON_NODE", text)
        self.assertIn("SLURM_JOB_GPUS", text)
        self.assertIn("CUDA_VISIBLE_DEVICES", text)
        self.assertIn(
            "cpython-3.12.11-linux-x86_64-gnu/bin/python3.12", text
        )
        self.assertIn("PYVENV_CONFIG=", text)
        self.assertIn(
            '[[ ! -w "$(dirname "$PRETRAINED")" ]]', text
        )
        self.assertGreaterEqual(text.count("verify_pretrained"), 3)
        self.assertIn("unset LABLESS_AUTOSUBMIT_FILE", text)
        self.assertIn(
            "export TORCH_HOME=/data/ryan.kim/nanopath/reruns/"
            "matched_cancer_fixed48_20260730/control/torch_home",
            text,
        )
        self.assertIn(
            "f433177089a681826f849f194ece3bb48"
            "f4d63fb38d32fc837e3dc7a4e5641fb",
            text,
        )
        self.assertIn("for RUN in slot1_plain slot1_fair B H P", text)
        executable_lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertFalse(
            any(line.startswith("sbatch ") for line in executable_lines)
        )
        self.assertFalse(
            any(line.startswith("srun ") for line in executable_lines)
        )


if __name__ == "__main__":
    unittest.main()
