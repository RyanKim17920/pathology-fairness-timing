from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import adoption_authorization as adoption


def _ancestors(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "production"
    manifest = root / "control/FIXED48_SOURCE_MANIFEST_V2.json"
    authorization = root / "authorization/AUTHORIZATION_MANIFEST_V3.json"
    feasibility = root / "control/FEASIBILITY_GATE_RECEIPT_V2.json"
    success = (
        root
        / "diagnostic/seed_32001/attempt_03/SEED_SUCCESS_RECEIPT.json"
    )
    for path in (manifest, authorization, feasibility, success):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{path.name}\n", encoding="utf-8")
    return {
        "root": root,
        "manifest": manifest,
        "authorization": authorization,
        "feasibility": feasibility,
        "success": success,
        "adoption": (
            root
            / "authorization/FIXED5_ADOPTION_AUTHORIZATION_V1.json"
        ),
    }


def _common(paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "fixed48_source_manifest": paths["manifest"],
        "authorization_manifest": paths["authorization"],
        "feasibility_gate": paths["feasibility"],
        "production_root": paths["root"],
    }


class AdoptionAuthorizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.paths = _ancestors(Path(self.temporary.name))
        self.calls: list[tuple[str, Path]] = []
        root_patcher = mock.patch.object(
            adoption, "PRODUCTION_ROOT", self.paths["root"]
        )
        root_patcher.start()
        self.addCleanup(root_patcher.stop)

        def record(label: str):
            def check(
                path: str | Path, *args: object, **kwargs: object
            ) -> dict:
                self.calls.append((label, Path(path).resolve()))
                return {}

            return check

        for name, label in (
            ("verify_fixed48_manifest", "manifest"),
            ("verify_fixed48_authorization", "authorization"),
            ("verify_fixed48_feasibility", "feasibility"),
            ("verify_seed_success", "success"),
        ):
            patcher = mock.patch.object(adoption, name, side_effect=record(label))
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_create_and_verify_binds_exact_canary_and_ancestors(self) -> None:
        output = adoption.create(
            self.paths["adoption"], **_common(self.paths)
        )
        receipt = adoption.verify(output, **_common(self.paths))

        self.assertEqual(receipt["status"], "authorized")
        self.assertEqual(receipt["adopted_seed"], 32001)
        self.assertEqual(
            receipt["fixed5_seeds"], [32001, 32002, 32003, 32004, 32005]
        )
        self.assertEqual(
            receipt["controller_seeds"], [32002, 32003, 32004, 32005]
        )
        self.assertIs(receipt["ancestor_controls_recomputed"], False)
        self.assertIs(receipt["values_inspected"], False)
        self.assertEqual(
            receipt["identities"]["seed1_success"]["canonical_path"],
            str(self.paths["success"].resolve()),
        )
        self.assertEqual(
            {label for label, _ in self.calls},
            {"manifest", "authorization", "feasibility", "success"},
        )

    def test_create_refuses_overwrite(self) -> None:
        adoption.create(self.paths["adoption"], **_common(self.paths))
        with self.assertRaisesRegex(FileExistsError, "exists"):
            adoption.create(self.paths["adoption"], **_common(self.paths))

    def test_create_rejects_redirected_destination(self) -> None:
        with self.assertRaisesRegex(ValueError, "destination differs"):
            adoption.create(
                Path(self.temporary.name) / "redirected.json",
                **_common(self.paths),
            )

    def test_rejects_whole_root_redirect(self) -> None:
        redirected = Path(self.temporary.name) / "redirected-production"
        redirected.mkdir()
        common = _common(self.paths)
        common["production_root"] = redirected
        with self.assertRaisesRegex(ValueError, "fixed study root"):
            adoption.create(self.paths["adoption"], **common)

    def test_rejects_each_redirected_control_path(self) -> None:
        for argument in (
            "fixed48_source_manifest",
            "authorization_manifest",
            "feasibility_gate",
        ):
            with self.subTest(argument=argument):
                redirected = (
                    Path(self.temporary.name) / f"redirected-{argument}.json"
                )
                redirected.write_text("{}\n", encoding="utf-8")
                common = _common(self.paths)
                common[argument] = redirected
                with self.assertRaisesRegex(ValueError, "path differs"):
                    adoption.create(self.paths["adoption"], **common)

    def test_rejects_symlinked_control_inputs(self) -> None:
        for argument in (
            "fixed48_source_manifest",
            "authorization_manifest",
            "feasibility_gate",
        ):
            with self.subTest(argument=argument):
                alias = Path(self.temporary.name) / f"{argument}.link"
                alias.symlink_to(_common(self.paths)[argument])
                common = _common(self.paths)
                common[argument] = alias
                with self.assertRaisesRegex(ValueError, "non-symlink"):
                    adoption.create(self.paths["adoption"], **common)

    def test_verify_detects_bound_ancestor_drift(self) -> None:
        adoption.create(self.paths["adoption"], **_common(self.paths))
        self.paths["manifest"].write_text("drift\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            adoption.verify(self.paths["adoption"], **_common(self.paths))

    def test_verify_detects_seed_success_drift(self) -> None:
        adoption.create(self.paths["adoption"], **_common(self.paths))
        self.paths["success"].write_text("drift\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            adoption.verify(self.paths["adoption"], **_common(self.paths))

    def test_verify_detects_receipt_field_tamper(self) -> None:
        output = adoption.create(
            self.paths["adoption"], **_common(self.paths)
        )
        receipt = json.loads(output.read_text(encoding="utf-8"))
        receipt["status"] = "tampered"
        output.write_text(
            json.dumps(
                receipt,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "status differs"):
            adoption.verify(output, **_common(self.paths))

    def test_requires_exactly_one_seed1_success(self) -> None:
        self.paths["success"].unlink()
        with self.assertRaisesRegex(ValueError, "exactly one success"):
            adoption.create(self.paths["adoption"], **_common(self.paths))

        duplicate = (
            self.paths["root"]
            / "diagnostic/seed_32001/attempt_04/SEED_SUCCESS_RECEIPT.json"
        )
        duplicate.parent.mkdir(parents=True)
        duplicate.write_text("{}\n", encoding="utf-8")
        self.paths["success"].write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly one success"):
            adoption.create(self.paths["adoption"], **_common(self.paths))

    def test_rejects_symlinked_seed1_success(self) -> None:
        target = Path(self.temporary.name) / "redirected-success.json"
        target.write_text("{}\n", encoding="utf-8")
        self.paths["success"].unlink()
        self.paths["success"].symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlink"):
            adoption.create(self.paths["adoption"], **_common(self.paths))

    def test_concurrent_destination_is_never_replaced(self) -> None:
        original_link = adoption.os.link
        competitor = b'{"competitor":true}\n'

        def race(source: str | Path, destination: str | Path) -> None:
            Path(destination).write_bytes(competitor)
            original_link(source, destination)

        with mock.patch.object(adoption.os, "link", side_effect=race):
            with self.assertRaises(FileExistsError):
                adoption.create(
                    self.paths["adoption"], **_common(self.paths)
                )
        self.assertEqual(self.paths["adoption"].read_bytes(), competitor)

    def test_receipt_encoding_is_canonical(self) -> None:
        output = adoption.create(
            self.paths["adoption"], **_common(self.paths)
        )
        raw = output.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        self.assertEqual(
            raw,
            json.dumps(
                parsed,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
        )


if __name__ == "__main__":
    unittest.main()
