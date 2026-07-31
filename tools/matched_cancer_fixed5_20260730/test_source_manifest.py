from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from .source_manifest import (
    create_manifest,
    validate_import_closure,
    verify_manifest,
)


class SourceManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _source(self, name: str, text: str = "VALUE = 1\n") -> Path:
        path = self.root / name
        path.write_text(text)
        return path.resolve()

    def test_create_verify_and_o_excl_refuses_overwrite(self) -> None:
        source = self._source("runtime.py")
        spec = {"runtime": source}
        manifest = create_manifest(self.root / "manifest.json", spec=spec)
        verified = verify_manifest(manifest, spec=spec)
        self.assertEqual(verified["status"], "frozen")
        self.assertEqual(verified["source_role_count"], 1)
        self.assertIs(verified["values_inspected"], False)
        with self.assertRaises(FileExistsError):
            create_manifest(manifest, spec=spec)

    def test_source_drift_fails_closed(self) -> None:
        source = self._source("runtime.py")
        spec = {"runtime": source}
        manifest = create_manifest(self.root / "manifest.json", spec=spec)
        source.write_text("VALUE = 2\n")
        with self.assertRaisesRegex(ValueError, "identity mismatch|drift"):
            verify_manifest(manifest, spec=spec)

    def test_manifest_tamper_and_noncanonical_json_fail_closed(self) -> None:
        source = self._source("runtime.py")
        spec = {"runtime": source}
        manifest = create_manifest(self.root / "manifest.json", spec=spec)
        raw = manifest.read_bytes()
        manifest.write_bytes(b" " + raw)
        with self.assertRaisesRegex(
            ValueError, "canonical|invalid|topology|identity"
        ):
            verify_manifest(manifest, spec=spec)

    def test_symlink_source_is_rejected(self) -> None:
        source = self._source("runtime.py")
        link = self.root / "redirect.py"
        link.symlink_to(source)
        with self.assertRaisesRegex(ValueError, "symlink|redirected"):
            create_manifest(
                self.root / "manifest.json", spec={"runtime": link}
            )

    def test_symlink_destination_ancestry_is_rejected(self) -> None:
        source = self._source("runtime.py")
        real = self.root / "real"
        real.mkdir()
        link = self.root / "link"
        link.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "ancestry"):
            create_manifest(link / "manifest.json", spec={"runtime": source})
        self.assertFalse((real / "manifest.json").exists())

    def test_existing_symlink_destination_is_never_followed(self) -> None:
        source = self._source("runtime.py")
        target = self._source("target.json", "do-not-overwrite\n")
        destination = self.root / "manifest.json"
        destination.symlink_to(target)
        with self.assertRaises(FileExistsError):
            create_manifest(destination, spec={"runtime": source})
        self.assertEqual(target.read_text(), "do-not-overwrite\n")

    def test_unallowlisted_repository_import_is_rejected(self) -> None:
        source = self._source(
            "runtime.py",
            "import tools.matched_cancer_fixed5_20260730.analyzer\n",
        )
        with self.assertRaisesRegex(ValueError, "unallowlisted local import"):
            validate_import_closure({"runtime": source})

    def test_exact_allowlisted_import_edge_is_recorded(self) -> None:
        imported = (
            Path(__file__).resolve().with_name("__init__.py")
        )
        source = self._source(
            "runtime.py",
            "import tools.matched_cancer_fixed5_20260730\n",
        )
        edges = validate_import_closure(
            {"runtime": source, "package": imported}
        )
        self.assertEqual(
            edges,
            [{"from_role": "runtime", "to_role": "package"}],
        )

    def test_duplicate_path_roles_are_rejected(self) -> None:
        source = self._source("runtime.py")
        with self.assertRaisesRegex(ValueError, "multiple semantic roles"):
            create_manifest(
                self.root / "manifest.json",
                spec={"one": source, "two": source},
            )

    def test_fifo_source_is_rejected_as_nonregular(self) -> None:
        fifo = self.root / "source.fifo"
        os.mkfifo(fifo)
        with self.assertRaisesRegex(ValueError, "non-regular|empty"):
            create_manifest(
                self.root / "manifest.json", spec={"runtime": fifo}
            )


if __name__ == "__main__":
    unittest.main()
