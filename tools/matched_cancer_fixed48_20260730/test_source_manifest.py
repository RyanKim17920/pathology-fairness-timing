from __future__ import annotations

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
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _source(self, name: str, text: str = "VALUE = 1\n") -> Path:
        path = self.root / name
        path.write_text(text)
        return path

    def test_create_verify_and_refuse_overwrite(self) -> None:
        source = self._source("runtime.py")
        spec = {"runtime": source}
        manifest = create_manifest(self.root / "manifest.json", spec=spec)
        verified = verify_manifest(manifest, spec=spec)
        self.assertEqual(verified["status"], "frozen")
        with self.assertRaises(FileExistsError):
            create_manifest(manifest, spec=spec)

    def test_source_drift_fails_closed(self) -> None:
        source = self._source("runtime.py")
        spec = {"runtime": source}
        manifest = create_manifest(self.root / "manifest.json", spec=spec)
        source.write_text("VALUE = 2\n")
        with self.assertRaisesRegex(ValueError, "identity mismatch|drift"):
            verify_manifest(manifest, spec=spec)

    def test_symlink_source_is_rejected(self) -> None:
        source = self._source("runtime.py")
        link = self.root / "redirect.py"
        link.symlink_to(source)
        with self.assertRaisesRegex(ValueError, "symlink"):
            create_manifest(self.root / "manifest.json", spec={"runtime": link})

    def test_unallowlisted_repository_import_is_rejected(self) -> None:
        source = self._source(
            "runtime.py",
            "import tools.matched_cancer_fixed48_20260730.serial_controller\n",
        )
        with self.assertRaisesRegex(ValueError, "unallowlisted local import"):
            validate_import_closure({"runtime": source})

    def test_unallowlisted_top_level_repository_import_is_rejected(self) -> None:
        source = self._source("runtime.py", "import hf_tiles\n")
        with self.assertRaisesRegex(ValueError, "unallowlisted local import"):
            validate_import_closure({"runtime": source})

    def test_duplicate_path_roles_are_rejected(self) -> None:
        source = self._source("runtime.py")
        with self.assertRaisesRegex(ValueError, "multiple semantic roles"):
            create_manifest(
                self.root / "manifest.json",
                spec={"one": source, "two": source},
            )


if __name__ == "__main__":
    unittest.main()
