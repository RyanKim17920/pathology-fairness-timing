from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from . import continuation_options_receipt as receipt


class ContinuationOptionsReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.document = self.root / "options.md"
        self.document.write_text(
            "\n".join((
                "before the fixed-five analyzer or any scientific result was opened",
                "at most one study GPU",
                "no optional extension or run-until-significance",
                "at most five new FM seeds",
                "downstream-label firewall",
            ))
            + "\n"
        )
        self.amendments = []
        for index in range(1, 9):
            path = self.root / f"amendment-{index}.md"
            path.write_text(f"amendment {index}\n")
            self.amendments.append(path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _create(self, destination: Path | None = None) -> Path:
        return receipt.create(
            destination or self.root / "receipt.json",
            document=self.document,
            amendments=self.amendments,
        )

    def test_create_verify_and_refuse_overwrite(self) -> None:
        path = self._create()
        value = receipt.verify(
            path, document=self.document, amendments=self.amendments
        )
        self.assertEqual(value["max_new_fm_seeds"], 5)
        self.assertEqual(value["max_concurrent_study_gpus"], 1)
        self.assertIs(value["scientific_values_opened"], False)
        with self.assertRaises(FileExistsError):
            self._create(path)

    def test_document_or_amendment_drift_fails_closed(self) -> None:
        path = self._create()
        self.document.write_text(self.document.read_text() + "drift\n")
        with self.assertRaisesRegex(ValueError, "identity mismatch|differ"):
            receipt.verify(
                path, document=self.document, amendments=self.amendments
            )

    def test_missing_value_blind_contract_phrase_is_rejected(self) -> None:
        self.document.write_text("not a frozen continuation contract\n")
        with self.assertRaisesRegex(ValueError, "lacks frozen contract"):
            self._create()

    def test_requires_exactly_eight_unique_amendments(self) -> None:
        with self.assertRaisesRegex(ValueError, "eight unique"):
            receipt.create(
                self.root / "receipt.json",
                document=self.document,
                amendments=self.amendments[:-1],
            )
        with self.assertRaisesRegex(ValueError, "eight unique"):
            receipt.create(
                self.root / "receipt-2.json",
                document=self.document,
                amendments=[*self.amendments[:-1], self.amendments[0]],
            )

    def test_symlink_destination_and_source_fail_closed(self) -> None:
        target = self.root / "target.json"
        target.write_text("preserve\n")
        destination = self.root / "receipt.json"
        destination.symlink_to(target)
        with self.assertRaises(FileExistsError):
            self._create(destination)
        self.assertEqual(target.read_text(), "preserve\n")
        document_link = self.root / "options-link.md"
        document_link.symlink_to(self.document)
        with self.assertRaisesRegex(ValueError, "symlink"):
            receipt.create(
                self.root / "other.json",
                document=document_link,
                amendments=self.amendments,
            )


if __name__ == "__main__":
    unittest.main()
