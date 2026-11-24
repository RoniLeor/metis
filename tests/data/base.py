"""Check the shared base of the guard data: the row file, the report files and the builder."""

import json
import tempfile
import unittest
from pathlib import Path

from metis.data.base import Builder, leaves, read_row, read_rows
from metis.data.models import Row
from tests.sample import ROWS, write_folder


class TinyBuilder(Builder):
    """A source with two rows and one report."""

    def rows(self) -> list[Row]:
        return ROWS[:2]

    def reports(self, rows: list[Row]) -> dict[str, object]:
        return {"count.json": {"rows": len(rows)}}


class RowFileTests(unittest.TestCase):
    """Rows and reports go to disk and come back the same."""

    def test_bad_rows(self) -> None:
        good: dict[str, object] = dict(ROWS[0].__dict__)
        cases: list[tuple[object, str]] = [
            ([], "each line must be a JSON object"),
            ({**good, "kind": "other"}, "kind in"),
            ({**good, "words": "3"}, "integer fields"),
            ({**good, "path": "notes"}, "path must be a list"),
            ({**good, "harmful": 1}, "harmful must be true or false"),
            ({**good, "group": 3}, "group a string"),
        ]
        raw: object
        message: str
        for raw, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                read_row(raw=raw, label="x:1")
        self.assertEqual(read_row(raw=good, label="x:1"), ROWS[0])

    def test_leaves(self) -> None:
        self.assertEqual(
            leaves(value={"a": ["x", {"b": "y"}], "c": 1}, path=[]), [(["a", 0], "x"), (["a", 1, "b"], "y")]
        )
        self.assertEqual(leaves(value=3, path=[]), [])

    def test_read_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(len(read_rows(path=write_folder(folder=Path(directory)) / "rows.jsonl")), 25)


class BuilderTests(unittest.TestCase):
    """write() saves the rows of a source and each of its reports."""

    def test_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder: Path = Path(directory) / "new" / "guard"
            rows: list[Row] = TinyBuilder().write(folder=folder)
            self.assertEqual(rows, ROWS[:2])
            self.assertEqual(read_rows(path=folder / "rows.jsonl"), rows)
            self.assertEqual(json.loads((folder / "count.json").read_text(encoding="utf-8")), {"rows": 2})
