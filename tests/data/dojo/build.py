"""Check the dojo source reader, the summary and the files that the build writes."""

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

from metis.data.base import read_rows
from metis.data.dojo.build import DojoBuilder, main, summary
from metis.data.dojo.models import Injected, read_source
from metis.data.dojo.rows import CAP, build
from metis.data.models import Row

from .sample import SOURCE


class SourceTests(unittest.TestCase):
    """The reader returns typed records, and names the file when a record is wrong."""

    def test_read_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path: Path = Path(directory) / "source.json"
            path.write_text(json.dumps(asdict(SOURCE)), encoding="utf-8")
            self.assertEqual(read_source(path=path), SOURCE)
            path.write_text(
                json.dumps({"agentdojo": "x", "leaves": [{"suite": "s"}], "injected": []}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "source.json: not a dojo source file"):
                read_source(path=path)
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must hold a JSON object"):
                read_source(path=path)


class SummaryTests(unittest.TestCase):
    """The counts come from the source and the rows."""

    def test_summary(self) -> None:
        counts: dict[str, object] = summary(source=SOURCE, rows=build(source=SOURCE))
        self.assertEqual(
            (counts["combinations"], counts["slots"], counts["short_twin_slots"]), (3, {"workspace": 2}, 1)
        )
        self.assertEqual((counts["whole_string_slots"], counts["cut_slots"]), (0, 0))
        whole: Injected = replace(SOURCE.injected[0], vector="v4", prefix=0, suffix=0)
        long: Injected = replace(SOURCE.injected[0], vector="v5", text=" ".join(["w"] * (CAP + 1)))
        more: dict[str, object] = summary(source=replace(SOURCE, injected=[whole, long]), rows=[])
        self.assertEqual((more["whole_string_slots"], more["cut_slots"]), (1, 1))
        self.assertEqual(
            counts["rows"], {"workspace field False": 1, "workspace field True": 2, "workspace twin False": 1}
        )
        self.assertEqual(counts["harmful_by_attack"], {"direct": 1, "injecagent": 1})


class BuildTests(unittest.TestCase):
    """The build writes rows that the engine reads, and the summary."""

    def test_write_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder: Path = Path(directory) / "guard"
            rows: list[Row] = DojoBuilder(source=SOURCE).write(folder=folder)
            self.assertEqual(read_rows(path=folder / "rows.jsonl"), rows)
            self.assertEqual(json.loads((folder / "summary.json").read_text())["combinations"], 3)

    def test_main_reports_a_missing_file(self) -> None:
        with mock.patch("sys.argv", ["build", "missing.json", "out"]), self.assertRaises(SystemExit):
            main()
