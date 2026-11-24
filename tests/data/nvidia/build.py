"""Check the guard build on a small file."""

import json
import tempfile
import unittest
from pathlib import Path

from metis.data.models import Row
from metis.data.nvidia.adapter import adapt
from metis.data.nvidia.build import NvidiaBuilder, spread
from metis.data.nvidia.models import Task
from metis.data.nvidia.splits import folds

from .sample import injection, record


def row_of(words: int) -> Row:
    """Return a harmful field row with the given number of words."""
    return Row(
        task_id=1,
        domain="d",
        template="t",
        goal="g",
        path=["f"],
        text="w " * words,
        words=words,
        harmful=True,
        kind="field",
    )


class GuardBuildTests(unittest.TestCase):
    """The build writes rows, splits and counts for the clean tasks."""

    def test_folds(self) -> None:
        tasks: list[Task] = [adapt(record=record(id=index, domain=f"d{index % 2}")) for index in range(6)]
        strategies: list[str] = [fold.strategy for fold in folds(tasks=tasks)]
        self.assertEqual(sorted(set(strategies)), ["domain", "random"])

    def test_spread(self) -> None:
        rows: list[Row] = [row_of(words=count) for count in range(1, 11)]
        self.assertEqual(spread(rows=rows, kind="field", harmful=True), {"median": 5.5, "p90": 10.0})
        self.assertEqual(spread(rows=rows, kind="padded", harmful=False), {})

    def test_build_writes_files_for_clean_tasks(self) -> None:
        bad: dict[str, object] = record(id=99, injection=injection(injection_text="absent from the environment"))
        rows: list[dict[str, object]] = [record(id=index, domain=f"d{index % 2}") for index in range(6)] + [bad]
        with tempfile.TemporaryDirectory() as directory:
            folder: Path = Path(directory)
            (folder / "train.jsonl").write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
            NvidiaBuilder(data=folder).write(folder=folder / "guard")
            lines: list[str] = (folder / "guard" / "rows.jsonl").read_text(encoding="utf-8").splitlines()
            summary: object = json.loads((folder / "guard" / "summary.json").read_text(encoding="utf-8"))
            splits: object = json.loads((folder / "guard" / "splits.json").read_text(encoding="utf-8"))
        self.assertNotIn(99, {json.loads(line)["task_id"] for line in lines})
        self.assertIsInstance(summary, dict)
        self.assertIsInstance(splits, list)
        if isinstance(splits, list):
            self.assertEqual({fold["strategy"] for fold in splits}, {"random", "domain"})
            self.assertTrue(all(set(fold["test_ids"]) <= set(range(6)) for fold in splits))
        if isinstance(summary, dict):
            self.assertEqual((summary["rows"], summary["harmful"], summary["clean_by_kind"]), (6, 6, {}))
            self.assertEqual(summary["augmented_rows"], {"bare": 6, "rewrap": 6})
            self.assertEqual(summary["words"]["harmful"], {"median": 12, "p90": 12.0})
        groups: set[str] = {json.loads(line)["group"] for line in lines}
        self.assertEqual(len(groups), 1)
        self.assertTrue(next(iter(groups)).startswith("goal:"))
