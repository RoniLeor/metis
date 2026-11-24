"""Check the split of guard rows into the parts of one fold."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from metis.data.split import ALL, Split, fold_ids, group_of, held_keys, load_split, sample
from tests.sample import EXTRA, ROWS, write_folder


class SplitTests(unittest.TestCase):
    """The split keeps the rows of each task on one side."""

    def test_load_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split: Split = load_split(
                folder=write_folder(folder=Path(directory)),
                strategy="template",
                fold="other templates",
                val_share=0.4,
                seed=0,
                augment=[],
            )
        self.assertEqual([row.task_id for row in split.test], [2, 2, 3])
        self.assertEqual(split.test[0], ROWS[2])
        val_groups: set[str] = {group_of(row=item) for item in split.val}
        self.assertEqual(len(val_groups), 2)
        self.assertTrue(all(item.kind == "bare" or not item.harmful for item in split.val))
        self.assertFalse(any(item.kind == "bare" for item in split.train + split.test))
        self.assertFalse(val_groups & {group_of(row=item) for item in split.train})
        self.assertEqual((sum(1 for item in split.val if item.kind == "bare"), len(split.train)), (4, 8))
        with tempfile.TemporaryDirectory() as directory:
            folder: Path = write_folder(folder=Path(directory))
            more: Split = load_split(
                folder=folder, strategy="template", fold="other templates", val_share=0.4, seed=0, augment=["bare"]
            )
        self.assertEqual(sum(1 for item in more.train if item.kind == "bare"), 6)
        self.assertEqual(more.val, split.val)
        self.assertEqual(more.test, split.test)

    def test_all_has_no_test_part(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split: Split = load_split(
                folder=write_folder(folder=Path(directory)),
                strategy=ALL,
                fold="ignored",
                val_share=0.4,
                seed=0,
                augment=[],
            )
        self.assertEqual(split.test, [])
        self.assertEqual({row.task_id for row in split.train + split.val}, {row.task_id for row in ROWS + EXTRA})

    def test_fold_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder: Path = write_folder(folder=Path(directory))
            self.assertEqual(fold_ids(path=folder / "splits.json", strategy="template", fold="other templates"), {2, 3})
            with self.assertRaisesRegex(ValueError, "no fold 'x'"):
                fold_ids(path=folder / "splits.json", strategy="template", fold="x")
            (folder / "splits.json").write_text(json.dumps([{"strategy": "s", "name": "n", "test_ids": "1"}]))
            with self.assertRaisesRegex(ValueError, "needs a list of task ids"):
                fold_ids(path=folder / "splits.json", strategy="s", fold="n")
            (folder / "splits.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "needs a list of folds"):
                fold_ids(path=folder / "splits.json", strategy="s", fold="n")


class HeldKeyTests(unittest.TestCase):
    """The holdout takes whole groups: a share of them, within a row limit."""

    def test_whole_groups_share_and_limit(self) -> None:
        ids: list[str] = [f"g{task}" for task in range(20) for _ in range(3)]
        held: set[str] = held_keys(keys=ids, seed=0, share=0.1, max_rows=400)
        self.assertEqual(len(held), 2)
        self.assertEqual(held, held_keys(keys=ids, seed=0, share=0.1, max_rows=400))
        self.assertLessEqual(len(held_keys(keys=ids, seed=0, share=0.5, max_rows=7)), 2)
        self.assertEqual(held_keys(keys=["a", "a", "b"], seed=0, share=0.1, max_rows=400), set())
        self.assertEqual(len(held_keys(keys=["a", "b", "a"], seed=0, share=0.5, max_rows=9)), 1)

    def test_group_of_falls_back_to_task(self) -> None:
        self.assertEqual((group_of(row=ROWS[0]), group_of(row=replace(ROWS[0], group="g1"))), ("1", "g1"))

    def test_sample(self) -> None:
        self.assertEqual(sample(rows=ROWS, limit=0, seed=0), ROWS)
        self.assertEqual(sample(rows=ROWS, limit=9, seed=0), ROWS)
        self.assertEqual(len(sample(rows=ROWS, limit=2, seed=0)), 2)
        self.assertEqual(sample(rows=ROWS, limit=2, seed=0), sample(rows=ROWS, limit=2, seed=0))
