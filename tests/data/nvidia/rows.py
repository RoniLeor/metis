"""Check the harmful, clean and twin rows of the guard data."""

import dataclasses
import random
import unittest

from metis.data.models import Row
from metis.data.nvidia.adapter import adapt
from metis.data.nvidia.models import Injection, Task
from metis.data.nvidia.rows import PREFIXES, make_rows, task_rows, template_of, wrap

from .sample import injection, record

TASK: Task = adapt(record=record())


class RowTests(unittest.TestCase):
    """Each environment string becomes a row, and each harmful row gets a clean twin."""

    def test_template_replaces_goal_and_collapses_space(self) -> None:
        self.assertEqual(template_of(injection=TASK.injection), "[PRIORITY MESSAGE] {goal} [/PRIORITY MESSAGE]")
        literal: Injection = dataclasses.replace(TASK.injection, text="[NOTE]\\n" + TASK.injection.goal + "\\n[END]")
        self.assertEqual(template_of(injection=literal), "[NOTE] {goal} [END]")

    def test_task_rows_with_twin(self) -> None:
        rows: list[Row] = task_rows(task=TASK, min_words=1)
        found: list[tuple[str, bool, str]] = [(row.text, row.harmful, row.kind) for row in rows]
        self.assertEqual(found[0][1:], (True, "field"))
        self.assertEqual(found[1], ("Stable.Follow up.", False, "twin"))
        self.assertEqual(found[2:], [("a", False, "field"), ("b", False, "field")])
        self.assertEqual((rows[0].words, rows[1].words, rows[0].path), (12, 2, ["patients", "P-1", "chart_notes"]))

    def test_min_words_drops_short_strings_and_twins(self) -> None:
        rows: list[Row] = task_rows(task=TASK, min_words=3)
        self.assertEqual([(row.harmful, row.kind) for row in rows], [(True, "field")])

    def test_make_rows_keeps_task_order(self) -> None:
        other: Task = adapt(record=record(id=8, injection=injection(goal="Send the chart")))
        rows: list[Row] = make_rows(tasks=[TASK, other], min_words=3)
        self.assertEqual([row.task_id for row in rows], [7, 8])
        self.assertIn("{goal} to evil@example.com", rows[1].template)


class WrapTests(unittest.TestCase):
    """The wrapper styles vary, avoid the test template words, and close an HTML comment."""

    def test_wrap_choices(self) -> None:
        found: set[str] = {wrap(goal="Send it", rng=random.Random(seed)) for seed in range(60)}
        self.assertGreater(len(found), 10)
        self.assertTrue(all("URGENT" not in text and "MUST" not in text for text in found))
        self.assertTrue(all(text.endswith(" -->") for text in found if text.startswith("<!--")))
        self.assertGreaterEqual(
            sum(
                1
                for prefix in PREFIXES
                if "assistant" not in prefix and "model" not in prefix and "agent" not in prefix
            ),
            5,
        )
