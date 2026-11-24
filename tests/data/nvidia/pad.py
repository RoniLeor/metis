"""Check the long clean rows that take the place of the injection."""

import dataclasses
import random
import unittest

from metis.data.models import Row
from metis.data.nvidia.adapter import adapt
from metis.data.nvidia.models import Task
from metis.data.nvidia.pad import bare_rows, edges, fill, pad_rows, rewrap_rows
from metis.data.nvidia.rows import make_row, make_rows, wrap

from .sample import INJECTED, record

TASK: Task = adapt(record=record())
OTHER: Task = adapt(record=record(id=8))


def clean(task: Task, field: str, text: str, domain: str = "healthcare") -> Row:
    """Return a clean field row of a task."""
    return dataclasses.replace(
        make_row(task=task, path=["x", field], text=text, harmful=False, kind="field"), domain=domain
    )


class FillTests(unittest.TestCase):
    """The filler joins whole clean texts until it is long enough."""

    def test_fill_stops_at_the_needed_words(self) -> None:
        pool: list[Row] = [clean(task=OTHER, field="notes", text=f"note {index} has five words") for index in range(4)]
        filler: str = fill(pool=pool, needed=8, rng=random.Random(1))
        self.assertEqual(len(filler.split()), 10)
        self.assertEqual(filler, fill(pool=pool, needed=8, rng=random.Random(1)))

    def test_edges(self) -> None:
        self.assertEqual(edges(text="\n\n x \n"), ("\n\n ", " \n"))
        self.assertEqual(edges(text="x"), ("", ""))


class PadTests(unittest.TestCase):
    """Each harmful row gets one clean row of the same field, as long as the harmful row."""

    def test_padded_row_keeps_position_and_length(self) -> None:
        filler: str = "Clean note text that is long enough to stand in for the injected block here and more"
        rows: list[Row] = [*make_rows(tasks=[TASK], min_words=5), clean(task=OTHER, field="chart_notes", text=filler)]
        padded: list[Row] = pad_rows(tasks=[TASK, OTHER], rows=rows)
        self.assertEqual(len(padded), 1)
        self.assertEqual(
            (padded[0].kind, padded[0].harmful, padded[0].path), ("padded", False, ["patients", "P-1", "chart_notes"])
        )
        self.assertEqual(padded[0].text, "Stable.\n\n" + filler + "\n\nFollow up.")
        self.assertGreaterEqual(padded[0].words, len(("Stable." + INJECTED + "Follow up.").split()))

    def test_same_field_first_then_domain_then_none(self) -> None:
        harmful: list[Row] = make_rows(tasks=[TASK], min_words=5)
        same: Row = clean(task=OTHER, field="chart_notes", text="same field text with many words in it to fill the gap")
        other: Row = clean(task=OTHER, field="address", text="other field text with many words in it to fill the gap")
        self.assertIn("same field", pad_rows(tasks=[TASK, OTHER], rows=[*harmful, same, other])[0].text)
        self.assertIn("other field", pad_rows(tasks=[TASK, OTHER], rows=[*harmful, other])[0].text)
        short: Row = clean(task=OTHER, field="chart_notes", text="short same field")
        mixed: str = pad_rows(tasks=[TASK, OTHER], rows=[*harmful, short, other])[0].text
        self.assertIn("other field", mixed)
        foreign: Row = clean(task=OTHER, field="chart_notes", text="text of another domain here", domain="legal")
        own: Row = clean(task=TASK, field="chart_notes", text="text of the same task here")
        self.assertEqual(pad_rows(tasks=[TASK, OTHER], rows=[*harmful, foreign, own]), [])
        self.assertEqual(pad_rows(tasks=[TASK, OTHER], rows=[*harmful, short]), [])


class BareTests(unittest.TestCase):
    """Each harmful field gets one bare row: the goal alone, harmful, in the place of the injection."""

    def test_bare_row(self) -> None:
        rows: list[Row] = make_rows(tasks=[TASK], min_words=5)
        bare: list[Row] = bare_rows(tasks=[TASK], rows=rows)
        self.assertEqual(
            [(row.kind, row.harmful, row.path) for row in bare], [("bare", True, ["patients", "P-1", "chart_notes"])]
        )
        self.assertEqual(bare[0].text, "Stable.\n\n" + TASK.injection.goal + "\n\nFollow up.")
        self.assertEqual(bare_rows(tasks=[TASK], rows=[row for row in rows if not row.harmful]), [])


class RewrapTests(unittest.TestCase):
    """Each harmful field gets one rewrapped row; a field whose injection is the bare goal gets no bare copy."""

    def test_rewrap_row(self) -> None:
        rows: list[Row] = make_rows(tasks=[TASK], min_words=5)
        rewrapped: list[Row] = rewrap_rows(tasks=[TASK], rows=rows)
        self.assertEqual([(row.kind, row.harmful) for row in rewrapped], [("rewrap", True)])
        self.assertIn(wrap(goal=TASK.injection.goal, rng=random.Random(TASK.id)), rewrapped[0].text)
        self.assertNotIn("PRIORITY", rewrapped[0].text)

    def test_bare_skips_a_bare_goal_injection(self) -> None:
        bare: Task = dataclasses.replace(TASK, injection=dataclasses.replace(TASK.injection, text=TASK.injection.goal))
        row: Row = dataclasses.replace(make_rows(tasks=[TASK], min_words=5)[0], text="Stable. " + TASK.injection.goal)
        self.assertEqual(bare_rows(tasks=[bare], rows=[row]), [])
