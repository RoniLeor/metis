"""Check the registered metrics."""

import unittest
from dataclasses import replace

from metis.data.models import Row
from metis.engine.metrics import METRICS, counts, recall_by, recall_by_goal
from metis.engine.results import Predictions
from tests.sample import ROWS, row

PREDICTIONS: Predictions = Predictions(rows=ROWS, scores=[0.1, 0.9, 0.8, 0.2, 0.6], threshold=0.5)


class MetricTests(unittest.TestCase):
    """Each metric uses the threshold and the row kinds."""

    def test_values(self) -> None:
        found: dict[str, float | None] = {name: METRICS.get(name)(PREDICTIONS) for name in METRICS.names()}
        self.assertEqual(PREDICTIONS.guesses(), [False, True, True, False, True])
        self.assertEqual(found["accuracy"], 0.4)
        self.assertEqual((found["precision"], found["recall"], found["f1"]), (1 / 3, 0.5, 0.4))
        self.assertEqual((found["fpr"], found["fpr_twin"], found["fpr_padded"]), (2 / 3, 1.0, 1.0))
        self.assertAlmostEqual(found["brier"] or 0.0, (0.01 + 0.01 + 0.64 + 0.64 + 0.36) / 5)

    def test_empty_parts_give_none(self) -> None:
        empty: Predictions = Predictions(rows=[], scores=[], threshold=0.5)
        self.assertIsNone(counts(predictions=empty, kind=None))
        self.assertEqual({METRICS.get(name)(empty) for name in METRICS.names()}, {None})
        only: Predictions = Predictions(rows=ROWS[:2], scores=[0.1, 0.9], threshold=0.5)
        self.assertIsNone(METRICS.get("fpr_twin")(only))


class GoalRecallTests(unittest.TestCase):
    """Recall is split by whether the train part has the goal group of the row."""

    def test_seen_and_unseen(self) -> None:
        self.assertEqual(
            recall_by_goal(predictions=PREDICTIONS, seen={"1"}), {"recall_seen_goal": 1.0, "recall_unseen_goal": 0.0}
        )
        self.assertEqual(recall_by_goal(predictions=PREDICTIONS, seen=set())["recall_seen_goal"], None)

    def test_a_goal_group_decides_before_the_task_id(self) -> None:
        grouped: Predictions = replace(PREDICTIONS, rows=[replace(row, group="g1") for row in ROWS])
        self.assertEqual(recall_by_goal(predictions=grouped, seen={"1"})["recall_seen_goal"], None)
        self.assertEqual(recall_by_goal(predictions=grouped, seen={"g1"})["recall_seen_goal"], 0.5)


class FieldRecallTests(unittest.TestCase):
    """Recall is split by the value of one row field, on the harmful rows only."""

    def test_recall_by(self) -> None:
        rows: list[Row] = [row(task_id=1, text="a", harmful=True), row(task_id=2, text="b", harmful=False)]
        rows.append(replace(rows[0], task_id=3, template="u"))
        self.assertEqual(recall_by(predicted=[True, True, False], rows=rows, field="task_id"), {"1": 1.0, "3": 0.0})
        self.assertEqual(recall_by(predicted=[True, True, False], rows=rows, field="template"), {"t": 1.0, "u": 0.0})
        self.assertEqual(recall_by(predicted=[], rows=[], field="domain"), {})
