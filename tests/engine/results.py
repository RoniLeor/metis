"""Check the score of the decisions."""

import unittest

from metis.engine.results import Score, score
from tests.sample import ROWS


class ScoreTests(unittest.TestCase):
    """The score counts the hits and the errors, and gives 0 for a rate without rows."""

    def test_score(self) -> None:
        found: Score = score(predicted=[False, True, True, False, True], rows=ROWS)
        self.assertEqual((found.tp, found.fp, found.fn, found.tn), (1, 2, 1, 1))
        self.assertEqual((found.precision, found.recall, found.f1, found.fpr), (1 / 3, 0.5, 0.4, 2 / 3))
        self.assertEqual(
            score(predicted=[], rows=[]), Score(tp=0, fp=0, fn=0, tn=0, precision=0.0, recall=0.0, f1=0.0, fpr=0.0)
        )
