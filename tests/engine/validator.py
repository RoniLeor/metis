"""Check validation and the choice of the best validated epoch."""

import unittest
from unittest import mock

from metis.cfg import Settings
from metis.data.models import Row
from metis.engine.validator import BestEpoch, validate
from tests.sample import ROWS, FakeBackend


class BestEpochTests(unittest.TestCase):
    """The best epoch keeps a copy of its state; epoch 0 can win; no validation keeps the last weights."""

    def test_best_epoch_and_restore(self) -> None:
        backend: FakeBackend = FakeBackend(settings=Settings())
        rows: list[Row] = [ROWS[0], ROWS[1]]
        validator: BestEpoch = BestEpoch(backend=backend, rows=rows, settings=Settings())
        validator.check(epoch=0, loss=None)
        backend.version = 1
        validator.on_epoch_end(epoch=0, loss=0.4)
        backend.version = 2
        validator.on_epoch_end(epoch=1, loss=0.1)
        self.assertEqual((validator.best_epoch, validator.best_state), (1, 1))
        self.assertEqual(
            validator.history[1],
            {"epoch": 1, "loss": 0.4, "rows": 2, "harmful": 1, "metrics": validator.history[1]["metrics"]},
        )
        validator.restore()
        self.assertEqual(backend.version, 1)

    def test_epoch_zero_wins_and_empty_restore(self) -> None:
        backend: FakeBackend = FakeBackend(settings=Settings())
        backend.version = 1
        validator: BestEpoch = BestEpoch(backend=backend, rows=[ROWS[0], ROWS[1]], settings=Settings())
        validator.check(epoch=0, loss=None)
        backend.version = 2
        validator.on_epoch_end(epoch=0, loss=0.1)
        self.assertEqual(validator.best_epoch, 0)
        empty: BestEpoch = BestEpoch(backend=backend, rows=[], settings=Settings())
        empty.restore()
        self.assertEqual(backend.version, 2)

    def test_validate_without_rows(self) -> None:
        backend: FakeBackend = FakeBackend(settings=Settings())
        with mock.patch.object(FakeBackend, "predict", side_effect=AssertionError("predict was called")):
            predictions, metrics = validate(backend=backend, rows=[], threshold=0.5, metrics=["f1", "brier"])
        self.assertEqual((predictions.scores, metrics), ([], {"f1": None, "brier": None}))


class ValidateTests(unittest.TestCase):
    """validate() scores the rows and computes each selected metric."""

    def test_validate(self) -> None:
        backend: FakeBackend = FakeBackend(settings=Settings())
        backend.version = 1
        predictions, metrics = validate(backend=backend, rows=ROWS, threshold=0.5, metrics=["accuracy", "recall"])
        self.assertEqual((predictions.scores[1], metrics), (0.9, {"accuracy": 1.0, "recall": 1.0}))
