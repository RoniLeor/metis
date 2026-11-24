"""Check the key=value command line that train and evaluate share."""

import contextlib
import io
import unittest
from pathlib import Path
from unittest import mock

from metis.cfg import Settings
from metis.engine.cli import run_cli
from metis.engine.results import Predictions, Run
from tests.sample import FakeBackend


def fake_run(settings: Settings) -> Run:
    """Return a run that holds the settings, without training."""
    return Run(
        save_dir=Path("runs") / settings.name,
        settings=settings,
        backend=FakeBackend(settings=settings),
        losses=[],
        predictions=Predictions(rows=[], scores=[], threshold=0.5),
        metrics={"f1": 0.5},
        report={},
        history=[],
        best_epoch=None,
    )


class CliTests(unittest.TestCase):
    """The pairs become settings; the run folder and the metrics are printed; a bad pair ends with an error."""

    def test_runs_the_action_with_the_settings(self) -> None:
        seen: list[Settings] = []

        def action(settings: Settings) -> Run:
            seen.append(settings)
            return fake_run(settings=settings)

        output: io.StringIO = io.StringIO()
        with mock.patch("sys.argv", ["train", "epochs=3", "name=demo"]), contextlib.redirect_stdout(output):
            run_cli(doc="demo", action=action)
        self.assertEqual((seen[0].epochs, seen[0].name), (3, "demo"))
        self.assertIn("saved to runs/demo", output.getvalue())
        self.assertIn('"f1": 0.5', output.getvalue())

    def test_bad_pair_and_action_error_exit(self) -> None:
        def fail(settings: Settings) -> Run:
            raise ValueError("no rows")

        errors: io.StringIO = io.StringIO()
        with mock.patch("sys.argv", ["train", "epochs"]), contextlib.redirect_stderr(errors):
            with self.assertRaises(SystemExit):
                run_cli(doc=None, action=fake_run)
        with mock.patch("sys.argv", ["train"]), contextlib.redirect_stderr(errors), self.assertRaises(SystemExit):
            run_cli(doc=None, action=fail)
        self.assertIn("no rows", errors.getvalue())
