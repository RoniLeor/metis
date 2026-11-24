"""Check the evaluate command: it scores every row, saves no weights, and breaks recall down by template and domain."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from metis.cfg import Settings
from metis.engine.evaluate import evaluate, main
from metis.engine.results import Run
from tests.sample import write_folder


class EvaluateTests(unittest.TestCase):
    """Every row of rows.jsonl is scored; the run has settings, metrics and predictions, but no checkpoint."""

    def test_evaluate(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch("tests.sample.FakeBackend.predict") as guess:
            folder: Path = write_folder(folder=Path(directory))
            guess.side_effect = lambda texts: [0.9 if "ignore" in text else 0.1 for text in texts]
            run: Run = evaluate(settings=Settings(backend="fake", data=folder, project=folder / "runs", name="dojo"))
            files: list[str] = sorted(path.name for path in run.save_dir.iterdir())
            metrics: dict[str, object] = json.loads((run.save_dir / "metrics.json").read_text())
        self.assertEqual(files, ["metrics.json", "predictions.jsonl", "settings.json"])
        self.assertEqual(len(run.predictions.rows), 25)
        self.assertEqual((run.metrics["recall"], run.metrics["fpr"]), (1.0, 0.0))
        self.assertEqual(
            metrics["report"], {"fitted_rows": 0, "recall_by_template": {"t": 1.0}, "recall_by_domain": {"d": 1.0}}
        )
        self.assertNotIn("checkpoint", run.settings.outputs)

    def test_a_bad_output_name_fails_before_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch("tests.sample.FakeBackend.predict") as guess:
            folder: Path = write_folder(folder=Path(directory))
            settings: Settings = Settings(backend="fake", data=folder, project=folder / "runs", outputs=["nope"])
            with self.assertRaisesRegex(KeyError, "unknown output 'nope'"):
                evaluate(settings=settings)
        guess.assert_not_called()

    def test_main_reports_a_bad_pair(self) -> None:
        with mock.patch("sys.argv", ["evaluate", "epochs"]), self.assertRaises(SystemExit):
            main()
