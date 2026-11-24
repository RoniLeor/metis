"""Check the run folder name and the output files of a run."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from metis.cfg import Settings
from metis.engine.outputs import OUTPUTS, check_names, next_dir, write_outputs
from metis.engine.results import Predictions, Run
from tests.sample import ROWS, FakeBackend


def run_in(folder: Path, outputs: list[str]) -> Run:
    """Return a finished run of the fake backend that saves into the folder."""
    settings: Settings = Settings(backend="fake", project=folder, outputs=outputs)
    predictions: Predictions = Predictions(rows=ROWS[:2], scores=[0.9, 0.1], threshold=0.5)
    return Run(
        save_dir=folder,
        settings=settings,
        backend=FakeBackend(settings=settings),
        losses=[0.5],
        predictions=predictions,
        metrics={"f1": 1.0},
        report={"note": "x"},
        history=[],
        best_epoch=1,
    )


class OutputTests(unittest.TestCase):
    """Each named output writes its file; the run folder gets the first free name."""

    def test_next_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project: Path = Path(directory)
            self.assertEqual(next_dir(project=project, name="exp"), project / "exp")
            (project / "exp").mkdir()
            (project / "exp2").mkdir()
            self.assertEqual(next_dir(project=project, name="exp"), project / "exp3")

    def test_write_outputs_writes_only_the_named_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder: Path = Path(directory)
            write_outputs(run=run_in(folder=folder, outputs=["settings", "metrics", "predictions"]))
            names: list[str] = sorted(path.name for path in folder.iterdir())
            settings: dict[str, object] = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
            metrics: dict[str, object] = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            lines: list[str] = (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(names, ["metrics.json", "predictions.jsonl", "settings.json"])
        self.assertEqual((settings["backend"], settings["project"]), ("fake", str(folder)))
        self.assertEqual((metrics["metrics"], metrics["losses"], metrics["best_epoch"]), ({"f1": 1.0}, [0.5], 1))
        self.assertEqual([json.loads(line)["score"] for line in lines], [0.9, 0.1])

    def test_unknown_output_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run: Run = run_in(folder=Path(directory), outputs=["settings"])
            with self.assertRaisesRegex(KeyError, "unknown output 'nope'"):
                write_outputs(run=replace(run, settings=replace(run.settings, outputs=["nope"])))
        self.assertEqual(OUTPUTS.names(), ["checkpoint", "metrics", "predictions", "settings"])

    def test_check_names(self) -> None:
        check_names(settings=Settings())
        with self.assertRaisesRegex(KeyError, "unknown metric 'nope'"):
            check_names(settings=Settings(select="nope"))
        with self.assertRaisesRegex(KeyError, "unknown output 'nope'"):
            check_names(settings=Settings(outputs=["nope"]))
