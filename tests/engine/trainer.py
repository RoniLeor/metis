"""Check the trainer with the fake backend: the run files, the callbacks and the masked train copies."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from metis.cfg import Settings
from metis.data.split import Split
from metis.engine.results import Run
from metis.engine.trainer import Trainer
from tests.sample import FakeBackend, write_folder


def settings(folder: Path, **changes: object) -> Settings:
    """Return settings for the fake backend on the tiny folder."""
    found: Settings = Settings(backend="fake", data=folder, project=folder / "runs", val_share=0.4)
    for key, value in changes.items():
        setattr(found, key, value)
    return found


class TrainerTests(unittest.TestCase):
    """A run trains on the train rows, scores the test rows and writes the selected files."""

    def test_train_writes_run(self) -> None:
        events: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            folder: Path = write_folder(folder=Path(directory))
            trainer: Trainer = Trainer(settings=settings(folder=folder, epochs=2))
            trainer.add_callback(event="on_train_start", callback=lambda **values: events.append("start"))
            trainer.add_callback(
                event="on_epoch_end", callback=lambda **values: events.append(f"epoch {values['epoch']}")
            )
            trainer.add_callback(event="on_train_end", callback=lambda **values: events.append("end"))
            trainer.add_callback(
                event="on_val_end", callback=lambda **values: events.append(f"val {values['entry']['epoch']}")
            )
            run: Run = trainer.train()
            files: list[str] = sorted(path.name for path in run.save_dir.iterdir())
            metrics: object = json.loads((run.save_dir / "metrics.json").read_text(encoding="utf-8"))
            lines: list[str] = (run.save_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(events, ["start", "val 0", "epoch 0", "val 1", "epoch 1", "val 2", "end"])
        self.assertEqual(json.loads(json.dumps(run.history))[0]["epoch"], 0)
        self.assertEqual(files, ["metrics.json", "predictions.jsonl", "settings.json", "weights"])
        self.assertEqual(run.save_dir.name, "exp")
        self.assertEqual((run.losses, run.report), ([0.5, 0.5], {"fitted_rows": 8}))
        self.assertEqual(([entry["epoch"] for entry in run.history], run.best_epoch), ([0, 1, 2], 1))
        self.assertEqual(len(trainer.callbacks["on_epoch_end"]), 2)
        backend: object = run.backend
        self.assertTrue(isinstance(backend, FakeBackend) and backend.version == 1 and backend.finished)
        self.assertIsInstance(metrics, dict)
        if isinstance(metrics, dict):
            self.assertEqual(metrics["metrics"]["accuracy"], 1.0)
        self.assertEqual(len(lines), 3)

    def test_split_applies_the_limit_to_train_and_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder: Path = write_folder(folder=Path(directory))
            whole: Split = Trainer(settings=settings(folder=folder)).split()
            small: Split = Trainer(settings=settings(folder=folder, limit=2)).split()
        self.assertEqual((len(whole.train), len(whole.test)), (8, 3))
        self.assertEqual((len(small.train), len(small.test), small.val), (2, 2, whole.val))

    def test_all_trains_without_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run: Run = Trainer(settings=settings(folder=write_folder(folder=Path(directory)), strategy="all")).train()
        self.assertEqual((run.predictions.rows, run.metrics["f1"], run.best_epoch), ([], None, 1))

    def test_bad_names_and_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder: Path = Path(directory)
            with self.assertRaisesRegex(KeyError, "unknown metric 'nope'"):
                Trainer(settings=settings(folder=folder, metrics=["nope"]))
            with self.assertRaisesRegex(KeyError, "unknown output 'nope'"):
                Trainer(settings=settings(folder=folder, outputs=["nope"]))
            with self.assertRaisesRegex(KeyError, "unknown event 'on_nothing'"):
                Trainer(settings=settings(folder=folder)).add_callback(event="on_nothing", callback=print)

    def test_failed_fit_keeps_only_the_progress(self) -> None:
        def fail(**values: object) -> None:
            raise RuntimeError("stop")

        with tempfile.TemporaryDirectory() as directory:
            folder: Path = write_folder(folder=Path(directory))
            trainer: Trainer = Trainer(settings=settings(folder=folder))
            trainer.add_callback(event="on_epoch_end", callback=fail)
            with self.assertRaisesRegex(RuntimeError, "stop"):
                trainer.train()
            files: list[str] = sorted(path.name for path in (folder / "runs" / "exp").iterdir())
            progress: object = json.loads((folder / "runs" / "exp" / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(trainer.callbacks["on_epoch_end"]), 2)
        self.assertEqual(files, ["metrics.json"])
        self.assertIsInstance(progress, dict)
        if isinstance(progress, dict):
            self.assertEqual((progress["metrics"], [entry["epoch"] for entry in progress["history"]]), (None, [0]))


class TrainerMaskTests(unittest.TestCase):
    """Masked copies go to the train part only; validation and test rows stay real."""

    def test_trainer_masks_only_train(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch("metis.data.masking.MIN_COUNT", 1):
            folder: Path = write_folder(folder=Path(directory))
            base: Settings = settings(folder=folder)
            plain: Run = Trainer(settings=base).train()
            masked: Run = Trainer(settings=replace(base, augment=["masked"])).train()
        before: object = plain.report["fitted_rows"]
        after: object = masked.report["fitted_rows"]
        self.assertTrue(isinstance(before, int) and isinstance(after, int) and after > before)
        self.assertFalse(any(item.kind == "masked" for item in masked.predictions.rows))
        self.assertEqual(masked.history[0]["rows"], plain.history[0]["rows"])
