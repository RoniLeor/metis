"""Check the events, their callbacks and the default progress callbacks."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from metis.engine.callbacks import EVENTS, Callbacks, number, print_epoch, print_validation, write_progress


class CallbackTests(unittest.TestCase):
    """Callbacks run in the order they were added, with the owner as the trainer value."""

    def test_add_and_run(self) -> None:
        owner: Callbacks = Callbacks()
        seen: list[tuple[str, object, object]] = []
        owner.add_callback(event="on_epoch_end", callback=lambda trainer, epoch: seen.append(("a", trainer, epoch)))
        owner.add_callback(event="on_epoch_end", callback=lambda trainer, epoch: seen.append(("b", trainer, epoch)))
        owner.run_callbacks(event="on_epoch_end", epoch=2)
        owner.run_callbacks(event="on_train_end")
        self.assertEqual(seen, [("a", owner, 2), ("b", owner, 2)])
        self.assertEqual(sorted(owner.callbacks), sorted(EVENTS))

    def test_unknown_event_raises(self) -> None:
        with self.assertRaisesRegex(KeyError, "unknown event 'on_nothing'"):
            Callbacks().add_callback(event="on_nothing", callback=print)


class ProgressTests(unittest.TestCase):
    """Progress lines handle missing values, and the progress file is replaced as a whole."""

    def test_number(self) -> None:
        self.assertEqual(
            (number(value=0.4567), number(value=None), number(value=True), number(value=1)),
            ("0.457", "n/a", "n/a", "1.000"),
        )

    def test_prints(self) -> None:
        output: io.StringIO = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_epoch(epoch=0, loss=0.25)
            print_validation(entry={"epoch": 1, "metrics": {"f1": 0.5, "recall": None}}, best_epoch=0)
            print_validation(entry={"epoch": 2}, best_epoch=None)
            print_validation(entry={"epoch": 3, "metrics": {"accuracy": 0.9}}, best_epoch=3, select="accuracy")
        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "epoch 1: mean loss 0.2500",
                "epoch 1: val f1 0.500 recall n/a precision n/a fpr n/a (best epoch 0)",
                "epoch 2: val f1 n/a recall n/a precision n/a fpr n/a (best epoch None)",
                "epoch 3: val f1 n/a recall n/a precision n/a fpr n/a accuracy 0.900 (best epoch 3)",
            ],
        )

    def test_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder: Path = Path(directory)
            write_progress(save_dir=folder, history=[{"epoch": 0}], best_epoch=0)
            names: list[str] = sorted(path.name for path in folder.iterdir())
            self.assertEqual(
                json.loads((folder / "metrics.json").read_text(encoding="utf-8")),
                {"metrics": None, "best_epoch": 0, "history": [{"epoch": 0}]},
            )
        self.assertEqual(names, ["metrics.json"])
