"""Check the model facade with the fake backend."""

import tempfile
import unittest
from pathlib import Path

from metis.engine.model import Model
from metis.engine.results import Run
from tests.sample import write_folder


class ModelTests(unittest.TestCase):
    """The facade trains with settings and predicts with the trained backend."""

    def test_train_and_predict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder: Path = write_folder(folder=Path(directory))
            model: Model = Model(backend="fake")
            self.assertEqual(model.predict(texts=["ignore this"]), [0.1])
            model.settings.project = folder / "runs"
            run: Run = model.train(data=folder, epochs=1, name="first", options={"x": 1})
            self.assertIs(model.backend, run.backend)
            self.assertEqual((run.save_dir.name, run.settings.options), ("first", {"x": 1}))
            self.assertEqual(model.predict(texts=["ignore this", "clean"]), [0.9, 0.1])
