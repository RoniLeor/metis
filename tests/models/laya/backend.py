"""Check the laya backend end to end. It loads the 0.84 GB checkpoint, so it runs only with METIS_LAYA_TEST=1."""

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from metis.cfg import Settings
from metis.data.models import Row
from metis.models.laya.backend import LayaBackend
from metis.models.laya.tuning import temperature_of, truncated
from tests.sample import ROWS


@unittest.skipUnless(os.environ.get("METIS_LAYA_TEST") == "1", "set METIS_LAYA_TEST=1 to load the laya checkpoint")
class LayaTests(unittest.TestCase):
    """One short fit refits the temperatures, scores each text and saves a checkpoint."""

    def test_fit_predict_save(self) -> None:
        backend: LayaBackend = LayaBackend(settings=Settings(options={"micro_batch": 2, "grad_accum": 1}))
        before: list[float] = backend.predict(texts=[ROWS[0].text])
        self.assertEqual(len(before), 1)
        rows: list[Row] = [replace(row, task_id=row.task_id + 10 * copy) for copy in range(8) for row in ROWS]
        epochs: list[int] = []
        losses: list[float] = backend.fit(
            rows=rows, epochs=1, seed=0, on_epoch_end=lambda epoch, loss: epochs.append(epoch)
        )
        backend.model.train()
        backend.model.encoder.eval()
        flags: list[bool] = [module.training for module in backend.model.modules()]
        backend.predict(texts=[ROWS[0].text])
        self.assertEqual([module.training for module in backend.model.modules()], flags)
        self.assertNotEqual(set(flags), {True})
        backend.finish()
        scores: list[float] = backend.predict(texts=[row.text for row in ROWS])
        report: dict[str, object] = backend.report()
        self.assertEqual((len(losses), epochs, len(scores)), (1, [0], 5))
        self.assertTrue(all(0.0 <= score <= 1.0 for score in scores))
        sizes: list[int] = [int(str(report[key])) for key in ("train_items", "calibration_items", "truncated_rows")]
        self.assertEqual((sizes[0] + sizes[1], sizes[1] > 0, sizes[2]), (40, True, 0))
        with tempfile.TemporaryDirectory() as directory:
            backend.save(path=Path(directory) / "weights")
            cfg: object = json.loads((Path(directory) / "weights" / "rl_agent_config.json").read_text(encoding="utf-8"))
        self.assertIsInstance(cfg, dict)
        if isinstance(cfg, dict):
            self.assertTrue(cfg["fine_tuned"])
            self.assertEqual(temperature_of(cfg=cfg), report["temperature"])

    def test_truncated_counts_long_texts(self) -> None:
        backend: LayaBackend = LayaBackend(settings=Settings())
        texts: list[str] = ["short text", "word " * 2000]
        found: int = truncated(tok=backend.tok, question=backend.question, texts=texts, max_share=0.5, **backend.limits)
        self.assertEqual(found, 1)
        with self.assertRaisesRegex(ValueError, "1 of 2 rows exceed"):
            truncated(tok=backend.tok, question=backend.question, texts=texts, max_share=0.1, **backend.limits)
