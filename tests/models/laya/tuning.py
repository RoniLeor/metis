"""Check the laya training config and the hold-out split. These need no checkpoint."""

import unittest
from dataclasses import replace

from laya.train import TrainConfig

from metis.data.models import Row
from metis.models.laya.spec import LayaOptions, read_options
from metis.models.laya.tuning import hold_out, train_config
from tests.sample import ROWS


class TuningTests(unittest.TestCase):
    """The config carries the options; the hold-out keeps whole goal groups and no augmented rows."""

    def test_train_config_carries_rl_options(self) -> None:
        options: LayaOptions = read_options(options={"loss": "rlcd", "rl_samples": 6, "sigma_start": 0.3})
        config: TrainConfig = train_config(options=options, epochs=2, seed=3)
        found: tuple[object, ...] = (config.loss, config.rl_samples, config.sigma_start, config.epochs, config.seed)
        self.assertEqual((found, config.log_every), (("rlcd", 6, 0.3, 2, 3), 50))
        config.validate()

    def test_hold_out_keeps_whole_groups(self) -> None:
        rows: list[Row] = [replace(row, task_id=row.task_id + 10 * copy) for copy in range(8) for row in ROWS]
        rows.extend([replace(row, kind="bare") for row in rows if row.harmful])
        fitting, held = hold_out(rows=rows, seed=0)
        groups: set[int] = {row.task_id for row in held}
        self.assertTrue(held)
        self.assertFalse(groups & {row.task_id for row in fitting})
        self.assertFalse(any(row.kind == "bare" for row in held))
        self.assertTrue(any(row.kind == "bare" for row in fitting))
        dropped: int = sum(1 for row in rows if row.kind == "bare" and row.task_id in groups)
        self.assertGreater(dropped, 0)
        self.assertEqual(len(fitting) + len(held) + dropped, len(rows))
        self.assertEqual((fitting, held), hold_out(rows=rows, seed=0))
