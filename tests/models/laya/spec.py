"""Check the laya options, cases, probabilities and temperatures. These need no checkpoint."""

import math
import unittest

from metis.models.laya.spec import (
    QUESTION_ID,
    LayaOptions,
    calibrated,
    case,
    noul_temperature,
    pick,
    read_options,
    true_probability,
)


class SpecTests(unittest.TestCase):
    """laya options, cases, probabilities and temperatures."""

    def test_options(self) -> None:
        self.assertEqual(read_options(options={}), LayaOptions())
        chosen: LayaOptions = read_options(options={"loss": "rlcd", "head_lr": 1, "gradient_checkpointing": True})
        self.assertEqual((chosen.loss, chosen.head_lr, chosen.gradient_checkpointing), ("rlcd", 1.0, True))
        cases: list[tuple[dict[str, object], str]] = [
            ({"nope": 1}, "unknown laya options"),
            ({"loss": "mse"}, "soft-ce or rlcd"),
            ({"micro_batch": "8"}, "micro_batch must be a int"),
            ({"freeze_encoder": 1}, "freeze_encoder must be a bool"),
            ({"max_truncated": 1.0}, "max_truncated"),
        ]
        options: dict[str, object]
        message: str
        for options, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                read_options(options=options)
        self.assertEqual(pick(options={}, key="micro_batch", kind=int, default=8), 8)

    def test_case_and_probability(self) -> None:
        question: dict[str, object] = {"type": "noul", "instructions": "x"}
        expected: dict[str, object] = {
            "state": {"prompt": "t"},
            "questions": {QUESTION_ID: question},
            "expected": {QUESTION_ID: True},
        }
        self.assertEqual(case(text="t", question=question, label=True), expected)
        self.assertEqual(true_probability(logits=[0.0, 0.0], temperature=1.0), 0.5)
        self.assertAlmostEqual(true_probability(logits=[0.0, 2.0], temperature=2.0), 1 / (1 + math.exp(-1)))

    def test_temperatures(self) -> None:
        cfg: dict[str, object] = {"temperature": [1.0, 1.0, 1.5], "temperature_by_options": {"noul:2": 2.0}}
        self.assertEqual(noul_temperature(cfg=cfg, bucket="noul:2", noul=2), 2.0)
        self.assertEqual(noul_temperature(cfg={"temperature": [1.0, 1.0, 1.5]}, bucket="noul:2", noul=2), 1.5)
        self.assertEqual(noul_temperature(cfg={}, bucket="noul:2", noul=2), 1.0)
        found: dict[str, object] = calibrated(
            cfg={**cfg, "max_len": 512}, fitted={"temperature": [1.0, 1.0, 0.8], "temperature_by_options": {}}
        )
        self.assertEqual(found, {"temperature": [1.0, 1.0, 0.8], "max_len": 512, "fine_tuned": True})
        kept: dict[str, object] = calibrated(
            cfg={}, fitted={"temperature": [1.0], "temperature_by_options": {"noul:2": 0.9}}
        )
        self.assertEqual(kept["temperature_by_options"], {"noul:2": 0.9})
