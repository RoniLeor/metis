"""Check the run settings and the key=value reader."""

import unittest
from pathlib import Path

from metis.cfg import Settings, from_pairs, parse_value, typed


class SettingsTests(unittest.TestCase):
    """Settings reject values that no run can use."""

    def test_validate(self) -> None:
        cases: list[tuple[Settings, str]] = [
            (Settings(epochs=0), "epochs"),
            (Settings(epochs=True), "epochs"),
            (Settings(limit=-1), "limit"),
            (Settings(threshold=1.0), "threshold"),
            (Settings(name="a/b"), "name"),
            (Settings(select="fpr"), "select must be one of"),
            (Settings(val_share=1.0), "val_share"),
            (Settings(augment=["field"]), "augment must name kinds"),
        ]
        settings: Settings
        message: str
        for settings, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                settings.validate()

    def test_values(self) -> None:
        self.assertEqual(parse_value(text="2"), 2)
        self.assertEqual(parse_value(text="rlcd"), "rlcd")
        self.assertEqual(typed(current=0.5, key="threshold", text="1"), 1.0)
        self.assertEqual(typed(current=Path("a"), key="data", text="b"), Path("b"))
        self.assertEqual(typed(current="x", key="fold", text="12"), "12")
        with self.assertRaisesRegex(ValueError, "epochs must be a int"):
            typed(current=1, key="epochs", text="two")
        with self.assertRaisesRegex(ValueError, "metrics must be a JSON list"):
            typed(current=["f1"], key="metrics", text="f1")

    def test_from_pairs(self) -> None:
        settings: Settings = from_pairs(
            pairs=["epochs=2", 'metrics=["f1"]', "loss=rlcd", "fold=legal", "data=x", 'augment=["bare","rewrap"]']
        )
        self.assertEqual(
            (settings.epochs, settings.metrics, settings.fold, settings.data), (2, ["f1"], "legal", Path("x"))
        )
        self.assertEqual((settings.options, settings.augment), ({"loss": "rlcd"}, ["bare", "rewrap"]))
        with self.assertRaisesRegex(ValueError, "expected key=value"):
            from_pairs(pairs=["epochs"])
        with self.assertRaisesRegex(ValueError, "epochs must be a positive integer"):
            from_pairs(pairs=["epochs=0"])
