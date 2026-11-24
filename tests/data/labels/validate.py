"""Check the validation of the ground-truth file."""

import unittest
from pathlib import Path

from metis.data.labels.validate import validate_labels

PATH: Path = Path("ground-truth.json")
STEP: dict[str, object] = {"function": "f", "args": {}, "result": "", "marked_args": {}, "marked_result": ""}
VARIANT: dict[str, object] = {
    "versions": ["v1"],
    "prompt": "p",
    "difficulty": "EASY",
    "output": "",
    "check": "",
    "steps": [STEP],
}
TOOL: dict[str, object] = {"name": "f", "description": "", "parameters": {}}
ATTACK: dict[str, object] = {"name": "direct", "template": "{goal}", "user": "the user", "model": None}
VALID: dict[str, object] = {
    "agentdojo": "0.1.35",
    "versions": ["v1"],
    "system": "s",
    "attacks": [ATTACK, {**ATTACK, "model": "GPT-4"}],
    "suites": {"s": {"tools": [TOOL]}},
    "tasks": {"s/t": [VARIANT]},
}


def with_variant(variant: object) -> dict[str, object]:
    """Return the valid file with one replaced variant."""
    return {**VALID, "tasks": {"s/t": [variant]}}


def with_step(step: object) -> dict[str, object]:
    """Return the valid file with one replaced step."""
    return with_variant(variant={**VARIANT, "steps": [step]})


class ValidateTests(unittest.TestCase):
    """Exercise the accepted file and each rejected shape."""

    def test_validate_labels_accepts_a_valid_file(self) -> None:
        validate_labels(truth=VALID, path=PATH)
        validate_labels(truth=with_variant(variant={**VARIANT, "steps": []}), path=PATH)
        validate_labels(truth={**VALID, "attacks": [], "suites": {"s": {"tools": []}}}, path=PATH)

    def test_validate_labels_rejects_bad_shapes(self) -> None:
        cases: list[tuple[object, str]] = [
            (None, "needs a tasks object"),
            ({**VALID, "tasks": []}, "needs a tasks object"),
            ({**VALID, "system": None}, "agentdojo and system strings"),
            ({**VALID, "agentdojo": 1}, "agentdojo and system strings"),
            ({**VALID, "versions": [1]}, "versions string array"),
            ({**VALID, "attacks": None}, "needs attacks"),
            ({**VALID, "attacks": [None]}, "needs attacks"),
            ({**VALID, "attacks": [{**ATTACK, "template": None}]}, "needs attacks"),
            ({**VALID, "attacks": [{**ATTACK, "model": 1}]}, "needs attacks"),
            ({**VALID, "suites": None}, "needs suites with tools"),
            ({**VALID, "suites": {"s": {}}}, "needs suites with tools"),
            ({**VALID, "suites": {"s": {"tools": [{**TOOL, "name": 1}]}}}, "needs suites with tools"),
            ({**VALID, "suites": {"s": {"tools": [{**TOOL, "parameters": []}]}}}, "needs suites with tools"),
            ({**VALID, "tasks": {"x/t": [VARIANT]}}, "task x/t needs a suite/task name"),
            ({**VALID, "tasks": {"s": [VARIANT]}}, "task s needs a suite/task name"),
            ({**VALID, "tasks": {"s/t": []}}, "task s/t needs a non-empty variant array"),
            (with_variant(variant=None), "variants with versions"),
            (with_variant(variant={**VARIANT, "versions": []}), "variants with versions"),
            (with_variant(variant={**VARIANT, "versions": [1]}), "variants with versions"),
            (with_variant(variant={**VARIANT, "prompt": None}), "prompt, difficulty, output and check"),
            (with_variant(variant={**VARIANT, "check": None}), "prompt, difficulty, output and check"),
            (with_variant(variant={**VARIANT, "steps": None}), "needs steps"),
            (with_step(step=None), "needs steps"),
            (with_step(step={**STEP, "function": 1}), "needs steps"),
            (with_step(step={**STEP, "result": None}), "needs steps"),
            (with_step(step={**STEP, "args": []}), "needs steps"),
            (with_step(step={**STEP, "marked_args": None}), "needs steps"),
            (with_step(step={**STEP, "marked_result": None}), "needs steps"),
        ]
        truth: object
        message: str
        for truth, message in cases:
            with self.subTest(truth=truth), self.assertRaisesRegex(ValueError, "ground-truth.json: .*" + message):
                validate_labels(truth=truth, path=PATH)
