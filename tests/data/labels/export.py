"""Check the offline exporter with a synthetic ground-truth file."""

import base64
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from metis.data.labels.export import export

TEXT: str = "</script><script>alert(1)</script><!--\u2028&αλφα <INFORMATION> <CANARY---bill--->"
STEP: dict[str, object] = {"function": "read", "args": {}, "result": TEXT, "marked_args": {}, "marked_result": TEXT}
VARIANT: dict[str, object] = {
    "versions": ["v1"],
    "prompt": TEXT,
    "difficulty": "EASY",
    "output": "",
    "check": "def utility(self) -> bool:\n    return 1 < 2\n",
    "steps": [STEP],
}
TRUTH: dict[str, object] = {
    "agentdojo": "0.1.35",
    "versions": ["v1"],
    "system": TEXT,
    "attacks": [{"name": "direct", "template": "TODO: {goal}", "user": "the user", "model": None}],
    "suites": {"banking": {"tools": [{"name": "read", "description": TEXT, "parameters": {}}]}},
    "tasks": {"banking/user_task_0": [VARIANT]},
}


def run_cli(data: Path, output: Path) -> subprocess.CompletedProcess[str]:
    """Run the exporter command line and capture its result."""
    command: list[str] = [sys.executable, "-m", "metis.data.labels.export", str(data), str(output)]
    return subprocess.run(command, capture_output=True, text=True, check=False)


class ExportTests(unittest.TestCase):
    """Exercise the export function and the command-line entry point."""

    def test_export_round_trips_text_and_matches_script_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data: Path = Path(directory)
            (data / "ground-truth.json").write_text(json.dumps(TRUTH), encoding="utf-8")
            output: Path = data / "viewer.html"
            export(data=data, output=output)
            html: str = output.read_text(encoding="utf-8")
            payload: str = html.split('id="records" type="application/json">')[1].split("</script>")[0]
            script: str = html.split("<script>")[1].split("</script>")[0]
            digest: str = base64.b64encode(hashlib.sha256(script.encode("utf-8")).digest()).decode("ascii")
            self.assertEqual(json.loads(payload), TRUTH)
            self.assertNotIn("<", payload)
            self.assertIn(f"'sha256-{digest}'", html)
            self.assertEqual((html.count("<script>"), html.count("</script>")), (1, 2))

    def test_cli_writes_requested_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data: Path = Path(directory)
            (data / "ground-truth.json").write_text(json.dumps(TRUTH), encoding="utf-8")
            output: Path = data / "viewer.html"
            result: subprocess.CompletedProcess[str] = run_cli(data=data, output=output)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.is_file())

    def test_cli_reports_bad_input_without_writing_output(self) -> None:
        cases: list[tuple[str | None, str]] = [
            (None, "ground-truth.json"),
            ("{", "Expecting property name"),
            ('{"tasks": []}', "ground-truth.json: needs a tasks object"),
        ]
        content: str | None
        message: str
        for content, message in cases:
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                data: Path = Path(directory)
                if content is not None:
                    (data / "ground-truth.json").write_text(content, encoding="utf-8")
                output: Path = data / "viewer.html"
                result: subprocess.CompletedProcess[str] = run_cli(data=data, output=output)
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)
                self.assertFalse(output.exists())
