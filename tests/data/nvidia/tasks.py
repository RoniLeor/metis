"""Check the build of the clean tasks, the issues and the NVIDIA page."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from metis.data.nvidia.models import Checked
from metis.data.nvidia.tasks import TEMPLATE, build, load, read_tasks

from .sample import injection, record


def write_lines(path: Path, rows: list[object]) -> None:
    """Write one JSON value on each line."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class BuildTests(unittest.TestCase):
    """The build writes the clean set, the issues and a page with its data."""

    def test_build_writes_outputs(self) -> None:
        bad: dict[str, object] = record(id=8, injection=injection(injection_text="not in the environment"))
        with tempfile.TemporaryDirectory() as directory:
            folder: Path = Path(directory)
            write_lines(path=folder / "train.jsonl", rows=[record(), bad])
            build(data=folder, output=folder / "out")
            kept: list[str] = (folder / "out" / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
            issues: object = json.loads((folder / "out" / "issues.json").read_text(encoding="utf-8"))
            html: str = (folder / "out" / "index.html").read_text(encoding="utf-8")
        self.assertEqual([json.loads(line)["id"] for line in kept], [7])
        self.assertIsInstance(issues, list)
        found: list[tuple[int, str, str]] = (
            [(row["task_id"], row["level"], row["code"]) for row in issues] if isinstance(issues, list) else []
        )
        expected: list[tuple[int, str, str]] = [
            (8, "error", "no-span"),
            (8, "warning", "goal-not-in-text"),
            (7, "warning", "repeated-prompt"),
            (8, "warning", "repeated-prompt"),
        ]
        self.assertEqual(found, expected)
        payload: object = json.loads(html.split('type="application/json">')[1].split("</script>")[0])
        self.assertIsInstance(payload, dict)
        if isinstance(payload, dict):
            self.assertEqual(
                (payload["records"], len(payload["toolsets"]), [task["kept"] for task in payload["tasks"]]),
                (2, 1, [True, False]),
            )

    def test_read_tasks_returns_records_issues_and_clean_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path: Path = Path(directory) / "train.jsonl"
            write_lines(path=path, rows=[record(), record(id=8, injection=injection(injection_text="absent"))])
            checked: Checked = read_tasks(path=path)
        self.assertEqual((len(checked.records), len(checked.tasks), [task.id for task in checked.kept]), (2, 2, [7]))
        self.assertIn((8, "no-span"), [(issue.task_id, issue.code) for issue in checked.issues])

    def test_load_rejects_a_line_that_is_not_an_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path: Path = Path(directory) / "train.jsonl"
            write_lines(path=path, rows=[record(), [1]])
            with self.assertRaisesRegex(ValueError, "train.jsonl:2: each line must be a JSON object"):
                load(path=path)

    def test_cli_reports_missing_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command: list[str] = [
                sys.executable,
                "-m",
                "metis.data.nvidia.tasks",
                directory,
                str(Path(directory) / "out"),
            ]
            result: subprocess.CompletedProcess[str] = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("train.jsonl", result.stderr)
            self.assertFalse((Path(directory) / "out").exists())

    def test_template_has_one_script_and_placeholders(self) -> None:
        html: str = TEMPLATE.read_text(encoding="utf-8")
        found: tuple[int, int, int, int] = (
            html.count("<script>"),
            html.count("</script>"),
            html.count("__RECORDS__"),
            html.count("__SCRIPT_HASH__"),
        )
        self.assertEqual(found, (1, 2, 1, 1))

    def test_build_rejects_repeated_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder: Path = Path(directory)
            write_lines(path=folder / "train.jsonl", rows=[record(), record()])
            with self.assertRaisesRegex(ValueError, r"record ids are not unique: \[7\]"):
                build(data=folder, output=folder / "out")
            self.assertFalse((folder / "out").exists())
