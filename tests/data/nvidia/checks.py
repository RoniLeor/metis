"""Check the data problems that the NVIDIA checks report."""

import dataclasses
import unittest

from metis.data.nvidia.adapter import adapt
from metis.data.nvidia.checks import (
    check,
    copy_issues,
    duplicates,
    schema_issues,
    span_issues,
    tasks_without_errors,
    tool_issues,
)
from metis.data.nvidia.models import Injection, Issue, Task

from .sample import record

TASK: Task = adapt(record=record())


def codes(issues: list[Issue]) -> list[tuple[str, str]]:
    """Return the level and code of each issue."""
    return [(issue.level, issue.code) for issue in issues]


class CheckTests(unittest.TestCase):
    """Each check reports its own problem and nothing for the clean sample."""

    def test_clean_sample_has_no_issue(self) -> None:
        self.assertEqual(check(task=TASK, record=record()), [])

    def test_copy_differs(self) -> None:
        issues: list[Issue] = copy_issues(
            task=TASK, record=record(target_tool="x", injection_vector="z", attack_category="y")
        )
        self.assertEqual(codes(issues), [("error", "copy-differs")] * 3)
        self.assertEqual(
            [issue.detail.split()[0] for issue in issues], ["target_tool", "injection_vector", "attack_category"]
        )

    def test_spans(self) -> None:
        self.assertEqual(codes(span_issues(task=dataclasses.replace(TASK, spans=[]))), [("error", "no-span")])
        many: Task = dataclasses.replace(TASK, spans=[["a"], ["b"]])
        self.assertEqual(codes(span_issues(task=many)), [("warning", "many-spans")])

    def test_schema(self) -> None:
        schema: dict[str, object] = {"properties": {"recipient": {}}, "required": ["recipient", "subject"]}
        issues: list[Issue] = schema_issues(task=TASK, schema=schema)
        self.assertEqual(codes(issues), [("error", "schema-mismatch")])
        self.assertIn("['body']", issues[0].detail)
        self.assertIn("['subject']", issues[0].detail)
        self.assertEqual(
            codes(schema_issues(task=TASK, schema={"properties": "x", "required": "x"})), [("error", "schema-mismatch")]
        )

    def test_unknown_tools(self) -> None:
        injection: Injection = dataclasses.replace(TASK.injection, target_tool="wire_money")
        task: Task = dataclasses.replace(TASK, injection=injection, required_tools=["read_mail"])
        self.assertEqual(codes(tool_issues(task=task)), [("error", "unknown-tool"), ("error", "unknown-tool")])

    def test_goal_not_in_text(self) -> None:
        task: Task = dataclasses.replace(TASK, injection=dataclasses.replace(TASK.injection, goal="other goal"))
        self.assertEqual(codes(check(task=task, record=record())), [("warning", "goal-not-in-text")])


class SetTests(unittest.TestCase):
    """Set-level checks and the clean split."""

    def test_duplicates_and_tasks_without_errors(self) -> None:
        other: Task = dataclasses.replace(TASK, id=8)
        third: Task = dataclasses.replace(TASK, id=9, prompt="Another prompt.")
        issues: list[Issue] = duplicates(tasks=[TASK, other, third])
        self.assertEqual([issue.task_id for issue in issues], [7, 8])
        error: Issue = Issue(task_id=8, code="no-span", level="error", detail="x")
        self.assertEqual(
            [task.id for task in tasks_without_errors(tasks=[TASK, other, third], issues=[*issues, error])], [7, 9]
        )
