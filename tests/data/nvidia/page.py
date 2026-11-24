"""Check the content of the NVIDIA page: the dataset facts, the task view and the payload."""

import unittest

from metis.data.nvidia.adapter import adapt
from metis.data.nvidia.models import Checked, Issue, Task
from metis.data.nvidia.page import blank_targets, constants, payload, task_view, toolsets, value_at

from .sample import injection, record


class SummaryTests(unittest.TestCase):
    """Dataset-level facts."""

    def test_constants_and_blank_targets(self) -> None:
        self.assertIsNone(value_at(record=record(), path=("responses_create_params", "missing")))
        found: dict[str, list[str]] = constants(records=[record(), record(license="other")])
        self.assertEqual(found["license"], ['"CC BY 4.0"', '"other"'])
        self.assertEqual(found["responses_create_params.max_output_tokens"], ["4096"])
        blank: Task = adapt(record=record(injection=injection(target_args={"recipient": " ", "body": ""})))
        self.assertEqual(blank_targets(tasks=[blank, adapt(record=record())]), 2)


class ViewTests(unittest.TestCase):
    """The page view of a task and the shared tool lists."""

    def test_task_view_and_toolsets(self) -> None:
        task: Task = adapt(record=record())
        issue: Issue = Issue(task_id=7, code="many-spans", level="warning", detail="x")
        view: dict[str, object] = task_view(task=task, issues=[issue], kept=True, toolset=0)
        self.assertNotIn("tools", view)
        self.assertEqual((view["toolset"], view["match_keys"], view["strict"]), (0, ["recipient"], True))
        self.assertEqual(view["issues"], [{"task_id": 7, "code": "many-spans", "level": "warning", "detail": "x"}])
        self.assertEqual(len(toolsets(tasks=[task, adapt(record=record(id=8))])), 1)

    def test_payload_holds_every_task_and_marks_the_kept_ones(self) -> None:
        kept: Task = adapt(record=record())
        dropped: Task = adapt(record=record(id=8))
        checked: Checked = Checked(records=[record(), record(id=8)], tasks=[kept, dropped], issues=[], kept=[kept])
        found: dict[str, object] = payload(checked=checked)
        tasks: object = found["tasks"]
        self.assertEqual((found["records"], found["blank_targets"]), (2, 0))
        self.assertIsInstance(tasks, list)
        if isinstance(tasks, list):
            self.assertEqual(
                [(task["id"], task["kept"], task["toolset"]) for task in tasks], [(7, True, 0), (8, False, 0)]
            )
        self.assertEqual(found["toolsets"], [[tool.__dict__ for tool in kept.tools]])
