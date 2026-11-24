"""Check that the guard splits group tasks and hold out templates, domains and attacker targets."""

import dataclasses
import unittest

from metis.data.models import Fold
from metis.data.nvidia.adapter import adapt
from metis.data.nvidia.models import Injection, Task
from metis.data.nvidia.splits import (
    bucket,
    domain_folds,
    goal_folds,
    groups,
    random_folds,
    target_keys,
    template_folds,
)

from .sample import record

BASE: Task = adapt(record=record())


def task(task_id: int, domain: str, goal: str, recipient: str, text: str) -> Task:
    """Return a sample task with another id, domain, goal, target value and injection text."""
    injection: Injection = dataclasses.replace(
        BASE.injection, goal=goal, text=text, target_args={"recipient": recipient}
    )
    return dataclasses.replace(BASE, id=task_id, domain=domain, injection=injection)


TASKS: list[Task] = [
    task(task_id=1, domain="hr", goal="g1", recipient="a@x", text="URGENT g1"),
    task(task_id=2, domain="hr", goal="g2", recipient="a@x", text="URGENT g2"),
    task(task_id=3, domain="legal", goal="g3", recipient="b@x", text="NOTE g3"),
    task(task_id=4, domain="legal", goal="g1", recipient="c@x", text="URGENT g1"),
]


class SplitTests(unittest.TestCase):
    """Every fold holds out whole tasks."""

    def test_random_folds_cover_each_task_once(self) -> None:
        folds: list[Fold] = random_folds(tasks=TASKS, parts=2, seed=0)
        self.assertEqual(sorted(task_id for fold in folds for task_id in fold.test_ids), [1, 2, 3, 4])
        self.assertEqual(folds, random_folds(tasks=TASKS, parts=2, seed=0))

    def test_template_fold_tests_the_other_templates(self) -> None:
        self.assertEqual(template_folds(tasks=TASKS), [Fold(strategy="template", name="other templates", test_ids=[3])])

    def test_domain_folds(self) -> None:
        self.assertEqual(
            [(fold.name, fold.test_ids) for fold in domain_folds(tasks=TASKS)], [("hr", [1, 2]), ("legal", [3, 4])]
        )

    def test_groups_join_shared_goals_and_target_values(self) -> None:
        self.assertEqual(target_keys(task=TASKS[0]), ["goal:g1", "value:a@x"])
        found: dict[int, str] = groups(tasks=TASKS)
        self.assertEqual(len({found[1], found[2], found[4]}), 1)
        self.assertNotEqual(found[1], found[3])

    def test_goal_folds_never_split_a_group(self) -> None:
        folds: list[Fold] = goal_folds(tasks=TASKS, parts=3)
        holder: dict[int, int] = {task_id: index for index, fold in enumerate(folds) for task_id in fold.test_ids}
        self.assertEqual(sorted(holder), [1, 2, 3, 4])
        self.assertEqual(len({holder[1], holder[2], holder[4]}), 1)
        self.assertEqual([bucket(key=key, parts=3) for key in ("a", "b", "c")], [1, 2, 1])
