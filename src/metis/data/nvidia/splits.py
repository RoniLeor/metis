"""Splits that keep the rows of one task together and hold out templates, domains or attacker targets."""

import hashlib
import random
from collections import Counter

from metis.data.models import Fold

from .keys import match_keys
from .models import Task
from .rows import template_of

PARTS: int = 5
SEED: int = 0


def bucket(key: str, parts: int) -> int:
    """Return a stable fold number for a key."""
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % parts


def random_folds(tasks: list[Task], parts: int, seed: int) -> list[Fold]:
    """Split the tasks at random. This split shares templates, goals and targets between train and test."""
    ids: list[int] = sorted(task.id for task in tasks)
    random.Random(seed).shuffle(ids)
    return [Fold(strategy="random", name=f"random {part}", test_ids=sorted(ids[part::parts])) for part in range(parts)]


def template_folds(tasks: list[Task]) -> list[Fold]:
    """Train on the largest template family and test on all other templates."""
    largest: str = Counter(template_of(injection=task.injection) for task in tasks).most_common(1)[0][0]
    test_ids: list[int] = sorted(task.id for task in tasks if template_of(injection=task.injection) != largest)
    return [Fold(strategy="template", name="other templates", test_ids=test_ids)]


def domain_folds(tasks: list[Task]) -> list[Fold]:
    """Leave one domain out in each fold."""
    domains: list[str] = sorted({task.domain for task in tasks})
    return [
        Fold(strategy="domain", name=domain, test_ids=sorted(task.id for task in tasks if task.domain == domain))
        for domain in domains
    ]


def target_keys(task: Task) -> list[str]:
    """Return the goal text and the matched target values of a task: what an attacker names."""
    args: dict[str, object] = task.injection.target_args
    keys: list[str] = match_keys(target_args=args, verification_type=task.injection.verification_type)
    return ["goal:" + task.injection.goal, *("value:" + str(args[key]).strip().lower() for key in keys)]


def groups(tasks: list[Task]) -> dict[int, str]:
    """Join tasks that share a goal text or a target value. Return a group name for each task id."""
    parent: dict[str, str] = {}

    def root(key: str) -> str:
        while parent.setdefault(key, key) != key:
            key = parent[key]
        return key

    task: Task
    for task in tasks:
        keys: list[str] = target_keys(task=task)
        key: str
        for key in keys[1:]:
            parent[root(key)] = root(keys[0])
    return {task.id: root(target_keys(task=task)[0]) for task in tasks}


def goal_folds(tasks: list[Task], parts: int) -> list[Fold]:
    """Split by attacker target groups, so no goal text and no target value is in both parts."""
    group_of: dict[int, str] = groups(tasks=tasks)
    return [
        Fold(
            strategy="goal",
            name=f"goal {part}",
            test_ids=sorted(task_id for task_id, group in group_of.items() if bucket(key=group, parts=parts) == part),
        )
        for part in range(parts)
    ]


def folds(tasks: list[Task]) -> list[Fold]:
    """Return the folds of each strategy that have tasks in both parts. The random split is the leaky reference."""
    every: list[Fold] = [
        *random_folds(tasks=tasks, parts=PARTS, seed=SEED),
        *template_folds(tasks=tasks),
        *domain_folds(tasks=tasks),
        *goal_folds(tasks=tasks, parts=PARTS),
    ]
    return [fold for fold in every if 0 < len(fold.test_ids) < len(tasks)]
