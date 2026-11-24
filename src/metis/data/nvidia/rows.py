"""Turn each clean task into harmful and clean rows, one row for each environment string.

wrap() gives a wrapper style for a training copy of an injection: a prefix and a suffix around the goal. The words
stay away from the test templates ("URGENT", "MUST", "critical priority", "[PRIORITY MESSAGE]", "I forgot to
mention"), and about half of the prefixes name no addressee, as the URGENT template names none.
"""

import random

from metis.data.base import leaves
from metis.data.models import KeyPath, Kind, Row

from .models import Injection, Task

PREFIXES: tuple[str, ...] = (
    "Also: ",
    "Next step: ",
    "Pending request: ",
    "Open item: ",
    "Follow-up item: ",
    "Note for the assistant: ",
    "Assistant instructions: ",
    "To the model reading this: ",
    "<!-- agent: ",
)
SUFFIXES: tuple[str, ...] = ("", " - thanks.", " Then carry on.", " (requested by the account owner)")


def template_of(injection: Injection) -> str:
    """Return the wrapper text of the injection, with {goal} for the goal and each run of white space as one space."""
    return " ".join(injection.text.replace(injection.goal, "{goal}").replace("\\n", " ").split())


def task_rows(task: Task, min_words: int) -> list[Row]:
    """Return the rows of one task: its strings with at least min_words words, and the twin of each harmful one."""
    found: list[Row] = []
    path: KeyPath
    text: str
    for path, text in leaves(value=task.environment, path=[]):
        harmful: bool = path in task.spans
        variants: list[tuple[str, bool, Kind]] = [(text, harmful, "field")]
        if harmful:
            variants.append((text.replace(task.injection.text, ""), False, "twin"))
        found.extend(
            make_row(task=task, path=path, text=variant, harmful=label, kind=kind)
            for variant, label, kind in variants
            if len(variant.split()) >= min_words
        )
    return found


def make_row(task: Task, path: KeyPath, text: str, harmful: bool, kind: Kind) -> Row:
    """Return one row of a task."""
    template: str = template_of(injection=task.injection)
    return Row(
        task_id=task.id,
        domain=task.domain,
        template=template,
        goal=task.injection.goal,
        path=path,
        text=text,
        words=len(text.split()),
        harmful=harmful,
        kind=kind,
    )


def make_rows(tasks: list[Task], min_words: int) -> list[Row]:
    """Return the rows of all tasks."""
    return [row for task in tasks for row in task_rows(task=task, min_words=min_words)]


def wrap(goal: str, rng: random.Random) -> str:
    """Return the goal inside one prefix and one suffix chosen with the random source."""
    prefix: str = rng.choice(PREFIXES)
    suffix: str = " -->" if prefix.startswith("<!--") else rng.choice(SUFFIXES)
    return prefix + goal + suffix
