"""Long clean rows: each harmful string with clean text of other tasks in place of the injection.

A padded row has the same field, position and separators as its harmful row, and at least as many words, so a
word-count limit cannot tell them apart.
"""

import random
from collections import defaultdict
from collections.abc import Callable

from metis.data.models import Kind, Row

from .models import Task
from .rows import make_row, wrap


def fill(pool: list[Row], needed: int, rng: random.Random) -> str:
    """Join clean texts from the pool, in a random order, until they have at least the needed number of words."""
    candidates: list[Row] = list(pool)
    rng.shuffle(candidates)
    chosen: list[str] = []
    total: int = 0
    row: Row
    for row in candidates:
        if total >= needed:
            break
        chosen.append(row.text)
        total += row.words
    return " ".join(chosen)


def edges(text: str) -> tuple[str, str]:
    """Return the white space at the start and at the end of a text."""
    return text[: len(text) - len(text.lstrip())], text[len(text.rstrip()) :]


def pad_rows(tasks: list[Task], rows: list[Row]) -> list[Row]:
    """Return one padded row for each harmful row.

    The clean text comes from field rows of other tasks in the same domain: from the same field name when those texts
    have enough words, else from the whole domain. The random order uses the task id as its seed, so each build gives
    the same rows.
    """
    by_field: defaultdict[tuple[str, str], list[Row]] = defaultdict(list)
    by_domain: defaultdict[str, list[Row]] = defaultdict(list)
    row: Row
    for row in rows:
        if row.kind == "field" and not row.harmful:
            by_field[(row.domain, str(row.path[-1]))].append(row)
            by_domain[row.domain].append(row)
    task_of: dict[int, Task] = {task.id: task for task in tasks}
    padded: list[Row] = []
    for row in rows:
        if not row.harmful:
            continue
        task: Task = task_of[row.task_id]
        needed: int = len(task.injection.text.split())
        same_field: list[Row] = [item for item in by_field[(row.domain, str(row.path[-1]))] if item.task_id != task.id]
        domain: list[Row] = [item for item in by_domain[row.domain] if item.task_id != task.id]
        pool: list[Row] = same_field if sum(item.words for item in same_field) >= needed else domain
        if sum(item.words for item in pool) < needed:
            continue
        lead, trail = edges(text=task.injection.text)
        filler: str = fill(pool=pool, needed=needed, rng=random.Random(task.id))
        text: str = row.text.replace(task.injection.text, lead + filler + trail)
        padded.append(make_row(task=task, path=row.path, text=text, harmful=False, kind="padded"))
    return padded


def copies(tasks: list[Task], rows: list[Row], kind: Kind, goal_text: Callable[[Task], str]) -> list[Row]:
    """Return one harmful copy of each harmful field, with goal_text(task) in place of the injection wrapper.

    A copy that equals its field is dropped: then the injection was already the goal alone.
    """
    task_of: dict[int, Task] = {task.id: task for task in tasks}
    found: list[Row] = []
    row: Row
    for row in rows:
        if row.harmful and row.kind == "field":
            task: Task = task_of[row.task_id]
            lead, trail = edges(text=task.injection.text)
            text: str = row.text.replace(task.injection.text, lead + goal_text(task) + trail)
            if text != row.text:
                found.append(make_row(task=task, path=row.path, text=text, harmful=True, kind=kind))
    return found


def bare_rows(tasks: list[Task], rows: list[Row]) -> list[Row]:
    """Return one bare row for each harmful row: the attacker goal alone in place of the injection, with no wrapper."""
    return copies(tasks=tasks, rows=rows, kind="bare", goal_text=lambda task: task.injection.goal)


def rewrap_rows(tasks: list[Task], rows: list[Row]) -> list[Row]:
    """Return one row for each harmful field with the goal in another wrapper style, for training only."""
    return copies(
        tasks=tasks,
        rows=rows,
        kind="rewrap",
        goal_text=lambda task: wrap(goal=task.injection.goal, rng=random.Random(task.id)),
    )
