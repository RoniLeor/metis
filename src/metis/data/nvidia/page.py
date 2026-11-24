"""The content of the NVIDIA page: the dataset facts, the tool lists and the view of each task."""

import json
from collections import Counter
from dataclasses import asdict

from metis.data.models import Tool

from .keys import match_keys, strict_keys
from .models import Checked, Issue, Task

CONSTANTS: tuple[tuple[str, ...], ...] = (
    ("license",),
    ("used_in",),
    ("agent_ref",),
    ("verifier_config",),
    ("responses_create_params", "tool_choice"),
    ("responses_create_params", "max_output_tokens"),
)


def value_at(record: dict[str, object], path: tuple[str, ...]) -> object:
    """Return the value at a key path, or None when a key is missing."""
    value: object = record
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def constants(records: list[dict[str, object]]) -> dict[str, list[str]]:
    """Return the distinct values, as JSON text, of each field that should be the same in all records."""
    found: dict[str, list[str]] = {}
    path: tuple[str, ...]
    for path in CONSTANTS:
        values: Counter[str] = Counter(
            json.dumps(value_at(record=record, path=path), sort_keys=True) for record in records
        )
        found[".".join(path)] = sorted(values)
    return found


def blank_targets(tasks: list[Task]) -> int:
    """Count the target argument values that are empty or only white space."""
    return sum(1 for task in tasks for value in task.injection.target_args.values() if not str(value).strip())


def toolsets(tasks: list[Task]) -> list[list[Tool]]:
    """Return each distinct tool list once, in the order of the tasks."""
    found: list[list[Tool]] = []
    task: Task
    for task in tasks:
        if task.tools not in found:
            found.append(task.tools)
    return found


def task_view(task: Task, issues: list[Issue], kept: bool, toolset: int) -> dict[str, object]:
    """Return one task for the page, with its verifier rule and its issues.

    Many tasks share one tool list, so the view names the index of its list in the page toolsets.
    """
    args: dict[str, object] = task.injection.target_args
    rule: str = task.injection.verification_type
    view: dict[str, object] = asdict(task)
    del view["tools"]
    view["toolset"] = toolset
    view["kept"] = kept
    view["match_keys"] = match_keys(target_args=args, verification_type=rule)
    view["strict"] = bool(strict_keys(target_args=args, verification_type=rule))
    view["issues"] = [asdict(issue) for issue in issues if issue.task_id == task.id]
    return view


def payload(checked: Checked) -> dict[str, object]:
    """Return the page data of the checked records: the dataset facts, the tool lists and every task view."""
    kept: set[int] = {task.id for task in checked.kept}
    shared: list[list[Tool]] = toolsets(tasks=checked.tasks)
    return {
        "records": len(checked.records),
        "constants": constants(records=checked.records),
        "blank_targets": blank_targets(tasks=checked.tasks),
        "toolsets": [[asdict(tool) for tool in toolset] for toolset in shared],
        "tasks": [
            task_view(task=task, issues=checked.issues, kept=task.id in kept, toolset=shared.index(task.tools))
            for task in checked.tasks
        ],
    }
