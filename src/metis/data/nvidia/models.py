"""Typed tasks of the NVIDIA indirect prompt injection dataset."""

from dataclasses import dataclass
from typing import Literal

from metis.data.models import KeyPath, Tool


@dataclass(frozen=True)
class Injection:
    """The attacker text in the environment and the tool call that it tries to cause."""

    goal: str
    text: str
    target_tool: str
    target_args: dict[str, object]
    verification_type: str
    vector: str
    category: str


@dataclass(frozen=True)
class Task:
    """One record: the prompts, the tools, the environment state and the injection in it."""

    id: int
    domain: str
    system: str
    prompt: str
    tools: list[Tool]
    environment: dict[str, object]
    required_tools: list[str]
    injection: Injection
    spans: list[KeyPath]


@dataclass(frozen=True)
class Issue:
    """One data problem. An error removes the task from the clean set. A warning keeps it."""

    task_id: int
    code: str
    level: Literal["error", "warning"]
    detail: str


@dataclass(frozen=True)
class Checked:
    """All records of a file, their tasks and issues, and the clean tasks without an error."""

    records: list[dict[str, object]]
    tasks: list[Task]
    issues: list[Issue]
    kept: list[Task]
