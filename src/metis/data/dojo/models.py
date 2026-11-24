"""The records that metis/data/dojo/extract/source.py writes from AgentDojo, and the reader of its file."""

import json
from dataclasses import dataclass
from pathlib import Path

from metis.data.models import KeyPath


@dataclass(frozen=True)
class Leaf:
    """One string of a suite's default environment.

    vector names the injection slot that the string holds, or is empty. prefix and suffix are the word counts of the
    text before and after the slot, so that a long string can be cut around it.
    """

    suite: str
    path: KeyPath
    text: str
    vector: str
    prefix: int
    suffix: int


@dataclass(frozen=True)
class Injected:
    """One environment string with an attack in an injection slot, as AgentDojo renders it."""

    suite: str
    vector: str
    injection_task: str
    goal: str
    attack: str
    path: KeyPath
    text: str
    prefix: int
    suffix: int


@dataclass(frozen=True)
class Source:
    """The AgentDojo version, the clean strings and the injected strings."""

    agentdojo: str
    leaves: list[Leaf]
    injected: list[Injected]


def read_source(path: Path) -> Source:
    """Read the file that metis/data/dojo/extract/source.py writes, or raise an error that names the file."""
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    try:
        if not isinstance(raw, dict):
            raise TypeError("the file must hold a JSON object")
        return Source(
            agentdojo=str(raw["agentdojo"]),
            leaves=[Leaf(**item) for item in raw["leaves"]],
            injected=[Injected(**item) for item in raw["injected"]],
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path}: not a dojo source file: {error}") from error
