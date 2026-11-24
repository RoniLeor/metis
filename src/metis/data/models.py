"""Typed models shared by every guard data source: the row, its key path, the folds and the tool."""

from dataclasses import dataclass
from typing import Literal, get_args

# The keys and list indexes from a JSON root to one string value.
KeyPath = list[str | int]
Kind = Literal["field", "twin", "padded", "bare", "rewrap"]  # The kinds that a rows file can hold.
KINDS: tuple[str, ...] = get_args(Kind)
# Kinds that are training copies, never test rows: goal-only and rewrapped copies of the harmful fields, and masked
# copies of the field, twin and padded rows (see masking.py). The augment setting selects them.
AUGMENTED: tuple[str, ...] = ("bare", "rewrap", "masked")
MASK_SOURCE_KINDS: tuple[str, ...] = tuple(kind for kind in KINDS if kind not in AUGMENTED)


@dataclass(frozen=True)
class Row:
    """One environment string. It is harmful when it holds the injection.

    The kind tells where a row comes from. A field is an environment string as it is. A twin is the harmful string
    with the injection removed. A padded row is the harmful string with clean text of other tasks in place of the
    injection, so it is as long as the harmful row and still clean. A bare row is the harmful string with only the
    attacker goal in place of the injection: harmful, but without a wrapper. A rewrap row has the goal in another
    wrapper style. A masked row is a training copy with the top style words removed (see masking.py). Bare, rewrap and
    masked rows are for validation and training only, never for test. The group joins tasks
    that share a goal text or a target value, so that a split by group keeps every goal on one side.
    """

    task_id: int
    domain: str
    template: str
    goal: str
    path: KeyPath
    text: str
    words: int
    harmful: bool
    kind: Kind | Literal["masked"]
    group: str = ""


@dataclass(frozen=True)
class Fold:
    """One split. The test part is these tasks; the train part is all other tasks."""

    strategy: str
    name: str
    test_ids: list[int]


@dataclass(frozen=True)
class Tool:
    """One tool that the agent can call: its name, its description and the JSON schema of its parameters."""

    name: str
    description: str
    parameters: dict[str, object]
