"""Typed AgentDojo ground truth, as the generator writes it and the viewer reads it."""

from dataclasses import dataclass

CANARY: str = "<CANARY---{vector}--->"
REPLAY_TIME: str = "<replay time>"


@dataclass(frozen=True)
class Call:
    """One tool call that a ground-truth replay made, with the text the tool returned."""

    function: str
    args: dict[str, object]
    result: str


@dataclass(frozen=True)
class Step:
    """One expected tool call and its output in the default environment.

    The marked fields hold the same call in the replay with canary injections. A canary shows an injection slot.
    """

    function: str
    args: dict[str, object]
    result: str
    marked_args: dict[str, object]
    marked_result: str


@dataclass
class Variant:
    """One task definition, shared by the benchmark versions that did not change it."""

    versions: list[str]
    prompt: str
    difficulty: str
    output: str
    check: str
    steps: list[Step]


@dataclass(frozen=True)
class Attack:
    """One attack template. `model` is None when the model name depends on the attacked pipeline."""

    name: str
    template: str
    user: str
    model: str | None
