"""Build ground-truth steps and variants from AgentDojo task replays."""

import datetime
import inspect
import json
import re
import textwrap
from collections.abc import Callable
from dataclasses import asdict

from .models import REPLAY_TIME, Call, Step, Variant


def mask_replay_time(text: str, days: set[datetime.date]) -> str:
    """Replace the timestamps that the replay itself wrote, so equal definitions compare equal."""
    dates: str = "|".join(sorted(day.isoformat() for day in days))
    return re.sub(pattern=rf"(?:{dates}) \d{{2}}:\d{{2}}:\d{{2}}(?:\.\d+)?", repl=REPLAY_TIME, string=text)


def build_steps(plain: list[Call], marked: list[Call], days: set[datetime.date]) -> list[Step]:
    """Pair each call of the plain replay with the same call of the replay with canary injections."""
    if [call.function for call in plain] != [call.function for call in marked]:
        raise ValueError("The replay with canary injections called different tools")
    return [
        Step(
            function=clean.function,
            args=clean.args,
            result=mask_replay_time(text=clean.result, days=days),
            marked_args=canary.args,
            marked_result=mask_replay_time(text=canary.result, days=days),
        )
        for clean, canary in zip(plain, marked, strict=True)
    ]


def add_variant(variants: list[Variant], candidate: Variant) -> None:
    """Add the candidate's versions to an equal definition, or keep the candidate as a new variant."""
    definition: str = json.dumps({**asdict(candidate), "versions": None}, sort_keys=True)
    variant: Variant
    for variant in variants:
        if json.dumps({**asdict(variant), "versions": None}, sort_keys=True) == definition:
            variant.versions.extend(candidate.versions)
            return
    variants.append(candidate)


def check_source(task_type: type, base: type, names: tuple[str, ...]) -> str:
    """Return the source of the named check methods, in the given order, that the task defines itself."""
    kept: list[Callable[..., object]] = [
        getattr(task_type, name) for name in names if getattr(task_type, name) is not getattr(base, name)
    ]
    return "\n".join(textwrap.dedent(inspect.getsource(method)) for method in kept)
