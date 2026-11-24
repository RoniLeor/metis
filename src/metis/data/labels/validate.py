"""Check the ground-truth file before the viewer embeds it."""

from pathlib import Path


def is_text_list(value: object) -> bool:
    """Return true for a list that holds only strings."""
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def has_texts(record: object, fields: tuple[str, ...]) -> bool:
    """Return true for an object where each named field is a string."""
    return isinstance(record, dict) and all(isinstance(record.get(field), str) for field in fields)


def valid_step(step: object) -> bool:
    """Return true for a step with a function name and the arguments and output of both replays."""
    return (
        isinstance(step, dict)
        and has_texts(record=step, fields=("function", "result", "marked_result"))
        and isinstance(step.get("args"), dict)
        and isinstance(step.get("marked_args"), dict)
    )


def valid_tools(suite: object) -> bool:
    """Return true for a suite whose tools have a name, a description and a parameters object."""
    tools: object = suite.get("tools") if isinstance(suite, dict) else None
    return isinstance(tools, list) and all(
        isinstance(tool, dict)
        and has_texts(record=tool, fields=("name", "description"))
        and isinstance(tool.get("parameters"), dict)
        for tool in tools
    )


def validate_variants(variants: object, key: str, path: Path) -> None:
    """Check the variants of one task."""
    if not isinstance(variants, list) or not variants:
        raise ValueError(f"{path}: task {key} needs a non-empty variant array")
    variant: object
    for variant in variants:
        if not isinstance(variant, dict) or not is_text_list(variant.get("versions")) or not variant["versions"]:
            raise ValueError(f"{path}: task {key} needs variants with versions")
        if not has_texts(record=variant, fields=("prompt", "difficulty", "output", "check")):
            raise ValueError(f"{path}: task {key} needs prompt, difficulty, output and check strings")
        steps: object = variant.get("steps")
        if not isinstance(steps, list) or any(not valid_step(step=step) for step in steps):
            raise ValueError(f"{path}: task {key} needs steps with function, args, result and their marked copies")


def validate_labels(truth: object, path: Path) -> None:
    """Check the fields the viewer reads from the ground-truth file."""
    if not isinstance(truth, dict) or not isinstance(truth.get("tasks"), dict):
        raise ValueError(f"{path}: needs a tasks object")
    if not has_texts(record=truth, fields=("agentdojo", "system")) or not is_text_list(truth.get("versions")):
        raise ValueError(f"{path}: needs agentdojo and system strings and a versions string array")
    attacks: object = truth.get("attacks")
    if not isinstance(attacks, list) or any(
        not has_texts(record=attack, fields=("name", "template", "user"))
        or not isinstance(attack.get("model"), str | None)
        for attack in attacks
    ):
        raise ValueError(f"{path}: needs attacks with name, template, user and an optional model")
    suites: object = truth.get("suites")
    if not isinstance(suites, dict) or any(not valid_tools(suite=suite) for suite in suites.values()):
        raise ValueError(f"{path}: needs suites with tools that have name, description and parameters")
    key: object
    variants: object
    for key, variants in truth["tasks"].items():
        if not isinstance(key, str) or key.count("/") != 1 or key.split("/")[0] not in suites:
            raise ValueError(f"{path}: task {key} needs a suite/task name with a listed suite")
        validate_variants(variants=variants, key=key, path=path)
