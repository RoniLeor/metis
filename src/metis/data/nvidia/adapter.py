"""Convert one raw NVIDIA record into a typed task."""

from metis.data.base import leaves
from metis.data.models import Tool

from .models import Injection, Task


def text(record: dict[str, object], name: str, label: str) -> str:
    """Return a string field, or raise an error that names the record and the field."""
    value: object = record.get(name)
    if not isinstance(value, str):
        raise ValueError(f"record {label}: {name} must be a string")
    return value


def mapping(record: dict[str, object], name: str, label: str) -> dict[str, object]:
    """Return an object field."""
    value: object = record.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"record {label}: {name} must be an object")
    return {str(key): item for key, item in value.items()}


def items(record: dict[str, object], name: str, label: str) -> list[object]:
    """Return an array field."""
    value: object = record.get(name)
    if not isinstance(value, list):
        raise ValueError(f"record {label}: {name} must be an array")
    return list(value)


def adapt_tool(raw: object, label: str) -> Tool:
    """Read a tool in the flat form or in the form nested under a function key."""
    if not isinstance(raw, dict):
        raise ValueError(f"record {label}: each tool must be an object")
    tool: dict[str, object] = {str(key): value for key, value in raw.items()}
    schema: dict[str, object] = mapping(record=tool, name="function", label=label) if "function" in tool else tool
    return Tool(
        name=text(record=schema, name="name", label=label),
        description=text(record=schema, name="description", label=label),
        parameters=mapping(record=schema, name="parameters", label=label),
    )


def adapt_injection(raw: dict[str, object], label: str) -> Injection:
    """Read the injection object. The goal must not be empty."""
    goal: str = text(record=raw, name="goal", label=label)
    if not goal.strip():
        raise ValueError(f"record {label}: goal must not be empty")
    return Injection(
        goal=goal,
        text=text(record=raw, name="injection_text", label=label),
        target_tool=text(record=raw, name="target_tool", label=label),
        target_args=mapping(record=raw, name="target_args", label=label),
        verification_type=text(record=raw, name="verification_type", label=label),
        vector=text(record=raw, name="vector", label=label),
        category=text(record=raw, name="category", label=label),
    )


def messages(params: dict[str, object], label: str) -> dict[str, str]:
    """Return the system and the user message texts."""
    found: dict[str, str] = {}
    message: object
    for message in items(record=params, name="input", label=label):
        if not isinstance(message, dict):
            raise ValueError(f"record {label}: each input message must be an object")
        role: str = str(message.get("role"))
        if role in found:
            raise ValueError(f"record {label}: input repeats the {role} message")
        found[role] = text(record=message, name="content", label=label)
    if set(found) != {"system", "user"}:
        raise ValueError(f"record {label}: input needs one system and one user message")
    return found


def adapt(record: dict[str, object]) -> Task:
    """Validate the fields that a task needs and normalize them."""
    label: str = str(record.get("id", "?"))
    task_id: object = record.get("id")
    if not isinstance(task_id, int) or isinstance(task_id, bool):
        raise ValueError(f"record {label}: id must be an integer")
    params: dict[str, object] = mapping(record=record, name="responses_create_params", label=label)
    texts: dict[str, str] = messages(params=params, label=label)
    required: list[object] = items(record=record, name="required_tools", label=label)
    if not all(isinstance(name, str) for name in required):
        raise ValueError(f"record {label}: required_tools must hold strings")
    environment: dict[str, object] = mapping(record=record, name="environment", label=label)
    injection: Injection = adapt_injection(raw=mapping(record=record, name="injection", label=label), label=label)
    return Task(
        id=task_id,
        domain=text(record=record, name="domain", label=label),
        system=texts["system"],
        prompt=texts["user"],
        tools=[adapt_tool(raw=tool, label=label) for tool in items(record=params, name="tools", label=label)],
        environment=environment,
        required_tools=[str(name) for name in required],
        injection=injection,
        spans=[
            path for path, value in leaves(value=environment, path=[]) if injection.text and injection.text in value
        ],
    )
