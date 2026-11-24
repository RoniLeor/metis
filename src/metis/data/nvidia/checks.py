"""Find data problems in the NVIDIA tasks and keep the tasks without errors."""

from collections import Counter

from .models import Issue, Task


def copy_issues(task: Task, record: dict[str, object]) -> list[Issue]:
    """Report each top-level record field that differs from its copy in the injection object."""
    copies: dict[str, str] = {
        "target_tool": task.injection.target_tool,
        "injection_vector": task.injection.vector,
        "attack_category": task.injection.category,
    }
    return [
        Issue(task_id=task.id, code="copy-differs", level="error", detail=f"{name} differs from the injection object")
        for name, value in copies.items()
        if record.get(name) != value
    ]


def span_issues(task: Task) -> list[Issue]:
    """Report an injection that is not in the environment, or that is in more than one string."""
    if not task.spans:
        detail: str = "injection text is not in the environment"
        return [Issue(task_id=task.id, code="no-span", level="error", detail=detail)]
    if len(task.spans) > 1:
        detail = f"injection text is in {len(task.spans)} strings"
        return [Issue(task_id=task.id, code="many-spans", level="warning", detail=detail)]
    return []


def schema_issues(task: Task, schema: dict[str, object]) -> list[Issue]:
    """Report target arguments that the target tool schema does not have, and required arguments that are missing."""
    properties: object = schema.get("properties", {})
    required: object = schema.get("required", [])
    names: list[str] = [str(key) for key in properties] if isinstance(properties, dict) else []
    needed: list[str] = [str(key) for key in required] if isinstance(required, list) else []
    extra: list[str] = [key for key in task.injection.target_args if key not in names]
    missing: list[str] = [key for key in needed if key not in task.injection.target_args]
    if not extra and not missing:
        return []
    detail: str = f"target arguments not in the schema: {extra}; required and missing: {missing}"
    return [Issue(task_id=task.id, code="schema-mismatch", level="error", detail=detail)]


def tool_issues(task: Task) -> list[Issue]:
    """Report target or required tools that the task does not list, and a target call outside the schema."""
    tools: dict[str, dict[str, object]] = {tool.name: tool.parameters for tool in task.tools}
    issues: list[Issue] = [
        Issue(task_id=task.id, code="unknown-tool", level="error", detail=f"{name} is not in the tools")
        for name in [task.injection.target_tool, *task.required_tools]
        if name not in tools
    ]
    if task.injection.target_tool in tools:
        issues.extend(schema_issues(task=task, schema=tools[task.injection.target_tool]))
    return issues


def check(task: Task, record: dict[str, object]) -> list[Issue]:
    """Return the problems of one task."""
    issues: list[Issue] = [*copy_issues(task=task, record=record), *span_issues(task=task), *tool_issues(task=task)]
    if task.injection.goal not in task.injection.text:
        detail: str = "the goal text is not inside the injection text"
        issues.append(Issue(task_id=task.id, code="goal-not-in-text", level="warning", detail=detail))
    return issues


def duplicates(tasks: list[Task]) -> list[Issue]:
    """Report the tasks that share a user prompt with another task."""
    counts: Counter[str] = Counter(task.prompt for task in tasks)
    return [
        Issue(task_id=task.id, code="repeated-prompt", level="warning", detail="another task has the same user prompt")
        for task in tasks
        if counts[task.prompt] > 1
    ]


def tasks_without_errors(tasks: list[Task], issues: list[Issue]) -> list[Task]:
    """Return the tasks that have no error."""
    errors: set[int] = {issue.task_id for issue in issues if issue.level == "error"}
    return [task for task in tasks if task.id not in errors]
