"""Write the AgentDojo ground truth of every task in every benchmark version as JSON.

Run with `make ground-truth`. AgentDojo is a temporary `uv run --with` package, not a project dependency.
"""

import argparse
import datetime
import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path

from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.functions_runtime import FunctionsRuntime, TaskEnvironment
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.task_suite.task_suite import TaskSuite
from agentdojo.types import ChatMessage, get_text_content_as_str

from metis.data.labels.build import add_variant, build_steps, check_source
from metis.data.labels.models import CANARY, Call, Variant
from metis.utils import write_text

from .inputs import attack_templates, suite_tools

VERSIONS: tuple[str, ...] = ("v1", "v1.1", "v1.1.1", "v1.1.2", "v1.2", "v1.2.1", "v1.2.2")
CHECKS: dict[type, tuple[str, ...]] = {
    BaseUserTask: ("utility_from_traces", "utility"),
    BaseInjectionTask: ("security_from_traces", "security"),
}
Task = BaseUserTask[TaskEnvironment] | BaseInjectionTask[TaskEnvironment]


def replay(suite: TaskSuite[TaskEnvironment], task: Task, prompt: str, injections: dict[str, str]) -> list[Call]:
    """Run the task's ground-truth calls on the default environment with the given injections."""
    environment: TaskEnvironment = suite.load_and_inject_default_environment(injections=injections)
    runtime: FunctionsRuntime = FunctionsRuntime(functions=suite.tools)
    pipeline: GroundTruthPipeline = GroundTruthPipeline(task=task)
    messages: list[ChatMessage] = list(pipeline.query(query=prompt, runtime=runtime, env=environment)[3])
    calls: list[Call] = []
    message: ChatMessage
    for message in messages:
        if message["role"] == "tool":
            text: str = get_text_content_as_str(content_blocks=message["content"] or [])
            calls.append(
                Call(function=message["tool_call"].function, args=dict(message["tool_call"].args), result=text)
            )
    return calls


def task_variant(suite: TaskSuite[TaskEnvironment], task: Task, version: str, days: set[datetime.date]) -> Variant:
    """Replay one task. User tasks replay twice: plain for the calls, with canaries to find the injection slots."""
    base: type = BaseUserTask if isinstance(task, BaseUserTask) else BaseInjectionTask
    prompt: str = task.PROMPT if isinstance(task, BaseUserTask) else task.GOAL
    plain: list[Call] = replay(suite=suite, task=task, prompt=prompt, injections={})
    marked: list[Call] = plain
    if isinstance(task, BaseUserTask):
        vectors: list[str] = list(suite.get_injection_vector_defaults())
        canaries: dict[str, str] = {vector: CANARY.format(vector=vector) for vector in vectors}
        marked = replay(suite=suite, task=task, prompt=prompt, injections=canaries)
    return Variant(
        versions=[version],
        prompt=prompt,
        difficulty=task.DIFFICULTY.name,
        output=task.GROUND_TRUTH_OUTPUT,
        check=check_source(task_type=type(task), base=base, names=CHECKS[base]),
        steps=build_steps(plain=plain, marked=marked, days=days),
    )


def main() -> None:
    """Replay all tasks and write the variants with the shared input: system prompt, tools and attack texts."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Output JSON file")
    args: argparse.Namespace = parser.parse_args()
    today: datetime.date = datetime.date.today()
    days: set[datetime.date] = {today, today + datetime.timedelta(days=1)}
    tasks: dict[str, list[Variant]] = {}
    latest: dict[str, TaskSuite[TaskEnvironment]] = {}  # The suites of the last version, for the shared input.
    version: str
    for version in VERSIONS:
        latest = get_suites(benchmark_version=version)
        suite_name: str
        suite: TaskSuite[TaskEnvironment]
        for suite_name, suite in latest.items():
            task_id: str
            task: Task
            for task_id, task in [*suite.user_tasks.items(), *suite.injection_tasks.items()]:
                variant: Variant = task_variant(suite=suite, task=task, version=version, days=days)
                add_variant(variants=tasks.setdefault(f"{suite_name}/{task_id}", []), candidate=variant)
    truth: dict[str, object] = {
        "agentdojo": importlib.metadata.version("agentdojo"),
        "versions": list(VERSIONS),
        "system": load_system_message(system_message_name=None),
        "attacks": [asdict(attack) for attack in attack_templates(suite=next(iter(latest.values())))],
        "suites": {
            name: {"tools": [asdict(tool) for tool in suite_tools(suite=suite)]} for name, suite in latest.items()
        },
        "tasks": {key: [asdict(variant) for variant in variants] for key, variants in tasks.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_text(path=args.output, text=json.dumps(truth, ensure_ascii=False))


if __name__ == "__main__":
    main()
