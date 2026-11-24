"""What the two AgentDojo commands share: tools, attack texts, reached vectors and environment strings."""

import json
import string

from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from agentdojo.attacks.attack_registry import ATTACKS, load_attack
from agentdojo.attacks.base_attacks import BaseAttack, FixedJailbreakAttack
from agentdojo.functions_runtime import TaskEnvironment
from agentdojo.task_suite.task_suite import TaskSuite

from metis.data.base import leaves
from metis.data.labels.models import CANARY, Attack
from metis.data.models import KeyPath, Tool

PIPELINE: str = "gpt-4o-2024-05-13"  # The model name that the attacks address.
OTHER: str = "command-r-plus"  # A second model name, to see if an attack text depends on the model.
# The key path of a string that holds the slot, and its words before and after the slot.
Slot = tuple[KeyPath, int, int]


def named_pipeline(name: str) -> GroundTruthPipeline:
    """Return a pipeline that only carries a model name. An attack reads the name to address the model."""
    pipeline: GroundTruthPipeline = GroundTruthPipeline(task=None)
    pipeline.name = name
    return pipeline


def fixed_attacks(suite: TaskSuite[TaskEnvironment], model: str = PIPELINE) -> dict[str, FixedJailbreakAttack]:
    """Load, by name, the attacks with a fixed text that needs only the goal, the user name and the model name."""
    found: dict[str, FixedJailbreakAttack] = {}
    name: str
    for name in ATTACKS:
        attack: BaseAttack = load_attack(attack_name=name, task_suite=suite, target_pipeline=named_pipeline(name=model))
        if isinstance(attack, FixedJailbreakAttack):
            fields: set[str] = {field for _, field, _, _ in string.Formatter().parse(attack.jailbreak) if field}
            if fields <= {"goal", "user", "model"}:
                found[name] = attack
    return found


def attack_templates(suite: TaskSuite[TaskEnvironment]) -> list[Attack]:
    """Return the text of each fixed attack. The model is None when the attack names each model in its own way."""
    others: dict[str, FixedJailbreakAttack] = fixed_attacks(suite=suite, model=OTHER)
    return [
        Attack(
            name=name,
            template=attack.jailbreak,
            user=attack.user_name,
            model=attack.model_name if attack.model_name == others[name].model_name else None,
        )
        for name, attack in fixed_attacks(suite=suite).items()
    ]


def reached(suite: TaskSuite[TaskEnvironment], attack: BaseAttack) -> list[str]:
    """Return the injection vectors that the ground truth of at least one user task reads."""
    found: set[str] = set()
    for task in suite.user_tasks.values():
        try:
            found |= set(attack.get_injection_candidates(user_task=task))
        except ValueError:  # AgentDojo raises it for a user task that reads no vector.
            continue
    return sorted(found)


def suite_tools(suite: TaskSuite[TaskEnvironment]) -> list[Tool]:
    """Describe the tools that the agent can call in the suite."""
    return [
        Tool(name=tool.name, description=tool.description, parameters=tool.parameters.model_json_schema())
        for tool in suite.tools
    ]


def strings(suite: TaskSuite[TaskEnvironment], injections: dict[str, str]) -> dict[str, tuple[KeyPath, str]]:
    """Return each string of the environment with these injections, keyed by its JSON path."""
    environment: TaskEnvironment = suite.load_and_inject_default_environment(injections=injections)
    return {json.dumps(path): (path, text) for path, text in leaves(value=environment.model_dump(mode="json"), path=[])}


def slots(suite: TaskSuite[TaskEnvironment], vectors: list[str]) -> dict[str, list[Slot]]:
    """Return the strings that hold each vector, found with one canary for each vector in one load."""
    canaries: dict[str, str] = {vector: CANARY.format(vector=vector) for vector in vectors}
    found: dict[str, list[Slot]] = {vector: [] for vector in vectors}
    for path, text in strings(suite=suite, injections=canaries).values():
        held: list[str] = [vector for vector, canary in canaries.items() if canary in text]
        if len(held) > 1:
            raise ValueError(f"{path}: one string holds the vectors {held}")
        for vector in held:
            before, _, after = text.partition(canaries[vector])
            found[vector].append((path, len(before.split()), len(after.split())))
    return found
