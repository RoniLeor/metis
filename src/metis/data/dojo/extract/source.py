"""Write the strings of the AgentDojo v1.2.2 environments, clean and with each attack in each reached injection slot.

Run with `make dojo-source`. AgentDojo is a temporary `uv run --with` package, not a project dependency. AgentDojo
puts the attack text into the environment YAML before it parses it, so the injected strings are made here, inside
AgentDojo, and are exactly the strings that a tool returns.
"""

import argparse
import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path

from agentdojo.attacks.base_attacks import FixedJailbreakAttack
from agentdojo.functions_runtime import TaskEnvironment
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.task_suite.task_suite import TaskSuite

from metis.data.dojo.models import Injected, Leaf
from metis.data.models import KeyPath
from metis.utils import write_text

from .inputs import Slot, fixed_attacks, reached, slots, strings

VERSION: str = "v1.2.2"


def attack_strings(
    suite: TaskSuite[TaskEnvironment], vectors: list[str], attacks: dict[str, FixedJailbreakAttack]
) -> dict[tuple[str, str], dict[str, tuple[KeyPath, str]]]:
    """Return the strings of the environment for each attack and goal, with the attack text in every vector.

    One load serves all vectors, because no string holds two vectors (slots() checks this).
    """
    found: dict[tuple[str, str], dict[str, tuple[KeyPath, str]]] = {}
    for attack_name, attack in attacks.items():
        for task_id, task in suite.injection_tasks.items():
            text: str = attack.jailbreak.format(goal=task.GOAL, user=attack.user_name, model=attack.model_name)
            found[(attack_name, task_id)] = strings(suite=suite, injections={vector: text for vector in vectors})
    return found


def injected(
    suite_name: str,
    suite: TaskSuite[TaskEnvironment],
    places: dict[str, list[Slot]],
    attacks: dict[str, FixedJailbreakAttack],
) -> list[Injected]:
    """Return the strings with each attack and each attacker goal of the suite in each vector, vector by vector."""
    values: dict[tuple[str, str], dict[str, tuple[KeyPath, str]]] = attack_strings(
        suite=suite, vectors=list(places), attacks=attacks
    )
    found: list[Injected] = []
    for vector, slot_list in places.items():
        for attack_name in attacks:
            for task_id, task in suite.injection_tasks.items():
                found.extend(
                    Injected(
                        suite=suite_name,
                        vector=vector,
                        injection_task=task_id,
                        goal=task.GOAL,
                        attack=attack_name,
                        path=path,
                        text=values[(attack_name, task_id)][json.dumps(path)][1],
                        prefix=before,
                        suffix=after,
                    )
                    for path, before, after in slot_list
                )
    return found


def suite_records(suite_name: str, suite: TaskSuite[TaskEnvironment]) -> tuple[list[Leaf], list[Injected]]:
    """Return the clean strings of the default environment and the injected strings of the reached vectors."""
    attacks: dict[str, FixedJailbreakAttack] = fixed_attacks(suite=suite)
    places: dict[str, list[Slot]] = slots(
        suite=suite, vectors=reached(suite=suite, attack=next(iter(attacks.values())))
    )
    marks: dict[str, tuple[str, int, int]] = {
        json.dumps(path): (vector, before, after) for vector, found in places.items() for path, before, after in found
    }
    clean: list[Leaf] = []
    for key, (path, text) in strings(suite=suite, injections={}).items():
        vector, before, after = marks.get(key, ("", 0, 0))
        clean.append(Leaf(suite=suite_name, path=path, text=text, vector=vector, prefix=before, suffix=after))
    return clean, injected(suite_name=suite_name, suite=suite, places=places, attacks=attacks)


def main() -> None:
    """Write the clean and injected strings of every suite as one JSON file."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Output JSON file")
    args: argparse.Namespace = parser.parse_args()
    clean: list[Leaf] = []
    found: list[Injected] = []
    for suite_name, suite in get_suites(benchmark_version=VERSION).items():
        suite_clean, suite_found = suite_records(suite_name=suite_name, suite=suite)
        clean.extend(suite_clean)
        found.extend(suite_found)
    source: dict[str, object] = {
        "agentdojo": importlib.metadata.version("agentdojo"),
        "leaves": [asdict(leaf) for leaf in clean],
        "injected": [asdict(item) for item in found],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_text(path=args.output, text=json.dumps(source, ensure_ascii=False))


if __name__ == "__main__":
    main()
