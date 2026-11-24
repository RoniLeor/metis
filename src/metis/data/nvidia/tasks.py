"""Clean the NVIDIA records and write the clean tasks, the issues and an offline HTML page."""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from metis.utils import render, write_json, write_lines, write_text

from .adapter import adapt
from .checks import check, duplicates, tasks_without_errors
from .models import Checked, Issue, Task
from .page import payload

SOURCE: str = "nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1"
TEMPLATE: Path = Path(__file__).with_name("viewer.html")


def load(path: Path) -> list[dict[str, object]]:
    """Read one JSON object from each line of the file."""
    records: list[dict[str, object]] = []
    number: int
    line: str
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        record: object = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{number}: each line must be a JSON object")
        records.append({str(key): value for key, value in record.items()})
    return records


def read_tasks(path: Path) -> Checked:
    """Read, adapt and check every record of the file. Repeated record ids raise an error."""
    records: list[dict[str, object]] = load(path=path)
    tasks: list[Task] = [adapt(record=record) for record in records]
    counts: Counter[int] = Counter(task.id for task in tasks)
    repeated: list[int] = sorted(task_id for task_id, count in counts.items() if count > 1)
    if repeated:
        raise ValueError(f"{path}: record ids are not unique: {repeated}")
    issues: list[Issue] = [issue for task, record in zip(tasks, records) for issue in check(task=task, record=record)]
    issues.extend(duplicates(tasks=tasks))
    return Checked(records=records, tasks=tasks, issues=issues, kept=tasks_without_errors(tasks=tasks, issues=issues))


def build(data: Path, output: Path) -> None:
    """Read DATA/train.jsonl and write tasks.jsonl, issues.json and index.html into OUTPUT."""
    path: Path = data / "train.jsonl"
    checked: Checked = read_tasks(path=path)
    output.mkdir(parents=True, exist_ok=True)
    write_lines(path=output / "tasks.jsonl", records=[asdict(task) for task in checked.kept])
    write_json(path=output / "issues.json", value=[asdict(issue) for issue in checked.issues])
    found: dict[str, object] = {"source": SOURCE, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    found.update(payload(checked=checked))
    template: str = TEMPLATE.read_text(encoding="utf-8")
    write_text(path=output / "index.html", text=render(template=template, payload=found))


def main() -> None:
    """Read the input and output directories from the command line."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Directory with train.jsonl")
    parser.add_argument("output", type=Path, help="Output directory")
    args: argparse.Namespace = parser.parse_args()
    try:
        build(data=args.data, output=args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
