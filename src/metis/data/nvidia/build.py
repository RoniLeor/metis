"""Write the guard rows, the fair splits and the counts for the clean NVIDIA tasks."""

import argparse
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from statistics import median

from metis.data.base import MIN_WORDS, Builder
from metis.data.models import AUGMENTED, Fold, Row

from .models import Task
from .pad import bare_rows, pad_rows, rewrap_rows
from .rows import make_rows
from .splits import folds, groups
from .tasks import read_tasks


def spread(rows: list[Row], kind: str, harmful: bool) -> dict[str, float]:
    """Return the median and the 90th percentile of the word counts of one kind of row."""
    counts: list[int] = sorted(row.words for row in rows if row.kind == kind and row.harmful == harmful)
    if not counts:
        return {}
    return {"median": median(counts), "p90": float(counts[min(len(counts) - 1, int(0.9 * len(counts)))])}


class NvidiaBuilder(Builder):
    """The NVIDIA guard rows of the clean tasks, with the fair splits and the counts."""

    def __init__(self, data: Path) -> None:
        self.tasks: list[Task] = read_tasks(path=data / "train.jsonl").kept

    def rows(self) -> list[Row]:
        """Return the field, twin, padded, bare and rewrap rows, each with the goal group of its task."""
        rows: list[Row] = make_rows(tasks=self.tasks, min_words=MIN_WORDS)
        rows.extend(pad_rows(tasks=self.tasks, rows=rows))
        rows.extend([*bare_rows(tasks=self.tasks, rows=rows), *rewrap_rows(tasks=self.tasks, rows=rows)])
        group_of: dict[int, str] = groups(tasks=self.tasks)
        return [replace(row, group=group_of[row.task_id]) for row in rows]

    def reports(self, rows: list[Row]) -> dict[str, object]:
        """Return splits.json and summary.json. The counts of rows and harmful rows leave out the training copies."""
        scored: list[Row] = [row for row in rows if row.kind not in AUGMENTED]
        chosen: list[Fold] = folds(tasks=self.tasks)
        summary: dict[str, object] = {
            "scope": f"environment strings with at least {MIN_WORDS} words, from {len(self.tasks)} clean tasks",
            "rows": len(scored),
            "harmful": sum(1 for row in scored if row.harmful),
            "augmented_rows": dict(Counter(row.kind for row in rows if row.kind in AUGMENTED)),
            "clean_by_kind": dict(Counter(row.kind for row in rows if not row.harmful)),
            "words": {
                "harmful": spread(rows=rows, kind="field", harmful=True),
                "padded": spread(rows=rows, kind="padded", harmful=False),
            },
        }
        return {"splits.json": [asdict(fold) for fold in chosen], "summary.json": summary}


def main() -> None:
    """Read the data directory and the output folder from the command line, and write the files."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Directory with train.jsonl")
    parser.add_argument("output", type=Path, help="Output folder")
    args: argparse.Namespace = parser.parse_args()
    try:
        NvidiaBuilder(data=args.data).write(folder=args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
