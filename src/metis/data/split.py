"""Split the guard rows of one source into the train, validation and test parts of one fold."""

import json
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .base import read_rows
from .models import AUGMENTED, Row

ALL: str = "all"  # The strategy with no test part: every task is on the train side, for a final model.


@dataclass(frozen=True)
class Split:
    """The train, validation and test rows of one fold. The rows of one task are on one side only."""

    train: list[Row]
    val: list[Row]
    test: list[Row]


def fold_ids(path: Path, strategy: str, fold: str) -> set[int]:
    """Return the test task ids of one fold in splits.json."""
    folds: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(folds, list):
        raise ValueError(f"{path}: needs a list of folds")
    for item in folds:
        if isinstance(item, dict) and item.get("strategy") == strategy and item.get("name") == fold:
            ids: object = item.get("test_ids")
            if not isinstance(ids, list) or not all(isinstance(task_id, int) for task_id in ids):
                raise ValueError(f"{path}: fold {fold!r} needs a list of task ids")
            return set(ids)
    raise ValueError(f"{path}: no fold {fold!r} in strategy {strategy!r}")


def group_of(row: Row) -> str:
    """Return the goal group of a row, or its task id when the data has no groups."""
    return row.group or str(row.task_id)


def load_split(folder: Path, strategy: str, fold: str, val_share: float, seed: int, augment: list[str]) -> Split:
    """Return the rows of one fold. The test part is the fold's tasks, without bare or rewrap rows.

    Validation takes whole goal groups of the train tasks: their bare rows (harmful, without a wrapper) and their clean
    rows. The other train tasks are the train part, with the augmented kinds that augment names.
    """
    rows: list[Row] = read_rows(path=folder / "rows.jsonl")
    tested: set[int] = set() if strategy == ALL else fold_ids(path=folder / "splits.json", strategy=strategy, fold=fold)
    side: list[Row] = [row for row in rows if row.task_id not in tested]
    held: set[str] = held_keys(keys=[group_of(row=row) for row in side], seed=seed, share=val_share, max_rows=len(side))
    return Split(
        train=[
            row for row in side if group_of(row=row) not in held and (row.kind not in AUGMENTED or row.kind in augment)
        ],
        val=[row for row in side if group_of(row=row) in held and (row.kind == "bare" or not row.harmful)],
        test=[row for row in rows if row.task_id in tested and row.kind not in AUGMENTED],
    )


def held_keys(keys: Sequence[str], seed: int, share: float, max_rows: int) -> set[str]:
    """Choose whole groups of rows: a share of the distinct keys, with at most max_rows rows.

    Rows with one key share text or a goal (a harmful field, its twin and its padded row), so a split must not cut them.
    """
    sizes: Counter[str] = Counter(keys)
    order: list[str] = sorted(sizes)
    random.Random(seed).shuffle(order)
    held: set[str] = set()
    rows: int = 0
    key: str
    for key in order:
        if len(held) >= int(len(order) * share):
            break
        if rows + sizes[key] <= max_rows:
            held.add(key)
            rows += sizes[key]
    return held


def sample(rows: list[Row], limit: int, seed: int) -> list[Row]:
    """Return at most limit rows, chosen with a fixed seed. A limit of 0 keeps all rows."""
    return rows if not limit or limit >= len(rows) else random.Random(seed).sample(rows, limit)
