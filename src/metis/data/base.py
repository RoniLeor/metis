"""The shared base of the guard data: the row, its file, and the builder that each source extends.

A source (NVIDIA, AgentDojo) extends Builder: it gives its rows and its report files, and write() saves them.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path

from metis.utils import write_json, write_lines

from .models import KINDS, KeyPath, Row

MIN_WORDS: int = 5  # A row needs at least this many words.


def read_row(raw: object, label: str) -> Row:
    """Check one rows.jsonl line and return it as a row."""
    if not isinstance(raw, dict):
        raise ValueError(f"{label}: each line must be a JSON object")
    texts: tuple[str, ...] = ("domain", "template", "goal", "text", "kind")
    numbers: tuple[str, ...] = ("task_id", "words")
    if not all(isinstance(raw.get(key), str) for key in texts) or raw.get("kind") not in KINDS:
        raise ValueError(f"{label}: needs text fields {texts} and a kind in {KINDS}")
    if not all(isinstance(raw.get(key), int) and not isinstance(raw.get(key), bool) for key in numbers):
        raise ValueError(f"{label}: needs integer fields {numbers}")
    path: object = raw.get("path")
    if not isinstance(path, list) or not all(isinstance(key, (str, int)) for key in path):
        raise ValueError(f"{label}: path must be a list of keys")
    if not isinstance(raw.get("harmful"), bool) or not isinstance(raw.get("group", ""), str):
        raise ValueError(f"{label}: harmful must be true or false, and group a string")
    return Row(
        task_id=raw["task_id"],
        domain=raw["domain"],
        template=raw["template"],
        goal=raw["goal"],
        path=list(path),
        text=raw["text"],
        words=raw["words"],
        harmful=raw["harmful"],
        kind=raw["kind"],
        group=raw.get("group", ""),
    )


def read_rows(path: Path) -> list[Row]:
    """Read every row of a rows.jsonl file."""
    lines: list[str] = path.read_text(encoding="utf-8").splitlines()
    return [read_row(raw=json.loads(line), label=f"{path}:{number}") for number, line in enumerate(lines, start=1)]


def leaves(value: object, path: KeyPath) -> list[tuple[KeyPath, str]]:
    """Return the key path and the text of each string in a JSON value."""
    if isinstance(value, str):
        return [(path, value)]
    children: list[tuple[str | int, object]] = []
    if isinstance(value, dict):
        children = [(str(key), child) for key, child in value.items()]
    elif isinstance(value, list):
        children = list(enumerate(value))
    return [leaf for key, child in children for leaf in leaves(value=child, path=[*path, key])]


class Builder(ABC):
    """Build the guard rows of one source and write them with the source's report files."""

    @abstractmethod
    def rows(self) -> list[Row]:
        """Return the guard rows of the source."""

    @abstractmethod
    def reports(self, rows: list[Row]) -> dict[str, object]:
        """Return each report file of the rows: the file name and its JSON value."""

    def write(self, folder: Path) -> list[Row]:
        """Write rows.jsonl and the report files into the folder, and return the rows."""
        rows: list[Row] = self.rows()
        folder.mkdir(parents=True, exist_ok=True)
        write_lines(path=folder / "rows.jsonl", records=[asdict(row) for row in rows])
        for name, value in self.reports(rows=rows).items():
            write_json(path=folder / name, value=value)
        return rows
