"""Small shared helpers: the name registry, the file writers and the HTML page render."""

import base64
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A named table of objects of one kind."""

    def __init__(self, kind: str) -> None:
        self.kind: str = kind
        self.items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Return a decorator that adds an object under a name. A name can be used only once."""

        def add(item: T) -> T:
            if name in self.items:
                raise ValueError(f"{self.kind} {name!r} is already registered")
            self.items[name] = item
            return item

        return add

    def get(self, name: str) -> T:
        """Return the object of a name, or raise an error that lists the known names."""
        if name not in self.items:
            raise KeyError(f"unknown {self.kind} {name!r}; known: {', '.join(sorted(self.items))}")
        return self.items[name]

    def names(self) -> list[str]:
        """Return the registered names in sorted order."""
        return sorted(self.items)


def write_text(path: Path, text: str) -> None:
    """Write the text to a temporary file, then put it in place of the path."""
    temporary: Path = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: object) -> None:
    """Write one JSON file. A value that JSON does not know, such as a path, becomes its text."""
    write_text(path=path, text=json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def write_lines(path: Path, records: list[dict[str, object]]) -> None:
    """Write a JSONL file: one JSON object per line."""
    write_text(path=path, text="".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))


def render(template: str, payload: object) -> str:
    """Embed the payload as inert data and allow only the one inline script of the template."""
    script: str = template.split("<script>")[1].split("</script>")[0]
    digest: bytes = hashlib.sha256(script.encode("utf-8")).digest()
    script_hash: str = base64.b64encode(digest).decode("ascii")
    data: str = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")
    return template.replace("__SCRIPT_HASH__", script_hash).replace("__RECORDS__", data)
