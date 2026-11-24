"""Check the shared helpers: the name registry, the file writers and the page render."""

import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from metis.utils import Registry, render, write_json, write_lines, write_text

TEMPLATE: str = (
    "<meta content=\"script-src 'sha256-__SCRIPT_HASH__'\">"
    '<script id="r" type="application/json">__RECORDS__</script><script>run()</script>'
)


class RenderTests(unittest.TestCase):
    """The renderer embeds inert data and the hash of the one inline script."""

    def test_render(self) -> None:
        payload: dict[str, object] = {"text": "</script><b>\u2028"}
        html: str = render(template=TEMPLATE, payload=payload)
        data: str = html.split('type="application/json">')[1].split("</script>")[0]
        digest: str = base64.b64encode(hashlib.sha256(b"run()").digest()).decode("ascii")
        self.assertEqual(json.loads(data), payload)
        self.assertNotIn("<", data)
        self.assertIn(f"'sha256-{digest}'", html)


class FileTests(unittest.TestCase):
    """Files are written whole, keep non-ASCII text, and leave no temporary file."""

    def test_write_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder: Path = Path(directory)
            write_json(path=folder / "a.json", value={"name": "café", "path": Path("x/y")})
            write_json(path=folder / "a.json", value={"name": "café", "path": Path("x/z")})
            text: str = (folder / "a.json").read_text(encoding="utf-8")
            names: list[str] = [path.name for path in folder.iterdir()]
        self.assertEqual(text, '{\n  "name": "café",\n  "path": "x/z"\n}\n')
        self.assertEqual(names, ["a.json"])

    def test_write_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder: Path = Path(directory)
            write_lines(path=folder / "rows.jsonl", records=[{"text": "naïve"}, {"text": "b"}])
            write_lines(path=folder / "empty.jsonl", records=[])
            lines: list[str] = (folder / "rows.jsonl").read_text(encoding="utf-8").splitlines()
            empty: str = (folder / "empty.jsonl").read_text(encoding="utf-8")
        self.assertEqual(
            ([json.loads(line) for line in lines], lines[0], empty),
            ([{"text": "naïve"}, {"text": "b"}], '{"text": "naïve"}', ""),
        )

    def test_write_text_replaces_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path: Path = Path(directory) / "a.txt"
            write_text(path=path, text="one")
            write_text(path=path, text="two")
            self.assertEqual((path.read_text(encoding="utf-8"), len(list(path.parent.iterdir()))), ("two", 1))


class RegistryTests(unittest.TestCase):
    """A decorator adds an object, and get finds it by name."""

    def test_register_get_and_names(self) -> None:
        table: Registry[int] = Registry(kind="number")
        table.register(name="two")(2)
        table.register(name="one")(1)
        self.assertEqual((table.get(name="two"), table.names()), (2, ["one", "two"]))

    def test_errors(self) -> None:
        table: Registry[int] = Registry(kind="number")
        table.register(name="one")(1)
        with self.assertRaisesRegex(ValueError, "number 'one' is already registered"):
            table.register(name="one")(2)
        with self.assertRaisesRegex(KeyError, "unknown number 'two'; known: one"):
            table.get(name="two")
