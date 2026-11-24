"""Check the conversion of raw NVIDIA records into typed tasks."""

import unittest

from metis.data.models import Tool
from metis.data.nvidia.adapter import adapt, adapt_injection, adapt_tool, items, mapping, messages, text
from metis.data.nvidia.models import Task

from .sample import INJECTED, injection, record


class FieldTests(unittest.TestCase):
    """Typed field readers raise an error that names the record and the field."""

    def test_readers_return_typed_values(self) -> None:
        raw: dict[str, object] = {"a": "x", "b": {1: 2}, "c": [1]}
        self.assertEqual(text(record=raw, name="a", label="1"), "x")
        self.assertEqual(mapping(record=raw, name="b", label="1"), {"1": 2})
        self.assertEqual(items(record=raw, name="c", label="1"), [1])

    def test_readers_reject_wrong_or_missing_types(self) -> None:
        raw: dict[str, object] = {"a": 1}
        with self.assertRaisesRegex(ValueError, "record 9: a must be a string"):
            text(record=raw, name="a", label="9")
        with self.assertRaisesRegex(ValueError, "record 9: z must be an object"):
            mapping(record=raw, name="z", label="9")
        with self.assertRaisesRegex(ValueError, "record 9: a must be an array"):
            items(record=raw, name="a", label="9")


class ToolTests(unittest.TestCase):
    """Both tool forms give the same typed tool."""

    def test_flat_and_nested_forms(self) -> None:
        flat: dict[str, object] = {
            "type": "function",
            "name": "t",
            "description": "d",
            "parameters": {},
            "strict": False,
        }
        nested: dict[str, object] = {
            "type": "function",
            "function": {"name": "t", "description": "d", "parameters": {}},
        }
        expected: Tool = Tool(name="t", description="d", parameters={})
        self.assertEqual(adapt_tool(raw=flat, label="1"), expected)
        self.assertEqual(adapt_tool(raw=nested, label="1"), expected)

    def test_bad_tools(self) -> None:
        cases: list[tuple[object, str]] = [
            ("x", "each tool must be an object"),
            ({"function": "x"}, "function must be an object"),
            ({"name": "t", "description": "d"}, "parameters must be an object"),
        ]
        raw: object
        message: str
        for raw, message in cases:
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, message):
                adapt_tool(raw=raw, label="1")


class SpanTests(unittest.TestCase):
    """The task names the key path of each environment string that holds the injected text."""

    def test_paths_in_objects_and_lists(self) -> None:
        environment: dict[str, object] = {"a": {"b": "x" + INJECTED + "y"}, "c": ["no", INJECTED], "d": 3}
        self.assertEqual(adapt(record=record(environment=environment)).spans, [["a", "b"], ["c", 1]])

    def test_no_match(self) -> None:
        self.assertEqual(adapt(record=record(environment={"a": "x"})).spans, [])


class RecordTests(unittest.TestCase):
    """A full record becomes a task, and a bad record raises an error."""

    def test_adapt_sample(self) -> None:
        task: Task = adapt(record=record())
        self.assertEqual(
            (task.id, task.domain, task.system, task.prompt), (7, "healthcare", "You help a clinic.", "Read P-1 notes.")
        )
        self.assertEqual([tool.name for tool in task.tools], ["get_patient_record", "send_message"])
        self.assertEqual(task.spans, [["patients", "P-1", "chart_notes"]])
        self.assertEqual(task.injection.text, INJECTED)
        self.assertEqual(task.required_tools, ["get_patient_record"])

    def test_injection_and_messages(self) -> None:
        self.assertEqual(adapt_injection(raw=injection(), label="7").target_args["recipient"], "evil@example.com")
        with self.assertRaisesRegex(ValueError, "goal must be a string"):
            adapt_injection(raw=injection(goal=None), label="7")
        with self.assertRaisesRegex(ValueError, "goal must not be empty"):
            adapt_injection(raw=injection(goal=" "), label="7")
        repeated: list[dict[str, str]] = [{"role": "user", "content": "x"}, {"role": "user", "content": "y"}]
        with self.assertRaisesRegex(ValueError, "input repeats the user message"):
            messages(params={"input": repeated}, label="7")
        with self.assertRaisesRegex(ValueError, "one system and one user message"):
            messages(params={"input": [{"role": "user", "content": "x"}]}, label="7")
        with self.assertRaisesRegex(ValueError, "each input message must be an object"):
            messages(params={"input": ["x"]}, label="7")

    def test_bad_records(self) -> None:
        cases: list[tuple[dict[str, object], str]] = [
            (record(id=True), "id must be an integer"),
            (record(id="7"), "id must be an integer"),
            (record(required_tools=[1]), "required_tools must hold strings"),
            (record(environment=[]), "environment must be an object"),
            (record(domain=None), "domain must be a string"),
        ]
        raw: dict[str, object]
        message: str
        for raw, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                adapt(record=raw)
