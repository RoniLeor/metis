"""Check the helpers that build the ground truth, with synthetic replays."""

import datetime
import unittest

from metis.data.labels.build import add_variant, build_steps, check_source, mask_replay_time
from metis.data.labels.models import CANARY, REPLAY_TIME, Call, Step, Variant

DAYS: set[datetime.date] = {datetime.date(2026, 11, 19), datetime.date(2026, 11, 20)}
SLOT: str = CANARY.format(vector="a")


class Base:
    """Stands in for an AgentDojo task base class."""

    def utility(self) -> bool:
        raise NotImplementedError

    def utility_from_traces(self) -> bool | None:
        return None


class Plain(Base):
    """Defines only the main check."""

    def utility(self) -> bool:
        return True


class Traced(Plain):
    """Inherits the main check and adds the check that takes precedence."""

    def utility_from_traces(self) -> bool | None:
        return False


def variant(versions: list[str], prompt: str, steps: list[Step], check: str = "") -> Variant:
    """Build a variant with the fields that these tests do not vary."""
    return Variant(versions=versions, prompt=prompt, difficulty="EASY", output="", check=check, steps=steps)


def step(args: dict[str, object]) -> Step:
    """Build a step whose two replays agree."""
    return Step(function="send", args=args, result="ok", marked_args=args, marked_result="ok")


class BuildTests(unittest.TestCase):
    """Exercise timestamp masking, step building, variant grouping and check source reading."""

    def test_mask_replay_time_covers_both_days_and_keeps_environment_dates(self) -> None:
        text: str = "a: 2026-11-19 14:52:18.368743\nb: 2026-11-20 00:00:01\nc: 2024-05-15 10:00:00\nd: 2026-11-19"
        expected: str = f"a: {REPLAY_TIME}\nb: {REPLAY_TIME}\nc: 2024-05-15 10:00:00\nd: 2026-11-19"
        self.assertEqual(mask_replay_time(text=text, days=DAYS), expected)

    def test_build_steps_keeps_both_replays_and_masks_both_outputs(self) -> None:
        plain: list[Call] = [Call(function="read", args={"channel": "External_0"}, result="at 2026-11-19 10:00:00")]
        marked: list[Call] = [
            Call(function="read", args={"channel": "External_" + SLOT}, result=SLOT + " at 2026-11-20 10:00:00.5")
        ]
        expected: Step = Step(
            function="read",
            args={"channel": "External_0"},
            result=f"at {REPLAY_TIME}",
            marked_args={"channel": "External_" + SLOT},
            marked_result=f"{SLOT} at {REPLAY_TIME}",
        )
        self.assertEqual(build_steps(plain=plain, marked=marked, days=DAYS), [expected])

    def test_build_steps_accepts_no_calls_and_rejects_other_tools(self) -> None:
        self.assertEqual(build_steps(plain=[], marked=[], days=DAYS), [])
        read: Call = Call(function="read", args={}, result="")
        send: Call = Call(function="send", args={}, result="")
        marked: list[Call]
        for marked in [[send], [], [read, read]]:
            with self.subTest(marked=marked), self.assertRaisesRegex(ValueError, "different tools"):
                build_steps(plain=[read], marked=marked, days=DAYS)

    def test_add_variant_groups_equal_definitions_in_version_order(self) -> None:
        first: Step = step(args={"to": "a", "amount": 1})
        reordered: Step = step(args={"amount": 1, "to": "a"})
        variants: list[Variant] = []
        add_variant(variants=variants, candidate=variant(versions=["v1"], prompt="p", steps=[first, first]))
        add_variant(variants=variants, candidate=variant(versions=["v1.1"], prompt="p", steps=[reordered, reordered]))
        add_variant(variants=variants, candidate=variant(versions=["v1.2"], prompt="q", steps=[first, first]))
        add_variant(variants=variants, candidate=variant(versions=["v1.2.1"], prompt="p", steps=[first]))
        add_variant(variants=variants, candidate=variant(versions=["v1.2.2"], prompt="p", steps=[first], check="x"))
        self.assertEqual([item.versions for item in variants], [["v1", "v1.1"], ["v1.2"], ["v1.2.1"], ["v1.2.2"]])
        self.assertEqual(variants[0].steps, [first, first])

    def test_check_source_keeps_task_methods_in_the_given_order(self) -> None:
        names: tuple[str, ...] = ("utility_from_traces", "utility")
        plain: str = check_source(task_type=Plain, base=Base, names=names)
        traced: str = check_source(task_type=Traced, base=Base, names=names)
        self.assertEqual(plain, "def utility(self) -> bool:\n    return True\n")
        self.assertEqual(traced, "def utility_from_traces(self) -> bool | None:\n    return False\n\n" + plain)
        self.assertEqual(check_source(task_type=Base, base=Base, names=names), "")
