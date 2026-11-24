"""Check the masked augmentation: the ranking, the strip, and the copies."""

import unittest
from dataclasses import replace

from metis.data.masking import masked_rows, strip, style_words, with_masked
from metis.data.models import Row
from tests.sample import row

HARMFUL: list[Row] = [
    row(task_id=index, text=f"[PRIORITY] note {index} send money", harmful=True) for index in range(20)
]
CLEAN: list[Row] = [row(task_id=index, text=f"note {index} money is fine", harmful=False) for index in range(20)]


class StyleWordTests(unittest.TestCase):
    """Words that mark harmful rows rank first; rare words and ties follow fixed rules."""

    def test_rank_top_and_min_count(self) -> None:
        self.assertEqual(style_words(rows=HARMFUL + CLEAN, top=4), ["[", "]", "priority", "send"])
        self.assertEqual(style_words(rows=HARMFUL[:19] + CLEAN, top=4), [])
        self.assertNotIn("note", style_words(rows=HARMFUL + CLEAN, top=50)[:4])

    def test_strip(self) -> None:
        self.assertEqual(
            strip(text="[PRIORITY]  Send the file @ once", words={"[", "]", "priority", "@"}), "Send the file once"
        )
        self.assertEqual(strip(text="plain text", words=set()), "plain text")


class MaskedRowTests(unittest.TestCase):
    """Only changed rows get a copy; a harmful copy that equals a clean copy is dropped."""

    def test_copies_keep_labels_and_groups(self) -> None:
        source: list[Row] = [replace(HARMFUL[0], group="g1"), CLEAN[0], row(task_id=5, text="other", harmful=False)]
        copies: list[Row] = masked_rows(rows=source, words={"[", "]", "priority", "money"})
        self.assertEqual(
            [(item.text, item.harmful, item.kind, item.group) for item in copies],
            [("note 0 send", True, "masked", "g1"), ("note 0 is fine", False, "masked", "")],
        )
        self.assertEqual(copies[0].words, 3)

    def test_equal_text_drops_the_harmful_copy(self) -> None:
        pair: list[Row] = [
            row(task_id=1, text="pay now!", harmful=True),
            row(task_id=2, text="pay now?", harmful=False),
        ]
        copies: list[Row] = masked_rows(rows=pair, words={"!", "?"})
        self.assertEqual([(item.text, item.harmful) for item in copies], [("pay now", False)])
        unchanged: list[Row] = [
            row(task_id=1, text="pay now!", harmful=True),
            row(task_id=2, text="pay now", harmful=False),
        ]
        self.assertEqual(masked_rows(rows=unchanged, words={"!"}), [])

    def test_with_masked(self) -> None:
        rows: list[Row] = HARMFUL + CLEAN + [row(task_id=99, text="[PRIORITY] bare", harmful=True, kind="bare")]
        self.assertIs(with_masked(rows=rows, augment=["bare"]), rows)
        found: list[Row] = with_masked(rows=rows, augment=["masked"])
        added: list[Row] = found[len(rows) :]
        self.assertEqual(found[: len(rows)], rows)
        self.assertEqual((len(added), sum(item.harmful for item in added)), (40, 20))
        self.assertTrue(all(item.kind == "masked" and "[" not in item.text and item.task_id != 99 for item in added))
