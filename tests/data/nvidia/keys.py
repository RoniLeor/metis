"""Check the match keys of the NeMo Gym trace check."""

import unittest

from metis.data.nvidia.keys import match_keys, strict_keys


class KeyTests(unittest.TestCase):
    """The strict keys of the verification type decide which arguments must match."""

    def test_strict_keys_and_fallback(self) -> None:
        args: dict[str, object] = {"recipient": "a", "body": "b"}
        self.assertEqual(strict_keys(target_args=args, verification_type="check_message_sent"), ["recipient"])
        self.assertEqual(match_keys(target_args=args, verification_type="check_message_sent"), ["recipient"])
        self.assertEqual(match_keys(target_args=args, verification_type="check_note_added"), ["recipient", "body"])
        self.assertEqual(match_keys(target_args={"body": "b"}, verification_type="check_message_sent"), ["body"])

    def test_no_arguments(self) -> None:
        self.assertEqual(strict_keys(target_args={}, verification_type="check_message_sent"), [])
        self.assertEqual(match_keys(target_args={}, verification_type="check_message_sent"), [])
