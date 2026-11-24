"""Check the backend loader."""

import unittest
from unittest import mock

from metis.models import load_backend
from tests.sample import FakeBackend


class LoaderTests(unittest.TestCase):
    """The loader finds registered backends and explains a missing one."""

    def test_load(self) -> None:
        self.assertIs(load_backend(name="fake"), FakeBackend)
        with self.assertRaisesRegex(ValueError, "lowercase letters"):
            load_backend(name="../x")
        with self.assertRaisesRegex(KeyError, "unknown backend 'nothing'"):
            load_backend(name="nothing")
        with self.assertRaisesRegex(KeyError, "unknown backend 'base'"):
            load_backend(name="base")

    def test_a_missing_package_is_not_an_unknown_backend(self) -> None:
        missing: ModuleNotFoundError = ModuleNotFoundError("no laya", name="laya")
        with mock.patch("importlib.import_module", side_effect=missing):
            with self.assertRaisesRegex(KeyError, "backend 'other' needs laya; run: uv sync --extra train"):
                load_backend(name="other")
        broken: ModuleNotFoundError = ModuleNotFoundError("no helper", name="metis.helper")
        with mock.patch("importlib.import_module", side_effect=broken), self.assertRaises(ModuleNotFoundError):
            load_backend(name="other")
