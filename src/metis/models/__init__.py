"""The backend registry. Add one with @backend("name") in metis/models/<name>/ (the package must import the class)."""

import importlib
import re
from collections.abc import Callable
from typing import TypeVar

from metis.cfg import Settings
from metis.utils import Registry

from .base import Backend

Factory = Callable[[Settings], Backend]
BACKENDS: Registry[Factory] = Registry(kind="backend")
B = TypeVar("B", bound=Factory)


def backend(name: str) -> Callable[[B], B]:
    """Return a decorator that registers a backend class under a name and keeps the class as it is."""

    def add(item: B) -> B:
        BACKENDS.register(name=name)(item)
        return item

    return add


def load_backend(name: str) -> Factory:
    """Return the factory of a backend. Its module loads only now, because it can import heavy optional packages."""
    if not re.fullmatch(r"[a-z_]+", name):
        raise ValueError(f"backend name must be lowercase letters, got {name!r}")
    if name not in BACKENDS.names():
        try:
            importlib.import_module(f"{__name__}.{name}")
        except ModuleNotFoundError as error:
            if error.name == f"{__name__}.{name}":
                raise KeyError(f"unknown backend {name!r}; known: {', '.join(BACKENDS.names())}") from error
            if error.name in ("laya", "torch", "transformers"):
                raise KeyError(f"backend {name!r} needs {error.name}; run: uv sync --extra train") from error
            raise
    return BACKENDS.get(name)
