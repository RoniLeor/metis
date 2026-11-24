"""The laya backend. Importing this package registers it, so it loads laya and torch."""

from .backend import LayaBackend

__all__ = ["LayaBackend"]
