"""The settings of one training run. Backend-specific values go in options, and the backend checks them."""

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

from metis.data.models import AUGMENTED

DEFAULT_METRICS: tuple[str, ...] = ("accuracy", "precision", "recall", "f1", "fpr", "fpr_twin", "fpr_padded", "brier")
DEFAULT_OUTPUTS: tuple[str, ...] = ("settings", "metrics", "predictions", "checkpoint")
# The epoch choice keeps the largest value, so only metrics where higher is better can select.
SELECTABLE: tuple[str, ...] = ("accuracy", "precision", "recall", "f1")


@dataclass
class Settings:
    """What to train, on which split, and what to measure and save."""

    backend: str = "laya"
    base: str = "convaiinnovations/laya"
    data: Path = Path("resources/data/nvidia/rows")
    strategy: str = "template"
    fold: str = "other templates"
    epochs: int = 1
    seed: int = 0
    device: str = "auto"
    threshold: float = 0.5
    limit: int = 0
    val_share: float = 0.1
    select: str = "f1"
    augment: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=lambda: list(DEFAULT_METRICS))
    outputs: list[str] = field(default_factory=lambda: list(DEFAULT_OUTPUTS))
    project: Path = Path("resources/runs")
    name: str = "exp"
    options: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        """Raise an error for a value that no run can use."""
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs < 1:
            raise ValueError(f"epochs must be a positive integer, got {self.epochs!r}")
        if isinstance(self.limit, bool) or not isinstance(self.limit, int) or self.limit < 0:
            raise ValueError(f"limit must be 0 (all rows) or a positive integer, got {self.limit!r}")
        if not set(self.augment) <= set(AUGMENTED):
            raise ValueError(f"augment must name kinds in {AUGMENTED}, got {self.augment!r}")
        if self.select not in SELECTABLE:
            raise ValueError(f"select must be one of {SELECTABLE}, got {self.select!r}")
        if not 0.0 <= self.val_share < 1.0:
            raise ValueError(f"val_share must be in [0, 1), got {self.val_share!r}")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"threshold must be between 0 and 1, got {self.threshold!r}")
        if not self.name or "/" in self.name:
            raise ValueError(f"name must be a plain folder name, got {self.name!r}")


def parse_value(text: str) -> object:
    """Read a command-line value as JSON, or keep it as text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def typed(current: object, key: str, text: str) -> object:
    """Read a value with the type of the current setting, or raise an error that names the key."""
    if isinstance(current, Path):
        return Path(text)
    if isinstance(current, str):
        return text
    value: object = parse_value(text=text)
    if isinstance(current, float) and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(current, list):
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{key} must be a JSON list of names, got {text!r}")
        return value
    if type(value) is not type(current):
        raise ValueError(f"{key} must be a {type(current).__name__}, got {text!r}")
    return value


def from_pairs(pairs: list[str]) -> Settings:
    """Build settings from key=value pairs. Keys that are not settings fields become backend options."""
    settings: Settings = Settings()
    names: set[str] = {item.name for item in fields(Settings)} - {"options"}
    pair: str
    key: str
    sign: str
    text: str
    for pair in pairs:
        key, sign, text = pair.partition("=")
        if not sign or not key:
            raise ValueError(f"expected key=value, got {pair!r}")
        if key in names:
            setattr(settings, key, typed(current=getattr(settings, key), key=key, text=text))
        else:
            settings.options[key] = parse_value(text=text)
    settings.validate()
    return settings
