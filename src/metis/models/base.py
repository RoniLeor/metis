"""What a backend must do: train on rows, give harmful probabilities, save itself and report facts."""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from metis.data.models import Row

EpochHook = Callable[[int, float], None]


class Backend(Protocol):
    """A model that the trainer can fit, ask and save."""

    def fit(self, rows: list[Row], epochs: int, seed: int, on_epoch_end: EpochHook) -> list[float]:
        """Train on the rows and return the mean loss of each epoch."""
        ...

    def finish(self) -> None:
        """Run the step after training and epoch selection, for example the fit of new temperatures."""
        ...

    def state(self) -> object:
        """Return a copy of the trained weights that does not change when training goes on."""
        ...

    def load_state(self, state: object) -> None:
        """Put back weights from state()."""
        ...

    def predict(self, texts: list[str]) -> list[float]:
        """Return the harmful probability of each text."""
        ...

    def save(self, path: Path) -> None:
        """Write the trained model to a folder."""
        ...

    def report(self) -> dict[str, object]:
        """Return facts about the last fit, for example how many rows were cut to the token limit."""
        ...
