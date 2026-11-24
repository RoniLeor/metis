"""Validation and the choice of the best epoch.

The trainer scores the validation rows before training (epoch 0) and after each epoch, and keeps a copy of the weights
with the best value of the selected metric. Epoch 0 can win: then the trained backend goes back to its base weights.
"""

from metis.cfg import Settings
from metis.data.models import Row
from metis.models.base import Backend

from .metrics import METRICS
from .results import Predictions


def validate(
    backend: Backend, rows: list[Row], threshold: float, metrics: list[str]
) -> tuple[Predictions, dict[str, float | None]]:
    """Score the rows with the backend and compute each named metric. No rows give None for each metric."""
    scores: list[float] = backend.predict(texts=[row.text for row in rows]) if rows else []
    predictions: Predictions = Predictions(rows=rows, scores=scores, threshold=threshold)
    return predictions, {name: METRICS.get(name)(predictions) for name in metrics}


class BestEpoch:
    """Validate after each epoch and keep the weights of the best one."""

    def __init__(self, backend: Backend, rows: list[Row], settings: Settings) -> None:
        self.backend: Backend = backend
        self.rows: list[Row] = rows
        self.settings: Settings = settings
        self.history: list[dict[str, object]] = []
        self.best_epoch: int | None = None
        self.best_value: float | None = None
        self.best_state: object = None

    def check(self, epoch: int, loss: float | None) -> None:
        """Score the validation rows for one epoch, and copy the weights when the selected metric improves."""
        names: list[str] = [*self.settings.metrics, self.settings.select]
        _predictions, metrics = validate(
            backend=self.backend, rows=self.rows, threshold=self.settings.threshold, metrics=names
        )
        value: float | None = metrics[self.settings.select]
        harmful: int = sum(1 for row in self.rows if row.harmful)
        self.history.append(
            {"epoch": epoch, "loss": loss, "rows": len(self.rows), "harmful": harmful, "metrics": metrics}
        )
        if value is not None and (self.best_value is None or value > self.best_value):
            self.best_state = None  # Free the old copy before the backend makes the new one.
            self.best_state = self.backend.state()
            self.best_value = value
            self.best_epoch = epoch

    def on_epoch_end(self, epoch: int, loss: float, **_: object) -> None:
        """Callback for the trainer: the backend counts epochs from 0, the history from 1."""
        self.check(epoch=epoch + 1, loss=loss)

    def restore(self) -> None:
        """Put back the weights of the best epoch."""
        if self.best_state is not None:
            self.backend.load_state(state=self.best_state)
