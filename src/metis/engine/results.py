"""The predictions of a backend on test rows, their score, and the record of one finished run."""

from dataclasses import dataclass
from pathlib import Path

from metis.cfg import Settings
from metis.data.models import Row
from metis.models.base import Backend


@dataclass(frozen=True)
class Score:
    """The counts and rates of the decisions on one set of rows."""

    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    fpr: float


def score(predicted: list[bool], rows: list[Row]) -> Score:
    """Count the hits and the errors of the decisions."""
    pairs: list[tuple[bool, bool]] = list(zip(predicted, (row.harmful for row in rows)))
    tp: int = sum(1 for guess, truth in pairs if guess and truth)
    fp: int = sum(1 for guess, truth in pairs if guess and not truth)
    fn: int = sum(1 for guess, truth in pairs if not guess and truth)
    tn: int = len(pairs) - tp - fp - fn
    precision: float = tp / (tp + fp) if tp + fp else 0.0
    recall: float = tp / (tp + fn) if tp + fn else 0.0
    f1: float = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr: float = fp / (fp + tn) if fp + tn else 0.0
    return Score(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1, fpr=fpr)


@dataclass(frozen=True)
class Predictions:
    """A harmful probability for each row. A row at or above the threshold counts as harmful."""

    rows: list[Row]
    scores: list[float]
    threshold: float

    def guesses(self) -> list[bool]:
        """Return the harmful or clean decision of each row."""
        return [score >= self.threshold for score in self.scores]


@dataclass(frozen=True)
class Run:
    """Everything that the outputs write: the settings, the losses, the metrics and the trained backend.

    history holds one entry per validated epoch (epoch 0 is the backend before training); best_epoch is the epoch
    whose weights the backend keeps, or None when there was no validation.
    """

    save_dir: Path
    settings: Settings
    backend: Backend
    losses: list[float]
    predictions: Predictions
    metrics: dict[str, float | None]
    report: dict[str, object]
    history: list[dict[str, object]]
    best_epoch: int | None
