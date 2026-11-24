"""Registered metrics. Add one with @metric("name"); settings select metrics by name."""

from collections import defaultdict
from collections.abc import Callable

from metis.data.models import Row
from metis.data.split import group_of
from metis.utils import Registry

from .results import Predictions, Score, score

Metric = Callable[[Predictions], float | None]
METRICS: Registry[Metric] = Registry(kind="metric")
metric: Callable[[str], Callable[[Metric], Metric]] = METRICS.register


def recall_by(predicted: list[bool], rows: list[Row], field: str) -> dict[str, float]:
    """Return the recall on the harmful rows for each value of one row field, such as template or domain."""
    found: defaultdict[str, list[bool]] = defaultdict(list)
    guess: bool
    row: Row
    for guess, row in zip(predicted, rows):
        if row.harmful:
            found[str(getattr(row, field))].append(guess)
    return {key: sum(hits) / len(hits) for key, hits in sorted(found.items())}


def counts(predictions: Predictions, kind: str | None) -> Score | None:
    """Score the decisions on all rows, or on the clean rows of one kind. None when there is no such row."""
    pairs: list[tuple[bool, Row]] = [
        (guess, row)
        for guess, row in zip(predictions.guesses(), predictions.rows)
        if kind is None or (row.kind == kind and not row.harmful)
    ]
    if not pairs:
        return None
    return score(predicted=[guess for guess, _ in pairs], rows=[row for _, row in pairs])


def value(predictions: Predictions, kind: str | None, name: str) -> float | None:
    """Return one field of the score, or None when there is no row to score."""
    found: Score | None = counts(predictions=predictions, kind=kind)
    return None if found is None else float(getattr(found, name))


@metric("accuracy")
def accuracy(predictions: Predictions) -> float | None:
    """The share of rows with the right decision."""
    found: Score | None = counts(predictions=predictions, kind=None)
    return None if found is None else (found.tp + found.tn) / len(predictions.rows)


@metric("precision")
def precision(predictions: Predictions) -> float | None:
    """The share of flagged rows that are harmful."""
    return value(predictions=predictions, kind=None, name="precision")


@metric("recall")
def recall(predictions: Predictions) -> float | None:
    """The share of harmful rows that are flagged."""
    return value(predictions=predictions, kind=None, name="recall")


@metric("f1")
def f1(predictions: Predictions) -> float | None:
    """The harmonic mean of precision and recall."""
    return value(predictions=predictions, kind=None, name="f1")


@metric("fpr")
def fpr(predictions: Predictions) -> float | None:
    """The share of clean rows that are flagged."""
    return value(predictions=predictions, kind=None, name="fpr")


@metric("fpr_twin")
def fpr_twin(predictions: Predictions) -> float | None:
    """The share of clean twin rows that are flagged. None when the rows have no twin."""
    return value(predictions=predictions, kind="twin", name="fpr")


@metric("fpr_padded")
def fpr_padded(predictions: Predictions) -> float | None:
    """The share of clean padded rows that are flagged. None when the rows have no padded row."""
    return value(predictions=predictions, kind="padded", name="fpr")


@metric("brier")
def brier(predictions: Predictions) -> float | None:
    """Mean squared distance between the probability and the label (0 is perfect)."""
    if not predictions.rows:
        return None
    pairs: list[tuple[float, bool]] = list(zip(predictions.scores, (row.harmful for row in predictions.rows)))
    return sum((probability - float(label)) ** 2 for probability, label in pairs) / len(pairs)


def recall_by_goal(predictions: Predictions, seen: set[str]) -> dict[str, float | None]:
    """Return the recall on harmful rows whose goal group the train part has, and on the others."""
    found: dict[str, float | None] = {}
    name: str
    for name, inside in (("recall_seen_goal", True), ("recall_unseen_goal", False)):
        hits: list[bool] = [
            guess
            for guess, row in zip(predictions.guesses(), predictions.rows)
            if row.harmful and (group_of(row=row) in seen) == inside
        ]
        found[name] = sum(hits) / len(hits) if hits else None
    return found
