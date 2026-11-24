"""The files of a run. Add one with @output("name"); the outputs setting selects them by name."""

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from metis.cfg import Settings
from metis.utils import Registry, write_json, write_lines

from .metrics import METRICS
from .results import Run

Output = Callable[[Run], None]
OUTPUTS: Registry[Output] = Registry(kind="output")
output: Callable[[str], Callable[[Output], Output]] = OUTPUTS.register


def check_names(settings: Settings) -> None:
    """Raise an error for an unknown metric or output name, before any work starts."""
    for name in [*settings.metrics, settings.select]:
        METRICS.get(name)
    for name in settings.outputs:
        OUTPUTS.get(name)


def next_dir(project: Path, name: str) -> Path:
    """Return the first free run folder: name, then name2, name3, and so on."""
    folder: Path = project / name
    number: int = 1
    while folder.exists():
        number += 1
        folder = project / f"{name}{number}"
    return folder


@output("settings")
def write_settings(run: Run) -> None:
    """Write the settings of the run, so the run can be repeated."""
    write_json(path=run.save_dir / "settings.json", value=asdict(run.settings))


@output("metrics")
def write_metrics(run: Run) -> None:
    """Write the test metrics, the loss and validation of each epoch, the chosen epoch and the backend report."""
    value: dict[str, object] = {"metrics": run.metrics, "losses": run.losses, "best_epoch": run.best_epoch}
    value.update(history=run.history, report=run.report)
    write_json(path=run.save_dir / "metrics.json", value=value)


@output("predictions")
def write_predictions(run: Run) -> None:
    """Write the probability and the label of each test row."""
    records: list[dict[str, object]] = [
        {"task_id": row.task_id, "path": row.path, "kind": row.kind, "harmful": row.harmful, "score": score}
        for row, score in zip(run.predictions.rows, run.predictions.scores)
    ]
    write_lines(path=run.save_dir / "predictions.jsonl", records=records)


@output("checkpoint")
def write_checkpoint(run: Run) -> None:
    """Write the trained model into the weights folder of the run."""
    run.backend.save(path=run.save_dir / "weights")


def write_outputs(run: Run) -> None:
    """Write each output that the settings of the run name."""
    name: str
    for name in run.settings.outputs:
        OUTPUTS.get(name)(run)
