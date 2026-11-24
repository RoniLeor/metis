"""Score a saved guard model on every row of a rows file: python -m metis.engine.evaluate key=value ...

Set base to the weights folder (for example base=resources/model/weights), data to the folder that holds
rows.jsonl, and name to the run name. Every row is a test row. The run writes settings.json, metrics.json and
predictions.jsonl, but no checkpoint.
"""

from dataclasses import replace
from pathlib import Path

from metis.cfg import Settings
from metis.data.base import read_rows
from metis.data.models import Row
from metis.models import load_backend
from metis.models.base import Backend

from .cli import run_cli
from .metrics import recall_by
from .outputs import check_names, next_dir, write_outputs
from .results import Run
from .validator import validate


def evaluate(settings: Settings) -> Run:
    """Score every row with the backend, save the run without its weights, and return it."""
    check_names(settings=settings)
    rows: list[Row] = read_rows(path=settings.data / "rows.jsonl")
    backend: Backend = load_backend(name=settings.backend)(settings)
    predictions, metrics = validate(backend=backend, rows=rows, threshold=settings.threshold, metrics=settings.metrics)
    guesses: list[bool] = predictions.guesses()
    report: dict[str, object] = {
        **backend.report(),
        "recall_by_template": recall_by(predicted=guesses, rows=rows, field="template"),
        "recall_by_domain": recall_by(predicted=guesses, rows=rows, field="domain"),
    }
    save_dir: Path = next_dir(project=settings.project, name=settings.name)
    save_dir.mkdir(parents=True)
    kept: Settings = replace(settings, outputs=[name for name in settings.outputs if name != "checkpoint"])
    run: Run = Run(
        save_dir=save_dir,
        settings=kept,
        backend=backend,
        losses=[],
        predictions=predictions,
        metrics=metrics,
        report=report,
        history=[],
        best_epoch=None,
    )
    write_outputs(run=run)
    return run


def main() -> None:
    """Score the rows from the key=value pairs of the command line."""
    run_cli(doc=__doc__, action=evaluate)


if __name__ == "__main__":
    main()
