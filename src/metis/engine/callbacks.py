"""The events of a training run, the callbacks that run at each event, and the default progress callbacks."""

from collections.abc import Callable
from pathlib import Path

from metis.utils import write_json

EVENTS: tuple[str, ...] = ("on_train_start", "on_epoch_end", "on_val_end", "on_train_end")
Callback = Callable[..., None]


class Callbacks:
    """The callbacks of each event in EVENTS. The trainer extends this class and is the trainer value of each call."""

    def __init__(self) -> None:
        self.callbacks: dict[str, list[Callback]] = {event: [] for event in EVENTS}

    def add_callback(self, event: str, callback: Callback) -> None:
        """Run the callback at the event."""
        if event not in self.callbacks:
            raise KeyError(f"unknown event {event!r}; known: {', '.join(EVENTS)}")
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str, **values: object) -> None:
        """Run every callback of the event with the trainer and the values."""
        callback: Callback
        for callback in self.callbacks[event]:
            callback(trainer=self, **values)


def number(value: object) -> str:
    """Return a metric value with three decimals, or n/a when there is none."""
    return f"{value:.3f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "n/a"


def print_epoch(epoch: int, loss: float, **_: object) -> None:
    """Print the mean loss of an epoch."""
    print(f"epoch {epoch + 1}: mean loss {loss:.4f}", flush=True)


def print_validation(entry: dict[str, object], best_epoch: int | None, select: str = "f1", **_: object) -> None:
    """Print the validation result of an epoch, with the selection metric, and the best epoch so far."""
    found: object = entry.get("metrics")
    metrics: dict[str, object] = found if isinstance(found, dict) else {}
    names: list[str] = list(dict.fromkeys(("f1", "recall", "precision", "fpr", select)))
    values: str = " ".join(f"{name} {number(value=metrics.get(name))}" for name in names)
    print(f"epoch {entry.get('epoch')}: val {values} (best epoch {best_epoch})", flush=True)


def write_progress(save_dir: Path, history: list[dict[str, object]], best_epoch: int | None, **_: object) -> None:
    """Write metrics.json with the validation so far; the test metrics stay null until the run ends."""
    write_json(path=save_dir / "metrics.json", value={"metrics": None, "best_epoch": best_epoch, "history": history})
