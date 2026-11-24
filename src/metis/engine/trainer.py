"""The trainer: load a split, fit a backend, measure it on the test rows, and write the outputs."""

from pathlib import Path

from metis.cfg import Settings
from metis.data.masking import with_masked
from metis.data.models import Row
from metis.data.split import Split, group_of, load_split, sample
from metis.models import load_backend
from metis.models.base import Backend

from .callbacks import Callback, Callbacks, print_epoch, print_validation, write_progress
from .metrics import recall_by_goal
from .outputs import check_names, next_dir, write_outputs
from .results import Run
from .validator import BestEpoch, validate


class Trainer(Callbacks):
    """Run one training job from settings. Callbacks run at the events in EVENTS."""

    def __init__(self, settings: Settings) -> None:
        settings.validate()
        check_names(settings=settings)
        self.settings: Settings = settings
        super().__init__()
        self.add_callback(event="on_epoch_end", callback=print_epoch)
        self.add_callback(event="on_val_end", callback=print_validation)
        self.add_callback(event="on_val_end", callback=write_progress)

    def start_validation(self, validator: BestEpoch, save_dir: Path) -> list[Callback]:
        """Validate epoch 0 and add the epoch hooks that validate and report each epoch. Return the added hooks."""

        def emit(**_: object) -> None:
            self.run_callbacks(
                event="on_val_end",
                entry=validator.history[-1],
                history=validator.history,
                best_epoch=validator.best_epoch,
                save_dir=save_dir,
                select=self.settings.select,
            )

        if not validator.rows:
            print("no validation rows: fewer than 10 goal groups or val_share=0; the last epoch is kept", flush=True)
            return []
        validator.check(epoch=0, loss=None)
        emit()
        hooks: list[Callback] = [validator.on_epoch_end, emit]
        for hook in hooks:
            self.add_callback(event="on_epoch_end", callback=hook)
        return hooks

    def split(self) -> Split:
        """Return the split of the settings, with the masked copies and the row limit on the train and test parts."""
        settings: Settings = self.settings
        split: Split = load_split(
            folder=settings.data,
            strategy=settings.strategy,
            fold=settings.fold,
            val_share=settings.val_share,
            seed=settings.seed,
            augment=settings.augment,
        )
        train: list[Row] = with_masked(rows=split.train, augment=settings.augment)
        return Split(
            train=sample(rows=train, limit=settings.limit, seed=settings.seed),
            val=split.val,
            test=sample(rows=split.test, limit=settings.limit, seed=settings.seed),
        )

    def fit(self, backend: Backend, rows: list[Row], validator: BestEpoch, save_dir: Path) -> list[float]:
        """Fit the backend with a validation after each epoch, keep the best epoch, and return the losses."""
        hooks: list[Callback] = self.start_validation(validator=validator, save_dir=save_dir)

        def on_epoch_end(epoch: int, loss: float) -> None:
            self.run_callbacks(event="on_epoch_end", epoch=epoch, loss=loss)

        try:
            return backend.fit(
                rows=rows, epochs=self.settings.epochs, seed=self.settings.seed, on_epoch_end=on_epoch_end
            )
        finally:
            for hook in hooks:
                self.callbacks["on_epoch_end"].remove(hook)

    def train(self) -> Run:
        """Fit the backend on the train rows, keep the best validated epoch, then measure it on the test rows."""
        settings: Settings = self.settings
        split: Split = self.split()
        backend: Backend = load_backend(name=settings.backend)(settings)
        save_dir: Path = next_dir(project=settings.project, name=settings.name)
        save_dir.mkdir(parents=True)
        self.run_callbacks(event="on_train_start", save_dir=save_dir)
        validator: BestEpoch = BestEpoch(backend=backend, rows=split.val, settings=settings)
        losses: list[float] = self.fit(backend=backend, rows=split.train, validator=validator, save_dir=save_dir)
        validator.restore()
        backend.finish()
        predictions, metrics = validate(
            backend=backend, rows=split.test, threshold=settings.threshold, metrics=settings.metrics
        )
        metrics.update(recall_by_goal(predictions=predictions, seen={group_of(row=row) for row in split.train}))
        run: Run = Run(
            save_dir=save_dir,
            settings=settings,
            backend=backend,
            losses=losses,
            predictions=predictions,
            metrics=metrics,
            report=backend.report(),
            history=validator.history,
            best_epoch=validator.best_epoch,
        )
        write_outputs(run=run)
        self.run_callbacks(event="on_train_end", run=run)
        return run
