"""The laya backend: fine-tune laya on its own prompt-injection question.

loss=rlcd is laya's policy gradient on the labels, not RL from an agent reward; held-out groups refit temperatures.
"""

from pathlib import Path
from typing import Any

from laya import fit_temperature_map, guard_questions
from laya import train as laya_train
from laya.common import uses_parallel_layout

from metis.cfg import Settings
from metis.data.models import Row
from metis.models import backend
from metis.models.base import EpochHook

from .spec import LIMITS, QUESTION_ID, LayaOptions, calibrated, case, read_options, true_probability
from .tuning import hold_out, temperature_of, train_config, truncated


@backend("laya")
class LayaBackend:
    """A laya checkpoint with the prompt-injection question."""

    def __init__(self, settings: Settings) -> None:
        self.options: LayaOptions = read_options(options=settings.options)
        self.device: Any = laya_train.resolve_device(device=settings.device)
        loaded: tuple[Any, Any, dict[str, Any]] = laya_train.load_checkpoint(model_dir=settings.base)
        self.model: Any = loaded[0].to(self.device)
        self.tok: Any = loaded[1]
        cfg: dict[str, Any] = loaded[2]
        self.limits: dict[str, int] = {name: int(cfg.get(name, size)) for name, size in LIMITS.items()}
        self.cfg: dict[str, object] = dict(cfg, **self.limits)
        self.parallel: bool = uses_parallel_layout(cfg=cfg)
        self.question: dict[str, object] = dict(guard_questions()[QUESTION_ID])
        self.temperature: float = temperature_of(cfg=self.cfg)
        self.facts: dict[str, object] = {}
        self.calibration: list[dict[str, object]] = []

    def items(self, texts: list[str], labels: list[bool]) -> list[dict[str, object]]:
        """Return one laya item for each text, or raise an error, so that scores stay aligned with rows."""
        rows: list[dict[str, object]] = [
            case(text=text, question=self.question, label=flag) for text, flag in zip(texts, labels)
        ]
        items, skipped = laya_train.items_from_rows(tok=self.tok, rows=rows, **self.limits)
        if skipped or len(items) != len(texts):
            raise ValueError(f"laya could not encode {len(texts) - len(items)} rows: {skipped}")
        return items

    def shared(self) -> dict[str, Any]:
        """Return the arguments that laya's training and scoring functions share."""
        return {"model": self.model, "tok": self.tok, "device": self.device, "parallel": self.parallel, **self.limits}

    def rows_items(self, rows: list[Row]) -> list[dict[str, object]]:
        """Return the labeled laya items of guard rows."""
        return self.items(texts=[row.text for row in rows], labels=[row.harmful for row in rows])

    def records(self, items: list[dict[str, object]]) -> list[tuple[int, Any, Any, int]]:
        """Return laya's (type, logits, target, options) record of each item, in item order."""
        return laya_train.calibration_records(items=items, **self.shared())

    def fit(self, rows: list[Row], epochs: int, seed: int, on_epoch_end: EpochHook) -> list[float]:
        """Train on the rows. hold_out() keeps whole goal groups back, for the calibration in finish()."""
        texts: list[str] = [row.text for row in rows]
        cut: int = truncated(
            tok=self.tok, question=self.question, texts=texts, max_share=self.options.max_truncated, **self.limits
        )
        fitting, held = hold_out(rows=rows, seed=seed)
        train: list[dict[str, object]] = self.rows_items(rows=fitting)
        self.calibration = self.rows_items(rows=held)
        share: float = sum(1 for row in fitting if row.harmful) / max(1, len(fitting))
        self.facts = {
            "train_items": len(train),
            "calibration_items": len(self.calibration),
            "truncated_rows": cut,
            "train_harmful_share": share,
        }
        config: laya_train.TrainConfig = train_config(options=self.options, epochs=epochs, seed=seed)
        return laya_train.train_model(items=train, config=config, on_epoch_end=on_epoch_end, **self.shared())

    def finish(self) -> None:
        """Fit new temperatures on the held-out rows, so the checkpoint does not keep the base calibration."""
        records: list[tuple[int, Any, Any, int]] = self.records(items=self.calibration)
        fitted: dict[str, Any] = fit_temperature_map(records=records)
        self.cfg = calibrated(cfg=self.cfg, fitted=fitted)
        self.temperature = temperature_of(cfg=self.cfg)
        report: object = laya_train.calibration_report(records=records, temperature=fitted["temperature"])
        self.facts.update(temperature=self.temperature, calibration=report)

    def state(self) -> object:
        """Return a CPU copy of the weights; copy=True also copies on a CPU device, where .cpu() would share memory."""
        return {key: value.detach().to("cpu", copy=True) for key, value in self.model.state_dict().items()}

    def load_state(self, state: object) -> None:
        """Put back weights from state()."""
        self.model.load_state_dict(state)

    def predict(self, texts: list[str]) -> list[float]:
        """Return P(true) of the injection question for each text, and put back the mode of each module."""
        modes: list[tuple[Any, bool]] = [(module, module.training) for module in self.model.modules()]
        records: list[tuple[int, Any, Any, int]] = self.records(
            items=self.items(texts=texts, labels=[False] * len(texts))
        )
        for module, training in modes:
            module.train(training)
        return [
            true_probability(logits=[float(v) for v in record[1]], temperature=self.temperature) for record in records
        ]

    def save(self, path: Path) -> None:
        """Write a checkpoint that laya.load reads, with the fitted temperatures."""
        laya_train.save_checkpoint(model=self.model, tok=self.tok, cfg=self.cfg, path=str(path))

    def report(self) -> dict[str, object]:
        """Return the facts of the last fit."""
        return dict(self.facts)
