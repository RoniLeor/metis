"""The model facade, in the Ultralytics style: Model("laya").train(...) and Model("laya").predict(...)."""

from dataclasses import replace
from pathlib import Path

from metis.cfg import Settings
from metis.models import load_backend
from metis.models.base import Backend

from .results import Run
from .trainer import Trainer


class Model:
    """One backend and its base weights. Training replaces the backend with the trained one."""

    def __init__(self, backend: str = "laya", base: str = "convaiinnovations/laya", device: str = "auto") -> None:
        self.settings: Settings = Settings(backend=backend, base=base, device=device)
        self.backend: Backend | None = None

    def train(
        self,
        data: Path = Path("resources/data/nvidia/rows"),
        strategy: str = "template",
        fold: str = "other templates",
        epochs: int = 1,
        limit: int = 0,
        name: str = "exp",
        options: dict[str, object] | None = None,
    ) -> Run:
        """Train on one fold of the guard data and return the run."""
        settings: Settings = replace(
            self.settings,
            data=data,
            strategy=strategy,
            fold=fold,
            epochs=epochs,
            limit=limit,
            name=name,
            options=dict(options or {}),
        )
        run: Run = Trainer(settings=settings).train()
        self.backend = run.backend
        return run

    def predict(self, texts: list[str]) -> list[float]:
        """Return the harmful probability of each text, with the trained backend or the base weights."""
        if self.backend is None:
            self.backend = load_backend(name=self.settings.backend)(self.settings)
        return self.backend.predict(texts=texts)
