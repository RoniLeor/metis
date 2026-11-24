"""Train a guard model: python -m metis.engine.train key=value ...

Keys that are settings fields set the run (for example epochs=2 strategy=domain fold=legal limit=200).
Other keys go to the backend (for laya: loss=rlcd micro_batch=8 grad_accum=4 gradient_checkpointing=true).
"""

from metis.cfg import Settings

from .cli import run_cli
from .results import Run
from .trainer import Trainer


def train(settings: Settings) -> Run:
    """Train one run from the settings."""
    return Trainer(settings=settings).train()


def main() -> None:
    """Train from the key=value pairs of the command line."""
    run_cli(doc=__doc__, action=train)


if __name__ == "__main__":
    main()
