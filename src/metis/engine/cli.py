"""The key=value command line that the train and evaluate commands share."""

import argparse
import json
from collections.abc import Callable

from metis.cfg import Settings, from_pairs

from .results import Run


def run_cli(doc: str | None, action: Callable[[Settings], Run]) -> None:
    """Read key=value pairs, run the action with their settings, and print the run folder and the metrics."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("pairs", nargs="*", help="key=value settings")
    args: argparse.Namespace = parser.parse_args()
    try:
        run: Run = action(from_pairs(pairs=list(args.pairs)))
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"saved to {run.save_dir}")
    print(json.dumps(run.metrics, indent=2))
