"""Export the AgentDojo ground truth as an offline HTML page."""

import argparse
import json
from pathlib import Path

from metis.utils import render, write_text

from .validate import validate_labels

TEMPLATE: Path = Path(__file__).with_name("viewer.html")


def export(data: Path, output: Path) -> None:
    """Embed the ground truth as inert data and allow only the viewer script."""
    truth_path: Path = data / "ground-truth.json"
    truth: object = json.loads(truth_path.read_text(encoding="utf-8"))
    validate_labels(truth=truth, path=truth_path)
    template: str = TEMPLATE.read_text(encoding="utf-8")
    write_text(path=output, text=render(template=template, payload=truth))


def main() -> None:
    """Read the input directory and output file from the command line."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Directory with ground-truth.json")
    parser.add_argument("output", type=Path, help="Output HTML file")
    args: argparse.Namespace = parser.parse_args()
    try:
        export(data=args.data, output=args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
