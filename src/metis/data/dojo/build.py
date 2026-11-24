"""Write the AgentDojo guard rows: python -m metis.data.dojo.build <source.json> <output folder>

The source file comes from metis/data/dojo/extract/source.py (make dojo-source). The output folder gets rows.jsonl,
which the engine reads, and summary.json with the counts.
"""

import argparse
from collections import Counter
from pathlib import Path

from metis.data.base import MIN_WORDS, Builder
from metis.data.models import Row

from .models import Source, read_source
from .rows import build, long_slots


def summary(source: Source, rows: list[Row]) -> dict[str, object]:
    """Return the counts of the source and the rows: per suite, per attack, the slots, and the cut slots.

    short_twin_slots counts the slots whose clean twin has fewer than MIN_WORDS words, so it has no row.
    """
    slots: set[tuple[str, str]] = {(item.suite, item.vector) for item in source.injected}
    whole: set[tuple[str, str]] = {
        (item.suite, item.vector) for item in source.injected if not item.prefix + item.suffix
    }
    short: set[tuple[str, str]] = {
        (leaf.suite, leaf.vector) for leaf in source.leaves if leaf.vector and len(leaf.text.split()) < MIN_WORDS
    }
    return {
        "agentdojo": source.agentdojo,
        "scope": f"default environment strings with at least {MIN_WORDS} words; one row per text",
        "combinations": len(source.injected),
        "slots": dict(sorted(Counter(suite for suite, _ in slots).items())),
        "whole_string_slots": len(whole),
        "cut_slots": len(long_slots(source=source)),
        "rows": dict(sorted(Counter(f"{row.domain} {row.kind} {row.harmful}" for row in rows).items())),
        "harmful_by_attack": dict(sorted(Counter(row.template for row in rows if row.harmful).items())),
        "short_twin_slots": len(short),
    }


class DojoBuilder(Builder):
    """The AgentDojo guard rows, with their counts."""

    def __init__(self, source: Source) -> None:
        self.source: Source = source

    def rows(self) -> list[Row]:
        """Return one row per AgentDojo string."""
        return build(source=self.source)

    def reports(self, rows: list[Row]) -> dict[str, object]:
        """Return summary.json."""
        return {"summary.json": summary(source=self.source, rows=rows)}


def main() -> None:
    """Read the arguments, write the files and print the row count."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="JSON file from metis/data/dojo/extract/source.py")
    parser.add_argument("output", type=Path, help="Output folder")
    args: argparse.Namespace = parser.parse_args()
    try:
        rows: list[Row] = DojoBuilder(source=read_source(path=args.source)).write(folder=args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"wrote {len(rows)} rows ({sum(row.harmful for row in rows)} harmful) to {args.output}")


if __name__ == "__main__":
    main()
