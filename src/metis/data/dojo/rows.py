"""Turn the AgentDojo strings into guard rows.

An injected string is harmful. The same slot in the default environment is its clean twin. Every other string with at
least MIN_WORDS words is a clean field. The template of a harmful row is the AgentDojo attack name.
"""

from metis.data.base import MIN_WORDS
from metis.data.models import Row

from .models import Injected, Leaf, Source

CAP: int = 150  # A slot with a string over this many words is cut, so that the guard's token limit keeps the slot.
CONTEXT: int = 40  # The words of host text that a cut keeps on each side of the slot.


def long_slots(source: Source) -> set[tuple[str, str]]:
    """Return the (suite, vector) of each slot with an injected string over CAP words."""
    return {(item.suite, item.vector) for item in source.injected if len(item.text.split()) > CAP}


def crop(text: str, prefix: int, suffix: int) -> str:
    """Return the slot with at most CONTEXT host words on each side; prefix and suffix are the host word counts."""
    words: list[str] = text.split()
    return " ".join(words[max(0, prefix - CONTEXT) : len(words) - max(0, suffix - CONTEXT)])


def harmful_row(item: Injected, cut: bool) -> Row:
    """Return the harmful row of one injected string, cut when its slot is long. The group joins one attacker goal."""
    text: str = crop(text=item.text, prefix=item.prefix, suffix=item.suffix) if cut else item.text
    return Row(
        task_id=0,
        domain=item.suite,
        template=item.attack,
        goal=item.goal,
        path=item.path,
        text=text,
        words=len(text.split()),
        harmful=True,
        kind="field",
        group=f"{item.suite}/{item.injection_task}",
    )


def clean_row(leaf: Leaf, cut: bool) -> Row:
    """Return the clean row of one default string: a twin when it holds a slot, else a field. Cut it when asked."""
    text: str = crop(text=leaf.text, prefix=leaf.prefix, suffix=leaf.suffix) if cut else leaf.text
    return Row(
        task_id=0,
        domain=leaf.suite,
        template="",
        goal="",
        path=leaf.path,
        text=text,
        words=len(text.split()),
        harmful=False,
        kind="twin" if leaf.vector else "field",
        group=leaf.vector,
    )


def unique(rows: list[Row]) -> list[Row]:
    """Keep the first row of each text, in order."""
    first: dict[str, Row] = {}
    for row in rows:
        first.setdefault(row.text, row)
    return list(first.values())


def build(source: Source) -> list[Row]:
    """Return the harmful rows, then the clean rows, one row per text. Twins come before fields of the same text.

    Every string of a long slot is cut the same way: each attack and the twin keep the same host words.
    Raise an error when one text is both harmful and clean.
    """
    long: set[tuple[str, str]] = long_slots(source=source)
    harmful: list[Row] = unique(
        rows=[harmful_row(item=item, cut=(item.suite, item.vector) in long) for item in source.injected]
    )
    leaves: list[Leaf] = sorted(source.leaves, key=lambda leaf: not leaf.vector)
    every: list[Row] = [clean_row(leaf=leaf, cut=(leaf.suite, leaf.vector) in long) for leaf in leaves]
    clean: list[Row] = unique(rows=[row for row in every if row.words >= MIN_WORDS])
    both: set[str] = {row.text for row in harmful} & {row.text for row in clean}
    if both:
        raise ValueError(f"{len(both)} texts are both harmful and clean, for example {sorted(both)[0][:80]!r}")
    return harmful + clean
