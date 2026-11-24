"""The masked augmentation: train copies of the rows with the top style words removed.

The words are ranked on the train part only, so no test word leaks. Clean rows get copies too, so that text with gaps
does not become a sign of an attack.
"""

import math
import re
from collections import Counter
from dataclasses import replace

from .models import MASK_SOURCE_KINDS, Row

PIECE: re.Pattern[str] = re.compile(r"[A-Za-z_]+|[^\sA-Za-z0-9]")  # A word or one punctuation mark.
SPACE: re.Pattern[str] = re.compile(r"\s+")
TOP: int = 50
MIN_COUNT: int = 20


def style_words(rows: list[Row], top: int) -> list[str]:
    """Return the top words by smoothed log odds of harmful against clean rows, from at least MIN_COUNT harmful rows."""
    harmful: Counter[str] = Counter()
    clean: Counter[str] = Counter()
    for row in rows:
        (harmful if row.harmful else clean).update({token.lower() for token in PIECE.findall(row.text)})
    total_harmful: int = sum(1 for row in rows if row.harmful)
    total_clean: int = len(rows) - total_harmful
    score: dict[str, float] = {
        word: math.log((count + 1) / (total_harmful + 2)) - math.log((clean[word] + 1) / (total_clean + 2))
        for word, count in harmful.items()
        if count >= MIN_COUNT
    }
    return sorted(score, key=lambda word: (-score[word], word))[:top]


def strip(text: str, words: set[str]) -> str:
    """Remove the tokens that are in words (any case), and collapse the white space."""
    kept: str = PIECE.sub(lambda match: "" if match.group(0).lower() in words else match.group(0), text)
    return SPACE.sub(" ", kept).strip()


def masked_rows(rows: list[Row], words: set[str]) -> list[Row]:
    """Return a masked copy of each row whose text changes.

    A harmful copy whose text equals a clean text (masked or not) is dropped, so that one text never has two labels.
    """
    copies: list[Row] = [
        replace(row, text=text, words=len(text.split()), kind="masked")
        for row in rows
        if (text := strip(text=row.text, words=words)) != row.text
    ]
    clean: set[str] = {row.text for row in rows + copies if not row.harmful}
    return [row for row in copies if not (row.harmful and row.text in clean)]


def with_masked(rows: list[Row], augment: list[str]) -> list[Row]:
    """Add masked copies of the field, twin and padded rows when augment names "masked"; print the removed words."""
    if "masked" not in augment:
        return rows
    sources: list[Row] = [row for row in rows if row.kind in MASK_SOURCE_KINDS]
    words: list[str] = style_words(rows=sources, top=TOP)
    print(f"masked: removed {len(words)} style words: {' '.join(words)}", flush=True)
    return rows + masked_rows(rows=sources, words=set(words))
