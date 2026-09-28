"""Split Vietnamese prose while retaining the original sentence text."""

from __future__ import annotations

import re


_BOUNDARY = re.compile(r"(?<=[.!?…])(?:[\"'”’)]*)\s+|\n+")


def split_sentences(text: str) -> list[str]:
    """Return nonempty sentences in source order without rewriting their words."""
    if not isinstance(text, str):
        raise TypeError("text phải là chuỗi")
    return [part.strip() for part in _BOUNDARY.split(text.strip()) if part.strip()]
