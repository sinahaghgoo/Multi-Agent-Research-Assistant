"""Text normalisation and chunking shared by the local corpus and fetched web pages."""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"[ \t ]+")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")


def normalize(text: str) -> str:
    """NFKC-normalise (fixes ligatures such as 'ﬁ'), join hyphenated line breaks, squeeze spaces."""
    text = unicodedata.normalize("NFKC", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    lines = [_WS.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def word_count(text: str) -> int:
    return len(text.split())


def _split_long(line: str, max_words: int) -> list[str]:
    """Split an over-long line at sentence boundaries, falling back to hard word splits."""
    if word_count(line) <= max_words:
        return [line]
    pieces: list[str] = []
    current: list[str] = []
    for sentence in _SENTENCE_END.split(line):
        words = sentence.split()
        while len(words) > max_words:
            if current:
                pieces.append(" ".join(current))
                current = []
            pieces.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if len(current) + len(words) > max_words and current:
            pieces.append(" ".join(current))
            current = []
        current.extend(words)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_lines(lines: list[str], max_words: int = 220, min_words: int = 40) -> list[str]:
    """Greedily pack lines into chunks of at most ``max_words`` words.

    Lines are kept intact where possible so that list items such as "(a) ..." stay
    readable. A short trailing chunk is merged into the previous one.
    """
    units: list[str] = []
    for line in lines:
        units.extend(_split_long(line, max_words))

    chunks: list[list[str]] = []
    current: list[str] = []
    size = 0
    for unit in units:
        n = word_count(unit)
        if current and size + n > max_words:
            chunks.append(current)
            current, size = [], 0
        current.append(unit)
        size += n
    if current:
        if chunks and size < min_words and word_count(" ".join(chunks[-1])) + size <= max_words * 1.3:
            chunks[-1].extend(current)
        else:
            chunks.append(current)
    return ["\n".join(c) for c in chunks]


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + " …"
