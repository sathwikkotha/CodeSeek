"""Token counting and the greedy unit-packing algorithm shared by the code and
docs chunkers: pack atomic units (lines, or sentences) into token-capped
chunks, optionally overlapping, and never split a unit itself unless that one
unit alone exceeds the cap (graceful degradation, last resort only).
"""

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _hard_split_by_words(text: str, max_tokens: int) -> list[str]:
    """Last-resort split for a single unit that alone exceeds max_tokens.
    Cuts on whitespace boundaries only -- never mid-word/mid-token."""
    words = text.split(" ")
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = current + [word]
        if current and count_tokens(" ".join(candidate)) > max_tokens:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current = candidate
    if current:
        pieces.append(" ".join(current))
    return pieces


def pack_units(
    units: list[str], max_tokens: int, overlap_ratio: float, joiner: str
) -> list[tuple[int, int, str]]:
    """Greedily pack `units` into chunks capped at max_tokens.

    Returns (start_index, end_index, text) tuples with indices inclusive into
    the original `units` list, so callers can map a chunk back to a source
    position (e.g. line numbers for code). `overlap_ratio` carries that
    fraction of the previous chunk's units into the next chunk's start.
    """
    results: list[tuple[int, int, str]] = []
    current: list[int] = []

    def flush() -> None:
        nonlocal current
        if current:
            text = joiner.join(units[i] for i in current)
            results.append((current[0], current[-1], text))
        overlap_count = max(1, round(len(current) * overlap_ratio)) if current and overlap_ratio > 0 else 0
        current = current[-overlap_count:] if overlap_count else []

    for idx, unit in enumerate(units):
        if count_tokens(unit) > max_tokens:
            flush()
            results.extend((idx, idx, piece) for piece in _hard_split_by_words(unit, max_tokens))
            continue

        prospective = current + [idx]
        if current and count_tokens(joiner.join(units[i] for i in prospective)) > max_tokens:
            flush()
        current.append(idx)

    flush()
    return results
