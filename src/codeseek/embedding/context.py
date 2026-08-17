"""Builds the text handed to the embedding model for a chunk -- kept separate
from ChunkPayload.text (what's stored/displayed/keyword-matched) so it can be
enriched without changing what the UI shows.

Motivation, found via a real eval failure: dataclass-style classes (e.g.
typer's `OptionInfo`/`ArgumentInfo`) are almost entirely typed field
declarations with no docstring or prose, so their raw-text embedding carries
little semantic signal -- a conceptual question about them loses to a
prose-heavy README chunk even after doc-overview damping. The identifier
itself carries the missing signal ('ArgumentInfo' IS the answer to "how does
typer represent an argument") but only if it's split into words a
general-purpose embedder can actually match against question phrasing.
"""

import re

from codeseek.store.schema import ChunkPayload

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def split_identifier(name: str) -> str:
    """'ArgumentInfo' -> 'Argument Info', 'parse_csv' -> 'parse csv'."""
    name = name.split("#part")[0]  # oversized-chunk suffix isn't part of the identifier
    name = name.replace("_", " ")
    return _CAMEL_BOUNDARY.sub(" ", name)


def embedding_text(chunk: ChunkPayload) -> str:
    """Text to embed for this chunk. Doc chunks are already prose, so they're
    left as-is; code chunks get a header of symbol type/name/words/path
    prepended ahead of the raw source."""
    if chunk.symbol_type == "doc":
        return chunk.text

    words = split_identifier(chunk.symbol_name)
    header = f"{chunk.symbol_type} {chunk.symbol_name} ({words}) in {chunk.path}"
    return f"{header}\n{chunk.text}"
