"""Paragraph-split, sentence-boundary-aware chunker for README/markdown docs.
~400-token target chunks with 15% overlap, so a fact sitting on a chunk
boundary is never truncated in both directions."""

import re
from dataclasses import dataclass

from codeseek.chunking.shared import pack_units

MAX_CHUNK_TOKENS = 400
OVERLAP_RATIO = 0.15

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class DocChunk:
    repo: str
    path: str
    chunk_index: int
    text: str


def split_into_sentences(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences: list[str] = []
    for paragraph in paragraphs:
        sentences.extend(s for s in _SENTENCE_BOUNDARY.split(paragraph) if s)
    return sentences


def chunk_doc_text(text: str, repo: str, path: str) -> list[DocChunk]:
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    packed = pack_units(sentences, MAX_CHUNK_TOKENS, OVERLAP_RATIO, joiner=" ")
    return [
        DocChunk(repo=repo, path=path, chunk_index=i, text=chunk_text)
        for i, (_, _, chunk_text) in enumerate(packed)
    ]
