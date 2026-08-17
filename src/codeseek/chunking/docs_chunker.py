"""Paragraph-split, sentence-boundary-aware chunker for README/markdown docs.
~400-token target chunks with 15% overlap, so a fact sitting on a chunk
boundary is never truncated in both directions."""

import re
from dataclasses import dataclass

from codeseek.chunking.shared import pack_units

MAX_CHUNK_TOKENS = 400
OVERLAP_RATIO = 0.15

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_HTML_TAG = re.compile(r"<[^>]+>")
_BADGE_LINE = re.compile(r"^\s*(\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)|!\[[^\]]*\]\([^)]*\))\s*$", re.MULTILINE)


def _strip_badge_and_html_noise(text: str) -> str:
    """READMEs conventionally open with a block of CI/coverage/PyPI badges and
    raw HTML alignment wrappers (<p align="center">, <img>...). That block
    embeds as a short, generic, surprisingly high-scoring chunk that beats
    genuinely relevant code for almost any query -- found by diagnosing why
    a docs chunk kept outranking the actual implementation in eval results.
    Neither badges nor HTML markup carry prose meaning, so both are dropped
    before chunking rather than filtered after the fact."""
    text = _BADGE_LINE.sub("", text)
    text = _HTML_TAG.sub(" ", text)
    return re.sub(r"\n{3,}", "\n\n", text)


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
    text = _strip_badge_and_html_noise(text)
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    packed = pack_units(sentences, MAX_CHUNK_TOKENS, OVERLAP_RATIO, joiner=" ")
    return [
        DocChunk(repo=repo, path=path, chunk_index=i, text=chunk_text)
        for i, (_, _, chunk_text) in enumerate(packed)
    ]
