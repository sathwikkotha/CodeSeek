"""Wires ingestion -> chunking -> embedding -> storage into one indexing pass
per repo: walk its files, chunk each one with the chunker matching its
category, embed every chunk with every configured model, upsert into each
model's collection."""

import logging
from pathlib import Path

from codeseek.chunking.code_chunker import chunk_python_source
from codeseek.chunking.docs_chunker import chunk_doc_text
from codeseek.chunking.generic_chunker import chunk_generic_source
from codeseek.chunking.js_ts_chunker import chunk_js_ts_source
from codeseek.chunking.notebook_chunker import chunk_notebook_source
from codeseek.config import GENERIC_LANGUAGE_BY_EXTENSION, RepoSpec
from codeseek.embedding.context import embedding_text
from codeseek.embedding.service import EmbeddingService
from codeseek.ingestion.walker import FileRecord, walk_repo
from codeseek.observability.timing import timed_stage
from codeseek.store.qdrant_store import QdrantStore
from codeseek.store.schema import ChunkPayload, collection_name

logger = logging.getLogger(__name__)


def _generic_language(full_path: Path) -> str:
    if full_path.suffix:
        return GENERIC_LANGUAGE_BY_EXTENSION.get(full_path.suffix, full_path.suffix.lstrip("."))
    return full_path.name.lower()  # extensionless: Dockerfile, Makefile, ...


def _file_to_chunks(repo_path: Path, record: FileRecord, repo: RepoSpec) -> list[ChunkPayload]:
    full_path = repo_path / record.path
    text = full_path.read_text(encoding="utf-8", errors="ignore")

    if record.category == "code" and full_path.suffix == ".py":
        return [
            ChunkPayload(
                repo=c.repo, language=c.language, path=c.path, symbol_name=c.symbol_name,
                symbol_type=c.symbol_type, start_line=c.start_line, end_line=c.end_line, text=c.text,
            )
            for c in chunk_python_source(text, repo=repo.name, path=record.path)
        ]

    if record.category == "code" and full_path.suffix in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
        return [
            ChunkPayload(
                repo=c.repo, language=c.language, path=c.path, symbol_name=c.symbol_name,
                symbol_type=c.symbol_type, start_line=c.start_line, end_line=c.end_line, text=c.text,
            )
            for c in chunk_js_ts_source(text, repo=repo.name, path=record.path, suffix=full_path.suffix)
        ]

    if record.category == "notebook":
        return [
            ChunkPayload(
                repo=c.repo, language=c.language, path=c.path, symbol_name=c.symbol_name,
                symbol_type=c.symbol_type, start_line=c.start_line, end_line=c.end_line, text=c.text,
            )
            for c in chunk_notebook_source(text, repo=repo.name, path=record.path)
        ]

    if record.category == "code_generic":
        return [
            ChunkPayload(
                repo=c.repo, language=c.language, path=c.path, symbol_name=c.symbol_name,
                symbol_type=c.symbol_type, start_line=c.start_line, end_line=c.end_line, text=c.text,
            )
            for c in chunk_generic_source(text, repo=repo.name, path=record.path, language=_generic_language(full_path))
        ]

    if record.category == "docs":
        return [
            ChunkPayload(
                repo=c.repo, language="docs", path=c.path, symbol_name=f"doc#{c.chunk_index}",
                symbol_type="doc", start_line=0, end_line=0, text=c.text,
            )
            for c in chunk_doc_text(text, repo=repo.name, path=record.path)
        ]

    return []


def index_repo(
    repo_path: Path, repo: RepoSpec, store: QdrantStore, embedding_service: EmbeddingService,
    corpus_name: str, batch_size: int = 64,
) -> int:
    chunks: list[ChunkPayload] = []
    skipped_files = 0

    with timed_stage("extract_and_chunk", repo=repo.name):
        for record in walk_repo(repo_path, repo.name, repo.language):
            try:
                chunks.extend(_file_to_chunks(repo_path, record, repo))
            except (OSError, UnicodeError) as exc:
                # One unreadable/corrupt file must not abort indexing the rest
                # of the repo -- log it and move on. Syntax errors inside a
                # valid file are already handled by chunk_python_source itself.
                skipped_files += 1
                logger.warning("Skipping unreadable file %s/%s: %r", repo.name, record.path, exc)

    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]

        with timed_stage("embed", repo=repo.name, batch_size=len(batch)):
            vectors_by_model = embedding_service.embed_all([embedding_text(c) for c in batch])

        with timed_stage("upsert", repo=repo.name, batch_size=len(batch)):
            for model_key, vectors in vectors_by_model.items():
                collection = collection_name(corpus_name, model_key)
                store.ensure_collection(collection, vector_size=len(vectors[0]))
                store.upsert_chunks(collection, batch, vectors)

        total += len(batch)

    logger.info(
        "Indexed %d chunks from %s (%d files skipped)", total, repo.name, skipped_files,
    )
    return total
