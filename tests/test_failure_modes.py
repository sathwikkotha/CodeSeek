"""Deliberately break things and confirm the system degrades with a clear
error or a partial, logged skip -- never a silent wrong answer.

Mirrors the project's own build-order milestone: kill the embedding step
mid-batch, point the store at somewhere unreachable, feed a corrupt file."""

import logging
from pathlib import Path

import pytest

from codeseek.config import RepoSpec
from codeseek.embedding.service import EmbeddingService
from codeseek.pipeline.index import index_repo
from codeseek.store.qdrant_store import QdrantStore


class FakeEmbedder:
    dimensions = 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_unreachable_qdrant_url_raises_clear_error_not_hang_or_silent_empty():
    # An unroutable address (TEST-NET-1, RFC 5737) with a short timeout --
    # fails fast and loud instead of hanging on the client's default timeout.
    store = QdrantStore(url="http://192.0.2.1:6333", timeout=2)
    with pytest.raises(Exception):
        store.ensure_collection("whatever", vector_size=3)


def test_index_repo_skips_unreadable_file_but_still_indexes_the_rest(tmp_path, caplog, monkeypatch):
    repo_path = tmp_path / "demo_repo"
    repo_path.mkdir()
    (repo_path / "good.py").write_text("def add(a, b):\n    return a + b\n")
    (repo_path / "bad.py").write_text("def broken():\n    return 1\n")

    # Simulate an unreadable/corrupt file portably (Windows chmod doesn't
    # restrict reads the way POSIX does, so a real permissions test isn't reliable here).
    original_read_text = Path.read_text

    def flaky_read_text(self, *args, **kwargs):
        if self.name == "bad.py":
            raise OSError("simulated unreadable file")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)

    store = QdrantStore(location=":memory:")
    embedding_service = EmbeddingService({"general": FakeEmbedder()})
    repo = RepoSpec("demo_repo", "local", "python")

    with caplog.at_level(logging.WARNING):
        total = index_repo(repo_path, repo, store, embedding_service, corpus_name="failtest")

    assert total == 1  # only good.py's chunk made it through
    assert any("bad.py" in r.message for r in caplog.records)


def test_index_repo_python_file_with_syntax_error_is_skipped_not_fatal(tmp_path):
    repo_path = tmp_path / "demo_repo"
    repo_path.mkdir()
    (repo_path / "good.py").write_text("def add(a, b):\n    return a + b\n")
    (repo_path / "broken.py").write_text("def broken(:\n    this is not valid python\n")

    store = QdrantStore(location=":memory:")
    embedding_service = EmbeddingService({"general": FakeEmbedder()})
    repo = RepoSpec("demo_repo", "local", "python")

    total = index_repo(repo_path, repo, store, embedding_service, corpus_name="failtest2")

    assert total == 1  # broken.py contributes zero chunks, doesn't crash the run


def test_embedding_failure_raises_clear_error_instead_of_corrupting_store(tmp_path):
    """Local models have no network to flake on, so there's no retry/backoff
    to test -- the honest failure mode here is a bug or an OOM inside encode().
    Confirm that surfaces as a loud exception rather than a partial, silently
    wrong upsert."""

    repo_path = tmp_path / "demo_repo"
    repo_path.mkdir()
    (repo_path / "good.py").write_text("def add(a, b):\n    return a + b\n")

    class BrokenEmbedder:
        dimensions = 3

        def embed(self, texts):
            raise RuntimeError("simulated embedding backend crash")

    store = QdrantStore(location=":memory:")
    embedding_service = EmbeddingService({"general": BrokenEmbedder()})
    repo = RepoSpec("demo_repo", "local", "python")

    with pytest.raises(RuntimeError, match="simulated embedding backend crash"):
        index_repo(repo_path, repo, store, embedding_service, corpus_name="failtest3")

    # nothing should have been committed to the store for this failed batch
    assert not store._client.collection_exists("failtest3__general")


def test_upsert_with_wrong_vector_dimension_raises_clear_error():
    from codeseek.store.schema import ChunkPayload

    store = QdrantStore(location=":memory:")
    store.ensure_collection("dimtest", vector_size=3)
    chunk = ChunkPayload(
        repo="demo", language="python", path="a.py", symbol_name="f",
        symbol_type="function", start_line=1, end_line=1, text="def f(): pass",
    )

    with pytest.raises(Exception):
        store.upsert_chunks("dimtest", [chunk], [[1.0, 0.0]])  # wrong dimension: 2 instead of 3
