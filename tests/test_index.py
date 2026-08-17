import json

from codeseek.config import RepoSpec
from codeseek.embedding.registry import OPENAI
from codeseek.embedding.service import EmbeddingService
from codeseek.pipeline.index import index_repo
from codeseek.store.qdrant_store import QdrantStore
from codeseek.store.schema import collection_name


class FakeEmbedder:
    dimensions = 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


def _make_mixed_repo(tmp_path):
    repo_dir = tmp_path / "mixedrepo"
    repo_dir.mkdir()
    (repo_dir / "main.py").write_text("def f():\n    return 1\n")
    (repo_dir / "README.md").write_text("# Mixed repo\n\nSome prose about this project.\n")
    (repo_dir / "main.go").write_text("package main\n\nfunc main() {}\n")
    (repo_dir / "notebook.ipynb").write_text(json.dumps({
        "cells": [
            {"cell_type": "markdown", "source": ["# Notebook heading\n"]},
            {"cell_type": "code", "source": ["import numpy as np\n", "x = np.array([1, 2, 3])\n"]},
        ],
    }))
    return repo_dir


def test_index_repo_indexes_every_supported_category_in_a_mixed_repo(tmp_path):
    """Regression test for a real bug: an all-notebook repo indexed to a
    single chunk (just its README) because .ipynb, .go, and friends were
    silently dropped by the walker with no error, log line, or feedback --
    the repo looked "indexed" but had nothing useful to search."""
    repo_dir = _make_mixed_repo(tmp_path)
    repo = RepoSpec(name="mixedrepo", url="local", language="unknown")
    store = QdrantStore(location=":memory:")
    embedding_service = EmbeddingService({OPENAI: FakeEmbedder()})

    total = index_repo(repo_dir, repo, store, embedding_service, corpus_name="indextest")

    # main.py -> 1 (one function), README.md -> 1 (one doc chunk), main.go -> 1
    # (one generic block), notebook.ipynb -> 2 (one markdown cell, one code cell)
    assert total == 5

    collection = collection_name("indextest", OPENAI)
    hits = store.vector_search(collection, query_vector=[1.0, 0.0, 0.0], limit=100)
    languages = sorted(h.payload["language"] for h in hits)
    assert languages == ["docs", "go", "notebook", "notebook", "python"]
