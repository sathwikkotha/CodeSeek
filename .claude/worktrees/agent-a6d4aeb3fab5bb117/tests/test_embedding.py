import hashlib

import pytest

from codeseek.embedding.service import EmbeddingService


class FakeEmbedder:
    """Deterministic across processes (unlike builtin hash(), which is
    salted per-run by PYTHONHASHSEED) and collision-resistant enough that
    two different literal strings never coincidentally embed identically."""

    def __init__(self, dimensions: int = 8):
        self.dimensions = dimensions
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        vectors = []
        for t in texts:
            digest = hashlib.sha256(t.encode()).digest()
            vectors.append([digest[i % len(digest)] / 255.0 for i in range(self.dimensions)])
        return vectors


def test_embedding_service_embeds_with_every_configured_model():
    service = EmbeddingService({"general": FakeEmbedder(), "code": FakeEmbedder(dimensions=16)})

    result = service.embed_all(["def f(): pass", "class C: pass"])

    assert set(result) == {"general", "code"}
    assert all(len(vectors) == 2 for vectors in result.values())
    assert all(len(vec) == 8 for vec in result["general"])
    assert all(len(vec) == 16 for vec in result["code"])


def test_embed_one_caches_repeated_identical_queries():
    embedder = FakeEmbedder()
    service = EmbeddingService({"general": embedder})

    v1 = service.embed_one("general", "where is jwt validation")
    v2 = service.embed_one("general", "where is jwt validation")
    v3 = service.embed_one("general", "a different query")

    assert v1 == v2
    assert embedder.calls == 2  # second identical call hit the cache, no re-embed
    assert service.cache_hits == 1
    assert service.cache_misses == 2
    assert v3 != v1


def test_embed_one_cache_evicts_oldest_when_full():
    embedder = FakeEmbedder()
    service = EmbeddingService({"general": embedder}, query_cache_size=2)

    service.embed_one("general", "q1")
    service.embed_one("general", "q2")
    service.embed_one("general", "q3")  # evicts q1
    service.embed_one("general", "q1")  # cache miss again -- was evicted

    assert service.cache_misses == 4
    assert service.cache_hits == 0


@pytest.mark.slow
def test_real_local_models_smoke():
    """Downloads and runs the two actual local embedding models once, to prove
    the real pipeline (not the fake) produces sane, differently-shaped vectors."""
    from codeseek.embedding.registry import CODE, GENERAL, build_default_embedders

    service = EmbeddingService(build_default_embedders())
    result = service.embed_all(["def add(a, b):\n    return a + b"])

    assert len(result[GENERAL][0]) == 384
    assert len(result[CODE][0]) == 768
