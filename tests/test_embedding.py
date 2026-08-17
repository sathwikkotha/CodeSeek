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


def test_openai_embedder_uses_injected_client():
    """OpenAIEmbedder against a fake client -- no network, no cost --
    verifying the request shape and response-parsing, not the real API."""
    from codeseek.embedding.openai_embedder import OpenAIEmbedder

    class FakeEmbeddingItem:
        def __init__(self, embedding):
            self.embedding = embedding

    class FakeEmbeddingsResponse:
        def __init__(self, vectors):
            self.data = [FakeEmbeddingItem(v) for v in vectors]

    class FakeOpenAIClient:
        def __init__(self):
            self.calls = []
            self.embeddings = self

        def create(self, model, input):
            self.calls.append((model, input))
            return FakeEmbeddingsResponse([[0.1, 0.2, 0.3] for _ in input])

    fake_client = FakeOpenAIClient()
    embedder = OpenAIEmbedder(model_name="text-embedding-3-small", client=fake_client)

    vectors = embedder.embed(["def f(): pass", "class C: pass"])

    assert vectors == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert fake_client.calls == [("text-embedding-3-small", ["def f(): pass", "class C: pass"])]
    assert embedder.dimensions == 1536


def test_openai_embedder_returns_empty_list_for_empty_input():
    """The API 400s on an empty input list -- guard against ever sending one."""
    from codeseek.embedding.openai_embedder import OpenAIEmbedder

    class FailingClient:
        def __init__(self):
            self.embeddings = self

        def create(self, model, input):
            raise AssertionError("should never be called with empty input")

    embedder = OpenAIEmbedder(client=FailingClient())
    assert embedder.embed([]) == []


@pytest.mark.slow
def test_real_openai_embedder_smoke():
    """Calls the real OpenAI embeddings API once, to prove the actual
    pipeline (not the fake) produces a sane vector. Costs a fraction of a
    cent and requires OPENAI_API_KEY."""
    from codeseek.embedding.registry import OPENAI, build_default_embedders

    service = EmbeddingService(build_default_embedders())
    result = service.embed_all(["def add(a, b):\n    return a + b"])

    assert len(result[OPENAI][0]) == 1536
