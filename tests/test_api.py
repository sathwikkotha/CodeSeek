from fastapi.testclient import TestClient

from codeseek.api.app import create_app
from codeseek.embedding.service import EmbeddingService
from codeseek.store.qdrant_store import QdrantStore
from codeseek.store.schema import ChunkPayload, collection_name


class FakeEmbedder:
    """Deterministic 3-d embedder: 'jwt' queries/text point along x,
    everything else points along y -- just enough signal to test ranking."""

    dimensions = 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] if "jwt" in t.lower() else [0.0, 1.0, 0.0] for t in texts]


def _make_app_with_data():
    store = QdrantStore(location=":memory:")
    embedding_service = EmbeddingService({"general": FakeEmbedder(), "code": FakeEmbedder()})
    corpus_name = "testcorpus"

    chunks = [
        ChunkPayload(
            repo="demo", language="python", path="auth.py", symbol_name="validate_jwt",
            symbol_type="function", start_line=1, end_line=5, text="def validate_jwt(token): jwt stuff",
        ),
        ChunkPayload(
            repo="demo", language="python", path="csv.py", symbol_name="parse_csv",
            symbol_type="function", start_line=1, end_line=5, text="def parse_csv(path): csv stuff",
        ),
        ChunkPayload(
            repo="other", language="javascript", path="auth.js", symbol_name="checkJwt",
            symbol_type="function", start_line=1, end_line=5, text="function checkJwt(token) { jwt }",
        ),
    ]
    for model_key in ("general", "code"):
        collection = collection_name(corpus_name, model_key)
        store.ensure_collection(collection, vector_size=3)
        vectors = embedding_service._embedders[model_key].embed([c.text for c in chunks])
        store.upsert_chunks(collection, chunks, vectors)

    app = create_app(store, embedding_service, corpus_name=corpus_name)
    return TestClient(app)


def test_health():
    client = _make_app_with_data()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_search_returns_results_for_every_requested_model():
    client = _make_app_with_data()
    resp = client.post("/search", json={"query": "jwt validation", "models": ["general", "code"], "top_k": 5})
    assert resp.status_code == 200

    body = resp.json()
    assert set(body["results_by_model"]) == {"general", "code"}
    for results in body["results_by_model"].values():
        symbol_names = {r["symbol_name"] for r in results}
        assert "validate_jwt" in symbol_names or "checkJwt" in symbol_names

    assert set(body["timings_by_model"]) == {"general", "code"}
    for timings in body["timings_by_model"].values():
        assert all(v >= 0 for v in timings.values())


def test_search_filters_by_repo():
    client = _make_app_with_data()
    resp = client.post("/search", json={"query": "jwt", "models": ["general"], "top_k": 5, "repo": "demo"})
    results = resp.json()["results_by_model"]["general"]
    assert all(r["repo"] == "demo" for r in results)


def test_search_filters_by_language():
    client = _make_app_with_data()
    resp = client.post(
        "/search", json={"query": "jwt", "models": ["general"], "top_k": 5, "language": "javascript"}
    )
    results = resp.json()["results_by_model"]["general"]
    assert results  # the javascript chunk should be found
    assert all(r["repo"] == "other" for r in results)  # only javascript chunk in the fixture is repo "other"


def test_search_against_unindexed_collection_returns_empty_not_error():
    store = QdrantStore(location=":memory:")
    embedding_service = EmbeddingService({"general": FakeEmbedder(), "code": FakeEmbedder()})
    app = create_app(store, embedding_service, corpus_name="never_indexed")
    client = TestClient(app)

    resp = client.post("/search", json={"query": "anything", "models": ["general"], "top_k": 5})
    assert resp.status_code == 200
    assert resp.json()["results_by_model"]["general"] == []
