import json
import time

from fastapi.testclient import TestClient

from codeseek.api.app import create_app
from codeseek.embedding.registry import OPENAI
from codeseek.embedding.service import EmbeddingService
from codeseek.store.qdrant_store import QdrantStore
from codeseek.store.schema import ChunkPayload, collection_name
from tests._openai_fakes import FakeOpenAIClient, content_turn, tool_call_turn


class FakeEmbedder:
    """Deterministic 3-d embedder: 'jwt' queries/text point along x,
    everything else points along y -- just enough signal to test ranking."""

    dimensions = 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] if "jwt" in t.lower() else [0.0, 1.0, 0.0] for t in texts]


def _wait_for_ingest(client: TestClient, job_id: str, timeout_s: float = 5.0) -> dict:
    """Poll GET /ingest/{job_id} until the background job leaves pending/running --
    the job runs on a real background thread even under TestClient, so this mirrors
    how a real caller (the Streamlit UI) has to observe completion."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        body = client.get(f"/ingest/{job_id}").json()
        if body["state"] in ("done", "error"):
            return body
        time.sleep(0.02)
    raise TimeoutError(f"ingest job {job_id} did not finish within {timeout_s}s")


def _make_app_with_data():
    store = QdrantStore(location=":memory:")
    embedding_service = EmbeddingService({OPENAI: FakeEmbedder()})
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
    collection = collection_name(corpus_name, OPENAI)
    store.ensure_collection(collection, vector_size=3)
    vectors = FakeEmbedder().embed([c.text for c in chunks])
    store.upsert_chunks(collection, chunks, vectors)

    app = create_app(store, embedding_service, corpus_name=corpus_name)
    return TestClient(app)


def test_health():
    client = _make_app_with_data()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_search_returns_results():
    client = _make_app_with_data()
    resp = client.post("/search", json={"query": "jwt validation", "top_k": 5})
    assert resp.status_code == 200

    body = resp.json()
    symbol_names = {r["symbol_name"] for r in body["results"]}
    assert "validate_jwt" in symbol_names or "checkJwt" in symbol_names
    assert all(v >= 0 for v in body["timings"].values())


def test_search_with_reranker_blends_reranked_order_via_rrf():
    """The reranker's opinion is fused with the pre-rerank vector+keyword
    order (RRF), not trusted outright -- see hybrid.py's rrf_fuse. Real bug
    this guards against: a cross-encoder trained on natural-language passage
    ranking demoted a genuinely strong code match (#2 of 30) to #23 because
    it reads less like a natural-language answer than a prose doc chunk.
    Fixture: five candidates with strictly distinct (non-tied) vector scores,
    a reranker that demotes the pre-rank leader to dead last and leaves the
    others' relative order intact -- the leader should end up mid-pack, not
    first (reranker had a real effect) and not buried last (fusion tempered
    it), while the candidate strong in both orderings wins outright."""
    import math

    store = QdrantStore(location=":memory:")

    class GradedEmbedder:
        """query vector = 0 degrees; chunks planted at increasing angles so
        cosine similarity strictly decreases -- no ties, unlike same-direction
        vectors that differ only in magnitude."""

        dimensions = 2
        _ANGLES_DEG = {"query": 0, "leader": 0, "item1": 10, "item2": 20, "item3": 30, "item4": 40}

        def embed(self, texts: list[str]) -> list[list[float]]:
            vectors = []
            for t in texts:
                deg = self._ANGLES_DEG[t]
                rad = math.radians(deg)
                vectors.append([math.cos(rad), math.sin(rad)])
            return vectors

    embedding_service = EmbeddingService({OPENAI: GradedEmbedder()})
    corpus_name = "reranktest"

    names_in_pre_rank_order = ["leader", "item1", "item2", "item3", "item4"]
    chunks = [
        ChunkPayload(
            repo="demo", language="python", path=f"{name}.py", symbol_name=name,
            symbol_type="function", start_line=1, end_line=5, text=name,
        )
        for name in names_in_pre_rank_order
    ]
    collection = collection_name(corpus_name, OPENAI)
    store.ensure_collection(collection, vector_size=2)
    vectors = GradedEmbedder().embed([c.text for c in chunks])
    store.upsert_chunks(collection, chunks, vectors)

    class FakeReranker:
        def rerank(self, query, hits, top_k):
            leader = next(h for h in hits if h.payload["symbol_name"] == "leader")
            rest = [h for h in hits if h.payload["symbol_name"] != "leader"]
            return (rest + [leader])[:top_k]  # demote the pre-rank leader to last

    app = create_app(store, embedding_service, corpus_name=corpus_name, reranker=FakeReranker())
    client = TestClient(app)

    resp = client.post("/search", json={"query": "query", "top_k": 5})
    results = resp.json()["results"]
    names = [r["symbol_name"] for r in results]

    assert names[0] == "item1"  # strong pre-rank (#2) AND post-rank (#1) -- wins outright
    leader_rank = names.index("leader")
    assert 0 < leader_rank < 4  # demoted from #1, but not buried at #5 either


def test_search_filters_by_repo():
    client = _make_app_with_data()
    resp = client.post("/search", json={"query": "jwt", "top_k": 5, "repo": "demo"})
    results = resp.json()["results"]
    assert all(r["repo"] == "demo" for r in results)


def test_search_filters_by_language():
    client = _make_app_with_data()
    resp = client.post("/search", json={"query": "jwt", "top_k": 5, "language": "javascript"})
    results = resp.json()["results"]
    assert results  # the javascript chunk should be found
    assert all(r["repo"] == "other" for r in results)  # only javascript chunk in the fixture is repo "other"


def test_list_repos_returns_names_and_counts():
    client = _make_app_with_data()
    resp = client.get("/repos")
    assert resp.status_code == 200

    repos = {r["name"]: r["chunk_count"] for r in resp.json()["repos"]}
    assert repos == {"demo": 2, "other": 1}


def test_list_repos_empty_for_unindexed_corpus():
    store = QdrantStore(location=":memory:")
    embedding_service = EmbeddingService({OPENAI: FakeEmbedder()})
    app = create_app(store, embedding_service, corpus_name="never_indexed")
    client = TestClient(app)

    resp = client.get("/repos")
    assert resp.status_code == 200
    assert resp.json()["repos"] == []


def test_delete_repo_removes_only_that_repos_points():
    client = _make_app_with_data()
    resp = client.delete("/repos/demo")
    assert resp.status_code == 200
    assert resp.json() == {"deleted": "demo"}

    repos = {r["name"] for r in client.get("/repos").json()["repos"]}
    assert repos == {"other"}


def test_ingest_clones_indexes_and_makes_repo_searchable(tmp_path, monkeypatch):
    fake_repo = tmp_path / "widgetlib"
    fake_repo.mkdir()
    (fake_repo / "widget.py").write_text(
        "def make_widget():\n    \"\"\"jwt widget factory.\"\"\"\n    return 1\n"
    )

    monkeypatch.setattr("codeseek.api.app.clone_or_update", lambda repo, dest_dir: fake_repo)

    store = QdrantStore(location=":memory:")
    embedding_service = EmbeddingService({OPENAI: FakeEmbedder()})
    app = create_app(store, embedding_service, corpus_name="ingesttest")
    client = TestClient(app)

    resp = client.post("/ingest", json={"url": "https://github.com/example/widgetlib.git"})
    assert resp.status_code == 202
    body = _wait_for_ingest(client, resp.json()["job_id"])
    assert body["state"] == "done"
    assert body["repo"] == "widgetlib"
    assert body["chunks_indexed"] == 1

    search_resp = client.post("/search", json={"query": "jwt", "top_k": 5})
    results = search_resp.json()["results"]
    assert any(r["repo"] == "widgetlib" and r["symbol_name"] == "make_widget" for r in results)


def test_ingest_respects_custom_name(tmp_path, monkeypatch):
    fake_repo = tmp_path / "cloned"
    fake_repo.mkdir()
    (fake_repo / "a.py").write_text("def f():\n    return 1\n")

    monkeypatch.setattr("codeseek.api.app.clone_or_update", lambda repo, dest_dir: fake_repo)

    store = QdrantStore(location=":memory:")
    embedding_service = EmbeddingService({OPENAI: FakeEmbedder()})
    app = create_app(store, embedding_service, corpus_name="ingesttest2")
    client = TestClient(app)

    resp = client.post("/ingest", json={"url": "https://github.com/example/repo.git", "name": "my-custom-name"})
    body = _wait_for_ingest(client, resp.json()["job_id"])
    assert body["repo"] == "my-custom-name"


def test_ingest_clone_failure_surfaces_as_a_failed_job_not_a_500(monkeypatch):
    import subprocess

    def _raise(repo, dest_dir):
        raise subprocess.CalledProcessError(128, ["git", "clone"], stderr="repository not found")

    monkeypatch.setattr("codeseek.api.app.clone_or_update", _raise)

    store = QdrantStore(location=":memory:")
    embedding_service = EmbeddingService({OPENAI: FakeEmbedder()})
    app = create_app(store, embedding_service, corpus_name="ingesttest3")
    client = TestClient(app)

    resp = client.post("/ingest", json={"url": "https://github.com/example/missing.git"})
    assert resp.status_code == 202  # accepted -- the failure happens in the background job

    body = _wait_for_ingest(client, resp.json()["job_id"])
    assert body["state"] == "error"
    assert "repository not found" in body["error"]


def test_ingest_status_for_unknown_job_is_404():
    store = QdrantStore(location=":memory:")
    embedding_service = EmbeddingService({OPENAI: FakeEmbedder()})
    app = create_app(store, embedding_service, corpus_name="ingesttest4")
    client = TestClient(app)

    resp = client.get("/ingest/does-not-exist")
    assert resp.status_code == 404


def test_search_against_unindexed_collection_returns_empty_not_error():
    store = QdrantStore(location=":memory:")
    embedding_service = EmbeddingService({OPENAI: FakeEmbedder()})
    app = create_app(store, embedding_service, corpus_name="never_indexed")
    client = TestClient(app)

    resp = client.post("/search", json={"query": "anything", "top_k": 5})
    assert resp.status_code == 200
    assert resp.json()["results"] == []


# --- /explain -----------------------------------------------------------
# A fake streaming OpenAI client scripted to return a fixed sequence of
# turns, so the loop's tool-dispatch and message-history building are
# exercised without ever hitting the real API.


def test_explain_calls_tools_and_returns_grounded_answer(tmp_path, monkeypatch):
    store = QdrantStore(location=":memory:")
    embedding_service = EmbeddingService({OPENAI: FakeEmbedder()})
    corpus_name = "explaintest"

    chunks = [
        ChunkPayload(
            repo="demo", language="python", path="auth.py", symbol_name="validate_jwt",
            symbol_type="function", start_line=1, end_line=1, text="def validate_jwt(token): jwt stuff",
        ),
    ]
    collection = collection_name(corpus_name, OPENAI)
    store.ensure_collection(collection, vector_size=3)
    store.upsert_chunks(collection, chunks, FakeEmbedder().embed([c.text for c in chunks]))

    repo_dir = tmp_path / "demo"
    repo_dir.mkdir()
    (repo_dir / "auth.py").write_text("def validate_jwt(token): jwt stuff\n")
    monkeypatch.setattr("codeseek.api.app.REPOS_DIR", tmp_path)

    fake_client = FakeOpenAIClient([
        tool_call_turn(("call_1", "search_code", '{"query": "jwt validation", "top_k": 5}')),
        content_turn("JWT is validated by validate_jwt. demo/auth.py:1-1"),
    ])

    app = create_app(store, embedding_service, corpus_name=corpus_name, openai_client=fake_client)
    client = TestClient(app)

    resp = client.post("/explain", json={"repo": "demo", "question": "How is JWT validated?"})
    assert resp.status_code == 200
    body = resp.json()

    assert body["answer"] == "JWT is validated by validate_jwt. demo/auth.py:1-1"
    assert body["error"] is None
    assert len(body["tool_calls"]) == 1
    assert body["tool_calls"][0]["tool"] == "search_code"
    assert "validate_jwt" in body["tool_calls"][0]["result_summary"]

    # the citation in the answer ("demo/auth.py:1-1") points at a real file
    # actually on disk in this fixture -- verify_citations should confirm it
    assert body["citation_checks"] == [
        {
            "citation": "demo/auth.py:1-1", "path": "auth.py", "start_line": 1,
            "end_line": 1, "valid": True, "reason": None,
        },
    ]
    # no usage chunk was scripted onto either turn -- Usage.add() is never called
    assert body["usage"]["requests"] == 0
    assert body["usage"]["cost_usd"] == 0.0


def test_explain_returns_404_for_unindexed_repo(tmp_path, monkeypatch):
    store = QdrantStore(location=":memory:")
    embedding_service = EmbeddingService({OPENAI: FakeEmbedder()})
    monkeypatch.setattr("codeseek.api.app.REPOS_DIR", tmp_path)

    app = create_app(store, embedding_service, corpus_name="explaintest2", openai_client=FakeOpenAIClient([]))
    client = TestClient(app)

    resp = client.post("/explain", json={"repo": "never-indexed", "question": "anything"})
    assert resp.status_code == 404


# --- /explain/stream (SSE) ------------------------------------------------


def _parse_sse_events(raw_text: str) -> list[dict]:
    events = []
    for line in raw_text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


def test_explain_stream_emits_tool_call_and_answer_delta_events_ending_in_done(tmp_path, monkeypatch):
    store = QdrantStore(location=":memory:")
    embedding_service = EmbeddingService({OPENAI: FakeEmbedder()})
    corpus_name = "explainstreamtest"

    chunks = [
        ChunkPayload(
            repo="demo", language="python", path="auth.py", symbol_name="validate_jwt",
            symbol_type="function", start_line=1, end_line=1, text="def validate_jwt(token): jwt stuff",
        ),
    ]
    collection = collection_name(corpus_name, OPENAI)
    store.ensure_collection(collection, vector_size=3)
    store.upsert_chunks(collection, chunks, FakeEmbedder().embed([c.text for c in chunks]))

    repo_dir = tmp_path / "demo"
    repo_dir.mkdir()
    (repo_dir / "auth.py").write_text("def validate_jwt(token): jwt stuff\n")
    monkeypatch.setattr("codeseek.api.app.REPOS_DIR", tmp_path)

    fake_client = FakeOpenAIClient([
        tool_call_turn(("call_1", "search_code", '{"query": "jwt validation"}')),
        content_turn("JWT is validated by validate_jwt. demo/auth.py:1-1"),
    ])

    app = create_app(store, embedding_service, corpus_name=corpus_name, openai_client=fake_client)
    client = TestClient(app)

    with client.stream("POST", "/explain/stream", json={"repo": "demo", "question": "How is JWT validated?"}) as resp:
        assert resp.status_code == 200
        events = _parse_sse_events(resp.read().decode("utf-8"))

    event_types = [e["type"] for e in events]
    assert "tool_call" in event_types
    assert "answer_delta" in event_types
    assert event_types[-1] == "done"

    tool_call_event = next(e for e in events if e["type"] == "tool_call")
    assert tool_call_event["tool"] == "search_code"

    done = events[-1]
    assert done["answer"] == "JWT is validated by validate_jwt. demo/auth.py:1-1"
    assert done["citation_checks"] == [
        {
            "citation": "demo/auth.py:1-1", "path": "auth.py", "start_line": 1,
            "end_line": 1, "valid": True, "reason": None,
        },
    ]


def test_explain_stream_returns_404_for_unindexed_repo(tmp_path, monkeypatch):
    store = QdrantStore(location=":memory:")
    embedding_service = EmbeddingService({OPENAI: FakeEmbedder()})
    monkeypatch.setattr("codeseek.api.app.REPOS_DIR", tmp_path)

    app = create_app(store, embedding_service, corpus_name="explainstreamtest2", openai_client=FakeOpenAIClient([]))
    client = TestClient(app)

    resp = client.post("/explain/stream", json={"repo": "never-indexed", "question": "anything"})
    assert resp.status_code == 404
