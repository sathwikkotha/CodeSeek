from codeseek.retrieval.hybrid import HybridRetriever, merge_hits
from codeseek.store.qdrant_store import QdrantStore, SearchHit
from codeseek.store.schema import ChunkPayload


def test_merge_hits_keyword_only_hit_gets_floor_score_not_top_score():
    vector_hits = [
        SearchHit(id="a", score=0.9, payload={"symbol_name": "a"}),
        SearchHit(id="b", score=0.5, payload={"symbol_name": "b"}),
    ]
    keyword_hits = [SearchHit(id="c", score=0.0, payload={"symbol_name": "JWTBearer"})]

    merged = merge_hits(vector_hits, keyword_hits, top_k=10)

    by_id = {h.id: h for h in merged}
    assert by_id["a"].source == "vector"
    assert by_id["c"].source == "keyword"
    assert by_id["c"].score == 0.5  # the weakest vector score, not 0 and not top
    # a keyword-only hit never outranks a genuinely strong semantic match
    assert merged[0].id == "a"


def test_merge_hits_dedupes_and_marks_both():
    vector_hits = [SearchHit(id="a", score=0.9, payload={})]
    keyword_hits = [SearchHit(id="a", score=0.0, payload={})]

    merged = merge_hits(vector_hits, keyword_hits, top_k=10)

    assert len(merged) == 1
    assert merged[0].source == "both"
    assert merged[0].score == 0.9  # vector score preserved, not overwritten


def test_merge_hits_respects_top_k():
    vector_hits = [SearchHit(id=str(i), score=float(i), payload={}) for i in range(20)]
    merged = merge_hits(vector_hits, [], top_k=5)
    assert len(merged) == 5
    assert [h.id for h in merged] == ["19", "18", "17", "16", "15"]


def _make_chunk(repo: str, symbol: str, text: str) -> ChunkPayload:
    return ChunkPayload(
        repo=repo, language="python", path=f"{symbol}.py", symbol_name=symbol,
        symbol_type="function", start_line=1, end_line=5, text=text,
    )


def test_qdrant_store_vector_and_keyword_search_end_to_end():
    store = QdrantStore(location=":memory:")
    collection = "test_collection"
    store.ensure_collection(collection, vector_size=3)

    chunks = [
        _make_chunk("demo", "validate_jwt", "def validate_jwt(token): ..."),
        _make_chunk("demo", "parse_csv", "def parse_csv(path): ..."),
    ]
    # simple 3-d vectors: first chunk points along x, second along y
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    store.upsert_chunks(collection, chunks, vectors)

    vector_hits = store.vector_search(collection, query_vector=[1.0, 0.0, 0.0], limit=5)
    assert vector_hits[0].payload["symbol_name"] == "validate_jwt"

    keyword_hits = store.keyword_search(collection, query_text="parse_csv", limit=5)
    assert any(h.payload["symbol_name"] == "parse_csv" for h in keyword_hits)


def test_hybrid_retriever_merges_real_store_results():
    store = QdrantStore(location=":memory:")
    collection = "test_hybrid"
    store.ensure_collection(collection, vector_size=3)

    chunks = [
        _make_chunk("demo", "validate_jwt", "def validate_jwt(token): ..."),
        _make_chunk("demo", "parse_csv", "def parse_csv(path): ..."),
    ]
    vectors = [[1.0, 0.0, 0.0], [0.0, 0.1, 0.0]]
    store.upsert_chunks(collection, chunks, vectors)

    retriever = HybridRetriever(store)
    # query vector close to validate_jwt, but text keyword targets parse_csv
    results = retriever.search(collection, query_vector=[0.9, 0.0, 0.0], query_text="parse_csv", top_k=5)

    symbol_names = {h.payload["symbol_name"] for h in results}
    assert "validate_jwt" in symbol_names  # strong vector match
    assert "parse_csv" in symbol_names  # surfaced via keyword path
