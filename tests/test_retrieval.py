from codeseek.retrieval.hybrid import HybridRetriever, MergedHit, fetch_k_for, merge_hits, rrf_fuse
from codeseek.store.qdrant_store import QdrantStore, SearchHit
from codeseek.store.schema import ChunkPayload


def test_fetch_k_for_over_fetches_so_damping_can_promote_lower_ranked_hits():
    assert fetch_k_for(10) == 30
    assert fetch_k_for(5) == 30  # floor of 30 even for a small top_k
    assert fetch_k_for(20) == 60  # scales past the floor for larger top_k


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


def test_merge_hits_damps_only_the_generic_doc_overview_chunk():
    vector_hits = [
        SearchHit(id="overview", score=0.80, payload={"symbol_type": "doc", "symbol_name": "doc#0"}),
        SearchHit(id="specific-doc", score=0.80, payload={"symbol_type": "doc", "symbol_name": "doc#1"}),
        SearchHit(id="code", score=0.80, payload={"symbol_type": "function", "symbol_name": "install"}),
    ]

    merged = {h.id: h for h in merge_hits(vector_hits, [], top_k=10)}

    assert merged["overview"].score == 0.80 * 0.85  # damped
    assert merged["specific-doc"].score == 0.80  # untouched -- not the overview chunk
    assert merged["code"].score == 0.80  # untouched -- not a doc chunk at all


def test_merge_hits_damping_can_change_the_ranking():
    # A slightly-ahead generic overview chunk should no longer beat a close
    # specific answer once damped -- this is the whole point of the fix.
    vector_hits = [
        SearchHit(id="overview", score=0.70, payload={"symbol_type": "doc", "symbol_name": "doc#0"}),
        SearchHit(id="real_answer", score=0.65, payload={"symbol_type": "function", "symbol_name": "install"}),
    ]

    merged = merge_hits(vector_hits, [], top_k=10)

    assert merged[0].id == "real_answer"


def test_merge_hits_damping_applies_to_keyword_floor_score_too():
    vector_hits = [SearchHit(id="a", score=0.9, payload={"symbol_name": "a"})]
    keyword_hits = [SearchHit(id="c", score=0.0, payload={"symbol_type": "doc", "symbol_name": "doc#0"})]

    merged = {h.id: h for h in merge_hits(vector_hits, keyword_hits, top_k=10)}

    assert merged["c"].score == 0.9 * 0.85  # floor score (0.9), then damped


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


def test_find_by_symbol_matches_exact_split_part_and_method_names():
    store = QdrantStore(location=":memory:")
    collection = "symbol_test"
    store.ensure_collection(collection, vector_size=3)

    chunks = [
        _make_chunk("demo", "OptionInfo", "class OptionInfo: ..."),
        ChunkPayload(
            repo="demo", language="python", path="models.py", symbol_name="OptionInfo.__init__",
            symbol_type="method", start_line=10, end_line=20, text="def __init__(self): ...",
        ),
        ChunkPayload(
            repo="demo", language="python", path="models.py", symbol_name="OptionInfo#part2",
            symbol_type="class", start_line=21, end_line=40, text="...continued...",
        ),
        _make_chunk("demo", "ArgumentInfo", "class ArgumentInfo: ..."),  # different symbol, must not match
        _make_chunk("other", "OptionInfo", "class OptionInfo: ..."),  # same name, different repo, must not match
    ]
    vectors = [[1.0, 0.0, 0.0]] * len(chunks)
    store.upsert_chunks(collection, chunks, vectors)

    matches = {h.payload["symbol_name"] for h in store.find_by_symbol(collection, repo="demo", name="OptionInfo")}
    assert matches == {"OptionInfo", "OptionInfo.__init__", "OptionInfo#part2"}


def test_find_by_symbol_returns_empty_for_unknown_name():
    store = QdrantStore(location=":memory:")
    collection = "symbol_test2"
    store.ensure_collection(collection, vector_size=3)
    store.upsert_chunks(collection, [_make_chunk("demo", "Foo", "class Foo: ...")], [[1.0, 0.0, 0.0]])

    assert store.find_by_symbol(collection, repo="demo", name="DoesNotExist") == []


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


class FakeReranker:
    """Deterministic stand-in for CrossEncoderReranker: reverses whatever
    order it's given, so tests can prove the reranker's output -- not the
    pre-rerank order -- is what the retriever actually returns."""

    def __init__(self):
        self.received_query = None
        self.received_hit_count = None

    def rerank(self, query, hits, top_k):
        self.received_query = query
        self.received_hit_count = len(hits)
        return list(reversed(hits))[:top_k]


def test_hybrid_retriever_with_reranker_uses_reranked_order_not_merge_order():
    store = QdrantStore(location=":memory:")
    collection = "test_rerank"
    store.ensure_collection(collection, vector_size=3)

    chunks = [_make_chunk("demo", f"fn{i}", f"def fn{i}(): pass") for i in range(5)]
    vectors = [[1.0 - i * 0.01, 0.0, 0.0] for i in range(5)]  # fn0 has the strongest vector score
    store.upsert_chunks(collection, chunks, vectors)

    reranker = FakeReranker()
    retriever = HybridRetriever(store, reranker=reranker)
    results = retriever.search(collection, query_vector=[1.0, 0.0, 0.0], query_text="fn0", top_k=3)

    # the reranker is called with the full fetch_k-sized pool (not just top_k), and
    # its output feeds RRF fusion rather than being trusted outright -- see
    # test_rrf_fuse_prevents_reranker_from_burying_a_strong_pre_rerank_candidate
    # for why the final order isn't simply "whatever the reranker said".
    assert reranker.received_query == "fn0"
    assert reranker.received_hit_count == 5
    assert len(results) == 3


def test_rrf_fuse_prevents_reranker_from_burying_a_strong_pre_rerank_candidate():
    """Regression test for a real bug found via live eval: a class chunk ranked
    #2 of 30 candidates by vector+keyword score was pushed to #23 by the
    cross-encoder reranker alone (it's trained on natural-language passage
    ranking, so it's biased toward prose over code). Fusing the reranker's
    order with the pre-rerank order by rank should stop a candidate that
    strong pre-rerank from being buried, even when the reranker actively
    demotes it -- without simply ignoring the reranker either."""
    pre = [MergedHit(id=str(i), score=1.0 - i * 0.01, payload={}, source="vector") for i in range(10)]
    demoted = pre[1]  # pre-rank #2 (index 1) -- the reranker demotes it to dead last
    post = [h for h in pre if h.id != demoted.id] + [demoted]

    fused = rrf_fuse(pre, post, top_k=10)
    fused_ids = [h.id for h in fused]

    assert fused_ids[0] == "0"  # strong pre AND post rank still wins outright
    assert fused_ids.index(demoted.id) < 9  # demoted item survives, doesn't collapse to last


def test_hybrid_retriever_without_reranker_falls_back_to_merge_order():
    store = QdrantStore(location=":memory:")
    collection = "test_no_rerank"
    store.ensure_collection(collection, vector_size=3)

    chunks = [_make_chunk("demo", "fn0", "def fn0(): pass"), _make_chunk("demo", "fn1", "def fn1(): pass")]
    # different *directions*, not just magnitude -- cosine distance only cares
    # about angle, so parallel vectors (e.g. [1,0,0] vs [0.5,0,0]) would tie.
    vectors = [[1.0, 0.0, 0.0], [0.7, 0.7, 0.0]]
    store.upsert_chunks(collection, chunks, vectors)

    retriever = HybridRetriever(store)  # no reranker
    results = retriever.search(collection, query_vector=[1.0, 0.0, 0.0], query_text="fn0", top_k=2)

    assert results[0].payload["symbol_name"] == "fn0"
