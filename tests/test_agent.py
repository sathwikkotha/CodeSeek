from pathlib import Path

from codeseek.agent.loop import ExplainResult, Usage, explain
from codeseek.agent.tools import build_tool_specs, build_tools
from codeseek.embedding.registry import OPENAI
from codeseek.embedding.service import EmbeddingService
from codeseek.store.qdrant_store import QdrantStore
from codeseek.store.schema import ChunkPayload, collection_name
from tests._openai_fakes import FakeOpenAIClient, content_turn, tool_call_turn


class _FakePromptTokensDetails:
    def __init__(self, cached_tokens: int):
        self.cached_tokens = cached_tokens


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.prompt_tokens_details = _FakePromptTokensDetails(cached_tokens)


def test_usage_add_accumulates_across_calls_including_cached_tokens():
    usage = Usage(model="gpt-5.4-mini")
    usage.add(_FakeUsage(prompt_tokens=1000, completion_tokens=100, cached_tokens=200))
    usage.add(_FakeUsage(prompt_tokens=2000, completion_tokens=50, cached_tokens=1800))

    assert usage.input_tokens == 3000
    assert usage.cached_input_tokens == 2000
    assert usage.output_tokens == 150
    assert usage.requests == 2


def test_usage_add_tolerates_missing_usage_object():
    usage = Usage()
    usage.add(None)  # some SDK responses may omit usage -- must not crash
    assert usage.requests == 0


def test_usage_cost_usd_uses_verified_pricing_and_cache_discount():
    # gpt-5.4-mini: $0.75/1M regular input, $0.075/1M cached input, $4.50/1M output
    usage = Usage(model="gpt-5.4-mini", input_tokens=10_000, cached_input_tokens=4_000, output_tokens=1_000)

    uncached_cost = 6_000 / 1_000_000 * 0.75
    cached_cost = 4_000 / 1_000_000 * 0.075
    output_cost = 1_000 / 1_000_000 * 4.50
    assert usage.cost_usd == uncached_cost + cached_cost + output_cost


def test_usage_cost_usd_unknown_model_returns_zero_not_a_crash():
    usage = Usage(model="some-future-model-not-in-the-table", input_tokens=1_000_000, output_tokens=1_000_000)
    assert usage.cost_usd == 0.0


class FakeEmbedder:
    dimensions = 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_build_tool_specs_has_the_three_expected_tools():
    specs = build_tool_specs()
    names = {spec["function"]["name"] for spec in specs}
    assert names == {"search_code", "follow_symbol", "read_file"}
    for spec in specs:
        assert spec["type"] == "function"
        assert "description" in spec["function"]
        assert "parameters" in spec["function"]


def _make_store_and_service(corpus_name: str = "agenttest"):
    store = QdrantStore(location=":memory:")
    embedding_service = EmbeddingService({OPENAI: FakeEmbedder()})
    collection = collection_name(corpus_name, OPENAI)
    chunks = [
        ChunkPayload(
            repo="demo", language="python", path="auth.py", symbol_name="validate_jwt",
            symbol_type="function", start_line=1, end_line=1, text="def validate_jwt(token): jwt stuff",
        ),
    ]
    store.ensure_collection(collection, vector_size=3)
    store.upsert_chunks(collection, chunks, FakeEmbedder().embed([c.text for c in chunks]))
    return store, embedding_service


def test_search_code_tool_returns_formatted_chunk(tmp_path):
    store, embedding_service = _make_store_and_service()
    tools = build_tools("demo", tmp_path, store, embedding_service, reranker=None, corpus_name="agenttest")

    result = tools["search_code"](query="jwt validation")
    assert "validate_jwt" in result
    assert "demo/auth.py:1-1" in result


def test_search_code_tool_no_results():
    store, embedding_service = _make_store_and_service()
    tools = build_tools("empty-repo", Path("."), store, embedding_service, reranker=None, corpus_name="agenttest")

    assert tools["search_code"](query="anything") == "No results found."


def _make_store_with_many_chunks(n: int, text_size: int, corpus_name: str = "agenttest"):
    """n near-identical chunks, all matching any query under FakeEmbedder --
    lets a test request more results than exist and observe the real cap."""
    store = QdrantStore(location=":memory:")
    embedding_service = EmbeddingService({OPENAI: FakeEmbedder()})
    collection = collection_name(corpus_name, OPENAI)
    chunks = [
        ChunkPayload(
            repo="demo", language="python", path=f"mod{i}.py", symbol_name=f"fn{i}",
            symbol_type="function", start_line=1, end_line=1, text="x" * text_size,
        )
        for i in range(n)
    ]
    store.ensure_collection(collection, vector_size=3)
    store.upsert_chunks(collection, chunks, FakeEmbedder().embed([c.text for c in chunks]))
    return store, embedding_service


def test_search_code_tool_clamps_top_k_above_max():
    from codeseek.agent.tools import MAX_SEARCH_TOP_K

    store, embedding_service = _make_store_with_many_chunks(n=MAX_SEARCH_TOP_K + 10, text_size=10)
    tools = build_tools("demo", Path("."), store, embedding_service, reranker=None, corpus_name="agenttest")

    result = tools["search_code"](query="anything", top_k=10_000)
    # one block per matched chunk, separated by "---" -- count blocks, not chars
    assert result.count("\n\n---\n\n") + 1 == MAX_SEARCH_TOP_K


def test_search_code_tool_result_is_truncated_when_oversized():
    from codeseek.agent.tools import MAX_TOOL_RESULT_CHARS

    store, embedding_service = _make_store_with_many_chunks(n=20, text_size=2000)
    tools = build_tools("demo", Path("."), store, embedding_service, reranker=None, corpus_name="agenttest")

    result = tools["search_code"](query="anything", top_k=20)
    assert result.endswith("[truncated]")
    assert len(result) <= MAX_TOOL_RESULT_CHARS + len("\n... [truncated]")


def test_follow_symbol_tool_exact_match_and_miss():
    store, embedding_service = _make_store_and_service()
    tools = build_tools("demo", Path("."), store, embedding_service, reranker=None, corpus_name="agenttest")

    found = tools["follow_symbol"](name="validate_jwt")
    assert "validate_jwt" in found

    missing = tools["follow_symbol"](name="does_not_exist")
    assert "No symbol named" in missing


def test_follow_symbol_tool_result_is_truncated_when_oversized():
    """A symbol split into many oversized #partN chunks used to have no cap
    at all -- find_by_symbol can return up to 20 matches, each up to ~400
    tokens, with nothing bounding the combined output."""
    from codeseek.agent.tools import MAX_TOOL_RESULT_CHARS

    store = QdrantStore(location=":memory:")
    embedding_service = EmbeddingService({OPENAI: FakeEmbedder()})
    collection = collection_name("agenttest", OPENAI)
    chunks = [
        ChunkPayload(
            repo="demo", language="python", path="big.py", symbol_name=f"BigClass#part{i}",
            symbol_type="class", start_line=i, end_line=i, text="x" * 2000,
        )
        for i in range(20)
    ]
    store.ensure_collection(collection, vector_size=3)
    store.upsert_chunks(collection, chunks, FakeEmbedder().embed([c.text for c in chunks]))
    tools = build_tools("demo", Path("."), store, embedding_service, reranker=None, corpus_name="agenttest")

    result = tools["follow_symbol"](name="BigClass")
    assert result.endswith("[truncated]")
    assert len(result) <= MAX_TOOL_RESULT_CHARS + len("\n... [truncated]")


def test_read_file_tool_full_and_ranged(tmp_path):
    (tmp_path / "mod.py").write_text("line1\nline2\nline3\nline4\n")
    store, embedding_service = _make_store_and_service()
    tools = build_tools("demo", tmp_path, store, embedding_service, reranker=None, corpus_name="agenttest")

    full = tools["read_file"](path="mod.py")
    assert "line1" in full and "line4" in full

    ranged = tools["read_file"](path="mod.py", start_line=2, end_line=3)
    assert "line2" in ranged and "line3" in ranged
    assert "line1" not in ranged and "line4" not in ranged


def test_read_file_tool_rejects_path_escaping_repo_root(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("do not read me")

    store, embedding_service = _make_store_and_service()
    tools = build_tools("demo", repo_root, store, embedding_service, reranker=None, corpus_name="agenttest")

    result = tools["read_file"](path="../secret.txt")
    assert "outside the repository" in result
    assert "do not read me" not in result


def test_read_file_tool_missing_file(tmp_path):
    store, embedding_service = _make_store_and_service()
    tools = build_tools("demo", tmp_path, store, embedding_service, reranker=None, corpus_name="agenttest")

    assert "not found" in tools["read_file"](path="nope.py")


def test_read_file_tool_truncates_oversized_reads(tmp_path):
    (tmp_path / "big.py").write_text("x" * 20000)
    store, embedding_service = _make_store_and_service()
    tools = build_tools("demo", tmp_path, store, embedding_service, reranker=None, corpus_name="agenttest")

    result = tools["read_file"](path="big.py")
    assert "[truncated]" in result
    assert len(result) < 20000


# --- agent loop mechanics (fake streaming OpenAI client, no network) ----


def test_explain_stops_as_soon_as_the_model_answers_without_tool_calls():
    store, embedding_service = _make_store_and_service()
    client = FakeOpenAIClient([content_turn("Direct answer, no tools needed.")])

    result = explain(
        "demo", Path("."), "some question", store, embedding_service, reranker=None,
        client=client, corpus_name="agenttest",
    )

    assert isinstance(result, ExplainResult)
    assert result.answer == "Direct answer, no tools needed."
    assert result.tool_calls == []
    assert len(client.calls) == 1


def test_explain_executes_multiple_tool_calls_across_iterations(tmp_path):
    store, embedding_service = _make_store_and_service()
    client = FakeOpenAIClient([
        tool_call_turn(("call_1", "search_code", '{"query": "jwt"}')),
        tool_call_turn(("call_2", "follow_symbol", '{"name": "validate_jwt"}')),
        content_turn("Found it via two tool calls."),
    ])

    result = explain(
        "demo", tmp_path, "how is jwt validated", store, embedding_service, reranker=None,
        client=client, corpus_name="agenttest",
    )

    assert result.answer == "Found it via two tool calls."
    assert [tc.tool for tc in result.tool_calls] == ["search_code", "follow_symbol"]
    assert len(client.calls) == 3


def test_explain_unknown_tool_name_reported_as_error_not_a_crash():
    store, embedding_service = _make_store_and_service()
    client = FakeOpenAIClient([
        tool_call_turn(("call_1", "delete_everything", "{}")),
        content_turn("Recovered after the bad tool call."),
    ])

    result = explain(
        "demo", Path("."), "question", store, embedding_service, reranker=None,
        client=client, corpus_name="agenttest",
    )

    assert result.answer == "Recovered after the bad tool call."
    assert "unknown tool" in result.tool_calls[0].result_summary.lower()


def test_explain_gives_up_after_max_iterations_of_pure_tool_calls():
    store, embedding_service = _make_store_and_service()
    # every turn calls a tool, never answers -- forces the loop to hit its cap
    turns = [tool_call_turn((f"call_{i}", "search_code", '{"query": "x"}')) for i in range(10)]
    client = FakeOpenAIClient(turns)

    result = explain(
        "demo", Path("."), "question", store, embedding_service, reranker=None,
        client=client, corpus_name="agenttest",
    )

    assert result.answer is None
    assert result.error == "max_iterations_exceeded"
    assert len(result.tool_calls) == 8  # MAX_ITERATIONS, not all 10 available turns consumed
    assert len(client.calls) == 8


# --- explain_stream() event sequence -------------------------------------


def test_explain_stream_emits_status_tool_and_answer_delta_events_in_order(tmp_path):
    from codeseek.agent.loop import explain_stream

    store, embedding_service = _make_store_and_service()
    client = FakeOpenAIClient([
        tool_call_turn(("call_1", "search_code", '{"query": "jwt"}')),
        content_turn("The answer is X."),
    ])

    events = list(explain_stream(
        "demo", tmp_path, "how is jwt validated", store, embedding_service, reranker=None,
        client=client, corpus_name="agenttest",
    ))

    event_types = [e["type"] for e in events]
    assert event_types == [
        "status",       # thinking, before turn 1
        "tool_call",    # search_code
        "tool_result",
        "status",       # thinking, before turn 2
        "status",       # generating, once content starts
        "answer_delta",
        "done",
    ]
    assert events[1]["tool"] == "search_code"
    assert events[1]["input"] == {"query": "jwt"}
    assert events[5]["text"] == "The answer is X."

    done = events[-1]["result"]
    assert isinstance(done, ExplainResult)
    assert done.answer == "The answer is X."
    assert [tc.tool for tc in done.tool_calls] == ["search_code"]


def test_explain_stream_tracks_usage_from_chunk_usage_fields(tmp_path):
    from codeseek.agent.loop import explain_stream
    from tests._openai_fakes import FakeUsage

    store, embedding_service = _make_store_and_service()
    client = FakeOpenAIClient([
        content_turn("Direct answer.", usage=FakeUsage(prompt_tokens=500, completion_tokens=20, cached_tokens=100)),
    ])

    events = list(explain_stream(
        "demo", tmp_path, "question", store, embedding_service, reranker=None,
        client=client, corpus_name="agenttest",
    ))

    result = events[-1]["result"]
    assert result.usage.input_tokens == 500
    assert result.usage.cached_input_tokens == 100
    assert result.usage.output_tokens == 20
    assert result.usage.requests == 1
