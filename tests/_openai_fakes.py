"""Fake OpenAI streaming chat-completion objects, shared by test_agent.py and
test_api.py so both can exercise explain()/explain_stream() -- and the
/explain and /explain/stream routes built on them -- without ever hitting
the real API. Mirrors the real SDK's streaming shape closely enough to
exercise the loop's chunk-accumulation logic (tool-call arguments arrive
fragmented across chunks, keyed by `index`), not just its end result.

Not a test_*.py file on purpose -- it's fixtures, not a test suite."""


class FakeToolCallFunction:
    def __init__(self, name: str | None = None, arguments: str | None = None):
        self.name = name
        self.arguments = arguments


class FakeDeltaToolCall:
    def __init__(self, index: int, call_id: str | None = None, name: str | None = None, arguments: str | None = None):
        self.index = index
        self.id = call_id
        self.function = FakeToolCallFunction(name, arguments)


class FakeDelta:
    def __init__(self, content: str | None = None, tool_calls: list[FakeDeltaToolCall] | None = None):
        self.content = content
        self.tool_calls = tool_calls


class FakeChunkChoice:
    def __init__(self, delta: FakeDelta):
        self.delta = delta


class FakeChunk:
    def __init__(self, delta: FakeDelta | None = None, usage=None):
        self.choices = [FakeChunkChoice(delta)] if delta is not None else []
        self.usage = usage


class FakePromptTokensDetails:
    def __init__(self, cached_tokens: int):
        self.cached_tokens = cached_tokens


class FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.prompt_tokens_details = FakePromptTokensDetails(cached_tokens)


def content_turn(text: str, usage: FakeUsage | None = None) -> list[FakeChunk]:
    """A turn where the model answers directly (no tool calls)."""
    chunks = [FakeChunk(delta=FakeDelta(content=text))]
    if usage is not None:
        chunks.append(FakeChunk(usage=usage))
    return chunks


def tool_call_turn(*calls: tuple[str, str, str], usage: FakeUsage | None = None) -> list[FakeChunk]:
    """A turn where the model calls one or more tools. Each call is
    (call_id, tool_name, json_arguments), streamed as an id+name delta
    followed by an arguments delta -- matching the real API's
    fragment-then-fill-in shape closely enough to exercise accumulation."""
    chunks = []
    for index, (call_id, name, arguments) in enumerate(calls):
        chunks.append(FakeChunk(delta=FakeDelta(tool_calls=[FakeDeltaToolCall(index, call_id, name, "")])))
        chunks.append(FakeChunk(delta=FakeDelta(tool_calls=[FakeDeltaToolCall(index, None, None, arguments)])))
    if usage is not None:
        chunks.append(FakeChunk(usage=usage))
    return chunks


class FakeOpenAIClient:
    """client.chat.completions.create(stream=True, ...) returning a scripted
    sequence of chunk streams, one per turn -- self.chat and self.completions
    both alias to self so the real attribute-chasing call shape works
    without nested classes."""

    def __init__(self, turns: list[list[FakeChunk]]):
        self._turns = list(turns)
        self.calls: list[dict] = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self._turns.pop(0))
