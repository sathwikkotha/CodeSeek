"""OpenAI tool-calling agent loop: given a question and a repo already
indexed, iteratively calls search_code/follow_symbol/read_file until it has
enough grounded evidence to answer, then writes a cited answer -- or says
plainly that it couldn't find one, rather than guessing.

This is the direct fix for the two structural gaps found in the pure-
retrieval pipeline: one-shot search can't chase a reference across chunks,
and returning raw chunks isn't an answer. The loop is a single agent calling
tools repeatedly (the Claude Code / Codex pattern), not multiple agents --
simpler, cheaper, and there's no clear win from splitting roles here."""

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from codeseek.agent.tools import build_tool_specs, build_tools
from codeseek.agent.verify import CitationCheck, verify_citations
from codeseek.config import CORPUS_NAME
from codeseek.embedding.service import EmbeddingService
from codeseek.retrieval.hybrid import Reranker
from codeseek.store.qdrant_store import QdrantStore

MODEL = "gpt-5.4-mini"
MAX_ITERATIONS = 8
MAX_COMPLETION_TOKENS = 4000


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: float
    cached_input_per_million: float
    output_per_million: float


# USD per 1M tokens, verified live against platform.openai.com/docs/pricing
# (2026-08-11) -- not estimated. Cached input is priced separately (a ~90%
# discount vs. regular input on this model family) and only applies to input
# tokens OpenAI recognizes as a repeat of a recently-seen prefix -- exactly
# what this loop produces every turn, since each iteration resends the prior
# turns' messages verbatim plus one addition.
PRICING = {
    "gpt-5.4-mini": ModelPricing(input_per_million=0.75, cached_input_per_million=0.075, output_per_million=4.50),
    "gpt-5.4-nano": ModelPricing(input_per_million=0.20, cached_input_per_million=0.02, output_per_million=1.25),
}

GROUNDING_SYSTEM_PROMPT = """You are a code-explanation agent working on exactly one repository: {repo_name}.

You have three tools: search_code (semantic + keyword search), follow_symbol \
(exact lookup by name), and read_file (raw source read). Use them to gather \
evidence before answering -- do not answer from general knowledge about what \
a library like this "usually" does; this repository's actual source is what \
you're being asked about.

Rules:
- Every factual claim about the code must be backed by a tool call you made \
this turn. Cite the source of each claim as `repo/path:start-end` immediately \
after the claim.
- If your tools don't turn up enough to answer confidently, say plainly "I \
couldn't find this in the indexed code" rather than guessing.
- Prefer follow_symbol when you already know the exact name of what you're \
looking for -- it's a precise lookup, not a similarity guess.
- Keep the final answer focused and specific to what was asked."""


@dataclass
class ToolCallRecord:
    tool: str
    input: dict
    result_summary: str


@dataclass
class Usage:
    """Cumulative token usage across every model call this question made --
    each iteration is billed as a full request (the whole growing message
    history is resent every turn), so this is the real cost driver, not the
    single-turn max_completion_tokens cap. cached_input_tokens is the subset
    of input_tokens OpenAI billed at the (much cheaper) cached-input rate --
    tracking it separately is what makes cost_usd() the real bill instead of
    a worst-case ceiling that assumes zero caching."""

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    model: str = MODEL

    def add(self, response_usage) -> None:
        if response_usage is None:
            return
        self.input_tokens += response_usage.prompt_tokens
        self.output_tokens += response_usage.completion_tokens
        self.requests += 1
        details = getattr(response_usage, "prompt_tokens_details", None)
        if details is not None:
            self.cached_input_tokens += getattr(details, "cached_tokens", None) or 0

    @property
    def cost_usd(self) -> float:
        price = PRICING.get(self.model)
        if price is None:
            return 0.0
        uncached = max(0, self.input_tokens - self.cached_input_tokens)
        return (
            uncached / 1_000_000 * price.input_per_million
            + self.cached_input_tokens / 1_000_000 * price.cached_input_per_million
            + self.output_tokens / 1_000_000 * price.output_per_million
        )


@dataclass
class ExplainResult:
    answer: str | None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    error: str | None = None
    usage: Usage = field(default_factory=Usage)
    citation_checks: list[CitationCheck] = field(default_factory=list)

    @property
    def all_citations_valid(self) -> bool:
        return all(c.valid for c in self.citation_checks)


def explain_stream(
    repo_name: str,
    repo_path: Path,
    question: str,
    store: QdrantStore,
    embedding_service: EmbeddingService,
    reranker: Reranker | None,
    client: OpenAI | None = None,
    corpus_name: str = CORPUS_NAME,
    model: str = MODEL,
) -> Iterator[dict]:
    """Same agent loop as explain(), but yields progress events as they
    happen instead of returning one final result once everything is done --
    this is what lets a UI show live "searching / reading / generating"
    status and type the answer out token-by-token, instead of one blocking
    spinner. Each model call is a real OpenAI stream; tool-call arguments
    arrive fragmented across chunks (by `index`) and are accumulated before
    a tool is actually run.

    Event shapes:
      {"type": "status", "phase": "thinking" | "generating"}
      {"type": "tool_call", "tool": str, "input": dict}
      {"type": "tool_result", "tool": str, "summary": str}
      {"type": "answer_delta", "text": str}
      {"type": "done", "result": ExplainResult}
    """
    client = client or OpenAI()
    tool_specs = build_tool_specs()
    dispatch = build_tools(repo_name, repo_path, store, embedding_service, reranker, corpus_name=corpus_name)

    messages: list[dict] = [
        {"role": "system", "content": GROUNDING_SYSTEM_PROMPT.format(repo_name=repo_name)},
        {"role": "user", "content": question},
    ]
    tool_calls: list[ToolCallRecord] = []
    usage = Usage(model=model)

    for _ in range(MAX_ITERATIONS):
        yield {"type": "status", "phase": "thinking"}

        # messages/tools are built as plain dicts (not the SDK's TypedDicts) so this
        # loop can grow the history generically -- structurally identical at runtime,
        # cast here rather than threading TypedDict types through the whole loop.
        stream = client.chat.completions.create(
            model=model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            tools=cast(list[ChatCompletionToolParam], tool_specs),
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            stream=True, stream_options={"include_usage": True},
        )

        content_parts: list[str] = []
        calls_acc: dict[int, dict] = {}
        announced_generating = False

        for chunk in stream:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage.add(chunk_usage)

            choices = getattr(chunk, "choices", None)
            if not choices:
                continue  # the trailing usage-only chunk has no choices

            delta = choices[0].delta
            if getattr(delta, "content", None):
                if not announced_generating:
                    yield {"type": "status", "phase": "generating"}
                    announced_generating = True
                content_parts.append(delta.content)
                yield {"type": "answer_delta", "text": delta.content}

            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    entry = calls_acc.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
                    if tc.id:
                        entry["id"] = tc.id
                    if tc.function is not None:
                        if tc.function.name:
                            entry["name"] += tc.function.name
                        if tc.function.arguments:
                            entry["arguments"] += tc.function.arguments

        content = "".join(content_parts) or None

        if not calls_acc:
            citation_checks = verify_citations(content, repo_name, repo_path)
            result = ExplainResult(answer=content, tool_calls=tool_calls, usage=usage, citation_checks=citation_checks)
            yield {"type": "done", "result": result}
            return

        ordered_calls = [calls_acc[i] for i in sorted(calls_acc)]
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in ordered_calls
            ],
        })

        for c in ordered_calls:
            try:
                arguments = json.loads(c["arguments"]) if c["arguments"] else {}
            except json.JSONDecodeError:
                arguments = {}

            yield {"type": "tool_call", "tool": c["name"], "input": arguments}

            handler = dispatch.get(c["name"])
            if handler is None:
                result_text = f"Error: unknown tool '{c['name']}'"
            else:
                try:
                    result_text = handler(**arguments)
                except Exception as exc:  # tool errors are recoverable -- feed back to the model, don't crash the loop
                    result_text = f"Error: {exc}"

            record = ToolCallRecord(tool=c["name"], input=arguments, result_summary=result_text[:200])
            tool_calls.append(record)
            yield {"type": "tool_result", "tool": c["name"], "summary": record.result_summary}
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": result_text})

    result = ExplainResult(answer=None, tool_calls=tool_calls, error="max_iterations_exceeded", usage=usage)
    yield {"type": "done", "result": result}


def explain(
    repo_name: str,
    repo_path: Path,
    question: str,
    store: QdrantStore,
    embedding_service: EmbeddingService,
    reranker: Reranker | None,
    client: OpenAI | None = None,
    corpus_name: str = CORPUS_NAME,
    model: str = MODEL,
) -> ExplainResult:
    """Blocking wrapper over explain_stream() for callers that just want the
    final answer (tests, scripts, non-streaming API callers) -- drains the
    generator and returns the result from its terminal "done" event."""
    result: ExplainResult | None = None
    for event in explain_stream(
        repo_name, repo_path, question, store, embedding_service, reranker,
        client=client, corpus_name=corpus_name, model=model,
    ):
        if event["type"] == "done":
            result = event["result"]
    assert result is not None, "explain_stream always yields a 'done' event before returning"
    return result
