"""The three tools the explain agent can call, plus their OpenAI function-
calling JSON schemas. Tools are built fresh per /explain request via
build_tools(), a factory closure bound to one session's repo -- not global
state, so concurrent requests for different repos never cross-contaminate."""

from pathlib import Path
from typing import Callable

from codeseek.config import CORPUS_NAME
from codeseek.embedding.registry import OPENAI
from codeseek.embedding.service import EmbeddingService
from codeseek.retrieval.hybrid import HybridRetriever, Reranker
from codeseek.store.qdrant_store import QdrantStore
from codeseek.store.schema import collection_name

MAX_READ_CHARS = 8000
MAX_TOOL_RESULT_CHARS = 8000  # applies to search_code / follow_symbol combined output
MAX_SEARCH_TOP_K = 20

# Every loop iteration resends the entire message history, so an uncapped
# tool result doesn't just cost once -- it's paid for again on every
# subsequent turn. Found via a real cost analysis: follow_symbol had no cap
# at all (find_by_symbol can return up to 20 chunks, unbounded total size),
# and search_code's top_k had no upper bound the model couldn't exceed.


def build_tool_specs() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search_code",
                "description": (
                    "Semantic + keyword search over the indexed repository's code and docs. "
                    "Use this first for any question about how something works or where "
                    "something is implemented."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language or code-like search query"},
                        "top_k": {
                            "type": "integer",
                            "description": f"Number of results to return (default 8, max {MAX_SEARCH_TOP_K})",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "follow_symbol",
                "description": (
                    "Look up a function, class, or method by its exact name -- a 'go to "
                    "definition' lookup. Use this when search_code surfaces a reference to a "
                    "symbol you need the actual definition of, or when you already know the "
                    "exact symbol name (e.g. a class name mentioned in another chunk, or a "
                    "specific method like 'ClassName.method_name')."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Exact symbol name, e.g. 'OptionInfo' or 'OptionInfo.__init__'",
                        },
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": (
                    "Read raw source directly from a file in the repository, optionally "
                    "bounded to a line range. Use this to see imports, a class's full "
                    "context, or code that search_code didn't surface as its own chunk."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Repo-relative file path, e.g. 'src/models.py'"},
                        "start_line": {"type": "integer", "description": "1-indexed start line (optional)"},
                        "end_line": {"type": "integer", "description": "1-indexed end line, inclusive (optional)"},
                    },
                    "required": ["path"],
                },
            },
        },
    ]


def _format_chunk(payload: dict) -> str:
    location = f"{payload.get('repo')}/{payload.get('path')}:{payload.get('start_line')}-{payload.get('end_line')}"
    symbol = payload.get("symbol_name", "")
    return f"{location} [{symbol}]\n{payload.get('text', '')}"


def _truncate(text: str, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


def build_tools(
    repo_name: str,
    repo_path: Path,
    store: QdrantStore,
    embedding_service: EmbeddingService,
    reranker: Reranker | None,
    corpus_name: str = CORPUS_NAME,
) -> dict[str, Callable[..., str]]:
    collection = collection_name(corpus_name, OPENAI)
    retriever = HybridRetriever(store, reranker=reranker)
    repo_root = repo_path.resolve()

    def search_code(query: str, top_k: int = 8) -> str:
        top_k = max(1, min(top_k, MAX_SEARCH_TOP_K))
        query_vector = embedding_service.embed_one(OPENAI, query)
        hits = retriever.search(collection, query_vector, query, top_k, repo=repo_name)
        if not hits:
            return "No results found."
        return _truncate("\n\n---\n\n".join(_format_chunk(h.payload) for h in hits))

    def follow_symbol(name: str) -> str:
        hits = store.find_by_symbol(collection, repo_name, name)
        if not hits:
            return f"No symbol named '{name}' found in {repo_name}."
        return _truncate("\n\n---\n\n".join(_format_chunk(h.payload) for h in hits))

    def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        # path is untrusted model output -- confine every read to the repo root.
        full_path = (repo_root / path).resolve()
        if not full_path.is_relative_to(repo_root):
            return f"Error: '{path}' is outside the repository."
        if not full_path.is_file():
            return f"Error: '{path}' not found."

        lines = full_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        start = max(0, (start_line or 1) - 1)
        end = min(len(lines), end_line or len(lines))
        snippet = "\n".join(lines[start:end])
        if len(snippet) > MAX_READ_CHARS:
            snippet = snippet[:MAX_READ_CHARS] + "\n... [truncated]"
        return f"{path}:{start + 1}-{end}\n{snippet}"

    return {"search_code": search_code, "follow_symbol": follow_symbol, "read_file": read_file}
