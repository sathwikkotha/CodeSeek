"""FastAPI app factory: /health and /search. Built as a factory (rather than a
module-level singleton) so tests can inject an in-memory store and fake
embedders instead of the real local models and on-disk Qdrant storage.

The /search route inlines the vector-search / keyword-search / merge steps
(rather than calling HybridRetriever.search as one opaque call) so it can time
each pipeline stage separately for the UI's latency panel."""

import json
import subprocess
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI as OpenAIClient
from pydantic import BaseModel

from codeseek.agent.loop import explain as run_explain
from codeseek.agent.loop import explain_stream as run_explain_stream
from codeseek.api.jobs import JobStore
from codeseek.config import CORPUS_NAME, REPOS_DIR, RepoSpec
from codeseek.embedding.registry import OPENAI
from codeseek.embedding.service import EmbeddingService
from codeseek.ingestion.clone import clone_or_update, infer_repo_name
from codeseek.pipeline.index import index_repo
from codeseek.retrieval.hybrid import Reranker, fetch_k_for, merge_hits, rrf_fuse
from codeseek.store.qdrant_store import QdrantStore, build_metadata_filter
from codeseek.store.schema import collection_name


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    repo: str | None = None
    language: str | None = None


class SearchResultItem(BaseModel):
    id: str
    score: float
    source: str
    repo: str
    path: str
    symbol_name: str
    symbol_type: str
    start_line: int
    end_line: int
    text: str


class StageTimings(BaseModel):
    embed_ms: float
    vector_search_ms: float
    keyword_search_ms: float
    merge_ms: float
    rerank_ms: float


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    timings: StageTimings


class IngestRequest(BaseModel):
    url: str
    name: str | None = None


class IngestJobResponse(BaseModel):
    job_id: str
    state: str  # "pending" | "running" | "done" | "error"
    repo: str | None = None
    chunks_indexed: int | None = None
    error: str | None = None


class RepoInfo(BaseModel):
    name: str
    chunk_count: int


class ReposResponse(BaseModel):
    repos: list[RepoInfo]


class ExplainRequest(BaseModel):
    repo: str
    question: str


class ToolCallInfo(BaseModel):
    tool: str
    input: dict
    result_summary: str


class UsageInfo(BaseModel):
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    requests: int
    cost_usd: float


class CitationCheckInfo(BaseModel):
    citation: str
    path: str
    start_line: int
    end_line: int
    valid: bool
    reason: str | None = None


class ExplainResponse(BaseModel):
    answer: str | None
    tool_calls: list[ToolCallInfo]
    error: str | None = None
    usage: UsageInfo
    citation_checks: list[CitationCheckInfo]


def create_app(
    store: QdrantStore, embedding_service: EmbeddingService, corpus_name: str = CORPUS_NAME,
    reranker: Reranker | None = None, openai_client: OpenAIClient | None = None,
) -> FastAPI:
    app = FastAPI(title="CodeSeek")
    ingest_jobs = JobStore()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/repos", response_model=ReposResponse)
    def list_repos() -> ReposResponse:
        """What's actually searchable right now -- the UI shows this up front
        so 'search the curated corpus' vs. 'search a repo you just added' is
        visibly the same list, not two different concepts."""
        collection = collection_name(corpus_name, OPENAI)
        repos = sorted(store.list_repos(collection), key=lambda r: r[1], reverse=True)
        return ReposResponse(repos=[RepoInfo(name=n, chunk_count=c) for n, c in repos])

    @app.delete("/repos/{name}")
    def delete_repo(name: str) -> dict:
        store.delete_repo(collection_name(corpus_name, OPENAI), name)
        return {"deleted": name}

    @app.post("/ingest", response_model=IngestJobResponse, status_code=202)
    def ingest(req: IngestRequest) -> IngestJobResponse:
        """Clone (or update) an arbitrary repo and index it into the same
        corpus everything else lives in. Returns immediately with a job id --
        the clone+chunk+embed pipeline runs on a background thread instead of
        holding the request open for however long that takes (minutes, for a
        real repo). Poll GET /ingest/{job_id} for progress."""
        name = req.name or infer_repo_name(req.url)

        def do_ingest() -> tuple[str, int]:
            repo = RepoSpec(name, req.url, "unknown")
            try:
                repo_path = clone_or_update(repo, REPOS_DIR)
            except subprocess.CalledProcessError as exc:
                detail = (exc.stderr or str(exc)).strip()
                raise RuntimeError(f"Couldn't clone '{req.url}': {detail}") from exc
            total = index_repo(repo_path, repo, store, embedding_service, corpus_name=corpus_name)
            return name, total

        job = ingest_jobs.create()
        ingest_jobs.run(job.id, do_ingest)
        return IngestJobResponse(job_id=job.id, state=job.state)

    @app.get("/ingest/{job_id}", response_model=IngestJobResponse)
    def ingest_status(job_id: str) -> IngestJobResponse:
        job = ingest_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"No ingest job '{job_id}'.")
        return IngestJobResponse(
            job_id=job.id, state=job.state, repo=job.repo, chunks_indexed=job.chunks_indexed, error=job.error,
        )

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest) -> SearchResponse:
        query_filter = build_metadata_filter(req.repo, req.language)
        collection = collection_name(corpus_name, OPENAI)
        fetch_k = fetch_k_for(req.top_k)

        t0 = time.perf_counter()
        query_vector = embedding_service.embed_one(OPENAI, req.query)
        t1 = time.perf_counter()
        vector_hits = store.vector_search(collection, query_vector, limit=fetch_k, query_filter=query_filter)
        t2 = time.perf_counter()
        keyword_hits = store.keyword_search(collection, req.query, limit=fetch_k, extra_filter=query_filter)
        t3 = time.perf_counter()
        candidates = merge_hits(vector_hits, keyword_hits, fetch_k)
        t4 = time.perf_counter()
        if reranker is not None:
            reranked = reranker.rerank(req.query, candidates, len(candidates))
            hits = rrf_fuse(candidates, reranked, req.top_k)
        else:
            hits = candidates[: req.top_k]
        t5 = time.perf_counter()

        timings = StageTimings(
            embed_ms=(t1 - t0) * 1000, vector_search_ms=(t2 - t1) * 1000,
            keyword_search_ms=(t3 - t2) * 1000, merge_ms=(t4 - t3) * 1000,
            rerank_ms=(t5 - t4) * 1000,
        )
        results = [
            SearchResultItem(
                id=h.id, score=h.score, source=h.source,
                repo=h.payload.get("repo", ""), path=h.payload.get("path", ""),
                symbol_name=h.payload.get("symbol_name", ""), symbol_type=h.payload.get("symbol_type", ""),
                start_line=h.payload.get("start_line", 0), end_line=h.payload.get("end_line", 0),
                text=h.payload.get("text", ""),
            )
            for h in hits
        ]

        return SearchResponse(results=results, timings=timings)

    @app.post("/explain", response_model=ExplainResponse)
    def explain(req: ExplainRequest) -> ExplainResponse:
        """Runs the agent loop (search/follow_symbol/read_file, iterated)
        against a repo already indexed via /ingest, and returns a grounded,
        cited answer -- not just a ranked list of chunks."""
        repo_path = REPOS_DIR / req.repo
        if not repo_path.is_dir():
            raise HTTPException(status_code=404, detail=f"Repo '{req.repo}' not found -- index it via /ingest first.")

        result = run_explain(
            repo_name=req.repo, repo_path=repo_path, question=req.question,
            store=store, embedding_service=embedding_service, reranker=reranker, client=openai_client,
            corpus_name=corpus_name,
        )
        return ExplainResponse(
            answer=result.answer,
            tool_calls=[
                ToolCallInfo(tool=tc.tool, input=tc.input, result_summary=tc.result_summary)
                for tc in result.tool_calls
            ],
            error=result.error,
            usage=UsageInfo(
                input_tokens=result.usage.input_tokens,
                cached_input_tokens=result.usage.cached_input_tokens,
                output_tokens=result.usage.output_tokens,
                requests=result.usage.requests,
                cost_usd=result.usage.cost_usd,
            ),
            citation_checks=[
                CitationCheckInfo(
                    citation=c.citation, path=c.path, start_line=c.start_line, end_line=c.end_line,
                    valid=c.valid, reason=c.reason,
                )
                for c in result.citation_checks
            ],
        )

    @app.post("/explain/stream")
    def explain_stream_route(req: ExplainRequest) -> StreamingResponse:
        """Same agent loop as /explain, but as Server-Sent Events -- lets the
        UI show live status ("searching", "reading a file", "generating")
        and type the answer out as real tokens arrive, instead of blocking
        on one request until the whole answer is done."""
        repo_path = REPOS_DIR / req.repo
        if not repo_path.is_dir():
            raise HTTPException(status_code=404, detail=f"Repo '{req.repo}' not found -- index it via /ingest first.")

        def event_source():
            try:
                for event in run_explain_stream(
                    repo_name=req.repo, repo_path=repo_path, question=req.question,
                    store=store, embedding_service=embedding_service, reranker=reranker, client=openai_client,
                    corpus_name=corpus_name,
                ):
                    if event["type"] == "done":
                        result = event["result"]
                        payload = {
                            "type": "done",
                            "answer": result.answer,
                            "error": result.error,
                            "usage": {
                                "input_tokens": result.usage.input_tokens,
                                "cached_input_tokens": result.usage.cached_input_tokens,
                                "output_tokens": result.usage.output_tokens,
                                "requests": result.usage.requests,
                                "cost_usd": result.usage.cost_usd,
                            },
                            "citation_checks": [
                                {
                                    "citation": c.citation, "path": c.path, "start_line": c.start_line,
                                    "end_line": c.end_line, "valid": c.valid, "reason": c.reason,
                                }
                                for c in result.citation_checks
                            ],
                            "tool_calls": [
                                {"tool": tc.tool, "input": tc.input, "result_summary": tc.result_summary}
                                for tc in result.tool_calls
                            ],
                        }
                    else:
                        payload = event
                    yield f"data: {json.dumps(payload)}\n\n"
            except Exception as exc:
                # Surface as an SSE frame -- an abrupt connection drop looks like a bug, not an error message.
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

        return StreamingResponse(event_source(), media_type="text/event-stream")

    return app
