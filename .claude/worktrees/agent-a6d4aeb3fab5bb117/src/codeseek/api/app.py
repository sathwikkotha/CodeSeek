"""FastAPI app factory: /health and /search. Built as a factory (rather than a
module-level singleton) so tests can inject an in-memory store and fake
embedders instead of the real local models and on-disk Qdrant storage.

The /search route inlines the vector-search / keyword-search / merge steps
(rather than calling HybridRetriever.search as one opaque call) so it can time
each pipeline stage separately for the UI's latency panel."""

import time

from fastapi import FastAPI
from pydantic import BaseModel

from codeseek.config import CORPUS_NAME
from codeseek.embedding.service import EmbeddingService
from codeseek.retrieval.hybrid import merge_hits
from codeseek.store.qdrant_store import QdrantStore, build_metadata_filter
from codeseek.store.schema import collection_name


class SearchRequest(BaseModel):
    query: str
    models: list[str] = ["general", "code"]
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


class SearchResponse(BaseModel):
    results_by_model: dict[str, list[SearchResultItem]]
    timings_by_model: dict[str, StageTimings]


def create_app(store: QdrantStore, embedding_service: EmbeddingService, corpus_name: str = CORPUS_NAME) -> FastAPI:
    app = FastAPI(title="CodeSeek")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest) -> SearchResponse:
        results_by_model: dict[str, list[SearchResultItem]] = {}
        timings_by_model: dict[str, StageTimings] = {}
        query_filter = build_metadata_filter(req.repo, req.language)

        for model_key in req.models:
            collection = collection_name(corpus_name, model_key)

            t0 = time.perf_counter()
            query_vector = embedding_service.embed_one(model_key, req.query)
            t1 = time.perf_counter()
            vector_hits = store.vector_search(collection, query_vector, limit=req.top_k, query_filter=query_filter)
            t2 = time.perf_counter()
            keyword_hits = store.keyword_search(collection, req.query, limit=req.top_k, extra_filter=query_filter)
            t3 = time.perf_counter()
            hits = merge_hits(vector_hits, keyword_hits, req.top_k)
            t4 = time.perf_counter()

            timings_by_model[model_key] = StageTimings(
                embed_ms=(t1 - t0) * 1000, vector_search_ms=(t2 - t1) * 1000,
                keyword_search_ms=(t3 - t2) * 1000, merge_ms=(t4 - t3) * 1000,
            )
            results_by_model[model_key] = [
                SearchResultItem(
                    id=h.id, score=h.score, source=h.source,
                    repo=h.payload.get("repo", ""), path=h.payload.get("path", ""),
                    symbol_name=h.payload.get("symbol_name", ""), symbol_type=h.payload.get("symbol_type", ""),
                    start_line=h.payload.get("start_line", 0), end_line=h.payload.get("end_line", 0),
                    text=h.payload.get("text", ""),
                )
                for h in hits
            ]

        return SearchResponse(results_by_model=results_by_model, timings_by_model=timings_by_model)

    return app
