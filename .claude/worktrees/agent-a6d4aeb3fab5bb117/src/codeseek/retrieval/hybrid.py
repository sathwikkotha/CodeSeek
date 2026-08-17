"""Merges vector search and keyword search results: a keyword hit is added
additively at the weakest vector score seen, never used to outrank or remove
a genuine semantic match. This is why a literal symbol name like 'JWTBearer'
can surface even though its embedding alone wouldn't win on cosine similarity."""

from dataclasses import dataclass, replace

from codeseek.store.qdrant_store import QdrantStore, SearchHit, build_metadata_filter


@dataclass(frozen=True)
class MergedHit:
    id: str
    score: float
    payload: dict
    source: str  # "vector" | "keyword" | "both"


def merge_hits(vector_hits: list[SearchHit], keyword_hits: list[SearchHit], top_k: int) -> list[MergedHit]:
    by_id: dict[str, MergedHit] = {
        hit.id: MergedHit(id=hit.id, score=hit.score, payload=hit.payload, source="vector")
        for hit in vector_hits
    }

    floor_score = min((hit.score for hit in vector_hits), default=0.0)

    for hit in keyword_hits:
        existing = by_id.get(hit.id)
        if existing is None:
            by_id[hit.id] = MergedHit(id=hit.id, score=floor_score, payload=hit.payload, source="keyword")
        else:
            by_id[hit.id] = replace(existing, source="both")

    merged = sorted(by_id.values(), key=lambda h: h.score, reverse=True)
    return merged[:top_k]


class HybridRetriever:
    def __init__(self, store: QdrantStore):
        self._store = store

    def search(
        self, collection: str, query_vector: list[float], query_text: str, top_k: int,
        repo: str | None = None, language: str | None = None,
    ) -> list[MergedHit]:
        query_filter = build_metadata_filter(repo, language)
        vector_hits = self._store.vector_search(collection, query_vector, limit=top_k, query_filter=query_filter)
        keyword_hits = self._store.keyword_search(collection, query_text, limit=top_k, extra_filter=query_filter)
        return merge_hits(vector_hits, keyword_hits, top_k)
