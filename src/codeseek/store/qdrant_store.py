"""Thin wrapper around QdrantClient: collection setup (cosine distance + a
full-text payload index for keyword-hybrid search), batched upsert, and the
two raw search primitives the retrieval layer merges."""

from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from codeseek.store.schema import ChunkPayload


@dataclass(frozen=True)
class SearchHit:
    id: str
    score: float
    payload: dict


def build_metadata_filter(repo: str | None, language: str | None) -> models.Filter | None:
    conditions = []
    if repo:
        conditions.append(models.FieldCondition(key="repo", match=models.MatchValue(value=repo)))
    if language:
        conditions.append(models.FieldCondition(key="language", match=models.MatchValue(value=language)))
    return models.Filter(must=conditions) if conditions else None


class QdrantStore:
    def __init__(
        self, path: str | None = None, url: str | None = None, location: str | None = None,
        timeout: float | None = None,
    ):
        """path: on-disk local storage dir (dev default). url: containerized Qdrant
        (load-test phase). location=':memory:' for ephemeral in-process storage (tests).
        timeout: seconds before a url-mode request gives up -- a real production
        knob, not just a test convenience; a hung connection is worse than a fast, clear error."""
        if url:
            self._client = QdrantClient(url=url, timeout=timeout)
        elif path:
            self._client = QdrantClient(path=path)
        else:
            self._client = QdrantClient(location=location or ":memory:")

    def ensure_collection(self, name: str, vector_size: int) -> None:
        if self._client.collection_exists(name):
            return
        self._client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        self._client.create_payload_index(
            collection_name=name, field_name="text", field_schema=models.PayloadSchemaType.TEXT,
        )
        self._client.create_payload_index(
            collection_name=name, field_name="symbol_name", field_schema=models.PayloadSchemaType.TEXT,
        )

    def upsert_chunks(self, collection: str, chunks: list[ChunkPayload], vectors: list[list[float]]) -> None:
        points = [
            models.PointStruct(id=chunk.point_id(), vector=vector, payload=chunk.as_payload_dict())
            for chunk, vector in zip(chunks, vectors)
        ]
        self._client.upsert(collection_name=collection, points=points)

    def vector_search(
        self, collection: str, query_vector: list[float], limit: int, query_filter: models.Filter | None = None
    ) -> list[SearchHit]:
        if not self._client.collection_exists(collection):
            return []
        results = self._client.query_points(
            collection_name=collection, query=query_vector, limit=limit, query_filter=query_filter,
        )
        return [SearchHit(id=str(p.id), score=p.score, payload=p.payload or {}) for p in results.points]

    def list_repos(self, collection: str, limit: int = 200) -> list[tuple[str, int]]:
        """Distinct repo names currently stored in `collection`, with point
        counts -- powers the UI's 'what's actually searchable' list."""
        if not self._client.collection_exists(collection):
            return []
        result = self._client.facet(collection_name=collection, key="repo", limit=limit, exact=True)
        return [(hit.value, hit.count) for hit in result.hits]

    def delete_repo(self, collection: str, repo: str) -> None:
        """Remove every point for `repo` from `collection` -- used before a
        clean re-index (e.g. after a chunker fix) so stale points from the
        old chunk boundaries don't linger alongside the new ones."""
        if not self._client.collection_exists(collection):
            return
        self._client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(key="repo", match=models.MatchValue(value=repo))])
            ),
        )

    def find_by_symbol(self, collection: str, repo: str, name: str, limit: int = 20) -> list[SearchHit]:
        """Exact-or-split-part match on symbol_name within one repo -- the
        agent's 'go to definition' tool: bypasses embedding uncertainty
        entirely when the caller already knows the symbol name. Matches a
        bare name, an oversized-chunk split ("Name#part2"), or a method of a
        class ("Name.method") -- same rule the eval harness uses to decide
        whether a retrieved chunk answers a question about "Name"."""
        if not self._client.collection_exists(collection):
            return []
        prefilter = models.Filter(
            must=[
                models.FieldCondition(key="repo", match=models.MatchValue(value=repo)),
                models.FieldCondition(key="symbol_name", match=models.MatchText(text=name)),
            ]
        )
        points, _ = self._client.scroll(collection_name=collection, scroll_filter=prefilter, limit=200)
        matches = [
            p
            for p in points
            if (symbol_name := (p.payload or {}).get("symbol_name", "")) == name
            or symbol_name.startswith(f"{name}#part")
            or symbol_name.startswith(f"{name}.")
        ]
        return [SearchHit(id=str(p.id), score=0.0, payload=p.payload or {}) for p in matches[:limit]]

    def keyword_search(
        self, collection: str, query_text: str, limit: int, extra_filter: models.Filter | None = None
    ) -> list[SearchHit]:
        """Pure payload filter match, no vector involved -- Qdrant scores filter-only
        matches as 1.0, which the retrieval layer replaces with the vector floor score."""
        if not self._client.collection_exists(collection):
            return []
        text_filter = models.Filter(
            should=[
                models.FieldCondition(key="text", match=models.MatchText(text=query_text)),
                models.FieldCondition(key="symbol_name", match=models.MatchText(text=query_text)),
            ]
        )
        must = [text_filter, extra_filter] if extra_filter else [text_filter]
        points, _ = self._client.scroll(
            collection_name=collection, scroll_filter=models.Filter(must=must), limit=limit,
        )
        return [SearchHit(id=str(p.id), score=0.0, payload=p.payload or {}) for p in points]
