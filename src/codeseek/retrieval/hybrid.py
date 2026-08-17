"""Merges vector search and keyword search results: a keyword hit is added
additively at the weakest vector score seen, never used to outrank or remove
a genuine semantic match. This is why a literal symbol name like 'JWTBearer'
can surface even though its embedding alone wouldn't win on cosine similarity."""

from dataclasses import dataclass, replace
from typing import Protocol

from codeseek.store.qdrant_store import QdrantStore, SearchHit, build_metadata_filter

# A README's first chunk ("doc#0") is conventionally a generic project overview --
# broad enough to score deceptively high against almost any question about that
# repo, beating genuinely specific answers (code or a later, narrower doc chunk).
# Diagnosed via live eval on a freshly-indexed repo, then measured before/after:
# damping ONLY doc#0 (not every doc chunk) improved both implementation-seeking
# questions (code MRR 0.250 -> 0.312) AND doc-appropriate questions (code MRR
# 0.167 -> 0.250) -- damping every doc chunk helped the former at the cost of the
# latter, which is why the target is this narrow.
DOC_OVERVIEW_DAMPING = 0.85

# Damping only re-ranks whatever candidates were already fetched -- it can't
# promote a genuinely-better 11th-ranked chunk that a plain top-k fetch never
# retrieved in the first place. Over-fetching a wider pool before damping and
# truncating to the caller's real top_k afterward lets it do that; measured
# directly (offline replay): general-model MRR on doc-appropriate questions
# rose from 0.188 (fetch == top_k) toward the 0.237 a 30-candidate pool gave.
def fetch_k_for(top_k: int) -> int:
    return max(top_k * 3, 30)


@dataclass(frozen=True)
class MergedHit:
    id: str
    score: float
    payload: dict
    source: str  # "vector" | "keyword" | "both"


class Reranker(Protocol):
    """Structural type -- CrossEncoderReranker (retrieval/reranker.py) satisfies
    this just by having a matching method, so this module never has to import
    that one (which would import MergedHit from here -- a cycle)."""

    def rerank(self, query: str, hits: list[MergedHit], top_k: int) -> list[MergedHit]: ...


def _is_generic_doc_overview(payload: dict) -> bool:
    return payload.get("symbol_type") == "doc" and payload.get("symbol_name") == "doc#0"


def _damped_score(score: float, payload: dict) -> float:
    return score * DOC_OVERVIEW_DAMPING if _is_generic_doc_overview(payload) else score


# The cross-encoder reranker (retrieval/reranker.py) is trained on natural-
# language passage ranking (MS MARCO), so it's systematically biased toward
# prose: given a natural-language question, a README paragraph "reads like"
# a good answer even when a sparse code chunk (e.g. a dataclass with no
# docstring) is the actually-correct one. Verified directly on a real query
# ("how does typer group multiple subcommands") -- the target class ranked
# #2 out of 30 candidates by vector+keyword score, and the reranker alone
# pushed it to #23, burying a genuinely strong match under several doc
# chunks. Fusing the reranker's ordering with the pre-rerank ordering by
# rank (not raw score -- the two scales aren't comparable) lets the
# reranker's opinion count without letting it fully override a candidate
# that scored strongly before it ever saw the query.
RRF_K = 60


def rrf_fuse(pre_rerank: list[MergedHit], post_rerank: list[MergedHit], top_k: int) -> list[MergedHit]:
    pre_rank = {hit.id: i for i, hit in enumerate(pre_rerank)}
    post_rank = {hit.id: i for i, hit in enumerate(post_rerank)}
    by_id = {hit.id: hit for hit in pre_rerank}
    by_id.update({hit.id: hit for hit in post_rerank})

    def rrf_score(hit_id: str) -> float:
        score = 0.0
        if hit_id in pre_rank:
            score += 1.0 / (RRF_K + pre_rank[hit_id])
        if hit_id in post_rank:
            score += 1.0 / (RRF_K + post_rank[hit_id])
        return score

    fused = sorted(by_id.values(), key=lambda h: rrf_score(h.id), reverse=True)
    return [replace(h, score=rrf_score(h.id)) for h in fused[:top_k]]


def merge_hits(vector_hits: list[SearchHit], keyword_hits: list[SearchHit], top_k: int) -> list[MergedHit]:
    by_id: dict[str, MergedHit] = {
        hit.id: MergedHit(id=hit.id, score=_damped_score(hit.score, hit.payload), payload=hit.payload, source="vector")
        for hit in vector_hits
    }

    floor_score = min((hit.score for hit in vector_hits), default=0.0)

    for hit in keyword_hits:
        existing = by_id.get(hit.id)
        if existing is None:
            by_id[hit.id] = MergedHit(
                id=hit.id, score=_damped_score(floor_score, hit.payload), payload=hit.payload, source="keyword",
            )
        else:
            by_id[hit.id] = replace(existing, source="both")

    merged = sorted(by_id.values(), key=lambda h: h.score, reverse=True)
    return merged[:top_k]


class HybridRetriever:
    def __init__(self, store: QdrantStore, reranker: Reranker | None = None):
        self._store = store
        self._reranker = reranker

    def search(
        self, collection: str, query_vector: list[float], query_text: str, top_k: int,
        repo: str | None = None, language: str | None = None,
    ) -> list[MergedHit]:
        query_filter = build_metadata_filter(repo, language)
        fetch_k = fetch_k_for(top_k)
        vector_hits = self._store.vector_search(collection, query_vector, limit=fetch_k, query_filter=query_filter)
        keyword_hits = self._store.keyword_search(collection, query_text, limit=fetch_k, extra_filter=query_filter)
        # Keep the wider fetch_k pool through the merge -- narrowing to top_k
        # happens after reranking (or immediately, if there's no reranker),
        # never before it, so reranking has real candidates to promote.
        candidates = merge_hits(vector_hits, keyword_hits, fetch_k)

        if self._reranker is not None:
            # Rerank the full pool (not truncated to top_k yet) so RRF has a
            # complete post-rerank ordering to fuse against.
            reranked = self._reranker.rerank(query_text, candidates, len(candidates))
            return rrf_fuse(candidates, reranked, top_k)
        return candidates[:top_k]
