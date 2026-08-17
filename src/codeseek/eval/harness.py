"""Runs the ground-truth question set through retrieval for each embedding
model and reports Recall@5, Recall@10, and MRR -- the real measurement this
project is built to produce, not a claim.

Deliberately does NOT filter by the ground-truth item's own repo: a real user
asking 'where is JWT validation implemented' doesn't tell the system which
repo to look in, so narrowing the search to the known-correct repo would
inflate recall relative to what a real query experiences.
"""

from dataclasses import dataclass

from codeseek.embedding.service import EmbeddingService
from codeseek.eval.ground_truth import GroundTruthItem
from codeseek.eval.metrics import reciprocal_rank, recall_at_k
from codeseek.retrieval.hybrid import HybridRetriever, Reranker
from codeseek.store.qdrant_store import QdrantStore
from codeseek.store.schema import collection_name


@dataclass(frozen=True)
class EvalResult:
    model_key: str
    recall_at_5: float
    recall_at_10: float
    mrr: float
    num_questions: int


def _is_relevant(payload: dict, item: GroundTruthItem) -> bool:
    # An oversized function/class is split into "Name#part1", "Name#part2", ...
    # by both chunkers (code_chunker.py, js_ts_chunker.py) -- any part counts
    # as finding the target symbol, not just an exact match on the bare name.
    # A ground-truth item targeting a class also counts a retrieved method of
    # that class ("ClassName.method_name[#partN]") as relevant -- since
    # method-level chunking (code_chunker.py) split classes into an overview
    # chunk plus one chunk per method, a question about the class is often
    # best answered by one specific method, not the overview.
    symbol_name = payload.get("symbol_name", "")
    name_matches = (
        symbol_name == item.symbol_name
        or symbol_name.startswith(f"{item.symbol_name}#part")
        or symbol_name.startswith(f"{item.symbol_name}.")
    )
    return payload.get("repo") == item.repo and payload.get("path") == item.path and name_matches


def run_eval(
    ground_truth: list[GroundTruthItem],
    store: QdrantStore,
    embedding_service: EmbeddingService,
    corpus_name: str,
    model_keys: list[str],
    top_k: int = 10,
    reranker: Reranker | None = None,
) -> list[EvalResult]:
    retriever = HybridRetriever(store, reranker=reranker)
    results: list[EvalResult] = []

    for model_key in model_keys:
        collection = collection_name(corpus_name, model_key)
        recall5, recall10, rr = [], [], []

        for item in ground_truth:
            query_vector = embedding_service.embed_one(model_key, item.question)
            hits = retriever.search(collection, query_vector, item.question, top_k=top_k)
            ranked_ids = [h.id for h in hits]
            relevant_ids = {h.id for h in hits if _is_relevant(h.payload, item)}

            recall5.append(recall_at_k(ranked_ids, relevant_ids, 5))
            recall10.append(recall_at_k(ranked_ids, relevant_ids, 10))
            rr.append(reciprocal_rank(ranked_ids, relevant_ids))

        n = len(ground_truth)
        results.append(EvalResult(
            model_key=model_key,
            recall_at_5=sum(recall5) / n,
            recall_at_10=sum(recall10) / n,
            mrr=sum(rr) / n,
            num_questions=n,
        ))

    return results
