"""Ablation study: measure Recall@5/Recall@10/MRR at each retrieval stage --
vector-only -> + keyword hybrid -> + cross-encoder rerank -> + RRF fusion
(the production default, HybridRetriever.search) -- against the same
self-eval ground truth the CI eval gate checks (scripts/self_eval_ground_truth.json,
questions about CodeSeek's own source).

This exists to turn the "measured before/after" claims that justify each
retrieval technique (in code comments across retrieval/hybrid.py) into one
reproducible, checked-in table instead of scattered anecdotes from whenever
each piece was built. Writes docs/eval_results.md.

Requires OPENAI_API_KEY (real embedding calls -- a few cents). Downloads the
local cross-encoder model on first run (no API cost, one-time).
"""

import os
import sys
import time
from pathlib import Path

from codeseek.config import ROOT_DIR, RepoSpec
from codeseek.embedding.registry import OPENAI, build_default_embedders
from codeseek.embedding.service import EmbeddingService
from codeseek.eval.ground_truth import GroundTruthItem, load_ground_truth
from codeseek.eval.metrics import recall_at_k, reciprocal_rank
from codeseek.pipeline.index import index_repo
from codeseek.retrieval.hybrid import MergedHit, fetch_k_for, merge_hits, rrf_fuse
from codeseek.retrieval.reranker import CrossEncoderReranker
from codeseek.store.qdrant_store import QdrantStore
from codeseek.store.schema import collection_name

GROUND_TRUTH_PATH = ROOT_DIR / "scripts" / "self_eval_ground_truth.json"
CORPUS_NAME = "codeseek_ablation"
TOP_K = 10
RESULTS_PATH = ROOT_DIR / "docs" / "eval_results.md"


def _is_relevant(payload: dict, item: GroundTruthItem) -> bool:
    symbol_name = payload.get("symbol_name", "")
    name_matches = (
        symbol_name == item.symbol_name
        or symbol_name.startswith(f"{item.symbol_name}#part")
        or symbol_name.startswith(f"{item.symbol_name}.")
    )
    return payload.get("repo") == item.repo and payload.get("path") == item.path and name_matches


def _score(per_question_hits: list[list], ground_truth: list[GroundTruthItem]) -> tuple[float, float, float]:
    recall5, recall10, rr = [], [], []
    for item, hits in zip(ground_truth, per_question_hits, strict=True):
        ranked_ids = [h.id for h in hits]
        relevant_ids = {h.id for h in hits if _is_relevant(h.payload, item)}
        recall5.append(recall_at_k(ranked_ids, relevant_ids, 5))
        recall10.append(recall_at_k(ranked_ids, relevant_ids, 10))
        rr.append(reciprocal_rank(ranked_ids, relevant_ids))
    n = len(ground_truth)
    return sum(recall5) / n, sum(recall10) / n, sum(rr) / n


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set -- ablation needs real embeddings to measure anything real.")
        return 1

    store = QdrantStore(location=":memory:")
    embedding_service = EmbeddingService(build_default_embedders())
    repo = RepoSpec(name="codeseek", url="local", language="python")
    repo_path = ROOT_DIR / "src" / "codeseek"

    total = index_repo(repo_path, repo, store, embedding_service, corpus_name=CORPUS_NAME)
    print(f"Indexed {total} chunks from codeseek's own source.\n")

    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)
    collection = collection_name(CORPUS_NAME, OPENAI)
    reranker = CrossEncoderReranker()
    fetch_k = fetch_k_for(TOP_K)

    results: dict[str, tuple[float, float, float]] = {}
    merged_by_question: list[list[MergedHit]] = []

    # Stage 1: vector-only.
    vector_hits = []
    for item in ground_truth:
        qvec = embedding_service.embed_one(OPENAI, item.question)
        vector_hits.append(store.vector_search(collection, qvec, limit=TOP_K))
    results["1. Vector-only (bi-encoder cosine similarity)"] = _score(vector_hits, ground_truth)

    # Stage 2: + keyword hybrid (merge_hits, includes doc-overview damping).
    hybrid_hits = []
    for item in ground_truth:
        qvec = embedding_service.embed_one(OPENAI, item.question)
        vhits = store.vector_search(collection, qvec, limit=fetch_k)
        khits = store.keyword_search(collection, item.question, limit=fetch_k)
        merged = merge_hits(vhits, khits, fetch_k)
        merged_by_question.append(merged)
        hybrid_hits.append(merged[:TOP_K])
    results["2. + keyword hybrid (vector + lexical merge)"] = _score(hybrid_hits, ground_truth)

    # Stage 3: + cross-encoder rerank, no RRF (trust the reranker's order outright).
    rerank_hits = []
    reranked_by_question = []
    for item, merged in zip(ground_truth, merged_by_question, strict=True):
        reranked = reranker.rerank(item.question, merged, len(merged))
        reranked_by_question.append(reranked)
        rerank_hits.append(reranked[:TOP_K])
    results["3. + cross-encoder rerank (no fusion)"] = _score(rerank_hits, ground_truth)

    # Stage 4: + RRF fusion of pre/post-rerank order -- the production default
    # (this is exactly what HybridRetriever.search does).
    rrf_hits = []
    for merged, reranked in zip(merged_by_question, reranked_by_question, strict=True):
        rrf_hits.append(rrf_fuse(merged, reranked, TOP_K))
    results["4. + RRF fusion (production default)"] = _score(rrf_hits, ground_truth)

    lines = [
        "# Retrieval ablation results",
        "",
        f"Measured {time.strftime('%Y-%m-%d')} against CodeSeek's own source tree "
        f"({total} chunks, {len(ground_truth)} ground-truth questions -- see "
        "`scripts/self_eval_ground_truth.json`). Reproduce with `python scripts/run_ablation.py`.",
        "",
        "Each row adds one stage on top of the row above it -- this is the actual pipeline in "
        "`retrieval/hybrid.py`, measured stage by stage, not four independent systems.",
        "",
        "| Stage | Recall@5 | Recall@10 | MRR |",
        "|---|---|---|---|",
    ]
    for name, (r5, r10, mrr) in results.items():
        lines.append(f"| {name} | {r5:.3f} | {r10:.3f} | {mrr:.3f} |")
    lines.append("")

    report = "\n".join(lines)
    print(report)
    Path(RESULTS_PATH).write_text(report, encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
