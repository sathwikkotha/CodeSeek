"""Harder retrieval benchmark than scripts/self_eval_ground_truth.json (which
the CI gate already saturates at MRR=1.0 -- see docs/eval_results.md's own
"reading this table honestly" section). This one uses more, more naturally
paraphrased questions (scripts/hard_eval_ground_truth.json) against the same
codeseek-self-referential corpus, so there's actual headroom to see whether a
retrieval-side change moves Recall@5/Recall@10/MRR up or down.

Always runs the *production* pipeline shape: vector + keyword hybrid, then
cross-encoder rerank + RRF fusion (HybridRetriever.search with a real
CrossEncoderReranker) -- the same path a real /search or agent tool_call hits.

Usage: python scripts/run_hard_eval.py --label "baseline"
Appends one row to docs/hard_eval_results.md so successive runs (baseline,
after each incremental change) build a comparable history instead of
overwriting each other.
"""

import argparse
import os
import sys
import time

from codeseek.config import ROOT_DIR, RepoSpec
from codeseek.embedding.registry import OPENAI, build_default_embedders
from codeseek.embedding.service import EmbeddingService
from codeseek.eval.ground_truth import load_ground_truth
from codeseek.eval.harness import run_eval
from codeseek.pipeline.index import index_repo
from codeseek.retrieval.hyde import HydeQueryExpander
from codeseek.retrieval.reranker import CrossEncoderReranker
from codeseek.store.qdrant_store import QdrantStore

GROUND_TRUTH_PATH = ROOT_DIR / "scripts" / "hard_eval_ground_truth.json"
CORPUS_NAME = "codeseek_hard_eval"
TOP_K = 10
RESULTS_PATH = ROOT_DIR / "docs" / "hard_eval_results.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run", help="Short label for this row, e.g. 'baseline' or 'after_bm25'")
    parser.add_argument("--note", default="", help="Optional one-line note about what changed for this run")
    parser.add_argument("--hyde", action="store_true", help="Expand each query with HyDE before embedding it")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set -- this needs real embeddings to measure anything real.")
        return 1

    store = QdrantStore(location=":memory:")
    embedding_service = EmbeddingService(build_default_embedders())
    repo = RepoSpec(name="codeseek", url="local", language="python")
    repo_path = ROOT_DIR / "src" / "codeseek"

    total = index_repo(repo_path, repo, store, embedding_service, corpus_name=CORPUS_NAME)
    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)

    reranker = CrossEncoderReranker()
    expander = HydeQueryExpander().expand if args.hyde else None
    result = run_eval(
        ground_truth, store, embedding_service, CORPUS_NAME, [OPENAI],
        top_k=TOP_K, reranker=reranker, query_expander=expander,
    )[0]

    print(f"Indexed {total} chunks, {result.num_questions} questions.")
    print(f"Recall@5={result.recall_at_5:.3f}  Recall@10={result.recall_at_10:.3f}  MRR={result.mrr:.3f}")

    row = (
        f"| {time.strftime('%Y-%m-%d %H:%M')} | {args.label} | {result.recall_at_5:.3f} | "
        f"{result.recall_at_10:.3f} | {result.mrr:.3f} | {args.note} |"
    )

    header = (
        "# Hard-eval retrieval results\n\n"
        f"Production pipeline (vector + keyword hybrid + cross-encoder rerank + RRF fusion) measured "
        f"against `scripts/hard_eval_ground_truth.json` (28 harder, naturally-paraphrased questions about "
        f"CodeSeek's own source, {total} chunks). Reproduce a row with "
        f"`python scripts/run_hard_eval.py --label <name> --note <what changed>`.\n\n"
        "| Timestamp | Label | Recall@5 | Recall@10 | MRR | Note |\n"
        "|---|---|---|---|---|---|\n"
    )

    if RESULTS_PATH.exists():
        existing = RESULTS_PATH.read_text(encoding="utf-8")
        if existing.strip().startswith("# Hard-eval"):
            RESULTS_PATH.write_text(existing.rstrip("\n") + "\n" + row + "\n", encoding="utf-8")
        else:
            RESULTS_PATH.write_text(header + row + "\n", encoding="utf-8")
    else:
        RESULTS_PATH.write_text(header + row + "\n", encoding="utf-8")

    print(f"\nAppended row to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
