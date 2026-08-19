"""CI quality gate for retrieval, run against CodeSeek's own source tree.

Indexes src/codeseek itself (real chunker, real embedder, an in-memory Qdrant
store) and checks that hybrid retrieval still finds the right file/symbol for
a fixed set of questions *about this project's own code* -- see
self_eval_ground_truth.json. Exits non-zero if Recall@5 or MRR drops below the
committed baseline.

This is what turns "we measured retrieval quality once, in a code comment"
into a tested contract: a PR that quietly regresses ranking (a bad reranker
tweak, a chunking change that splits a symbol differently) fails CI the same
way a broken unit test would, instead of shipping unnoticed.

Requires OPENAI_API_KEY (real embedding calls against a few hundred chunks --
a few cents per run). Skips cleanly, exit 0, when the key isn't set, so a
fork's PR (no access to repo secrets) doesn't fail CI on something it can't run.
"""

import os
import sys

from codeseek.config import ROOT_DIR, RepoSpec
from codeseek.embedding.registry import OPENAI, build_default_embedders
from codeseek.embedding.service import EmbeddingService
from codeseek.eval.ground_truth import load_ground_truth
from codeseek.eval.harness import run_eval
from codeseek.pipeline.index import index_repo
from codeseek.store.qdrant_store import QdrantStore

GROUND_TRUTH_PATH = ROOT_DIR / "scripts" / "self_eval_ground_truth.json"
CORPUS_NAME = "codeseek_self_eval"

# Baseline measured 2026-08-19 against this exact ground-truth set: hybrid
# retrieval (no rerank -- run_eval doesn't pass one) scored Recall@5=1.000,
# MRR=1.000 on all 10 questions (see docs/eval_results.md for the full,
# stage-by-stage ablation). Thresholds sit below that perfect score to leave
# room for legitimate embedding-model version drift without being so loose
# the gate can't catch a real regression.
MIN_RECALL_AT_5 = 0.90
MIN_MRR = 0.85


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set -- skipping eval gate (nothing to grade it against).")
        return 0

    store = QdrantStore(location=":memory:")
    embedding_service = EmbeddingService(build_default_embedders())
    repo = RepoSpec(name="codeseek", url="local", language="python")
    repo_path = ROOT_DIR / "src" / "codeseek"

    total = index_repo(repo_path, repo, store, embedding_service, corpus_name=CORPUS_NAME)
    print(f"Indexed {total} chunks from codeseek's own source ({repo_path}).")

    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)
    result = run_eval(ground_truth, store, embedding_service, CORPUS_NAME, [OPENAI])[0]

    print(
        f"Recall@5={result.recall_at_5:.3f}  Recall@10={result.recall_at_10:.3f}  "
        f"MRR={result.mrr:.3f}  (n={result.num_questions})"
    )

    if result.recall_at_5 < MIN_RECALL_AT_5 or result.mrr < MIN_MRR:
        print(f"FAIL: below committed baseline (Recall@5 >= {MIN_RECALL_AT_5}, MRR >= {MIN_MRR}).")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
