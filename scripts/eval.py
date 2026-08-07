"""CLI entry point for Milestone 6: run the ground-truth question set against
the real indexed corpus and report Recall@5, Recall@10, and MRR per embedding
model. Writes data/eval_results.json for the Streamlit eval dashboard.

Usage:
    python scripts/eval.py
"""

import json
import logging
from dataclasses import asdict

from codeseek.config import CORPUS_NAME, DATA_DIR, QDRANT_PATH
from codeseek.embedding.registry import CODE, GENERAL, build_default_embedders
from codeseek.embedding.service import EmbeddingService
from codeseek.eval.ground_truth import load_ground_truth
from codeseek.eval.harness import run_eval
from codeseek.store.qdrant_store import QdrantStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.json"
RESULTS_PATH = DATA_DIR / "eval_results.json"


def main() -> None:
    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)
    store = QdrantStore(path=str(QDRANT_PATH))
    embedding_service = EmbeddingService(build_default_embedders())

    results = run_eval(ground_truth, store, embedding_service, CORPUS_NAME, [GENERAL, CODE])

    print(f"{'model':<10}{'recall@5':>10}{'recall@10':>11}{'mrr':>8}{'n':>5}")
    for r in results:
        print(f"{r.model_key:<10}{r.recall_at_5:>10.2f}{r.recall_at_10:>11.2f}{r.mrr:>8.2f}{r.num_questions:>5}")

    RESULTS_PATH.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    logger.info("Wrote %s", RESULTS_PATH)


if __name__ == "__main__":
    main()
