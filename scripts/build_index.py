"""CLI entry point for Milestone 4: index every cloned repo in the corpus
into the real on-disk Qdrant store, using the real local embedding models.

Usage:
    python scripts/ingest.py       # clone/update repos first
    python scripts/build_index.py  # then index them
"""

import logging

from codeseek.config import CORPUS, CORPUS_NAME, QDRANT_PATH, REPOS_DIR
from codeseek.embedding.registry import build_default_embedders
from codeseek.embedding.service import EmbeddingService
from codeseek.observability.timing import configure_json_logging
from codeseek.pipeline.index import index_repo
from codeseek.store.qdrant_store import QdrantStore

configure_json_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    store = QdrantStore(path=str(QDRANT_PATH))
    embedding_service = EmbeddingService(build_default_embedders())

    for repo in CORPUS:
        repo_path = REPOS_DIR / repo.name
        if not repo_path.exists():
            logger.warning("Skipping %s: not cloned (run scripts/ingest.py first)", repo.name)
            continue
        index_repo(repo_path, repo, store, embedding_service, corpus_name=CORPUS_NAME)


if __name__ == "__main__":
    main()
