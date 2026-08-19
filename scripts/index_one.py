"""Index a single arbitrary repo on demand: clone it, chunk it, embed it,
upsert into the same shared corpus every other repo lives in -- searchable
immediately afterward (filter by repo name in the UI or /search).

Reuses every existing building block (clone_or_update, walk_repo, both
chunkers, both embedders, the Qdrant store) -- this is a thin CLI wrapper
around the same index_repo() call scripts/build_index.py makes per repo.

Usage:
    python scripts/index_one.py https://github.com/psf/black.git
    python scripts/index_one.py https://github.com/psf/black.git --name black
"""

import argparse

from codeseek.config import CORPUS_NAME, QDRANT_PATH, REPOS_DIR, RepoSpec
from codeseek.embedding.registry import build_default_embedders
from codeseek.embedding.service import EmbeddingService
from codeseek.ingestion.clone import clone_or_update, infer_repo_name
from codeseek.observability.timing import configure_json_logging
from codeseek.pipeline.index import index_repo
from codeseek.store.qdrant_store import QdrantStore

configure_json_logging()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="GitHub (or any git) repo URL")
    parser.add_argument("--name", default=None, help="Override the inferred repo name")
    parser.add_argument(
        "--language", default="unknown",
        help="Informational only -- chunkers detect per-file language themselves",
    )
    args = parser.parse_args()

    name = args.name or infer_repo_name(args.url)
    repo = RepoSpec(name, args.url, args.language)

    print(f"Cloning {repo.url} as '{name}'...")
    repo_path = clone_or_update(repo, REPOS_DIR)

    print("Loading embedding models and indexing (this is the slow part)...")
    store = QdrantStore(path=str(QDRANT_PATH))
    embedding_service = EmbeddingService(build_default_embedders())
    total = index_repo(repo_path, repo, store, embedding_service, corpus_name=CORPUS_NAME)

    print(f"\nIndexed {total} chunks from '{name}'.")
    print(f"Search it now with the repo filter set to: {name}")


if __name__ == "__main__":
    main()
