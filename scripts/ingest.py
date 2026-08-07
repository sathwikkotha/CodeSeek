"""CLI entry point for Milestone 1: clone the corpus, walk it, write the manifest.

Usage:
    python scripts/ingest.py
"""

import logging

from codeseek.config import CORPUS, MANIFEST_PATH
from codeseek.ingestion.clone import clone_all
from codeseek.ingestion.manifest import write_manifest
from codeseek.ingestion.walker import walk_repo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    repo_paths = clone_all(CORPUS)
    cloned_by_name = {p.name: p for p in repo_paths}

    def all_records():
        for repo in CORPUS:
            repo_path = cloned_by_name.get(repo.name)
            if repo_path is None:
                logger.warning("Skipping %s: not cloned", repo.name)
                continue
            yield from walk_repo(repo_path, repo.name, repo.language)

    count = write_manifest(all_records(), MANIFEST_PATH)
    logger.info("Wrote %d file records to %s", count, MANIFEST_PATH)


if __name__ == "__main__":
    main()
