"""Shallow-clones (or updates) a repo into data/repos/<name>."""

import logging
import subprocess
from pathlib import Path

from codeseek.config import REPOS_DIR, RepoSpec

logger = logging.getLogger(__name__)


def infer_repo_name(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")


def clone_or_update(repo: RepoSpec, dest_dir: Path = REPOS_DIR) -> Path:
    """Ensure a shallow local copy of `repo` exists under dest_dir, return its path."""
    repo_path = dest_dir / repo.name

    if repo_path.exists():
        logger.info("Updating %s", repo.name)
        subprocess.run(
            ["git", "-C", str(repo_path), "pull", "--depth", "1"],
            check=True, capture_output=True, text=True,
        )
        return repo_path

    logger.info("Cloning %s from %s", repo.name, repo.url)
    dest_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo.url, str(repo_path)],
        check=True, capture_output=True, text=True,
    )
    return repo_path
