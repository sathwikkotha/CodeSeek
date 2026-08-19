"""Shallow-clones (or updates) a repo into data/repos/<name>."""

import logging
import subprocess
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

from codeseek.config import REPOS_DIR, RepoSpec

logger = logging.getLogger(__name__)


def infer_repo_name(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")


# A bad URL and a dropped connection both raise CalledProcessError -- retrying
# a bad URL just delays the same failure by a few seconds, which is an
# acceptable cost for not needing to parse git's stderr to tell them apart.
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8), reraise=True)
def _run_git(args: list[str]) -> None:
    subprocess.run(args, check=True, capture_output=True, text=True)


def clone_or_update(repo: RepoSpec, dest_dir: Path = REPOS_DIR) -> Path:
    """Ensure a shallow local copy of `repo` exists under dest_dir, return its path."""
    repo_path = dest_dir / repo.name

    if repo_path.exists():
        logger.info("Updating %s", repo.name)
        _run_git(["git", "-C", str(repo_path), "pull", "--depth", "1"])
        return repo_path

    logger.info("Cloning %s from %s", repo.name, repo.url)
    dest_dir.mkdir(parents=True, exist_ok=True)
    _run_git(["git", "clone", "--depth", "1", repo.url, str(repo_path)])
    return repo_path
