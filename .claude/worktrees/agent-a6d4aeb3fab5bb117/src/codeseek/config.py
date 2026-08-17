"""Project-wide configuration: paths and the corpus of repositories to ingest."""

from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
REPOS_DIR = DATA_DIR / "repos"
MANIFEST_PATH = DATA_DIR / "manifest.jsonl"
CHUNKS_DIR = DATA_DIR / "chunks"
QDRANT_PATH = DATA_DIR / "qdrant_storage"

# All repos are indexed into one shared corpus (one pair of collections,
# <CORPUS_NAME>__general / <CORPUS_NAME>__code), filterable by repo/language at query time.
CORPUS_NAME = "codeseek"


@dataclass(frozen=True)
class RepoSpec:
    name: str
    url: str
    language: str  # primary language, used for chunker routing later


# Corpus: real, well-known, permissively-licensed repos across Python and
# JS/TS. Deliberately avoids huge monorepos (django, pytest, nestjs) to keep
# local CPU-only embedding time bounded.
CORPUS: list[RepoSpec] = [
    # Python
    RepoSpec("fastapi", "https://github.com/fastapi/fastapi.git", "python"),
    RepoSpec("httpx", "https://github.com/encode/httpx.git", "python"),
    RepoSpec("pydantic", "https://github.com/pydantic/pydantic.git", "python"),
    RepoSpec("requests", "https://github.com/psf/requests.git", "python"),
    RepoSpec("flask", "https://github.com/pallets/flask.git", "python"),
    RepoSpec("click", "https://github.com/pallets/click.git", "python"),
    RepoSpec("rich", "https://github.com/Textualize/rich.git", "python"),
    RepoSpec("black", "https://github.com/psf/black.git", "python"),
    RepoSpec("attrs", "https://github.com/python-attrs/attrs.git", "python"),
    RepoSpec("starlette", "https://github.com/encode/starlette.git", "python"),
    RepoSpec("httpie", "https://github.com/httpie/cli.git", "python"),
    RepoSpec("sqlmodel", "https://github.com/fastapi/sqlmodel.git", "python"),
    RepoSpec("uvicorn", "https://github.com/encode/uvicorn.git", "python"),
    RepoSpec("poetry-core", "https://github.com/python-poetry/poetry-core.git", "python"),
    # JS/TS
    RepoSpec("axios", "https://github.com/axios/axios.git", "javascript"),
    RepoSpec("express", "https://github.com/expressjs/express.git", "javascript"),
    RepoSpec("lodash", "https://github.com/lodash/lodash.git", "javascript"),
    RepoSpec("chalk", "https://github.com/chalk/chalk.git", "javascript"),
    RepoSpec("commander.js", "https://github.com/tj/commander.js.git", "javascript"),
    RepoSpec("dayjs", "https://github.com/iamkun/dayjs.git", "javascript"),
    RepoSpec("zod", "https://github.com/colinhacks/zod.git", "typescript"),
    RepoSpec("yargs", "https://github.com/yargs/yargs.git", "javascript"),
    RepoSpec("koa", "https://github.com/koajs/koa.git", "javascript"),
    RepoSpec("nanoid", "https://github.com/ai/nanoid.git", "javascript"),
]

# File extensions routed to the AST/code chunker vs. the docs chunker.
CODE_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
DOC_EXTENSIONS = {".md", ".rst"}

# Directories skipped entirely while walking a cloned repo.
IGNORED_DIR_NAMES = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
    ".mypy_cache", ".pytest_cache", "site-packages", ".tox", "coverage",
}

# Locale codes under a "docs/" folder treated as translations to skip (keep
# "docs/en/" or a locale-less docs/). Covers the language set fastapi ships.
NON_ENGLISH_DOC_LOCALES = {
    "de", "es", "fr", "hi", "ja", "ko", "pt", "ru", "tr", "uk", "zh", "zh-hant",
}
