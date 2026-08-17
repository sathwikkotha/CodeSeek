"""Walks a cloned repo's file tree and routes each file to 'code', 'docs', or skips it."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Literal

from codeseek.config import CODE_EXTENSIONS, DOC_EXTENSIONS, IGNORED_DIR_NAMES, NON_ENGLISH_DOC_LOCALES

Category = Literal["code", "docs"]


@dataclass(frozen=True)
class FileRecord:
    repo: str
    language: str
    category: Category
    path: str  # relative to repo root
    size_bytes: int


def _route(path: Path) -> Category | None:
    if path.suffix in CODE_EXTENSIONS:
        return "code"
    if path.suffix in DOC_EXTENSIONS or path.name.upper().startswith("README"):
        return "docs"
    return None


def _is_translated_doc(path: Path) -> bool:
    """A 'docs/<locale>/...' folder (fastapi, and many i18n'd OSS repos) holds
    the same content translated into ~20 languages -- redundant for retrieval
    and a large multiple of the real indexing cost. Keep 'docs/en/' (or a
    locale-less docs/), skip the rest."""
    parts = path.parts
    for i, part in enumerate(parts[:-1]):
        if part == "docs" and parts[i + 1] in NON_ENGLISH_DOC_LOCALES:
            return True
    return False


def walk_repo(repo_path: Path, repo_name: str, language: str) -> Iterator[FileRecord]:
    """Yield a FileRecord for every code/docs file in repo_path, skipping ignored dirs."""
    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIR_NAMES for part in path.parts):
            continue
        if _is_translated_doc(path.relative_to(repo_path)):
            continue

        category = _route(path)
        if category is None:
            continue

        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size == 0:
            continue

        yield FileRecord(
            repo=repo_name,
            language=language,
            category=category,
            path=path.relative_to(repo_path).as_posix(),
            size_bytes=size,
        )


def file_record_to_dict(record: FileRecord) -> dict:
    return asdict(record)
