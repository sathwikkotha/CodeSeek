"""Walks a cloned repo's file tree and routes each file to 'code' (AST/tree-
sitter chunking), 'docs', 'notebook', 'code_generic' (line-based chunking for
any other plausible source file), or skips it."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Literal

from codeseek.config import (
    CODE_EXTENSIONS,
    DOC_EXTENSIONS,
    GENERIC_CODE_EXTENSIONS,
    GENERIC_CODE_FILENAMES,
    IGNORED_DIR_NAMES,
    NON_ENGLISH_DOC_LOCALES,
    NOTEBOOK_EXTENSIONS,
    SKIP_EXTENSIONS,
    SKIP_FILENAMES,
)

Category = Literal["code", "docs", "notebook", "code_generic"]


@dataclass(frozen=True)
class FileRecord:
    repo: str
    language: str
    category: Category
    path: str  # relative to repo root
    size_bytes: int


def _route(path: Path) -> Category | None:
    name = path.name
    suffix = path.suffix

    if name in SKIP_FILENAMES or suffix in SKIP_EXTENSIONS or name.endswith((".min.js", ".min.css")):
        return None
    if suffix in NOTEBOOK_EXTENSIONS:
        return "notebook"
    if suffix in DOC_EXTENSIONS or name.upper().startswith("README"):
        return "docs"
    if suffix in CODE_EXTENSIONS:
        return "code"
    if suffix in GENERIC_CODE_EXTENSIONS or name in GENERIC_CODE_FILENAMES:
        return "code_generic"
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
