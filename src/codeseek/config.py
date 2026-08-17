"""Project-wide configuration: paths and the corpus of repositories to ingest."""

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]

# Loads OPENAI_API_KEY (and anything else) from a .env file at the project
# root into the environment -- imported first by every entry point via this
# module, so this runs before any OpenAI client gets constructed.
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
REPOS_DIR = DATA_DIR / "repos"
MANIFEST_PATH = DATA_DIR / "manifest.jsonl"
CHUNKS_DIR = DATA_DIR / "chunks"
QDRANT_PATH = DATA_DIR / "qdrant_storage"

# Every indexed repo lives in one shared Qdrant store (one collection,
# <CORPUS_NAME>__openai), but retrieval and the explain agent always scope a
# given question to a single repo via a payload filter -- there is
# deliberately no more fixed, curated multi-repo corpus. Index whatever repo
# you want via /ingest; this is just the storage namespace.
CORPUS_NAME = "codeseek"


@dataclass(frozen=True)
class RepoSpec:
    name: str
    url: str
    language: str  # primary language, used for chunker routing later


# File extensions with a dedicated AST (Python) or tree-sitter (JS/TS) parser
# -- these get real symbol-level chunking (one chunk per function/class).
CODE_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
DOC_EXTENSIONS = {".md", ".rst", ".txt", ".adoc"}
NOTEBOOK_EXTENSIONS = {".ipynb"}

# Every other plausible source-code extension: no parser exists for these, so
# they get line-based generic chunking (real, citeable line ranges, but no
# symbol extraction) instead of being dropped outright -- this is what makes
# "index almost any file" true without needing a tree-sitter grammar per
# language.
GENERIC_CODE_EXTENSIONS = {
    ".go", ".rs", ".java", ".kt", ".kts", ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp",
    ".cs", ".rb", ".php", ".swift", ".scala", ".sh", ".bash", ".zsh", ".ps1",
    ".sql", ".html", ".htm", ".css", ".scss", ".less", ".yaml", ".yml", ".json",
    ".toml", ".ini", ".cfg", ".xml", ".proto", ".graphql", ".lua", ".pl", ".r",
    ".ex", ".exs", ".hs", ".clj", ".dart", ".vue", ".svelte", ".groovy", ".m",
    ".vb", ".jl", ".nim", ".zig", ".sol",
}

# Well-known extensionless filenames worth indexing as generic code even
# though they have no suffix to route on.
GENERIC_CODE_FILENAMES = {"Dockerfile", "Makefile", "Rakefile", "Gemfile", "Procfile", "Vagrantfile"}

# Friendly language label per generic extension -- used for the `language`
# payload field (search filtering, syntax highlighting). Falls back to the
# bare extension name for anything not listed here.
GENERIC_LANGUAGE_BY_EXTENSION = {
    ".go": "go", ".rs": "rust", ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".cs": "csharp", ".rb": "ruby", ".php": "php", ".swift": "swift", ".scala": "scala",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".ps1": "powershell",
    ".sql": "sql", ".html": "html", ".htm": "html", ".css": "css", ".scss": "scss",
    ".less": "less", ".yaml": "yaml", ".yml": "yaml", ".json": "json", ".toml": "toml",
    ".ini": "ini", ".cfg": "ini", ".xml": "xml", ".proto": "protobuf", ".graphql": "graphql",
    ".lua": "lua", ".pl": "perl", ".r": "r", ".ex": "elixir", ".exs": "elixir",
    ".hs": "haskell", ".clj": "clojure", ".dart": "dart", ".vue": "vue", ".svelte": "svelte",
    ".groovy": "groovy", ".m": "objective-c", ".vb": "vbnet", ".jl": "julia",
    ".nim": "nim", ".zig": "zig", ".sol": "solidity",
}

# Binary, generated, or otherwise low-value noise -- skipped even though
# ingestion is otherwise permissive by default. A denylist, not an allowlist:
# anything not explicitly recognized above and not matched here still falls
# through to generic-code or docs chunking rather than being dropped.
SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".bmp", ".tiff",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".webm", ".ogg", ".flac",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".zip", ".tar", ".gz", ".tgz", ".7z", ".rar", ".pyc", ".pyo", ".so", ".dll",
    ".dylib", ".exe", ".o", ".a", ".class", ".jar", ".whl", ".pdf",
    ".lock", ".map",
}
SKIP_FILENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "Cargo.lock", "composer.lock", "Gemfile.lock", "Pipfile.lock",
}

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
