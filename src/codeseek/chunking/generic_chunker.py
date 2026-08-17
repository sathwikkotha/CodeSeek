"""Line-based fallback chunker for source-code files with no dedicated AST or
tree-sitter parser (Go, Rust, Java, YAML, SQL, HTML, config files, and dozens
more) -- packs raw lines into token-capped chunks with real, accurate line
ranges. No symbol extraction: chunks are addressed by position (`block#N`),
not by function/class name, since there's no parser here to find one."""

from dataclasses import dataclass

from codeseek.chunking.shared import pack_units

MAX_CHUNK_TOKENS = 400
OVERLAP_RATIO = 0.1


@dataclass(frozen=True)
class CodeChunk:
    repo: str
    language: str
    path: str
    symbol_name: str
    symbol_type: str
    start_line: int
    end_line: int
    text: str


def chunk_generic_source(source: str, repo: str, path: str, language: str) -> list[CodeChunk]:
    lines = source.splitlines()
    if not lines:
        return []

    packed = pack_units(lines, MAX_CHUNK_TOKENS, OVERLAP_RATIO, joiner="\n")
    return [
        CodeChunk(
            repo=repo, language=language, path=path, symbol_name=f"block#{i}",
            symbol_type="block", start_line=start + 1, end_line=end + 1, text=text,
        )
        for i, (start, end, text) in enumerate(packed)
    ]
