"""AST-aware chunker for Python source: one chunk per top-level function/class,
docstring and decorators included, sliced directly from the source lines."""

import ast
from dataclasses import dataclass

from codeseek.chunking.shared import count_tokens, pack_units

MAX_CHUNK_TOKENS = 400
OVERSIZED_OVERLAP_RATIO = 0.1  # only applies when a single function/class must be split

_SYMBOL_TYPE = {
    ast.FunctionDef: "function",
    ast.AsyncFunctionDef: "async_function",
    ast.ClassDef: "class",
}


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


def chunk_python_source(source: str, repo: str, path: str) -> list[CodeChunk]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines()
    chunks: list[CodeChunk] = []

    for node in ast.iter_child_nodes(tree):
        symbol_type = _SYMBOL_TYPE.get(type(node))
        if symbol_type is None:
            continue

        start_line = node.lineno
        if node.decorator_list:
            start_line = node.decorator_list[0].lineno
        end_line = node.end_lineno

        node_lines = lines[start_line - 1 : end_line]
        text = "\n".join(node_lines)

        if count_tokens(text) <= MAX_CHUNK_TOKENS:
            chunks.append(CodeChunk(
                repo=repo, language="python", path=path,
                symbol_name=node.name, symbol_type=symbol_type,
                start_line=start_line, end_line=end_line, text=text,
            ))
            continue

        # Oversized: the same graceful-degradation logic as the docs chunker --
        # split on line boundaries only, never mid-statement.
        packed = pack_units(node_lines, MAX_CHUNK_TOKENS, OVERSIZED_OVERLAP_RATIO, joiner="\n")
        for part_num, (rel_start, rel_end, piece_text) in enumerate(packed, start=1):
            chunks.append(CodeChunk(
                repo=repo, language="python", path=path,
                symbol_name=f"{node.name}#part{part_num}", symbol_type=symbol_type,
                start_line=start_line + rel_start, end_line=start_line + rel_end,
                text=piece_text,
            ))

    return chunks
