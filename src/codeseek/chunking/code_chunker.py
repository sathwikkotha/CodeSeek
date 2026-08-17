"""AST-aware chunker for Python source: one chunk per top-level function, plus
-- for a top-level class -- one chunk per method (individually addressable and
retrievable) and one "overview" chunk (class header, docstring, field
declarations, and every method's signature line, but not method bodies, which
live in their own chunks). Everything is sliced directly from the source
lines, never synthesized.

Method-level chunking exists because a whole-class chunk was found to be the
single biggest ceiling on retrieval quality: a class with 15 methods used to
become one chunk (or, once oversized, arbitrary token-count-sliced pieces
that could cut a method in half), so a question about one specific method
could never retrieve just that method. It also directly helps a failure mode
found via live eval: dataclass-style classes with no docstring (e.g. typer's
`OptionInfo`) embed almost no semantic signal as a raw full-body dump, but
their `__init__` parameter list -- now its own dedicated chunk -- and their
overview (which includes every field declaration and method signature) both
carry far more of the meaning a natural-language question is actually
looking for.
"""

import ast
from dataclasses import dataclass

from codeseek.chunking.shared import count_tokens, pack_units

MAX_CHUNK_TOKENS = 400
OVERSIZED_OVERLAP_RATIO = 0.1  # only applies when a single unit must be split

_METHOD_TYPE = {
    ast.FunctionDef: "method",
    ast.AsyncFunctionDef: "async_method",
}
_TOP_LEVEL_TYPE = {
    ast.FunctionDef: "function",
    ast.AsyncFunctionDef: "async_function",
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


def _line_span(node: ast.AST) -> tuple[int, int]:
    start = node.lineno
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        start = decorators[0].lineno
    return start, node.end_lineno


def _is_docstring(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _signature_span(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[int, int]:
    """Just the 'def name(...):' header (and any decorators) -- the body has
    its own dedicated chunk, so the overview only needs to advertise it."""
    start, end = _line_span(node)
    if not node.body:
        return start, end
    header_end = node.body[0].lineno - 1
    return start, max(header_end, start)


def _emit(
    repo: str, path: str, symbol_name: str, symbol_type: str, start_line: int, end_line: int, text: str,
) -> list[CodeChunk]:
    if count_tokens(text) <= MAX_CHUNK_TOKENS:
        return [CodeChunk(
            repo=repo, language="python", path=path, symbol_name=symbol_name,
            symbol_type=symbol_type, start_line=start_line, end_line=end_line, text=text,
        )]

    packed = pack_units(text.split("\n"), MAX_CHUNK_TOKENS, OVERSIZED_OVERLAP_RATIO, joiner="\n")
    return [
        CodeChunk(
            repo=repo, language="python", path=path, symbol_name=f"{symbol_name}#part{part_num}",
            symbol_type=symbol_type, start_line=start_line + rel_start, end_line=start_line + rel_end,
            text=piece_text,
        )
        for part_num, (rel_start, rel_end, piece_text) in enumerate(packed, start=1)
    ]


def _render_spans(spans: list[tuple[int, int]], lines: list[str]) -> str:
    """Join selected (1-indexed, inclusive) line ranges into one excerpt,
    merging overlapping/adjacent ones and marking real gaps so the result
    doesn't read as continuous source when it isn't (a method signature
    immediately followed by another signature, body elided)."""
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    blocks = ["\n".join(lines[start - 1 : end]) for start, end in merged]
    return "\n    ...\n".join(blocks)


def _chunk_class(node: ast.ClassDef, lines: list[str], repo: str, path: str) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    class_start, class_end = _line_span(node)
    overview_spans: list[tuple[int, int]] = []

    header_end = node.body[0].lineno - 1 if node.body else class_end
    overview_spans.append((class_start, max(header_end, class_start)))

    body = list(node.body)
    if body and _is_docstring(body[0]):
        overview_spans.append(_line_span(body[0]))
        body = body[1:]

    for member in body:
        member_type = type(member)
        if member_type in _METHOD_TYPE:
            overview_spans.append(_signature_span(member))
            m_start, m_end = _line_span(member)
            m_text = "\n".join(lines[m_start - 1 : m_end])
            chunks.extend(_emit(
                repo, path, f"{node.name}.{member.name}", _METHOD_TYPE[member_type], m_start, m_end, m_text,
            ))
        elif isinstance(member, (ast.Assign, ast.AnnAssign)):
            # class-level field declarations (dataclass-style attributes) --
            # exactly the content that carries the most meaning for a class
            # with no docstring and no methods beyond __init__.
            overview_spans.append(_line_span(member))
        # nested classes and other statement kinds are left out of the
        # overview and aren't separately chunked -- a known, rare gap.

    overview_text = _render_spans(overview_spans, lines)
    chunks.extend(_emit(repo, path, node.name, "class", class_start, class_end, overview_text))
    return chunks


def chunk_python_source(source: str, repo: str, path: str) -> list[CodeChunk]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines()
    chunks: list[CodeChunk] = []

    for node in ast.iter_child_nodes(tree):
        if type(node) is ast.ClassDef:
            chunks.extend(_chunk_class(node, lines, repo, path))
            continue

        symbol_type = _TOP_LEVEL_TYPE.get(type(node))
        if symbol_type is None:
            continue

        start_line, end_line = _line_span(node)
        text = "\n".join(lines[start_line - 1 : end_line])
        chunks.extend(_emit(repo, path, node.name, symbol_type, start_line, end_line, text))

    return chunks
