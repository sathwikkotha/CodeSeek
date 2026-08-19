"""tree-sitter-based chunker for JS/TS source: one chunk per top-level
function/class (including `export`-wrapped and `const foo = () => {}` arrow
function declarations), mirroring code_chunker.py's Python approach exactly --
same CodeChunk shape, same oversized-node splitting via pack_units."""

from dataclasses import dataclass

import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Node, Parser

from codeseek.chunking.shared import count_tokens, pack_units

MAX_CHUNK_TOKENS = 400
OVERSIZED_OVERLAP_RATIO = 0.1

_LANGUAGE_BY_SUFFIX = {
    ".js": ("javascript", Language(tsjavascript.language())),
    ".jsx": ("javascript", Language(tsjavascript.language())),
    ".mjs": ("javascript", Language(tsjavascript.language())),
    ".cjs": ("javascript", Language(tsjavascript.language())),
    ".ts": ("typescript", Language(tstypescript.language_typescript())),
    ".tsx": ("typescript", Language(tstypescript.language_tsx())),
}

_FUNCTION_VALUE_TYPES = {"arrow_function", "function_expression"}


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


def _declarator_symbol(node: Node) -> tuple[str, str] | None:
    """A single `const foo = () => {}` / `var foo = function() {}` declarator."""
    declarators = [c for c in node.named_children if c.type == "variable_declarator"]
    if len(declarators) != 1:
        return None
    declarator = declarators[0]
    value = declarator.child_by_field_name("value")
    name = declarator.child_by_field_name("name")
    if value is None or name is None or name.text is None or value.type not in _FUNCTION_VALUE_TYPES:
        return None
    return name.text.decode("utf-8"), "function"


def _node_name(name: Node | None) -> str:
    if name is None or name.text is None:
        return "anonymous"
    return name.text.decode("utf-8")


def _classify(node: Node) -> tuple[str, str] | None:
    if node.type == "function_declaration":
        return _node_name(node.child_by_field_name("name")), "function"
    if node.type == "class_declaration":
        return _node_name(node.child_by_field_name("name")), "class"
    if node.type in ("lexical_declaration", "variable_declaration"):
        return _declarator_symbol(node)
    return None


def chunk_js_ts_source(source: str, repo: str, path: str, suffix: str) -> list[CodeChunk]:
    langinfo = _LANGUAGE_BY_SUFFIX.get(suffix)
    if langinfo is None:
        return []
    language_name, ts_language = langinfo

    parser = Parser(ts_language)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    lines = source.splitlines()
    chunks: list[CodeChunk] = []

    for node in tree.root_node.named_children:
        slice_node = node
        target = node

        if node.type == "export_statement":
            decl = node.child_by_field_name("declaration")
            if decl is not None:
                target = decl
            else:
                value = node.child_by_field_name("value")
                if value is not None and value.type in _FUNCTION_VALUE_TYPES | {"class"}:
                    default_classified: tuple[str, str] = ("default", "class" if value.type == "class" else "function")
                    chunks.extend(_emit(slice_node, default_classified, lines, repo, language_name, path))
                continue

        classified: tuple[str, str] | None = _classify(target)
        if classified is None:
            continue
        chunks.extend(_emit(slice_node, classified, lines, repo, language_name, path))

    return chunks


def _emit(
    node: Node, classified: tuple[str, str], lines: list[str], repo: str, language: str, path: str,
) -> list[CodeChunk]:
    symbol_name, symbol_type = classified
    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    node_lines = lines[start_line - 1 : end_line]
    text = "\n".join(node_lines)

    if count_tokens(text) <= MAX_CHUNK_TOKENS:
        return [CodeChunk(
            repo=repo, language=language, path=path, symbol_name=symbol_name, symbol_type=symbol_type,
            start_line=start_line, end_line=end_line, text=text,
        )]

    packed = pack_units(node_lines, MAX_CHUNK_TOKENS, OVERSIZED_OVERLAP_RATIO, joiner="\n")
    return [
        CodeChunk(
            repo=repo, language=language, path=path, symbol_name=f"{symbol_name}#part{part_num}",
            symbol_type=symbol_type, start_line=start_line + rel_start, end_line=start_line + rel_end,
            text=piece_text,
        )
        for part_num, (rel_start, rel_end, piece_text) in enumerate(packed, start=1)
    ]
