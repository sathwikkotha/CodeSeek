"""Extracts Jupyter notebook (.ipynb) cells as chunks -- source code and
markdown, never execution outputs (often huge base64 image/data blobs with no
semantic value for code Q&A). A notebook is JSON, not a text file with stable
per-cell line numbers the way .py source is, so chunks carry no citeable line
range (start_line=end_line=0) -- the same simplification the docs chunker
already makes for prose chunks -- and are addressed by cell position
instead."""

import json
from dataclasses import dataclass

from codeseek.chunking.shared import count_tokens, pack_units

MAX_CHUNK_TOKENS = 400
OVERSIZED_OVERLAP_RATIO = 0.1

_CELL_TYPE_TO_SYMBOL_TYPE = {"code": "notebook_code", "markdown": "notebook_markdown"}


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


def chunk_notebook_source(source: str, repo: str, path: str) -> list[CodeChunk]:
    try:
        notebook = json.loads(source)
    except json.JSONDecodeError:
        return []
    if not isinstance(notebook, dict):
        return []

    chunks: list[CodeChunk] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if not isinstance(cell, dict):
            continue
        symbol_type = _CELL_TYPE_TO_SYMBOL_TYPE.get(cell.get("cell_type"))
        if symbol_type is None:
            continue

        raw_source = cell.get("source", [])
        text = ("".join(raw_source) if isinstance(raw_source, list) else str(raw_source)).strip()
        if not text:
            continue

        symbol_name = f"cell[{index}]:{cell['cell_type']}"
        if count_tokens(text) <= MAX_CHUNK_TOKENS:
            chunks.append(CodeChunk(
                repo=repo, language="notebook", path=path, symbol_name=symbol_name,
                symbol_type=symbol_type, start_line=0, end_line=0, text=text,
            ))
            continue

        packed = pack_units(text.split("\n"), MAX_CHUNK_TOKENS, OVERSIZED_OVERLAP_RATIO, joiner="\n")
        chunks.extend(
            CodeChunk(
                repo=repo, language="notebook", path=path, symbol_name=f"{symbol_name}#part{part_num}",
                symbol_type=symbol_type, start_line=0, end_line=0, text=piece_text,
            )
            for part_num, (_, _, piece_text) in enumerate(packed, start=1)
        )

    return chunks
