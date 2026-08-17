from codeseek.embedding.context import embedding_text, split_identifier
from codeseek.store.schema import ChunkPayload


def test_split_identifier_splits_camel_case():
    assert split_identifier("ArgumentInfo") == "Argument Info"
    assert split_identifier("TyperGroup") == "Typer Group"


def test_split_identifier_splits_snake_case():
    assert split_identifier("parse_csv") == "parse csv"


def test_split_identifier_handles_acronyms():
    assert split_identifier("HTTPClient") == "HTTP Client"


def test_split_identifier_strips_oversized_chunk_suffix():
    assert split_identifier("ArgumentInfo#part1") == "Argument Info"


def _chunk(**overrides) -> ChunkPayload:
    defaults = dict(
        repo="typer", language="python", path="typer/models.py", symbol_name="ArgumentInfo",
        symbol_type="class", start_line=1, end_line=10, text="class ArgumentInfo(ParameterInfo):\n    ...",
    )
    defaults.update(overrides)
    return ChunkPayload(**defaults)


def test_embedding_text_prepends_split_identifier_and_path_for_code_chunks():
    text = embedding_text(_chunk())
    assert text.startswith("class ArgumentInfo (Argument Info) in typer/models.py\n")
    assert "class ArgumentInfo(ParameterInfo):" in text


def test_embedding_text_leaves_doc_chunks_unchanged():
    chunk = _chunk(symbol_name="doc#0", symbol_type="doc", path="README.md", text="# Typer\n\nTyper is a library.")
    assert embedding_text(chunk) == chunk.text
