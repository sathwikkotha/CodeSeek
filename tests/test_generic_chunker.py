from codeseek.chunking.generic_chunker import chunk_generic_source


def test_chunk_generic_source_returns_empty_list_for_empty_input():
    assert chunk_generic_source("", repo="demo", path="main.go", language="go") == []


def test_chunk_generic_source_produces_a_single_chunk_for_small_file():
    source = "package main\n\nfunc main() {\n\tprintln(\"hi\")\n}\n"
    chunks = chunk_generic_source(source, repo="demo", path="main.go", language="go")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.repo == "demo"
    assert chunk.language == "go"
    assert chunk.path == "main.go"
    assert chunk.symbol_name == "block#0"
    assert chunk.symbol_type == "block"
    assert chunk.start_line == 1
    assert chunk.end_line == source.count("\n")
    assert "func main" in chunk.text


def test_chunk_generic_source_line_ranges_are_accurate_and_contiguous():
    lines = [f"line{i}" for i in range(1, 21)]
    source = "\n".join(lines)

    chunks = chunk_generic_source(source, repo="demo", path="data.yaml", language="yaml")

    assert chunks[0].start_line == 1
    for chunk in chunks:
        # every line the chunk claims to span is actually present in its text
        assert lines[chunk.start_line - 1] in chunk.text
        assert lines[chunk.end_line - 1] in chunk.text


def test_chunk_generic_source_splits_oversized_file_into_multiple_blocks():
    source = "\n".join(f"x = {i}" for i in range(2000))  # far beyond the 400-token cap
    chunks = chunk_generic_source(source, repo="demo", path="big.sql", language="sql")

    assert len(chunks) > 1
    assert [c.symbol_name for c in chunks] == [f"block#{i}" for i in range(len(chunks))]
