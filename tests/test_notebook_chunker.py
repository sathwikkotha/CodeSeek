import json

from codeseek.chunking.notebook_chunker import chunk_notebook_source


def _notebook(cells: list[dict]) -> str:
    return json.dumps({"cells": cells, "metadata": {}, "nbformat": 4})


def test_chunk_notebook_source_extracts_code_and_markdown_cells():
    source = _notebook([
        {"cell_type": "markdown", "source": ["# Title\n", "Some prose.\n"]},
        {"cell_type": "code", "source": ["import pandas as pd\n", "df = pd.read_csv('x.csv')\n"]},
    ])

    chunks = chunk_notebook_source(source, repo="demo", path="nb.ipynb")

    assert len(chunks) == 2
    assert chunks[0].symbol_type == "notebook_markdown"
    assert chunks[0].symbol_name == "cell[0]:markdown"
    assert "Title" in chunks[0].text
    assert chunks[1].symbol_type == "notebook_code"
    assert chunks[1].symbol_name == "cell[1]:code"
    assert "pandas" in chunks[1].text
    assert all(c.language == "notebook" for c in chunks)
    assert all(c.start_line == 0 and c.end_line == 0 for c in chunks)


def test_chunk_notebook_source_skips_execution_outputs_and_empty_cells():
    source = _notebook([
        {
            "cell_type": "code",
            "source": ["1 + 1\n"],
            "outputs": [{"data": {"image/png": "base64garbage" * 1000}}],
        },
        {"cell_type": "code", "source": []},  # empty cell, nothing to index
        {"cell_type": "raw", "source": ["not code or markdown\n"]},  # unsupported cell type
    ])

    chunks = chunk_notebook_source(source, repo="demo", path="nb.ipynb")

    assert len(chunks) == 1
    assert chunks[0].text == "1 + 1"
    assert all("base64garbage" not in c.text for c in chunks)


def test_chunk_notebook_source_returns_empty_list_for_malformed_json():
    assert chunk_notebook_source("not json at all {{{", repo="demo", path="nb.ipynb") == []
    assert chunk_notebook_source('"just a string"', repo="demo", path="nb.ipynb") == []


def test_chunk_notebook_source_splits_oversized_cell_into_parts():
    huge_code = "\n".join(f"x{i} = {i}" for i in range(2000))
    source = _notebook([{"cell_type": "code", "source": [huge_code]}])

    chunks = chunk_notebook_source(source, repo="demo", path="nb.ipynb")

    assert len(chunks) > 1
    assert chunks[0].symbol_name == "cell[0]:code#part1"
    assert chunks[1].symbol_name == "cell[0]:code#part2"
