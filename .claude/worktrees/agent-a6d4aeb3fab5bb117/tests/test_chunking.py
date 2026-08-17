from codeseek.chunking.code_chunker import chunk_python_source
from codeseek.chunking.docs_chunker import chunk_doc_text
from codeseek.chunking.shared import count_tokens, pack_units


SAMPLE_SOURCE = '''\
import os


@decorator
def small_function(x):
    """Adds one."""
    return x + 1


class Thing:
    """A class."""

    def method(self):
        return None
'''


def test_chunk_python_source_captures_whole_functions_with_decorator_and_docstring():
    chunks = chunk_python_source(SAMPLE_SOURCE, repo="demo", path="mod.py")
    names = {c.symbol_name: c for c in chunks}

    assert set(names) == {"small_function", "Thing"}

    fn = names["small_function"]
    assert fn.symbol_type == "function"
    assert "@decorator" in fn.text
    assert "Adds one" in fn.text
    assert "return x + 1" in fn.text

    cls = names["Thing"]
    assert cls.symbol_type == "class"
    assert "def method" in cls.text  # methods stay inside their class chunk


def test_chunk_python_source_ignores_module_level_statements():
    chunks = chunk_python_source(SAMPLE_SOURCE, repo="demo", path="mod.py")
    assert not any("import os" in c.text for c in chunks)


def test_chunk_python_source_handles_syntax_error_gracefully():
    assert chunk_python_source("def f(:\n", repo="demo", path="broken.py") == []


def test_oversized_function_is_split_never_mid_line():
    # A single function whose body alone exceeds the token cap.
    body_lines = "\n".join(f"    x{i} = {i}" for i in range(2000))
    source = f"def big():\n{body_lines}\n    return x0\n"

    chunks = chunk_python_source(source, repo="demo", path="big.py")

    assert len(chunks) > 1
    assert all(c.symbol_name.startswith("big#part") for c in chunks)
    # every emitted line is a complete, untouched source line
    all_lines = {line for c in chunks for line in c.text.splitlines()}
    assert "    x0 = 0" in all_lines
    assert "    return x0" in all_lines


def test_pack_units_respects_token_cap():
    units = [f"line {i}" for i in range(500)]
    packed = pack_units(units, max_tokens=50, overlap_ratio=0.0, joiner="\n")
    assert all(count_tokens(text) <= 50 for _, _, text in packed)
    # every unit shows up in the reconstruction
    reconstructed = "\n".join(text for _, _, text in packed)
    assert "line 0" in reconstructed and "line 499" in reconstructed


def test_docs_chunker_overlaps_between_chunks():
    sentence = "This is a moderately long sentence about the project. "
    text = sentence * 200  # forces multiple chunks

    chunks = chunk_doc_text(text, repo="demo", path="README.md")

    assert len(chunks) > 1
    # consecutive chunks share at least one sentence (the overlap)
    first_sentences = set(chunks[0].text.split(". "))
    second_sentences = set(chunks[1].text.split(". "))
    assert first_sentences & second_sentences


def test_docs_chunker_empty_input():
    assert chunk_doc_text("   \n\n  ", repo="demo", path="README.md") == []
