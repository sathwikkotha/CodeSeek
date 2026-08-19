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

    assert set(names) == {"small_function", "Thing", "Thing.method"}

    fn = names["small_function"]
    assert fn.symbol_type == "function"
    assert "@decorator" in fn.text
    assert "Adds one" in fn.text
    assert "return x + 1" in fn.text


def test_chunk_class_emits_a_dedicated_chunk_per_method():
    chunks = chunk_python_source(SAMPLE_SOURCE, repo="demo", path="mod.py")
    names = {c.symbol_name: c for c in chunks}

    method = names["Thing.method"]
    assert method.symbol_type == "method"
    assert "return None" in method.text


def test_chunk_class_overview_has_the_docstring_and_method_signature_but_not_the_body():
    chunks = chunk_python_source(SAMPLE_SOURCE, repo="demo", path="mod.py")
    names = {c.symbol_name: c for c in chunks}

    overview = names["Thing"]
    assert overview.symbol_type == "class"
    assert "A class." in overview.text
    assert "def method(self):" in overview.text
    assert "return None" not in overview.text  # body lives in Thing.method's own chunk


def test_chunk_python_source_ignores_module_level_statements():
    chunks = chunk_python_source(SAMPLE_SOURCE, repo="demo", path="mod.py")
    assert not any("import os" in c.text for c in chunks)


def test_chunk_python_source_handles_syntax_error_gracefully():
    assert chunk_python_source("def f(:\n", repo="demo", path="broken.py") == []


DATACLASS_STYLE_SOURCE = '''\
class OptionInfo(ParameterInfo):
    def __init__(
        self,
        *,
        default=None,
        prompt=False,
    ):
        self.default = default
        self.prompt = prompt
'''


def test_chunk_class_with_no_docstring_dataclass_style_still_produces_a_useful_overview():
    # Mirrors the real typer.models.OptionInfo shape found via live eval: no
    # docstring, no class-level fields, everything defined inside __init__.
    chunks = chunk_python_source(DATACLASS_STYLE_SOURCE, repo="demo", path="models.py")
    names = {c.symbol_name: c for c in chunks}

    assert set(names) == {"OptionInfo", "OptionInfo.__init__"}
    overview = names["OptionInfo"]
    assert "class OptionInfo(ParameterInfo):" in overview.text
    assert "def __init__(" in overview.text
    assert "default=None" in overview.text  # full multi-line signature, not just "def __init__("

    init_chunk = names["OptionInfo.__init__"]
    assert init_chunk.symbol_type == "method"
    assert "self.default = default" in init_chunk.text


ASYNC_METHOD_SOURCE = '''\
class Client:
    async def fetch(self, url):
        return await self._get(url)
'''


def test_chunk_class_captures_async_methods_with_their_own_symbol_type():
    chunks = chunk_python_source(ASYNC_METHOD_SOURCE, repo="demo", path="client.py")
    names = {c.symbol_name: c for c in chunks}

    assert names["Client.fetch"].symbol_type == "async_method"
    assert "await self._get(url)" in names["Client.fetch"].text


FIELDS_ONLY_SOURCE = '''\
class Config:
    """Settings."""

    timeout: int = 30
    retries: int = 3
'''


def test_chunk_class_with_only_fields_and_no_methods():
    chunks = chunk_python_source(FIELDS_ONLY_SOURCE, repo="demo", path="config.py")
    names = {c.symbol_name: c for c in chunks}

    assert set(names) == {"Config"}  # no methods -> no separate method chunks
    assert "timeout: int = 30" in names["Config"].text
    assert "retries: int = 3" in names["Config"].text


def test_oversized_method_is_split_and_named_with_qualified_symbol():
    body_lines = "\n".join(f"        x{i} = {i}" for i in range(2000))
    source = f"class Big:\n    def method(self):\n{body_lines}\n        return x0\n"

    chunks = chunk_python_source(source, repo="demo", path="big.py")
    method_chunks = [c for c in chunks if c.symbol_name.startswith("Big.method#part")]

    assert len(method_chunks) > 1
    assert all(c.symbol_type == "method" for c in method_chunks)
    all_lines = {line for c in method_chunks for line in c.text.splitlines()}
    assert "        x0 = 0" in all_lines
    assert "        return x0" in all_lines


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


def test_docs_chunker_strips_badges_and_html_before_first_chunk():
    readme = """\
<p align="center">
  <a href="https://example.com"><img src="https://example.com/logo.svg" alt="Logo"></a>
</p>
<p align="center">
[![Test](https://example.com/test.svg)](https://example.com/actions)
[![Coverage](https://example.com/cov.svg)](https://example.com/cov)
</p>

Widgetlib is a library for building great widgets that developers love using.
It converts plain Python functions into fully validated widget pipelines.
"""
    chunks = chunk_doc_text(readme, repo="demo", path="README.md")
    assert chunks
    assert "img" not in chunks[0].text
    assert "<p" not in chunks[0].text
    assert "Widgetlib is a library" in chunks[0].text


def test_docs_chunker_empty_input():
    assert chunk_doc_text("   \n\n  ", repo="demo", path="README.md") == []
