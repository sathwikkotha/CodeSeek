from codeseek.chunking.js_ts_chunker import chunk_js_ts_source
from codeseek.chunking.shared import count_tokens

JS_SAMPLE = """\
import x from "y";

export function add(a, b) {
  return a + b;
}

const multiply = (a, b) => {
  return a * b;
};

export default class Calculator {
  constructor() {
    this.value = 0;
  }
}

export const divide = function(a, b) {
  return a / b;
};
"""


def test_chunk_js_captures_named_export_function():
    chunks = chunk_js_ts_source(JS_SAMPLE, repo="demo", path="calc.js", suffix=".js")
    by_name = {c.symbol_name: c for c in chunks}

    add = by_name["add"]
    assert add.symbol_type == "function"
    assert add.language == "javascript"
    assert "export function add" in add.text
    assert "return a + b" in add.text


def test_chunk_js_captures_const_arrow_function():
    chunks = chunk_js_ts_source(JS_SAMPLE, repo="demo", path="calc.js", suffix=".js")
    by_name = {c.symbol_name: c for c in chunks}

    multiply = by_name["multiply"]
    assert multiply.symbol_type == "function"
    assert "=>" in multiply.text


def test_chunk_js_captures_export_default_class():
    chunks = chunk_js_ts_source(JS_SAMPLE, repo="demo", path="calc.js", suffix=".js")
    by_name = {c.symbol_name: c for c in chunks}

    calculator = by_name["Calculator"]
    assert calculator.symbol_type == "class"
    assert "export default class Calculator" in calculator.text
    assert "constructor" in calculator.text  # method stays inside the class chunk


def test_chunk_js_captures_export_const_function_expression():
    chunks = chunk_js_ts_source(JS_SAMPLE, repo="demo", path="calc.js", suffix=".js")
    by_name = {c.symbol_name: c for c in chunks}

    divide = by_name["divide"]
    assert divide.symbol_type == "function"
    assert "export const divide" in divide.text


def test_chunk_js_ignores_bare_import():
    chunks = chunk_js_ts_source(JS_SAMPLE, repo="demo", path="calc.js", suffix=".js")
    assert not any("import x" in c.text for c in chunks)


def test_chunk_ts_captures_generic_class_and_ignores_interface():
    source = """\
interface Foo {
  bar: string;
}

export class Widget<T> {
  private value: T;
  method(): void {}
}
"""
    chunks = chunk_js_ts_source(source, repo="demo", path="widget.ts", suffix=".ts")
    names = {c.symbol_name: c for c in chunks}

    assert set(names) == {"Widget"}
    assert names["Widget"].language == "typescript"
    assert names["Widget"].symbol_type == "class"
    assert "method" in names["Widget"].text


def test_chunk_js_handles_malformed_source_gracefully():
    source = "function broken(: {\n  this is not valid javascript at all ]]][[[\n"
    chunks = chunk_js_ts_source(source, repo="demo", path="broken.js", suffix=".js")
    assert chunks == []  # tree-sitter tolerates it, nothing recognizable is emitted


def test_chunk_js_unsupported_extension_returns_empty():
    assert chunk_js_ts_source("const x = 1;", repo="demo", path="x.mjs", suffix=".mjs") == []


def test_chunk_js_oversized_function_is_split_never_mid_line():
    body_lines = "\n".join(f"  const x{i} = {i};" for i in range(2000))
    source = f"function big() {{\n{body_lines}\n  return x0;\n}}\n"

    chunks = chunk_js_ts_source(source, repo="demo", path="big.js", suffix=".js")

    assert len(chunks) > 1
    assert all(c.symbol_name.startswith("big#part") for c in chunks)
    assert all(count_tokens(c.text) <= 400 for c in chunks)
    all_lines = {line for c in chunks for line in c.text.splitlines()}
    assert "  const x0 = 0;" in all_lines
