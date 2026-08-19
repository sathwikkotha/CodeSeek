from codeseek.agent.verify import verify_citations


def _write_repo(tmp_path, files: dict[str, str]):
    repo_root = tmp_path / "demo"
    repo_root.mkdir()
    for rel_path, content in files.items():
        full = repo_root / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
    return repo_root


def test_no_citations_returns_empty_list(tmp_path):
    repo_root = _write_repo(tmp_path, {"a.py": "x = 1\n"})
    assert verify_citations("No citations in this answer at all.", "demo", repo_root) == []


def test_no_answer_returns_empty_list(tmp_path):
    repo_root = _write_repo(tmp_path, {"a.py": "x = 1\n"})
    assert verify_citations(None, "demo", repo_root) == []


def test_valid_citation_with_repo_prefix(tmp_path):
    repo_root = _write_repo(tmp_path, {"auth.py": "\n".join(f"line{i}" for i in range(1, 11)) + "\n"})

    checks = verify_citations("Validated here. `demo/auth.py:2-5`", "demo", repo_root)

    assert len(checks) == 1
    assert checks[0].valid is True
    assert checks[0].path == "auth.py"
    assert checks[0].start_line == 2
    assert checks[0].end_line == 5


def test_valid_citation_without_repo_prefix(tmp_path):
    """The model doesn't consistently include the repo name -- observed both
    forms in real answers -- so a bare path:start-end must verify too."""
    repo_root = _write_repo(tmp_path, {"auth.py": "\n".join(f"line{i}" for i in range(1, 11)) + "\n"})

    checks = verify_citations("Validated here. `auth.py:2-5`", "demo", repo_root)

    assert len(checks) == 1
    assert checks[0].valid is True


def test_citation_to_nonexistent_file_is_invalid(tmp_path):
    repo_root = _write_repo(tmp_path, {"auth.py": "line1\n"})

    checks = verify_citations("See `demo/does_not_exist.py:1-5`", "demo", repo_root)

    assert len(checks) == 1
    assert checks[0].valid is False
    assert checks[0].reason == "file not found"


def test_citation_with_out_of_bounds_line_range_is_invalid(tmp_path):
    repo_root = _write_repo(tmp_path, {"auth.py": "line1\nline2\nline3\n"})

    checks = verify_citations("See `demo/auth.py:100-200`", "demo", repo_root)

    assert len(checks) == 1
    assert checks[0].valid is False
    assert checks[0].reason == "line range out of bounds"


def test_citation_path_escaping_repo_root_is_rejected(tmp_path):
    # a sibling file outside the repo root the model shouldn't be able to cite
    (tmp_path / "secret.py").write_text("SECRET = 1\n", encoding="utf-8")
    repo_root = _write_repo(tmp_path, {"auth.py": "line1\n"})

    checks = verify_citations("See `demo/../secret.py:1-1`", "demo", repo_root)

    assert len(checks) == 1
    assert checks[0].valid is False
    assert checks[0].reason == "path escapes the repo"


def test_duplicate_citations_are_only_checked_once(tmp_path):
    repo_root = _write_repo(tmp_path, {"auth.py": "line1\nline2\n"})

    answer = "First mention `demo/auth.py:1-1`. Second mention `demo/auth.py:1-1` again."
    checks = verify_citations(answer, "demo", repo_root)

    assert len(checks) == 1


def test_citation_with_zero_zero_range_is_valid_if_file_exists(tmp_path):
    """notebook_chunker.py and the docs branch of pipeline/index.py both store
    start_line=end_line=0 (no stable, citeable line range for a notebook cell
    or a doc chunk) -- the agent cites those chunks exactly as shown, so
    "path:0-0" must verify on file existence alone, not fail every time."""
    repo_root = _write_repo(tmp_path, {"notebook.ipynb": "{}"})

    checks = verify_citations("See `demo/notebook.ipynb:0-0`", "demo", repo_root)

    assert len(checks) == 1
    assert checks[0].valid is True
    assert checks[0].reason is None


def test_citation_with_zero_zero_range_to_missing_file_is_invalid(tmp_path):
    repo_root = _write_repo(tmp_path, {"auth.py": "line1\n"})

    checks = verify_citations("See `demo/does_not_exist.ipynb:0-0`", "demo", repo_root)

    assert len(checks) == 1
    assert checks[0].valid is False
    assert checks[0].reason == "file not found"


def test_citation_with_only_start_zero_still_bounds_checked(tmp_path):
    """The 0-0 special case is exact -- start=0 paired with a nonzero end isn't
    the notebook/doc convention, so it still goes through the real bounds check."""
    repo_root = _write_repo(tmp_path, {"auth.py": "line1\nline2\n"})

    checks = verify_citations("See `demo/auth.py:0-5`", "demo", repo_root)

    assert len(checks) == 1
    assert checks[0].valid is False
    assert checks[0].reason == "line range out of bounds"


def test_multiple_distinct_citations_all_checked(tmp_path):
    repo_root = _write_repo(tmp_path, {"a.py": "line1\nline2\n", "b.py": "line1\n"})

    answer = "See `demo/a.py:1-2` and also `demo/b.py:1-1` and `demo/missing.py:1-1`."
    checks = verify_citations(answer, "demo", repo_root)

    assert len(checks) == 3
    by_path = {c.path: c.valid for c in checks}
    assert by_path == {"a.py": True, "b.py": True, "missing.py": False}
