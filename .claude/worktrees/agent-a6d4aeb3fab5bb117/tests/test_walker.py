from pathlib import Path

from codeseek.ingestion.manifest import read_manifest, write_manifest
from codeseek.ingestion.walker import walk_repo


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "fakerepo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "core.py").write_text("def f():\n    return 1\n")
    (repo / "README.md").write_text("# Fake repo\n")
    (repo / "pkg" / "__pycache__").mkdir()
    (repo / "pkg" / "__pycache__" / "core.cpython-311.pyc").write_bytes(b"\x00\x01")
    (repo / "pkg" / "empty.py").touch()  # zero-byte file, should be skipped
    (repo / "pkg" / "data.json").write_text("{}")  # unrouted extension, should be skipped
    return repo


def test_walk_repo_routes_and_skips(tmp_path):
    repo = _make_repo(tmp_path)
    records = list(walk_repo(repo, repo_name="fakerepo", language="python"))
    paths = {r.path.replace("\\", "/"): r.category for r in records}

    assert paths == {
        "pkg/core.py": "code",
        "README.md": "docs",
    }


def test_walk_repo_skips_translated_docs_but_keeps_english(tmp_path):
    repo = tmp_path / "fakerepo"
    (repo / "docs" / "en").mkdir(parents=True)
    (repo / "docs" / "de").mkdir(parents=True)
    (repo / "docs" / "zh-hant").mkdir(parents=True)
    (repo / "docs" / "en" / "index.md").write_text("# English docs\n", encoding="utf-8")
    (repo / "docs" / "de" / "index.md").write_text("# Deutsche Dokumentation\n", encoding="utf-8")
    (repo / "docs" / "zh-hant" / "index.md").write_text("# 文件\n", encoding="utf-8")

    records = list(walk_repo(repo, repo_name="fakerepo", language="python"))
    paths = {r.path.replace("\\", "/") for r in records}

    assert paths == {"docs/en/index.md"}


def test_manifest_roundtrip(tmp_path):
    repo = _make_repo(tmp_path)
    records = walk_repo(repo, repo_name="fakerepo", language="python")

    out = tmp_path / "manifest.jsonl"
    count = write_manifest(records, out)
    assert count == 2

    rows = list(read_manifest(out))
    assert {r["path"].replace("\\", "/") for r in rows} == {"pkg/core.py", "README.md"}
    assert all(r["repo"] == "fakerepo" for r in rows)
