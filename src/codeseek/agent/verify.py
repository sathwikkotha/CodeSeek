"""Deterministic, zero-cost verification that every citation the agent wrote
actually points at real source -- not an LLM call, so this step cannot
itself hallucinate. This is what turns "the agent claims it's grounded"
into "the agent's claims are checked against the actual repo."

Deliberately narrow: it confirms a citation's file exists and its line range
is within bounds, not that the cited lines truly support the adjacent claim
-- that would need semantic judgment (another LLM call), which trades away
the "deterministic" property this is built for. A citation passing this
check means "this location is real"; it does not mean "this location proves
the sentence next to it." Catching the former is still the highest-value,
cheapest check available, since a citation pointing at a nonexistent file or
line is the most damaging and most mechanically detectable failure mode.
"""

import re
from dataclasses import dataclass
from pathlib import Path

# Matches "path/to/file.ext:START-END", optionally prefixed with the repo
# name (the model doesn't consistently include it -- observed both forms in
# real answers). The repo prefix is stripped by name, not inferred from the
# pattern, since a generic "first path segment is the repo" rule can't tell
# a repo prefix apart from a genuine subdirectory.
_CITATION_PATTERN = re.compile(r"([A-Za-z0-9_./\-]+\.[A-Za-z0-9]+):(\d+)-(\d+)")


@dataclass(frozen=True)
class CitationCheck:
    citation: str
    path: str
    start_line: int
    end_line: int
    valid: bool
    reason: str | None = None


def verify_citations(answer: str | None, repo_name: str, repo_path: Path) -> list[CitationCheck]:
    if not answer:
        return []

    repo_root = repo_path.resolve()
    prefix = f"{repo_name}/"
    checks: list[CitationCheck] = []
    seen: set[str] = set()

    for match in _CITATION_PATTERN.finditer(answer):
        citation = match.group(0)
        if citation in seen:
            continue
        seen.add(citation)

        raw_path, start_str, end_str = match.groups()
        path = raw_path.removeprefix(prefix)
        start_line, end_line = int(start_str), int(end_str)

        full_path = (repo_root / path).resolve()
        if not full_path.is_relative_to(repo_root):
            checks.append(CitationCheck(citation, path, start_line, end_line, False, "path escapes the repo"))
            continue
        if not full_path.is_file():
            checks.append(CitationCheck(citation, path, start_line, end_line, False, "file not found"))
            continue

        line_count = sum(1 for _ in full_path.open(encoding="utf-8", errors="ignore"))
        if start_line < 1 or end_line < start_line or start_line > line_count:
            checks.append(CitationCheck(citation, path, start_line, end_line, False, "line range out of bounds"))
            continue

        checks.append(CitationCheck(citation, path, start_line, end_line, True))

    return checks
