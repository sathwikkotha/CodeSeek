"""Writes/reads the ingestion manifest: one JSON line per (repo, file) discovered."""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from codeseek.ingestion.walker import FileRecord, file_record_to_dict


def write_manifest(records: Iterable[FileRecord], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(file_record_to_dict(record)) + "\n")
            count += 1
    return count


def read_manifest(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
