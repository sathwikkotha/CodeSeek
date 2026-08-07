"""A ground-truth item: a natural-language question paired with the
repo/path/symbol that should be retrieved for it -- the eval harness's source
of truth for Recall@k and MRR."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GroundTruthItem:
    question: str
    repo: str
    path: str
    symbol_name: str


def load_ground_truth(path: Path) -> list[GroundTruthItem]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [GroundTruthItem(**item) for item in data]
