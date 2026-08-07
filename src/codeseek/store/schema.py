"""Payload shape stored alongside every vector, and the collection-naming
convention: one collection per embedding model (e.g. 'general', 'code')."""

import uuid
from dataclasses import dataclass

# Fixed namespace so the same logical chunk always maps to the same point ID
# across re-ingestion runs -- upserting updates it in place instead of duplicating.
_ID_NAMESPACE = uuid.UUID("5b6a8f0c-6b3f-4a1a-9b8b-6b7c9b9e6f10")


@dataclass(frozen=True)
class ChunkPayload:
    repo: str
    language: str
    path: str
    symbol_name: str
    symbol_type: str  # "function" | "async_function" | "class" | "doc"
    start_line: int
    end_line: int
    text: str
    license: str = ""
    stars: int = 0

    def point_id(self) -> str:
        key = f"{self.repo}:{self.path}:{self.symbol_name}:{self.start_line}:{self.end_line}"
        return str(uuid.uuid5(_ID_NAMESPACE, key))

    def as_payload_dict(self) -> dict:
        return {
            "repo": self.repo, "language": self.language, "path": self.path,
            "symbol_name": self.symbol_name, "symbol_type": self.symbol_type,
            "start_line": self.start_line, "end_line": self.end_line,
            "text": self.text, "license": self.license, "stars": self.stars,
        }


def collection_name(corpus_name: str, model_key: str) -> str:
    return f"{corpus_name}__{model_key}"
