"""The embedder interface every embedding backend implements, so the service
and tests can swap a real model for a fake one without touching call sites."""

from typing import Protocol


class Embedder(Protocol):
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...
