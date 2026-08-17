"""Wraps OpenAI's embeddings API behind the Embedder interface. Unlike the
local sentence-transformers embedders, this makes a network call per batch and
costs money per call -- callers should batch generously (EmbeddingService
already does) rather than embedding one text at a time."""

from openai import OpenAI

_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


class OpenAIEmbedder:
    def __init__(self, model_name: str = "text-embedding-3-small", client: OpenAI | None = None):
        self.model_name = model_name
        self.dimensions = _DIMENSIONS[model_name]
        self._client = client or OpenAI()

    def embed(self, texts: list[str]) -> list[list[float]]:
        # The API 400s on an empty input list -- callers occasionally build
        # empty batches (e.g. no chunks left after filtering).
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in response.data]
