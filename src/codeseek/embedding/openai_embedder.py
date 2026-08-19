"""Wraps OpenAI's embeddings API behind the Embedder interface. Unlike the
local sentence-transformers embedders, this makes a network call per batch and
costs money per call -- callers should batch generously (EmbeddingService
already does) rather than embedding one text at a time."""

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

# Retry only failures that are plausibly transient (dropped connection, a 429,
# a 5xx blip) -- never a 400/401/etc, which will fail identically every retry
# and just delays surfacing a real problem (bad key, malformed input).
_RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)


class OpenAIEmbedder:
    def __init__(self, model_name: str = "text-embedding-3-small", client: OpenAI | None = None):
        self.model_name = model_name
        self.dimensions = _DIMENSIONS[model_name]
        self._client = client or OpenAI()

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def embed(self, texts: list[str]) -> list[list[float]]:
        # The API 400s on an empty input list -- callers occasionally build
        # empty batches (e.g. no chunks left after filtering).
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in response.data]
