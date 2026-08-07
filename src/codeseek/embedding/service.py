"""Embeds every chunk through every configured model -- the 'embed EVERY
chunk twice' step of the pipeline. embed_one (the query path) caches repeated
identical queries, since a search UI/eval harness re-sends the same question
often and re-embedding it is pure wasted compute."""

from collections import OrderedDict

from codeseek.embedding.base import Embedder


class EmbeddingService:
    def __init__(self, embedders: dict[str, Embedder], query_cache_size: int = 256):
        self._embedders = embedders
        self._query_cache: OrderedDict[tuple[str, str], list[float]] = OrderedDict()
        self._query_cache_size = query_cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    def embed_all(self, texts: list[str]) -> dict[str, list[list[float]]]:
        return {key: embedder.embed(texts) for key, embedder in self._embedders.items()}

    def embed_one(self, model_key: str, text: str) -> list[float]:
        key = (model_key, text)
        cached = self._query_cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            self._query_cache.move_to_end(key)
            return cached

        self.cache_misses += 1
        vector = self._embedders[model_key].embed([text])[0]
        self._query_cache[key] = vector
        if len(self._query_cache) > self._query_cache_size:
            self._query_cache.popitem(last=False)
        return vector
