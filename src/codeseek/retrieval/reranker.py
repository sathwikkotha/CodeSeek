"""Cross-encoder re-ranking: a second, more accurate (but slower) scoring pass
over a small candidate pool.

The first-stage embedding search (a bi-encoder) embeds the query and each
chunk *independently* and compares the two vectors afterward -- it never lets
the query and the chunk actually attend to each other. A cross-encoder feeds
(query, chunk) through one model *together*, which is measurably more
accurate at judging "does this specific chunk answer this specific question" --
at the cost of one full forward pass per candidate, which is why it only ever
runs on the already-narrow fetch_k pool (~30 candidates), never the whole
corpus."""

from dataclasses import replace

from sentence_transformers import CrossEncoder

from codeseek.retrieval.hybrid import MergedHit

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[MergedHit], top_k: int) -> list[MergedHit]:
        if not hits:
            return []

        pairs = [(query, hit.payload.get("text", "")) for hit in hits]
        scores = self._model.predict(pairs)

        rescored = [replace(hit, score=float(score)) for hit, score in zip(hits, scores)]
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:top_k]
