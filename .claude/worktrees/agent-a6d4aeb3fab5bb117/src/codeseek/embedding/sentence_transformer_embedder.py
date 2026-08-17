"""Wraps a local sentence-transformers model behind the Embedder interface."""

from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, dimensions: int, trust_remote_code: bool = False, batch_size: int = 32):
        self.model_name = model_name
        self.dimensions = dimensions
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=False, normalize_embeddings=True,
        )
        return vectors.tolist()
