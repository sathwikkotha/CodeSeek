"""Production ASGI entry point: `uvicorn codeseek.api.main:app`.
Wires the real local embedding models and on-disk Qdrant storage."""

from codeseek.api.app import create_app
from codeseek.config import QDRANT_PATH
from codeseek.embedding.registry import build_default_embedders
from codeseek.embedding.service import EmbeddingService
from codeseek.store.qdrant_store import QdrantStore

store = QdrantStore(path=str(QDRANT_PATH))
embedding_service = EmbeddingService(build_default_embedders())
app = create_app(store, embedding_service)
