"""Production ASGI entry point: `uvicorn codeseek.api.main:app`.
Wires OpenAI embeddings + the OpenAI-backed explain agent, Qdrant storage, and
the local cross-encoder reranker."""

import os

from openai import OpenAI

from codeseek.api.app import create_app
from codeseek.config import QDRANT_PATH
from codeseek.embedding.registry import build_default_embedders
from codeseek.embedding.service import EmbeddingService
from codeseek.observability.timing import configure_json_logging
from codeseek.retrieval.reranker import CrossEncoderReranker
from codeseek.store.qdrant_store import QdrantStore

# Without this, an /ingest request's per-batch chunk/embed/upsert stage logs
# are silently dropped (no handler on the "codeseek.pipeline" logger), so a
# genuinely-still-working multi-minute indexing run looks indistinguishable
# from a hung one from outside the process -- found the hard way while
# debugging what looked like a stuck clone+index request.
configure_json_logging()

# QDRANT_URL (e.g. "http://qdrant:6333" in docker-compose.yml) switches to a
# containerized Qdrant; unset, this stays the on-disk single-machine default
# every other entry point (scripts/index_one.py, tests) already uses.
qdrant_url = os.environ.get("QDRANT_URL")
store = QdrantStore(url=qdrant_url) if qdrant_url else QdrantStore(path=str(QDRANT_PATH))
embedding_service = EmbeddingService(build_default_embedders())
reranker = CrossEncoderReranker()
openai_client = OpenAI()  # reads OPENAI_API_KEY, loaded from .env via config's import
app = create_app(store, embedding_service, reranker=reranker, openai_client=openai_client)
