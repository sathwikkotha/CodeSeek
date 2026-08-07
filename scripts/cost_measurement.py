"""Milestone 8: cost measurement + one real optimization pass.

Both embedding models are local (no per-call API fee), so '$/1000 chunks' and
'$/query' are honestly $0 in API cost -- the real cost is CPU compute time.
This measures that instead, and one deliberate optimization:

  1. Embedding throughput at several batch sizes (the indexing-side cost) --
     the "$/1000 chunks" analogue for a local model.
  2. Repeated-query caching (already wired into EmbeddingService.embed_one) --
     the "$/query" analogue: a before/after on re-asking the same question.

Usage:
    python scripts/cost_measurement.py
"""

import time

from codeseek.chunking.code_chunker import chunk_python_source
from codeseek.config import REPOS_DIR
from codeseek.embedding.registry import GENERAL, build_default_embedders
from codeseek.embedding.service import EmbeddingService

BATCH_SIZES = [1, 8, 32, 128]


def _sample_chunk_texts(n: int = 256) -> list[str]:
    """Real chunk text pulled straight from the already-cloned httpx repo --
    no synthetic filler."""
    texts = []
    for py_file in (REPOS_DIR / "httpx" / "httpx").glob("*.py"):
        source = py_file.read_text(encoding="utf-8", errors="ignore")
        for chunk in chunk_python_source(source, repo="httpx", path=py_file.name):
            texts.append(chunk.text)
            if len(texts) >= n:
                return texts
    return texts


def measure_indexing_throughput() -> None:
    texts = _sample_chunk_texts(256)
    embedder = build_default_embedders()[GENERAL]

    print(f"Indexing-side throughput -- embedding {len(texts)} real httpx chunks at each batch size:")
    print(f"{'batch_size':<12}{'total_s':>10}{'chunks/sec':>13}{'ms/chunk':>11}")
    for batch_size in BATCH_SIZES:
        embedder.batch_size = batch_size
        start = time.perf_counter()
        embedder.embed(texts)
        elapsed = time.perf_counter() - start
        print(f"{batch_size:<12}{elapsed:>10.2f}{len(texts) / elapsed:>13.1f}{elapsed / len(texts) * 1000:>11.2f}")
    print()


def measure_query_cache_speedup() -> None:
    service = EmbeddingService({"general": build_default_embedders()[GENERAL]})
    query = "where is jwt validation implemented"
    n = 30

    start = time.perf_counter()
    for _ in range(n):
        service.embed_one("general", query)
    elapsed_cached = time.perf_counter() - start

    # simulate the uncached baseline: a fresh service (empty cache) re-embeds every time
    uncached_service = EmbeddingService({"general": build_default_embedders()[GENERAL]}, query_cache_size=0)
    start = time.perf_counter()
    for i in range(n):
        uncached_service.embed_one("general", f"{query} {i}")  # force a cache miss each time
    elapsed_uncached = time.perf_counter() - start

    print(f"Query-side cost -- same query asked {n} times:")
    print(f"  without caching (every ask re-embeds): {elapsed_uncached:.3f}s total, {elapsed_uncached / n * 1000:.2f} ms/query")
    print(f"  with caching (repeats are free):        {elapsed_cached:.3f}s total, {elapsed_cached / n * 1000:.2f} ms/query")
    print(f"  cache hits: {service.cache_hits}, cache misses: {service.cache_misses}")
    print(f"  speedup on repeated queries: {elapsed_uncached / elapsed_cached:.1f}x")


if __name__ == "__main__":
    measure_indexing_throughput()
    measure_query_cache_speedup()
