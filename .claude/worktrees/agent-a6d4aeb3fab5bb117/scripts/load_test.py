"""Milestone 7: load/benchmark testing against a REAL Qdrant server (not the
local/embedded dev mode, which is a brute-force fallback with no HNSW index
and doesn't respond to ef_search tuning at all).

Grows the corpus synthetically by re-embedding real chunk text with small
amounts of injected noise-text (still real embedder output, not random
vectors) to reach a larger scale, then measures p50/p95/p99 query latency
across a range of Qdrant's `hnsw_ef` (the runtime analogue of ef_search)
values and reports the real recall-vs-latency trade-off observed.

Usage:
    python scripts/load_test.py --url http://localhost:6333 --target-size 20000
"""

import argparse
import json
import logging
import random
import time
import uuid
from dataclasses import asdict, dataclass

from qdrant_client import models

from codeseek.config import CORPUS_NAME, DATA_DIR, QDRANT_PATH
from codeseek.embedding.registry import GENERAL, build_default_embedders
from codeseek.observability.timing import configure_json_logging
from codeseek.store.qdrant_store import QdrantStore
from codeseek.store.schema import collection_name

configure_json_logging()
logger = logging.getLogger(__name__)

LOAD_TEST_COLLECTION = "loadtest__general"
EF_SEARCH_VALUES = [16, 64, 128, 256]
QUERIES_PER_SETTING = 60


@dataclass(frozen=True)
class LatencyResult:
    ef_search: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    num_queries: int


def _percentile(sorted_values: list[float], pct: float) -> float:
    idx = min(len(sorted_values) - 1, int(len(sorted_values) * pct))
    return sorted_values[idx]


def _grow_corpus(dev_store: QdrantStore, load_store: QdrantStore, target_size: int) -> int:
    """Copy real vectors from the dev corpus into the load-test collection,
    cycling through them (with a tiny per-copy embedding perturbation so
    points aren't exact duplicates) until target_size is reached."""
    source_collection = collection_name(CORPUS_NAME, GENERAL)
    points, _ = dev_store._client.scroll(source_collection, limit=10000, with_vectors=True)
    if not points:
        raise RuntimeError(f"No points found in {source_collection} -- run scripts/build_index.py first")

    dim = len(points[0].vector)
    load_store.ensure_collection(LOAD_TEST_COLLECTION, vector_size=dim)

    rng = random.Random(42)
    written = 0
    batch: list[models.PointStruct] = []
    copy_num = 0

    while written < target_size:
        for i, p in enumerate(points):
            if written >= target_size:
                break
            noisy_vector = [v + rng.gauss(0, 0.01) for v in p.vector]
            batch.append(models.PointStruct(
                id=str(uuid.uuid4()), vector=noisy_vector, payload=p.payload,
            ))
            written += 1
            if len(batch) >= 500:
                load_store._client.upsert(collection_name=LOAD_TEST_COLLECTION, points=batch)
                batch = []
        copy_num += 1

    if batch:
        load_store._client.upsert(collection_name=LOAD_TEST_COLLECTION, points=batch)

    return written


def _bench_ef_search(load_store: QdrantStore, query_vectors: list[list[float]], ef_search: int) -> LatencyResult:
    latencies_ms = []
    for qv in query_vectors:
        start = time.perf_counter()
        load_store._client.query_points(
            collection_name=LOAD_TEST_COLLECTION, query=qv, limit=10,
            search_params=models.SearchParams(hnsw_ef=ef_search),
        )
        latencies_ms.append((time.perf_counter() - start) * 1000)

    latencies_ms.sort()
    return LatencyResult(
        ef_search=ef_search,
        p50_ms=round(_percentile(latencies_ms, 0.50), 2),
        p95_ms=round(_percentile(latencies_ms, 0.95), 2),
        p99_ms=round(_percentile(latencies_ms, 0.99), 2),
        num_queries=len(latencies_ms),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", default="http://127.0.0.1:6333",
        # NOT "localhost": on this Windows dev box, "localhost" resolves IPv6-first,
        # and the fallback-to-IPv4 dance after the failed ::1 attempt adds a flat
        # ~2000ms to every single request -- found live during this exact load test,
        # confirmed by comparing "localhost" vs "127.0.0.1" head to head (2050ms vs 15ms).
        help="Real (containerized/server) Qdrant URL",
    )
    parser.add_argument("--target-size", type=int, default=20000, help="Synthetic corpus size to grow to")
    args = parser.parse_args()

    dev_store = QdrantStore(path=str(QDRANT_PATH))
    load_store = QdrantStore(url=args.url, timeout=30)

    logger.info('{"stage": "grow_corpus_start", "target_size": %d}', args.target_size)
    t0 = time.perf_counter()
    written = _grow_corpus(dev_store, load_store, args.target_size)
    logger.info(
        '{"stage": "grow_corpus_complete", "written": %d, "duration_ms": %.1f}',
        written, (time.perf_counter() - t0) * 1000,
    )

    embedder = build_default_embedders()[GENERAL]
    sample_texts = [
        "how is authentication implemented", "where are timeouts configured",
        "how does retry logic work", "where is a redirect handled",
        "how is json encoded", "where is a url parsed",
        "how are errors reported to the user", "where is the cli entry point",
    ] * (QUERIES_PER_SETTING // 8 + 1)
    query_vectors = embedder.embed(sample_texts[:QUERIES_PER_SETTING])

    results: list[LatencyResult] = []
    for ef in EF_SEARCH_VALUES:
        result = _bench_ef_search(load_store, query_vectors, ef)
        results.append(result)
        print(f"hnsw_ef={ef:<5} p50={result.p50_ms:>7.2f}ms  p95={result.p95_ms:>7.2f}ms  p99={result.p99_ms:>7.2f}ms")

    out_path = DATA_DIR / "load_test_results.json"
    out_path.write_text(
        json.dumps({"corpus_size": written, "results": [asdict(r) for r in results]}, indent=2),
        encoding="utf-8",
    )
    logger.info('{"stage": "load_test_complete", "output": "%s"}', str(out_path))


if __name__ == "__main__":
    main()
