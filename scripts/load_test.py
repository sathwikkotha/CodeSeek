"""Load test: index a growing set of real public repos, one at a time, and
measure indexing throughput and query latency at each corpus-size checkpoint.

This exists to validate (or correct) docs/scaling_design.md's claims with
real numbers instead of just architectural reasoning -- specifically whether
query latency stays roughly flat as corpus size grows (expected: Qdrant's
vector index is sub-linear) or degrades (a real signal that keyword_search's
scroll-based implementation, not the vector path, is the actual bottleneck --
see store/qdrant_store.py's keyword_search, which uses .scroll() with a text
filter, not an inverted index).

Repos are shallow-cloned real public libraries, small enough to keep this
affordable (a few thousand chunks total, well under $0.10 of embedding calls
at text-embedding-3-small pricing) while still producing a real, multi-
thousand-chunk corpus to measure against. Writes docs/scaling_results.md.

Requires OPENAI_API_KEY. Clones into a temp directory, not data/repos/ --
this is a benchmark run, not something meant to persist as searchable state.
"""

import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

from codeseek.config import ROOT_DIR as PROJECT_ROOT
from codeseek.config import RepoSpec
from codeseek.embedding.registry import OPENAI, build_default_embedders
from codeseek.embedding.service import EmbeddingService
from codeseek.ingestion.clone import clone_or_update
from codeseek.pipeline.index import index_repo
from codeseek.retrieval.hybrid import HybridRetriever
from codeseek.retrieval.reranker import CrossEncoderReranker
from codeseek.store.qdrant_store import QdrantStore
from codeseek.store.schema import collection_name

# Small, well-known, permissively-licensed Python libraries -- enough to
# produce a real multi-thousand-chunk corpus without an expensive or slow run.
REPOS = [
    RepoSpec("typer", "https://github.com/tiangolo/typer.git", "python"),
    RepoSpec("click", "https://github.com/pallets/click.git", "python"),
    RepoSpec("itsdangerous", "https://github.com/pallets/itsdangerous.git", "python"),
    RepoSpec("httpx", "https://github.com/encode/httpx.git", "python"),
    RepoSpec("requests", "https://github.com/psf/requests.git", "python"),
    RepoSpec("gunicorn", "https://github.com/benoitc/gunicorn.git", "python"),
]

# Generic questions, not tied to any one repo -- run unfiltered across the
# whole growing corpus at each checkpoint, mirroring a real user who doesn't
# know (or say) which indexed repo has the answer.
QUERIES = [
    "how are command line arguments parsed",
    "how is an HTTP request sent and a response returned",
    "how does this library handle configuration options",
    "how are errors and exceptions raised and handled",
    "how is a decorator used to register a function",
]

CORPUS_NAME = "codeseek_load_test"
TOP_K = 10


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set -- load test needs real embeddings to measure anything real.")
        return 1

    tmp_dir = Path(tempfile.mkdtemp(prefix="codeseek_load_test_"))
    store = QdrantStore(location=":memory:")
    embedding_service = EmbeddingService(build_default_embedders())
    reranker = CrossEncoderReranker()
    retriever = HybridRetriever(store, reranker=reranker)
    collection = collection_name(CORPUS_NAME, OPENAI)

    checkpoints = []
    total_chunks = 0

    try:
        for repo in REPOS:
            print(f"\n=== {repo.name} ===")
            t0 = time.perf_counter()
            repo_path = clone_or_update(repo, tmp_dir)
            clone_s = time.perf_counter() - t0

            t1 = time.perf_counter()
            chunks_added = index_repo(repo_path, repo, store, embedding_service, corpus_name=CORPUS_NAME)
            index_s = time.perf_counter() - t1
            total_chunks += chunks_added
            throughput = chunks_added / index_s if index_s > 0 else 0.0
            print(
                f"cloned in {clone_s:.1f}s, indexed {chunks_added} chunks in {index_s:.1f}s "
                f"({throughput:.1f} chunks/s)"
            )

            latencies_ms = []
            for query in QUERIES:
                t2 = time.perf_counter()
                qvec = embedding_service.embed_one(OPENAI, query)
                retriever.search(collection, qvec, query, TOP_K)
                latencies_ms.append((time.perf_counter() - t2) * 1000)

            checkpoints.append({
                "repos_indexed": repo.name,
                "total_chunks": total_chunks,
                "index_throughput_chunks_per_s": throughput,
                "query_latency_p50_ms": statistics.median(latencies_ms),
                "query_latency_max_ms": max(latencies_ms),
            })
            print(f"query latency: p50={statistics.median(latencies_ms):.0f}ms max={max(latencies_ms):.0f}ms")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    lines = [
        "# Scaling load test results",
        "",
        f"Measured {time.strftime('%Y-%m-%d')} by indexing {len(REPOS)} real public repos one at a time "
        "into a single corpus and re-measuring query latency (embed + vector search + keyword search + "
        "merge + cross-encoder rerank + RRF fusion -- the full production `/search` path) after each one. "
        "Reproduce with `python scripts/load_test.py`. Written to validate (or correct) the reasoning in "
        "`docs/scaling_design.md` with real numbers.",
        "",
        "| Repo added | Total chunks | Index throughput (chunks/s) | Query p50 (ms) | Query max (ms) |",
        "|---|---|---|---|---|",
    ]
    for cp in checkpoints:
        lines.append(
            f"| {cp['repos_indexed']} | {cp['total_chunks']} | "
            f"{cp['index_throughput_chunks_per_s']:.1f} | {cp['query_latency_p50_ms']:.0f} | "
            f"{cp['query_latency_max_ms']:.0f} |"
        )
    lines.append("")

    report = "\n".join(lines)
    print("\n" + report)
    results_path = PROJECT_ROOT / "docs" / "scaling_results.md"
    results_path.write_text(report, encoding="utf-8")
    print(f"Wrote {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
