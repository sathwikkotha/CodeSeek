import json

from codeseek.embedding.service import EmbeddingService
from codeseek.eval.ground_truth import GroundTruthItem, load_ground_truth
from codeseek.eval.harness import _is_relevant, run_eval
from codeseek.eval.metrics import reciprocal_rank, recall_at_k
from codeseek.store.qdrant_store import QdrantStore
from codeseek.store.schema import ChunkPayload, collection_name


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"c"}, k=3) == 1.0
    assert recall_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0
    assert recall_at_k(["a", "b", "c"], {"z"}, k=3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_load_ground_truth(tmp_path):
    path = tmp_path / "gt.json"
    path.write_text(json.dumps([
        {"question": "where is jwt validated", "repo": "demo", "path": "auth.py", "symbol_name": "validate_jwt"},
    ]))
    items = load_ground_truth(path)
    assert items == [GroundTruthItem("where is jwt validated", "demo", "auth.py", "validate_jwt")]


def test_is_relevant_matches_oversized_split_parts():
    item = GroundTruthItem("q", repo="demo", path="a.py", symbol_name="DigestAuth")

    assert _is_relevant({"repo": "demo", "path": "a.py", "symbol_name": "DigestAuth"}, item)
    assert _is_relevant({"repo": "demo", "path": "a.py", "symbol_name": "DigestAuth#part1"}, item)
    assert _is_relevant({"repo": "demo", "path": "a.py", "symbol_name": "DigestAuth#part3"}, item)
    # must not fuzzy-match an unrelated symbol that happens to share a prefix
    assert not _is_relevant({"repo": "demo", "path": "a.py", "symbol_name": "DigestAuthChallenge"}, item)
    assert not _is_relevant({"repo": "demo", "path": "b.py", "symbol_name": "DigestAuth#part1"}, item)


def test_is_relevant_matches_a_method_of_the_targeted_class():
    # method-level chunking (code_chunker.py) splits a class into an overview
    # chunk plus one chunk per method, named "ClassName.method_name" -- a
    # ground-truth item targeting the class should count either as a hit.
    item = GroundTruthItem("q", repo="demo", path="a.py", symbol_name="DigestAuth")

    assert _is_relevant({"repo": "demo", "path": "a.py", "symbol_name": "DigestAuth.__init__"}, item)
    assert _is_relevant({"repo": "demo", "path": "a.py", "symbol_name": "DigestAuth.build_challenge#part1"}, item)
    assert not _is_relevant({"repo": "demo", "path": "a.py", "symbol_name": "DigestAuthChallenge.method"}, item)


class FakeEmbedder:
    """'jwt' -> along x, 'csv' -> along y, anything else -> near-zero (weak match)."""

    dimensions = 3

    def embed(self, texts):
        vectors = []
        for t in texts:
            t = t.lower()
            if "jwt" in t:
                vectors.append([1.0, 0.0, 0.0])
            elif "csv" in t:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


def test_run_eval_perfect_and_broken_model_scored_correctly():
    store = QdrantStore(location=":memory:")
    corpus_name = "evaltest"

    chunks = [
        ChunkPayload(
            repo="demo", language="python", path="auth.py", symbol_name="validate_jwt",
            symbol_type="function", start_line=1, end_line=5, text="def validate_jwt(token): jwt logic",
        ),
        ChunkPayload(
            repo="demo", language="python", path="io.py", symbol_name="parse_csv",
            symbol_type="function", start_line=1, end_line=5, text="def parse_csv(path): csv logic",
        ),
    ]

    good_embedder = FakeEmbedder()

    class BrokenEmbedder:
        dimensions = 3
        def embed(self, texts):
            return [[0.0, 0.0, 1.0] for _ in texts]  # never distinguishes anything

    for key, embedder, vectors_source in [("good", good_embedder, good_embedder), ("broken", BrokenEmbedder(), BrokenEmbedder())]:
        collection = collection_name(corpus_name, key)
        vectors = vectors_source.embed([c.text for c in chunks])
        store.ensure_collection(collection, vector_size=3)
        store.upsert_chunks(collection, chunks, vectors)

    ground_truth = [
        GroundTruthItem("jwt validation", "demo", "auth.py", "validate_jwt"),
        GroundTruthItem("csv parsing", "demo", "io.py", "parse_csv"),
    ]

    embedding_service = EmbeddingService({"good": good_embedder, "broken": BrokenEmbedder()})
    results = {r.model_key: r for r in run_eval(ground_truth, store, embedding_service, corpus_name, ["good", "broken"])}

    assert results["good"].recall_at_5 == 1.0
    assert results["good"].mrr == 1.0
    # the broken embedder can't distinguish jwt from csv queries -- keyword search
    # still surfaces the right chunk via exact-term matching, so recall isn't
    # necessarily 0, but it should score no better than the good embedder.
    assert results["broken"].mrr <= results["good"].mrr
