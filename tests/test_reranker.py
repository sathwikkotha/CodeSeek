import pytest

from codeseek.retrieval.hybrid import MergedHit


def _hit(id_: str, text: str, score: float = 0.5) -> MergedHit:
    return MergedHit(id=id_, score=score, payload={"text": text, "symbol_type": "function"}, source="vector")


@pytest.mark.slow
def test_real_cross_encoder_ranks_the_actually_relevant_chunk_first():
    """Downloads and runs the real cross-encoder once, on a case deliberately
    designed so a bi-encoder (cosine similarity) would likely be fooled by
    surface word overlap, to prove the cross-encoder is doing real semantic
    matching, not just repeating the input order."""
    from codeseek.retrieval.reranker import CrossEncoderReranker

    hits = [
        _hit("wrong", "def parse_csv(path):\n    \"\"\"Parse a CSV file into rows.\"\"\"\n    return open(path).readlines()"),
        _hit("right", "def validate_jwt(token):\n    \"\"\"Decode and verify a JWT signature, raise if invalid.\"\"\"\n    return jwt.decode(token)"),
    ]

    reranker = CrossEncoderReranker()
    reranked = reranker.rerank("how is JWT signature verification implemented", hits, top_k=2)

    assert reranked[0].id == "right"
