"""Recall@k and MRR: per-question retrieval metrics. Both operate on a single
question's ranked result IDs plus the set of IDs that count as a correct
answer for it; the harness averages these across the full ground-truth set."""


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return 1.0 if set(ranked_ids[:k]) & relevant_ids else 0.0


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, item_id in enumerate(ranked_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0
