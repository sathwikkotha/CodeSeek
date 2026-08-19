# Retrieval ablation results

Measured 2026-08-19 against CodeSeek's own source tree (138 chunks, 10 ground-truth questions -- see `scripts/self_eval_ground_truth.json`). Reproduce with `python scripts/run_ablation.py`.

Each row adds one stage on top of the row above it -- this is the actual pipeline in `retrieval/hybrid.py`, measured stage by stage, not four independent systems.

| Stage | Recall@5 | Recall@10 | MRR |
|---|---|---|---|
| 1. Vector-only (bi-encoder cosine similarity) | 1.000 | 1.000 | 1.000 |
| 2. + keyword hybrid (vector + lexical merge) | 1.000 | 1.000 | 1.000 |
| 3. + cross-encoder rerank (no fusion) | 1.000 | 1.000 | 0.803 |
| 4. + RRF fusion (production default) | 1.000 | 1.000 | 1.000 |

## Reading this table honestly

This 10-question, 138-chunk self-eval set is small and deliberately
unambiguous (each question targets one clearly-named function/method) --
that's what makes it cheap and stable enough to run on every CI push, but it
also means most stages hit a ceiling of 1.000 rather than showing a gradual
curve. Its job is regression detection (did this PR make retrieval *worse*),
not benchmarking how hard retrieval can get.

The one real signal in this run is stage 3 vs. 4, and it's the exact failure
mode `retrieval/hybrid.py` is written to guard against: reranking alone
**drops MRR from 1.000 to 0.803** -- the cross-encoder, trained on
natural-language passage ranking, pushed at least one correct answer further
down the list even though hybrid retrieval had already ranked it first. RRF
fusion (stage 4) recovers to 1.000 by blending the reranker's opinion with
the pre-rerank order instead of trusting it outright. That's not a
theoretical justification anymore -- it's what happened, on this run, on
this codebase.

For a harder, more discriminating benchmark (one that would show a real
curve instead of a ceiling), see `docs/scaling_results.md`, which measures
across several full external repos instead of a 10-question curated set.
