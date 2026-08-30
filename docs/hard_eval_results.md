# Hard-eval retrieval results

Production pipeline (vector + keyword hybrid + cross-encoder rerank + RRF fusion) measured against `scripts/hard_eval_ground_truth.json` (28 harder, naturally-paraphrased questions about CodeSeek's own source, 138 chunks). Reproduce a row with `python scripts/run_hard_eval.py --label <name> --note <what changed>`.

| Timestamp | Label | Recall@5 | Recall@10 | MRR | Note |
|---|---|---|---|---|---|
| 2026-08-30 20:18 | baseline | 0.786 | 0.929 | 0.608 | current production pipeline, before any changes |
| 2026-08-30 20:25 | after_comment_enrichment | 0.786 | 0.964 | 0.623 | extract leading # / JSDoc comments into embedding_text, previously invisible to both ast and tree-sitter node spans |
| 2026-08-30 20:28 | after_bm25 | 0.750 | 0.964 | 0.614 | replace flat floor-score for keyword-only hits with real BM25 relevance scaled into [floor, ceiling] |
| 2026-08-30 20:35 | after_hyde | 0.786 | 0.893 | 0.578 | HyDE-expand each query for the vector leg only; keyword/BM25 leg still sees the raw question |

## Reading this table honestly

**Methodology caveat, read this first:** this corpus is CodeSeek's own source
tree, and each of these four changes edited that same source tree (new
functions like `_leading_comment`, `_bm25_relevance_for_keyword_only`, and
`HydeQueryExpander.expand` are themselves new indexed chunks). Chunk count
climbed 138 -> 140 -> 142 -> 148 across these runs for exactly that reason.
Some of the run-to-run delta is attributable to this corpus drift, not purely
to the retrieval-logic change being tested -- a clean re-measurement would
freeze the corpus and only vary the retrieval code, which these four
back-to-back runs did not do. Directionally the results are still real and
reproducible (each stage is additive on top of the last, same as
`docs/eval_results.md`'s ablation), just not a perfectly isolated ablation.

**Comment/JSDoc enrichment (embedding/context.py, code_chunker.py,
js_ts_chunker.py) helped, modestly:** MRR 0.608 -> 0.623, Recall@10 0.929 ->
0.964, Recall@5 unchanged. Consistent with the mechanism -- surfacing
previously-invisible `#`/JSDoc comments gives the embedding more of the
natural-language signal a paraphrased question can match against. Low risk,
worth keeping regardless of the small sample size.

**Real BM25 keyword scoring (retrieval/hybrid.py) was roughly flat-to-negative
here:** MRR 0.623 -> 0.614, Recall@5 0.786 -> 0.750. This is a real, measured
result, not a hypothesis -- on this benchmark, replacing the flat floor score
with BM25-scaled scores for keyword-only hits did not help, and even
recall@5 dipped slightly. plausible reason: with only ~28 questions and a
138-148 chunk pool, the fetch_k candidate set BM25 is fit over each query is
tiny and noisy, so its relative rankings within that pool aren't necessarily
more trustworthy than the old flat floor. This does not mean the mechanism is
wrong in general (a larger corpus/eval set could show a different result) --
it means this specific implementation, measured on this specific benchmark,
isn't a clear win, and shouldn't be kept on the strength of the theoretical
argument alone.

**HyDE query expansion (retrieval/hyde.py) actively hurt here:** MRR 0.614 ->
0.578, Recall@10 0.964 -> 0.893 -- worse than even the original baseline
(0.608). Plausible reason: these ground-truth questions ask about CodeSeek's
own, fairly idiosyncratic internal retrieval/chunking implementation, not
general programming concepts -- a small, cheap model (`gpt-5.4-nano`)
generating a "plausible" hypothetical answer for a question this specific is
liable to generate a generic-sounding guess that drifts *away* from this
particular codebase's actual wording, rather than a bridge toward it. This is
the opposite of the failure mode HyDE is meant to fix, and is a real
measured result: **HyDE is not a safe default here** and stays behind an
opt-in `--hyde` flag, not wired into the production `/search` or agent path.

The one change from the original four-item plan that this benchmark
structurally cannot measure is **identifier-query routing**
(`_looks_like_identifier` in retrieval/hybrid.py) -- every ground-truth
question here is a full natural-language sentence by design, so routing
(which only triggers on a bare symbol name with no spaces) never fires on
this set. It's verified instead by three dedicated unit tests in
`tests/test_retrieval.py` proving the mechanism itself works correctly.
