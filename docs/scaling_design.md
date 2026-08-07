# Scaling Design: Sharding, Access Control, Incremental Re-indexing

This document is a design, not a build — it covers the three things CodeSeek's
project scope explicitly excluded from v1 (issue/PR/wiki ingestion aside,
which is a data-pipeline problem, not a retrieval one). It describes how the
existing architecture extends past a single-corpus, single-machine deployment,
and why each choice follows directly from decisions already made in the code.

## 1. Sharding strategy

**Shard by repository, not by language or a hash of the point ID.**

Today, every repo lands in one pair of shared collections
(`<corpus>__general` / `<corpus>__code`, see `store/schema.py:collection_name`),
filtered at query time by the `repo` field already on every point's payload
(`ChunkPayload.repo`, threaded through `store/qdrant_store.py:build_metadata_filter`).
That's fine at the scale this project runs at (tens of thousands of points on
one machine). Past that, the natural shard boundary is the repository itself:

- **Co-location.** A repo's chunks are almost always queried together (a
  repo-filtered search, or an unfiltered search where a repo either
  contributes results or doesn't). Sharding by repo keeps a query's relevant
  data on as few shards as possible — the opposite of hashing by point ID,
  which would scatter one repo's ~1-2k chunks evenly across every shard and
  turn every query into a full fan-out regardless of filters.
- **Natural re-indexing unit.** Re-indexing a repo (Section 3) becomes a
  single-shard operation: touch the shard that repo lives on, nothing else.
  A hash-based shard would spray those writes across the whole cluster.
- **Natural deletion/eviction unit.** Removing a repo from the corpus (a
  license change, a user request, decommissioning) is "drop this shard's
  data for this repo," not a scan-and-delete across every shard.
- **Placement.** New repos are assigned to the least-loaded shard by current
  point count (a simple bin-packing heuristic is enough here — repo sizes are
  known at ingestion time from the manifest step, so this doesn't need to be
  dynamic). A repo is never split across shards; if one repo alone would
  overwhelm a shard, that's a signal to raise the per-shard capacity target,
  not to split a single repository's chunks apart.

Qdrant supports this natively via multi-node collections with a
`shard_key` on upsert/query — the existing `QdrantStore.upsert_chunks` /
`vector_search` / `keyword_search` calls would pass `repo` as the shard key
instead of relying on payload filtering alone once the deployment moves off a
single node.

## 2. Repo-level access control

The mechanism already exists — it just isn't gated by identity yet.

Every search already flows through `build_metadata_filter(repo, language)`
in `store/qdrant_store.py`, which becomes a Qdrant `Filter(must=[...])`
applied to both `vector_search` and `keyword_search`
(`retrieval/hybrid.py:HybridRetriever.search`). Access control is adding one
more mandatory condition to that same filter, not a new subsystem:

1. An auth layer (API key or session token, checked in the FastAPI
   `/search` route in `api/app.py`) resolves the caller's **allowed repo
   list** — a simple per-token allowlist for a first version; a real ACL
   table (user → org → repo grants) once there's more than one tenant.
2. That list is intersected with whatever `repo` filter the caller
   requested. If the caller didn't specify a repo, the allowed-list becomes
   an OR'd `should` filter (any of these repos), not a bypass — a caller
   with access to 3 of 24 repos still only ever sees results from those 3
   in an unfiltered search.
3. Because the check is expressed as a Qdrant filter rather than a
   post-retrieval filter, a caller can never see a chunk from a repo they
   don't have access to even transiently (no "fetch top-k then discard" step
   that could leak a snippet through a bug) — enforcement happens at the
   vector engine, not in application code that's easier to get wrong.

This is deliberately boring: no new data model, no new query path — the
`repo` payload field and the filter-merging logic in `hybrid.py` were already
built for the language/repo filters in the Streamlit search UI, so ACL reuses
exactly that.

## 3. Incremental re-indexing via webhook

Today, `scripts/build_index.py` re-chunks and re-embeds an entire repo from
scratch every run. That's fine for a one-off corpus build; it doesn't scale
to "re-index on every commit" across 24+ repos.

**Flow:**

1. A GitHub webhook (`push` event) hits a small endpoint that enqueues
   `{repo, commit_sha, changed_files}` — GitHub's payload already lists
   added/modified/removed files per commit, so no local diffing against the
   old tree is needed.
2. For each **added or modified** file: re-run exactly the per-file path
   `pipeline/index.py:_file_to_chunks` already implements — walk it through
   the chunker matching its category (`chunk_python_source` /
   `chunk_js_ts_source` / `chunk_doc_text`), embed the resulting chunks, and
   upsert.
3. **Upsert is already idempotent for changed content**, for free:
   `ChunkPayload.point_id()` derives a deterministic `uuid5` from
   `repo:path:symbol_name:start_line:end_line`. If a function's body changed
   but its name/line range didn't move, the upsert overwrites the same point
   ID — no duplicate, no stale copy alongside the new one. If line numbers
   shifted (a function moved because code was added above it), it upserts
   under a new ID and the old one becomes an orphan — which is why step 4
   below always re-processes the whole changed file rather than trying to
   diff at the chunk level: recomputing every chunk in a changed file and
   deleting whatever pre-existing points in that repo+path aren't in the new
   set is simpler and cheaper to reason about than tracking chunk-level
   moves.
4. For each changed or **removed** file: scroll-delete every existing point
   matching `repo == X AND path == Y` (the same `build_metadata_filter`
   shape used everywhere else) before upserting the freshly computed set.
   This is what actually makes step 3's "new ID on a line shift" case safe —
   the old point for that file is gone regardless of whether its ID changed.
5. A repo whose default branch moved backward (a force-push) is handled the
   same way as any other diff — GitHub's webhook payload still enumerates
   the file-level delta, so there's no special case.

**Why this is cheap relative to a full rebuild:** a typical commit touches a
handful of files, not the whole repo — re-embedding is bounded by the
already-measured throughput (~14 chunks/sec/model from `scripts/cost_measurement.py`)
against a handful of files' worth of chunks, not the repo's full chunk count.
