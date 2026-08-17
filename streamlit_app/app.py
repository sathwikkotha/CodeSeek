import json
import re
import time

import httpx
import streamlit as st

st.set_page_config(page_title="CodeSeek", page_icon="🔎", layout="wide")
st.session_state.setdefault("eval_log", [])

ICON_LOGO = '<svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline></svg>'
ICON_CHAT = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>'
ICON_BRANCH = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="6" y1="3" x2="6" y2="15"></line><circle cx="18" cy="6" r="3"></circle><circle cx="6" cy="18" r="3"></circle><path d="M18 9a9 9 0 0 1-9 9"></path></svg>'
ICON_LAYERS = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>'
ICON_TOOL = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"></path></svg>'

st.markdown(
    """
    <style>
    :root { --cs-font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }

    /* hide Streamlit's own chrome so this reads as a built page, not a template */
    #MainMenu, footer, [data-testid="stToolbarActions"], [data-testid="stAppDeployButton"] { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent; }
    html, body, [class*="css"] { font-family: var(--cs-font); }

    .cs-hero-wrap { position: relative; padding: 0.25rem 0 1.6rem; }
    .cs-hero-wrap::before {
      content: ""; position: absolute; top: -60px; left: -5%; width: 55%; height: 240px;
      background: radial-gradient(ellipse at center, rgba(88,166,255,0.16), transparent 72%);
      pointer-events: none; z-index: -1;
    }
    .cs-eyebrow { display: flex; align-items: center; gap: 0.5rem; color: #7ee787; font-size: 0.78rem;
                  font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; margin-bottom: 0.7rem; }
    .cs-brand { display: flex; align-items: center; gap: 0.65rem; font-size: 2.4rem; font-weight: 800;
                letter-spacing: -0.02em; color: #e6edf3; margin-bottom: 0.6rem; }
    .cs-brand svg { color: #58a6ff; }
    .cs-lede { color: #8b949e; font-size: 1.02rem; max-width: 72ch; line-height: 1.65; }
    .cs-lede strong { color: #c9d1d9; font-weight: 600; }

    .cs-panel-head { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.3rem; }
    .cs-panel-head svg { flex-shrink: 0; color: #58a6ff; }
    .cs-panel-head.cs-accent-2 svg { color: #a371f7; }
    .cs-panel-title { font-size: 1.08rem; font-weight: 700; color: #e6edf3; }
    .cs-panel-sub { color: #8b949e; font-size: 0.89rem; margin: 0 0 1.1rem 1.85rem; line-height: 1.55; }

    div[data-testid="stForm"] {
      border: 1px solid #30363d; border-radius: 12px; padding: 1.35rem 1.4rem;
      background: linear-gradient(180deg, #131720, #0d1117);
      transition: border-color 0.2s ease;
    }
    div[data-testid="stForm"]:hover { border-color: #3d444d; }

    .repo-badge { display: inline-block; font-family: ui-monospace, "SF Mono", Consolas, monospace;
                  font-size: 0.8rem; background: #161b22; border: 1px solid #30363d; color: #7ee787;
                  padding: 0.2rem 0.65rem; border-radius: 999px; margin: 0.15rem 0.3rem 0.15rem 0; }
    .repo-badge .count { color: #8b949e; }

    .cs-answer { border: 1px solid #30363d; border-radius: 12px; padding: 1.4rem 1.6rem;
                 background: linear-gradient(180deg, #10161f, #0d1117); line-height: 1.7;
                 color: #e6edf3; font-size: 1rem; }
    .cs-tool-call { font-family: ui-monospace, "SF Mono", Consolas, monospace; font-size: 0.82rem;
                    color: #7ee787; margin-bottom: 0.2rem; }
    .cs-tool-result { color: #8b949e; font-size: 0.82rem; white-space: pre-wrap; margin: 0 0 0.9rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Connection")
api_url = st.sidebar.text_input("CodeSeek API URL", "http://localhost:8000")


TOOL_ICONS = {"search_code": "🔍", "follow_symbol": "🧭", "read_file": "📄"}


def _tool_call_label(tool: str, args: dict) -> str:
    icon = TOOL_ICONS.get(tool, "🔧")
    if tool == "search_code":
        return f'{icon} Searching for "{args.get("query", "")}"'
    if tool == "follow_symbol":
        return f'{icon} Looking up `{args.get("name", "")}`'
    if tool == "read_file":
        target = args.get("path", "")
        start, end = args.get("start_line"), args.get("end_line")
        if start and end:
            target += f":{start}-{end}"
        return f"{icon} Reading `{target}`"
    return f"{icon} Calling {tool}"


def _parse_sse_line(line: str) -> dict | None:
    if not line.startswith("data: "):
        return None
    return json.loads(line[len("data: "):])


_GITHUB_FILE_VIEW_URL = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/(?:blob|tree)/[^/]+(?:/.*)?/?$"
)


def _normalize_repo_url(url: str) -> tuple[str, bool]:
    """Auto-fix the single most common paste mistake: a GitHub file- or
    branch-view link (.../blob/main/notebook.ipynb, .../tree/main/src) copied
    from the browser address bar instead of the actual clonable repo URL.
    Without this, resubmitting the same pasted link fails identically every
    time, since the form doesn't clear the field on a failed submit -- the
    user has no way to "retry" without realizing they must retype the URL."""
    match = _GITHUB_FILE_VIEW_URL.match(url.strip())
    if not match:
        return url, False
    return f"https://github.com/{match['owner']}/{match['repo']}.git", True


def _http_error_detail(e: httpx.HTTPError) -> str:
    """FastAPI puts the actually-useful message in the JSON body's "detail"
    field -- str(e) alone is just a generic "Server error '422 ...'" line."""
    response = getattr(e, "response", None)
    if response is not None:
        try:
            detail = response.json().get("detail")
            if detail:
                return str(detail)
        except (ValueError, KeyError):
            pass
    return str(e)


def _eval_verdict(rec: dict) -> str:
    """A deterministic groundedness label, not a correctness judgment -- it
    only knows whether the file:line citations the agent made are real, not
    whether the claims attached to them are. No LLM re-grades the answer:
    that would add cost and latency, and reintroduce (one layer up) the
    exact hallucination risk this project's citation check exists to catch."""
    if rec["error"]:
        return "⛔ Error"
    if not rec["answered"]:
        return "⚠ No answer"
    if rec["citations_total"] == 0:
        return "○ No citations made"
    if rec["citations_valid"] == rec["citations_total"]:
        return "✓ Grounded"
    return f"⚠ Partial ({rec['citations_valid']}/{rec['citations_total']})"


@st.cache_data(ttl=5)
def fetch_repos(url: str) -> list[dict] | None:
    try:
        resp = httpx.get(f"{url}/repos", timeout=10.0)
        resp.raise_for_status()
        return resp.json()["repos"]
    except httpx.HTTPError:
        return None


st.markdown('<div class="cs-hero-wrap">', unsafe_allow_html=True)
st.markdown(f'<div class="cs-eyebrow">{ICON_LAYERS} Point it at a repo, then ask</div>', unsafe_allow_html=True)
st.markdown(f'<div class="cs-brand">{ICON_LOGO} CodeSeek</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="cs-lede">Index any public GitHub repo, then ask it questions in plain English. '
    'An agent searches the codebase, follows references, and reads source directly — the same '
    'way you would — before writing a <strong>grounded, cited answer</strong>, not just a list of '
    'matching snippets.</div>',
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)

repos = fetch_repos(api_url)
if repos is None:
    st.error(f"Can't reach the CodeSeek API at `{api_url}`. Is `uvicorn codeseek.api.main:app` running?")
    st.stop()

repo_names = [r["name"] for r in repos]

with st.expander(f"Indexed repos — {len(repos)} ({sum(r['chunk_count'] for r in repos):,} chunks)", expanded=not repos):
    if repos:
        st.markdown(
            "".join(f'<span class="repo-badge">{r["name"]} <span class="count">{r["chunk_count"]}</span></span>' for r in repos),
            unsafe_allow_html=True,
        )
    st.write("")
    st.markdown(f'<div class="cs-panel-head cs-accent-2">{ICON_BRANCH}<span class="cs-panel-title">Index a new repo</span></div>', unsafe_allow_html=True)
    with st.form("ingest_form"):
        ingest_url = st.text_input("GitHub repo URL", placeholder="https://github.com/psf/black.git")
        ingest_name = st.text_input("Name (optional -- inferred from the URL)")
        ingest_submitted = st.form_submit_button("Clone + Index", use_container_width=True)

    if ingest_submitted:
        if not ingest_url:
            st.warning("Enter a repo URL first.")
        else:
            resolved_url, was_fixed = _normalize_repo_url(ingest_url)
            if was_fixed:
                st.info(
                    f"That looked like a link to a file or branch view, not the repo itself -- "
                    f"using `{resolved_url}` instead."
                )

            # st.status collapses (unmounting its children) as soon as .update()
            # sets a terminal state without re-passing expanded=True -- so the
            # error text is rendered after the `with` block exits, as a
            # page-level element, instead of nested inside the status box
            # where it would disappear the instant the box collapses.
            error_detail = None
            with st.status("Cloning and indexing...", expanded=True) as status:
                try:
                    ingest_resp = httpx.post(
                        f"{api_url}/ingest", json={"url": resolved_url, "name": ingest_name or None}, timeout=1800.0,
                    )
                    ingest_resp.raise_for_status()
                    ingest_body = ingest_resp.json()
                    status.update(label=f"Indexed {ingest_body['chunks_indexed']} chunks from '{ingest_body['repo']}'.", state="complete")
                except httpx.HTTPError as e:
                    status.update(label="Indexing failed.", state="error")
                    error_detail = _http_error_detail(e)

            if error_detail:
                st.error(error_detail)
            else:
                fetch_repos.clear()
                st.rerun()

st.write("")

if not repo_names:
    st.info("No repos indexed yet -- open the panel above and index one to get started.")
    st.stop()

tab_ask, tab_evals, tab_search = st.tabs(["💬  Ask", "📊  Evals", "🔍  Raw search (debug retrieval)"])

with tab_ask:
    st.markdown(f'<div class="cs-panel-head">{ICON_CHAT}<span class="cs-panel-title">Ask about a repo</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="cs-panel-sub">One repo at a time -- the agent searches, follows references, and reads source before answering.</div>', unsafe_allow_html=True)
    with st.form("explain_form"):
        ask_repo = st.selectbox("Repo", repo_names)
        question = st.text_area(
            "Question", placeholder="How does this library represent a configuration option internally?",
            label_visibility="collapsed", height=90,
        )
        ask_submitted = st.form_submit_button("Explain", type="primary", use_container_width=True)

    if ask_submitted and question:
        st.divider()
        status = st.status("🤔 Thinking...", expanded=True)
        final: dict = {}

        def _stream_answer():
            try:
                with httpx.stream(
                    "POST", f"{api_url}/explain/stream", json={"repo": ask_repo, "question": question}, timeout=180.0,
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        event = _parse_sse_line(line)
                        if event is None:
                            continue
                        etype = event["type"]
                        if etype == "status":
                            status.update(label="✍️ Writing the answer..." if event["phase"] == "generating" else "🤔 Thinking...")
                        elif etype == "tool_call":
                            status.write(_tool_call_label(event["tool"], event["input"]))
                        elif etype == "answer_delta":
                            yield event["text"]
                        elif etype == "error":
                            final["error"] = event.get("message", "Unknown error")
                        elif etype == "done":
                            final.update(event)
            except httpx.HTTPError as e:
                final["error"] = _http_error_detail(e)

        t0 = time.perf_counter()
        answer_text = st.write_stream(_stream_answer())
        latency_s = time.perf_counter() - t0

        if final.get("error"):
            status.update(label="Failed", state="error")
            st.error(f"The agent didn't finish cleanly: {final['error']}")
        else:
            status.update(label="✓ Done", state="complete")
            if not answer_text:
                st.warning("No answer was produced.")

        checks = final.get("citation_checks", [])
        if checks:
            bad = [c for c in checks if not c["valid"]]
            if bad:
                st.error(
                    f"⚠ {len(bad)} of {len(checks)} citation(s) don't check out — the file or line "
                    f"range doesn't exist: " + ", ".join(f"`{c['citation']}` ({c['reason']})" for c in bad)
                )
            else:
                st.caption(f"✓ All {len(checks)} citation(s) verified against the real repo (file + line range exist).")

        usage = final.get("usage")
        if usage:
            cache_pct = (usage["cached_input_tokens"] / usage["input_tokens"] * 100) if usage["input_tokens"] else 0
            u1, u2, u3, u4 = st.columns(4)
            u1.metric("Cost", f"${usage['cost_usd']:.4f}")
            u2.metric("Model requests", usage["requests"])
            u3.metric("Input tokens", f"{usage['input_tokens']:,}")
            u4.metric("Cached", f"{cache_pct:.0f}%")

        tool_calls = final.get("tool_calls", [])
        if tool_calls:
            with st.expander(f"How it got there — {len(tool_calls)} tool call(s)"):
                for tc in tool_calls:
                    st.markdown(f'<div class="cs-tool-call">{ICON_TOOL} {tc["tool"]}({tc["input"]})</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="cs-tool-result">{tc["result_summary"]}</div>', unsafe_allow_html=True)

        usage_for_log = usage or {}
        st.session_state.eval_log.append({
            "time": time.strftime("%H:%M:%S"),
            "repo": ask_repo,
            "question": question,
            "error": final.get("error"),
            "answered": bool(final.get("answer")),
            "citations_valid": sum(1 for c in checks if c["valid"]),
            "citations_total": len(checks),
            "tool_calls": len(tool_calls),
            "requests": usage_for_log.get("requests", 0),
            "cost_usd": usage_for_log.get("cost_usd", 0.0),
            "latency_s": latency_s,
        })
    elif ask_submitted:
        st.warning("Enter a question first.")

with tab_evals:
    st.markdown(f'<div class="cs-panel-head">{ICON_LAYERS}<span class="cs-panel-title">Evals</span></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cs-panel-sub">Every question asked this session, scored by <strong>citation-verification pass '
        'rate</strong> — whether each <code>file:line</code> the agent cited actually exists in the repo. That measures '
        'groundedness, not semantic correctness: a wrong claim can still cite a real line, and nothing here re-judges the '
        'answer with another LLM call — that would cost money and reintroduce, one layer up, the exact hallucination risk '
        'this check exists to catch.</div>',
        unsafe_allow_html=True,
    )

    log = st.session_state.eval_log
    if not log:
        st.info("Ask a question in the Ask tab to start building eval history for this session.")
    else:
        answered = [r for r in log if r["answered"] and not r["error"]]
        cited = [r for r in answered if r["citations_total"] > 0]
        fully_grounded = [r for r in cited if r["citations_valid"] == r["citations_total"]]
        total_citations = sum(r["citations_total"] for r in log)
        total_valid = sum(r["citations_valid"] for r in log)
        grounded_rate = (len(fully_grounded) / len(cited) * 100) if cited else 0.0
        citation_pass_rate = (total_valid / total_citations * 100) if total_citations else 0.0
        total_cost = sum(r["cost_usd"] for r in log)
        avg_latency = sum(r["latency_s"] for r in log) / len(log)

        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("Questions asked", len(log))
        e2.metric("Fully grounded", f"{grounded_rate:.0f}%")
        e3.metric("Citation pass rate", f"{citation_pass_rate:.0f}%")
        e4.metric("Total cost", f"${total_cost:.4f}")
        e5.metric("Avg latency", f"{avg_latency:.1f}s")

        st.write("")
        st.dataframe(
            [
                {
                    "Time": r["time"],
                    "Repo": r["repo"],
                    "Question": r["question"] if len(r["question"]) <= 80 else r["question"][:77] + "...",
                    "Verdict": _eval_verdict(r),
                    "Citations": f'{r["citations_valid"]}/{r["citations_total"]}' if r["citations_total"] else "—",
                    "Tool calls": r["tool_calls"],
                    "Cost": f'${r["cost_usd"]:.4f}',
                    "Latency": f'{r["latency_s"]:.1f}s',
                }
                for r in reversed(log)
            ],
            use_container_width=True,
            hide_index=True,
        )

        if st.button("Clear eval history"):
            st.session_state.eval_log = []
            st.rerun()

with tab_search:
    st.markdown(f'<div class="cs-panel-head">{ICON_LAYERS}<span class="cs-panel-title">Raw retrieval</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="cs-panel-sub">Bypasses the agent -- shows exactly what hybrid search + reranking returns, for debugging retrieval quality directly.</div>', unsafe_allow_html=True)
    with st.form("search_form"):
        query = st.text_input("Question", placeholder="how is a command-line progress bar implemented", label_visibility="collapsed")
        f1, f2 = st.columns(2)
        repo_filter = f1.selectbox("Search within", ["All indexed repos"] + repo_names)
        language_filter = f2.text_input("Language (optional)", placeholder="python / javascript")
        top_k = st.slider("Results", min_value=1, max_value=50, value=10)
        submitted = st.form_submit_button("Search", use_container_width=True)

    if submitted and query:
        payload = {
            "query": query,
            "top_k": int(top_k),
            "repo": repo_filter if repo_filter in repo_names else None,
            "language": language_filter or None,
        }
        try:
            resp = httpx.post(f"{api_url}/search", json=payload, timeout=30.0)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            st.error(f"Couldn't reach the CodeSeek API at `{api_url}`: {e}")
            st.stop()

        body = resp.json()
        results, timings = body["results"], body["timings"]

        st.divider()
        total_ms = sum(timings.values())
        t1, t2, t3, t4, t5, t6 = st.columns(6)
        t1.metric("Total", f"{total_ms:.0f} ms")
        t2.metric("Embed", f"{timings['embed_ms']:.1f} ms")
        t3.metric("Vector search", f"{timings['vector_search_ms']:.1f} ms")
        t4.metric("Keyword search", f"{timings['keyword_search_ms']:.1f} ms")
        t5.metric("Merge", f"{timings['merge_ms']:.2f} ms")
        t6.metric("Rerank", f"{timings['rerank_ms']:.1f} ms")

        if not results:
            st.info("No results for this query/filter combination.")

        for r in results:
            with st.container(border=True):
                header_col, score_col = st.columns([5, 1])
                header_col.markdown(f"**{r['repo']}** · `{r['path']}` · **{r['symbol_name']}** ({r['symbol_type']})")
                score_col.markdown(f"score `{r['score']:.3f}` · via *{r['source']}*")
                lang = "python" if r["path"].endswith(".py") else ("javascript" if r["path"].endswith((".js", ".ts")) else None)
                st.code(r["text"], language=lang)
    elif submitted:
        st.warning("Enter a query first.")
