import json
import os
import re
import time

import httpx
import streamlit as st

st.set_page_config(page_title="CodeSeek", page_icon="◈", layout="wide")
st.session_state.setdefault("eval_log", [])
st.session_state.setdefault("chat_history", {})  # repo name -> list of {"role", "content", ...}
st.session_state.setdefault("active_repo", None)
st.session_state.setdefault("view_mode", "chat")  # "chat" | "search" | "evals"

# In Docker Compose, "localhost" from inside the UI container doesn't reach the
# API container -- CODESEEK_API_URL (set to the API's service name there) overrides
# the bare-metal default. Still just a starting value the sidebar field can edit.
DEFAULT_API_URL = os.environ.get("CODESEEK_API_URL", "http://localhost:8000")

ICON_TOOL = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"></path></svg>'

st.markdown(
    """
    <style>
    :root { --cs-font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }

    /* hide Streamlit's own chrome so this reads as a built page, not a template */
    #MainMenu, footer, [data-testid="stToolbarActions"], [data-testid="stAppDeployButton"] { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent; }
    html, body, [class*="css"] { font-family: var(--cs-font); }

    .cs-sidebar-brand { font-size: 1.15rem; font-weight: 700; letter-spacing: 0.02em; text-transform: uppercase;
                         color: #e6edf3; margin: 0 0 1.1rem 0; }
    .cs-sidebar-caption { color: #6e7681; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
                           text-transform: uppercase; margin: 0.2rem 0 0.4rem 0.1rem; }

    /* sidebar repo/nav buttons read as a flat list, not a grid of boxed buttons */
    section[data-testid="stSidebar"] div[data-testid="stButton"] button {
      text-align: left; justify-content: flex-start; border-color: transparent; background: transparent;
      font-weight: 400;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover { background: #161b22; border-color: #30363d; }
    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
      background: #1f2937; border-color: #30363d; color: #58a6ff; font-weight: 600;
    }

    /* the Ask / Raw search / Evals nav -- a plain, typographic list, not iconography */
    section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 0.05rem; }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
      padding: 0.3rem 0.4rem; border-radius: 6px; width: 100%;
    }

    .cs-tool-call { font-family: ui-monospace, "SF Mono", Consolas, monospace; font-size: 0.82rem;
                    color: #7ee787; margin-bottom: 0.2rem; }
    .cs-tool-result { color: #8b949e; font-size: 0.82rem; white-space: pre-wrap; margin: 0 0 0.9rem 0; }

    /* bordered containers (the "index a repo" card, search-result cards) share one
       card language across the page instead of each looking like a different widget */
    [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] {
      background: linear-gradient(180deg, #131720, #0d1117); border-radius: 10px;
    }
    div[data-testid="stTextInput"] input {
      background: #0d1117; border-color: #30363d;
    }
    div[data-testid="stTextInput"] input:focus { border-color: #58a6ff; box-shadow: none; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _tool_call_label(tool: str, args: dict) -> str:
    if tool == "search_code":
        return f'Searching for "{args.get("query", "")}"'
    if tool == "follow_symbol":
        return f'Looking up `{args.get("name", "")}`'
    if tool == "read_file":
        target = args.get("path", "")
        start, end = args.get("start_line"), args.get("end_line")
        if start and end:
            target += f":{start}-{end}"
        return f"Reading `{target}`"
    return f"Calling {tool}"


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
    from the browser address bar instead of the actual clonable repo URL."""
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
    whether the claims attached to them are."""
    if rec["error"]:
        return "Error"
    if not rec["answered"]:
        return "No answer"
    if rec["citations_total"] == 0:
        return "No citations made"
    if rec["citations_valid"] == rec["citations_total"]:
        return "Grounded"
    return f"Partial ({rec['citations_valid']}/{rec['citations_total']})"


@st.cache_data(ttl=5)
def fetch_repos(url: str) -> list[dict] | None:
    try:
        resp = httpx.get(f"{url}/repos", timeout=10.0)
        resp.raise_for_status()
        return resp.json()["repos"]
    except httpx.HTTPError:
        return None
    except (ValueError, KeyError):
        # A 2xx with an empty/malformed body (e.g. the API mid-restart) isn't an
        # httpx.HTTPError -- resp.json() raises json.JSONDecodeError (a ValueError)
        # instead. Treated the same as "can't reach it": show the connection error,
        # not an unhandled traceback that leaves the whole page stuck.
        return None


def _render_assistant_turn(turn: dict) -> None:
    """Renders one assistant chat turn's citation/cost/tool-call footnotes --
    shared by the live streaming turn and every past turn replayed from
    session_state, so both look identical."""
    if turn.get("error"):
        st.error(f"The agent didn't finish cleanly: {turn['error']}")

    checks = turn.get("citations") or []
    if checks:
        bad = [c for c in checks if not c["valid"]]
        if bad:
            st.error(
                f"{len(bad)} of {len(checks)} citation(s) don't check out — the file or line "
                f"range doesn't exist: " + ", ".join(f"`{c['citation']}` ({c['reason']})" for c in bad)
            )
        else:
            st.caption(f"All {len(checks)} citation(s) verified against the real repo (file + line range exist).")

    usage = turn.get("usage")
    if usage:
        cache_pct = (usage["cached_input_tokens"] / usage["input_tokens"] * 100) if usage["input_tokens"] else 0
        u1, u2, u3, u4 = st.columns(4)
        u1.metric("Cost", f"${usage['cost_usd']:.4f}")
        u2.metric("Model requests", usage["requests"])
        u3.metric("Input tokens", f"{usage['input_tokens']:,}")
        u4.metric("Cached", f"{cache_pct:.0f}%")

    tool_calls = turn.get("tool_calls") or []
    if tool_calls:
        with st.expander(f"How it got there — {len(tool_calls)} tool call(s)"):
            for tc in tool_calls:
                st.markdown(f'<div class="cs-tool-call">{ICON_TOOL} {tc["tool"]}({tc["input"]})</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="cs-tool-result">{tc["result_summary"]}</div>', unsafe_allow_html=True)


def _render_ingest_form(api_url: str, key_prefix: str) -> None:
    """The clone+index form, stacked layout -- used by the sidebar's always-
    available, compact copy. key_prefix keeps widget keys unique from the
    empty-state panel's own copy, since both can be on the page at once."""
    ingest_url = st.text_input(
        "GitHub repo URL", placeholder="https://github.com/psf/black.git", key=f"{key_prefix}_ingest_url",
    )
    ingest_name = st.text_input(
        "Name (optional -- inferred from the URL)", key=f"{key_prefix}_ingest_name",
    )
    _submit_ingest(api_url, ingest_url, ingest_name, key_prefix)


def _submit_ingest(api_url: str, ingest_url: str, ingest_name: str, key_prefix: str) -> None:
    """The "Clone + Index" button and the clone/index/poll flow behind it --
    factored out from _render_ingest_form so the empty-state panel can lay its
    two inputs out side by side instead of stacked, while still sharing this
    submission logic exactly."""
    if st.button("Clone + Index", use_container_width=True, type="primary", key=f"{key_prefix}_ingest_submit"):
        if not ingest_url:
            st.warning("Enter a repo URL first.")
            return

        resolved_url, was_fixed = _normalize_repo_url(ingest_url)
        if was_fixed:
            st.info(f"That looked like a link to a file or branch view -- using `{resolved_url}` instead.")

        # /ingest returns immediately with a job id (the actual clone+index
        # work runs on a background thread server-side) -- st.status collapses
        # (unmounting its children) as soon as .update() sets a terminal state
        # without re-passing expanded=True, so the error text is rendered after
        # the `with` block exits instead of disappearing with the collapsed box.
        error_detail = None
        job_body = None
        with st.status("Cloning and indexing...", expanded=True) as status:
            try:
                ingest_resp = httpx.post(
                    f"{api_url}/ingest", json={"url": resolved_url, "name": ingest_name or None}, timeout=30.0,
                )
                ingest_resp.raise_for_status()
                job_id = ingest_resp.json()["job_id"]

                job_body = {"state": "pending"}
                while job_body["state"] in ("pending", "running"):
                    time.sleep(1.0)
                    status_resp = httpx.get(f"{api_url}/ingest/{job_id}", timeout=10.0)
                    status_resp.raise_for_status()
                    job_body = status_resp.json()
                    if job_body["state"] == "running":
                        status.update(label=f"Cloning and indexing `{resolved_url}`...")

                if job_body["state"] == "error":
                    status.update(label="Indexing failed.", state="error")
                    error_detail = job_body["error"]
                else:
                    status.update(
                        label=f"Indexed {job_body['chunks_indexed']} chunks from '{job_body['repo']}'.",
                        state="complete",
                    )
            except httpx.HTTPError as e:
                status.update(label="Indexing failed.", state="error")
                error_detail = _http_error_detail(e)
            except (ValueError, KeyError):
                # A 2xx with an empty/malformed body -- same gap as fetch_repos.
                status.update(label="Indexing failed.", state="error")
                error_detail = "The API returned an unexpected response. Is it still running?"

        if error_detail:
            st.error(error_detail)
        else:
            fetch_repos.clear()
            st.session_state["active_repo"] = job_body["repo"]
            st.session_state["view_mode"] = "chat"
            st.rerun()


# ==================================================================== sidebar
with st.sidebar:
    st.markdown('<div class="cs-sidebar-brand">CodeSeek</div>', unsafe_allow_html=True)

    with st.expander("Connection"):
        api_url = st.text_input("CodeSeek API URL", DEFAULT_API_URL, label_visibility="collapsed")

    repos = fetch_repos(api_url)
    if repos is None:
        st.error(f"Can't reach the CodeSeek API at `{api_url}`. Is `uvicorn codeseek.api.main:app` running?")
        st.stop()
    repo_names = [r["name"] for r in repos]

    with st.expander("Index a new repo", expanded=False):
        _render_ingest_form(api_url, key_prefix="sidebar")

    st.divider()

    if not repo_names:
        st.caption("No repos indexed yet.")
    else:
        st.markdown(
            f'<div class="cs-sidebar-caption">Repos — {len(repos)} · {sum(r["chunk_count"] for r in repos):,} chunks</div>',
            unsafe_allow_html=True,
        )
        for r in repos:
            is_active = r["name"] == st.session_state["active_repo"] and st.session_state["view_mode"] == "chat"
            if st.button(
                r["name"], key=f"repo_btn_{r['name']}", use_container_width=True,
                type="primary" if is_active else "secondary", help=f"{r['chunk_count']} chunks indexed",
            ):
                st.session_state["active_repo"] = r["name"]
                st.session_state["view_mode"] = "chat"

    st.divider()
    st.markdown('<div class="cs-sidebar-caption">Navigate</div>', unsafe_allow_html=True)
    view_mode = st.radio(
        "View", options=["chat", "search", "evals"],
        format_func=lambda m: {"chat": "Ask", "search": "Raw search", "evals": "Evals"}[m],
        label_visibility="collapsed", key="view_mode",
    )

if not repo_names:
    # True first-run state: nothing indexed yet at all. Explains what CodeSeek is and
    # the three-step workflow, rather than a single bare info line above blank space.
    st.markdown("### Search any codebase in plain English")
    st.write(
        "CodeSeek indexes a real repository — every function, class, and doc page — with "
        "AST-aware chunking and hybrid (semantic + keyword) retrieval. When you ask a question, "
        "an agent searches the code, follows symbol references, and reads source directly before "
        "answering, then cites the exact file and line range for every claim it makes — checked "
        "against the real repo before the answer ever reaches you."
    )

    st.markdown("**How it works**")
    step_cols = st.columns(3)
    with step_cols[0]:
        st.markdown("**1. Index a repository**")
        st.caption("Paste any public GitHub URL below.")
    with step_cols[1]:
        st.markdown("**2. Ask a question**")
        st.caption("Plain English — no query syntax, no filters required.")
    with step_cols[2]:
        st.markdown("**3. Get a cited answer**")
        st.caption("Every file:line reference is verified against the real source.")

    st.divider()
    with st.container(border=True):
        st.markdown("**Index your first repository**")
        url_col, name_col = st.columns(2)
        with url_col:
            ingest_url = st.text_input(
                "GitHub repo URL", placeholder="https://github.com/psf/black.git", key="empty_state_ingest_url",
            )
        with name_col:
            ingest_name = st.text_input(
                "Name (optional — inferred from the URL)", key="empty_state_ingest_name",
            )
        _submit_ingest(api_url, ingest_url, ingest_name, key_prefix="empty_state")
    st.stop()

if st.session_state["active_repo"] not in repo_names:
    st.session_state["active_repo"] = repo_names[0]

# ================================================================= main panel
if view_mode == "chat":
    active_repo = st.session_state["active_repo"]
    active_repo_info = next((r for r in repos if r["name"] == active_repo), None)
    history = st.session_state["chat_history"].setdefault(active_repo, [])

    if history:
        st.caption(f"`{active_repo}`")

    if not history:
        # First-run welcome state -- explains what CodeSeek actually does and gives a
        # running start, instead of a blank page above the input bar. Disappears the
        # moment a real conversation starts (same as ChatGPT's empty-state behavior).
        chunk_count = active_repo_info["chunk_count"] if active_repo_info else 0
        st.markdown(f"### {active_repo}")
        st.caption(f"{chunk_count:,} chunks indexed")
        st.write(
            "CodeSeek's agent searches this repository, follows symbol references, and reads "
            "source directly before answering — not a one-shot similarity lookup. Every citation "
            "(`file:line`) it writes is checked against the real repo before being shown to you, "
            "so a wrong answer can't hide behind a source that doesn't actually exist."
        )
        st.markdown("**Try asking:**")
        suggestions = [
            "What does this repository do, at a high level?",
            "Where is the main entry point defined?",
            "How is configuration or setup handled here?",
            "Walk me through the overall architecture.",
        ]
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            if cols[i % 2].button(suggestion, use_container_width=True, key=f"suggest_{i}"):
                st.session_state["pending_question"] = suggestion
        st.divider()

    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn["role"] == "assistant":
                _render_assistant_turn(turn)

    question = st.chat_input(f"Ask about {active_repo}...", key=f"chat_input_{active_repo}")
    if not question:
        question = st.session_state.pop("pending_question", None)

    if question:
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            status = st.status("Thinking…", expanded=True)
            final: dict = {}

            def _stream_answer():
                try:
                    with httpx.stream(
                        "POST", f"{api_url}/explain/stream", json={"repo": active_repo, "question": question}, timeout=180.0,
                    ) as resp:
                        resp.raise_for_status()
                        for line in resp.iter_lines():
                            event = _parse_sse_line(line)
                            if event is None:
                                continue
                            etype = event["type"]
                            if etype == "status":
                                status.update(label="Writing the answer…" if event["phase"] == "generating" else "Thinking…")
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
            else:
                status.update(label="Done", state="complete")
                if not answer_text:
                    st.warning("No answer was produced.")

            checks = final.get("citation_checks", [])
            usage = final.get("usage")
            tool_calls = final.get("tool_calls", [])
            turn = {
                "role": "assistant",
                "content": answer_text or "_No answer was produced._",
                "error": final.get("error"),
                "citations": checks,
                "usage": usage,
                "tool_calls": tool_calls,
            }
            _render_assistant_turn(turn)
            history.append(turn)

        usage_for_log = usage or {}
        st.session_state.eval_log.append({
            "time": time.strftime("%H:%M:%S"),
            "repo": active_repo,
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

elif view_mode == "search":
    st.subheader("Raw retrieval")
    st.caption("Bypasses the agent -- shows exactly what hybrid search + reranking returns, for debugging retrieval quality directly.")

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
            body = resp.json()
        except httpx.HTTPError as e:
            st.error(f"Couldn't reach the CodeSeek API at `{api_url}`: {e}")
            st.stop()
        except (ValueError, KeyError):
            st.error("The API returned an unexpected response. Is it still running?")
            st.stop()
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

elif view_mode == "evals":
    st.subheader("Evals")
    st.caption(
        "Every question asked this session, scored by citation-verification pass rate -- whether each file:line "
        "the agent cited actually exists in the repo. That measures groundedness, not semantic correctness: a "
        "wrong claim can still cite a real line, and nothing here re-judges the answer with another LLM call -- "
        "that would cost money and reintroduce, one layer up, the exact hallucination risk this check exists to catch."
    )

    log = st.session_state.eval_log
    if not log:
        st.info("Ask a question in the Ask view to start building eval history for this session.")
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
