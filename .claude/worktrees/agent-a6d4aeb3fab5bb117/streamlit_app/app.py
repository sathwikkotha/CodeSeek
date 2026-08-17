"""Surface 1 -- Search UI. Calls the CodeSeek FastAPI /search endpoint and
renders the general-purpose and code-specific model rankings side by side
(in tabs), plus the real per-stage latency for the query just run."""

import httpx
import streamlit as st

st.set_page_config(page_title="CodeSeek", page_icon="🔎", layout="wide")

st.sidebar.header("Connection")
api_url = st.sidebar.text_input("CodeSeek API URL", "http://localhost:8000")

st.title("🔎 CodeSeek")
st.caption("Semantic code search over real repositories -- compare a general-purpose vs. a code-specific embedding model.")

with st.form("search_form"):
    query = st.text_input("Ask a question about the indexed codebases", placeholder="where is JWT validation implemented")
    col1, col2, col3 = st.columns(3)
    repo_filter = col1.text_input("Filter by repo (optional)")
    language_filter = col2.text_input("Filter by language (optional)")
    top_k = col3.number_input("Top K", min_value=1, max_value=50, value=10)
    submitted = st.form_submit_button("Search", type="primary")

if submitted and query:
    payload = {
        "query": query,
        "models": ["general", "code"],
        "top_k": int(top_k),
        "repo": repo_filter or None,
        "language": language_filter or None,
    }
    try:
        resp = httpx.post(f"{api_url}/search", json=payload, timeout=30.0)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        st.error(f"Couldn't reach the CodeSeek API at `{api_url}`. Is `uvicorn codeseek.api.main:app` running?\n\n{e}")
        st.stop()

    body = resp.json()
    results_by_model = body["results_by_model"]
    timings_by_model = body["timings_by_model"]

    model_labels = {"general": "General-purpose model (all-MiniLM-L6-v2)", "code": "Code-specific model (st-codesearch-distilroberta)"}
    tabs = st.tabs([model_labels.get(k, k) for k in results_by_model])

    for tab, model_key in zip(tabs, results_by_model):
        with tab:
            results = results_by_model[model_key]
            timings = timings_by_model.get(model_key, {})

            if timings:
                total_ms = sum(timings.values())
                t1, t2, t3, t4, t5 = st.columns(5)
                t1.metric("Total", f"{total_ms:.0f} ms")
                t2.metric("Embed", f"{timings['embed_ms']:.1f} ms")
                t3.metric("Vector search", f"{timings['vector_search_ms']:.1f} ms")
                t4.metric("Keyword search", f"{timings['keyword_search_ms']:.1f} ms")
                t5.metric("Merge", f"{timings['merge_ms']:.2f} ms")

            if not results:
                st.info("No results. Has this corpus been indexed yet? (`python scripts/build_index.py`)")

            for r in results:
                with st.container(border=True):
                    header_col, score_col = st.columns([5, 1])
                    header_col.markdown(f"**{r['repo']}** · `{r['path']}` · **{r['symbol_name']}** ({r['symbol_type']})")
                    score_col.markdown(f"score `{r['score']:.3f}` · via *{r['source']}*")
                    lang = "python" if r["path"].endswith(".py") else ("javascript" if r["path"].endswith((".js", ".ts")) else None)
                    st.code(r["text"], language=lang)
elif submitted:
    st.warning("Enter a query first.")
