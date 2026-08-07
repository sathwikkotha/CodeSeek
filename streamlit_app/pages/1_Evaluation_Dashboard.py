"""Surface 2 -- Evaluation dashboard. Reads the eval harness's actual output
(data/eval_results.json, written by scripts/eval.py) and reports Recall@5,
Recall@10, and MRR per embedding model -- the real measurement this project
is built to produce, not a claim."""

import json

import altair as alt
import pandas as pd
import streamlit as st

from codeseek.config import DATA_DIR

METRIC_LABELS = {"recall_at_5": "Recall@5", "recall_at_10": "Recall@10", "mrr": "MRR"}
METRIC_COLORS = ["#2a78d6", "#eb6834", "#1aa172"]  # blue / orange / aqua -- validated categorical triple

st.set_page_config(page_title="CodeSeek -- Evaluation", page_icon="📊", layout="wide")

st.title("📊 Evaluation Dashboard")
st.caption("Recall@5 / Recall@10 / MRR from the ground-truth question set, run through each embedding model.")

results_path = DATA_DIR / "eval_results.json"

if not results_path.exists():
    st.warning(f"No eval results found at `{results_path}`. Run `python scripts/eval.py` first.")
    st.stop()

results = json.loads(results_path.read_text(encoding="utf-8"))
df = pd.DataFrame(results).set_index("model_key")

st.caption(f"n = {df['num_questions'].iloc[0]} ground-truth questions (data/ground_truth.json)")

col1, col2 = st.columns([1, 2])
with col1:
    st.dataframe(df[["recall_at_5", "recall_at_10", "mrr"]].style.format("{:.2f}"), use_container_width=True)
with col2:
    long_df = (
        df.reset_index()[["model_key", "recall_at_5", "recall_at_10", "mrr"]]
        .melt(id_vars="model_key", var_name="metric", value_name="value")
    )
    long_df["metric"] = long_df["metric"].map(METRIC_LABELS)

    chart = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X("model_key:N", title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("metric:N", sort=list(METRIC_LABELS.values())),
            y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "metric:N", sort=list(METRIC_LABELS.values()),
                scale=alt.Scale(domain=list(METRIC_LABELS.values()), range=METRIC_COLORS),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["model_key", "metric", alt.Tooltip("value:Q", format=".2f")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)

st.markdown(
    "A self-authored ground-truth set skews toward the question types the author thought to ask -- "
    "this is a real limitation, not hidden. A larger or independently-authored set would be the natural next step."
)
