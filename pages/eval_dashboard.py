"""
Streamlit Evaluation Dashboard — visualize Ragas evaluation results.

Loads eval_results.csv and displays:
  • Aggregate metric cards (st.metric)
  • Per-query score table (st.dataframe)
  • Failing query highlights (any metric < threshold)
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Eval Dashboard",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
_METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
}
_METRIC_HELP = {
    "faithfulness": "Is the answer grounded in retrieved context? (penalizes hallucination)",
    "answer_relevancy": "Does the answer address the original query?",
    "context_precision": "Are relevant chunks ranked higher than irrelevant ones?",
    "context_recall": "Did retrieval capture all necessary information?",
}
_DEFAULT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

@st.cache_data
def load_eval_results(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def main():
    st.title("📊 RAG Evaluation Dashboard")
    st.caption("Ragas metrics for the retrieval + generation pipeline")

    # File selector
    default_path = "eval_results.csv"
    csv_path = st.text_input("Evaluation CSV path", value=default_path)

    df = load_eval_results(csv_path)

    if df is None or df.empty:
        st.warning(
            f"No evaluation results found at `{csv_path}`. "
            "Run `python evaluate.py --generate` or `python evaluate.py --dataset <path>` first."
        )
        return

    # Detect available metric columns
    available = [c for c in _METRIC_COLS if c in df.columns]
    if not available:
        st.error("CSV does not contain expected Ragas metric columns.")
        st.write("Found columns:", list(df.columns))
        return

    # ----- Aggregate Metric Cards -----------------------------------------
    st.subheader("Aggregate Scores")

    cols = st.columns(len(available))
    for i, col_name in enumerate(available):
        mean_val = df[col_name].mean()
        min_val = df[col_name].min()
        label = _METRIC_LABELS.get(col_name, col_name)
        help_text = _METRIC_HELP.get(col_name, "")

        cols[i].metric(
            label=label,
            value=f"{mean_val:.3f}",
            delta=f"min: {min_val:.3f}",
            delta_color="off",
            help=help_text,
        )

    st.divider()

    # ----- Threshold control ----------------------------------------------
    threshold = st.slider(
        "Failure threshold (highlight queries below this score)",
        min_value=0.0,
        max_value=1.0,
        value=_DEFAULT_THRESHOLD,
        step=0.05,
    )

    # ----- Failing Queries ------------------------------------------------
    st.subheader("⚠️ Failing Queries")

    failing_mask = df[available].min(axis=1) < threshold
    failing = df[failing_mask]

    if failing.empty:
        st.success(f"✅ All {len(df)} queries passed the {threshold:.2f} threshold!")
    else:
        st.error(f"❌ {len(failing)} of {len(df)} queries failed (any metric < {threshold:.2f})")

        # Show failing queries with color-coded metric values
        display_cols = ["user_input"] if "user_input" in df.columns else []
        if not display_cols and "question" in df.columns:
            display_cols = ["question"]
        display_cols.extend(available)

        styled = failing[display_cols].style.applymap(
            lambda v: f"background-color: #f8514930; color: #f85149" if isinstance(v, (int, float)) and v < threshold else "",
            subset=available,
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()

    # ----- Full Results Table ---------------------------------------------
    st.subheader("All Query Results")

    # Sort options
    sort_col = st.selectbox("Sort by", options=["(default)"] + available)
    if sort_col != "(default)" and sort_col in df.columns:
        df_sorted = df.sort_values(sort_col, ascending=True)
    else:
        df_sorted = df

    # Metric filter
    filter_metric = st.selectbox("Highlight metric", options=["(none)"] + available)

    display_all_cols = [c for c in df_sorted.columns if c not in ["retrieved_contexts"]]
    display_df = df_sorted[display_all_cols]

    if filter_metric != "(none)" and filter_metric in display_df.columns:
        styled_all = display_df.style.background_gradient(
            subset=[filter_metric],
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
        )
        st.dataframe(styled_all, use_container_width=True, hide_index=True)
    else:
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ----- Summary stats --------------------------------------------------
    st.subheader("Distribution")

    if available:
        chart_data = df[available].describe().loc[["mean", "std", "min", "25%", "50%", "75%", "max"]]
        st.dataframe(chart_data.T.style.format("{:.3f}"), use_container_width=True)


if __name__ == "__main__":
    main()
