"""
Streamlit chat UI with integrated trace viewer.

Start with:
    streamlit run app.py
"""
from __future__ import annotations

import json

import httpx
import streamlit as st

from config import BACKEND_URL

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Engineering Ops Assistant",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "traces" not in st.session_state:
    st.session_state.traces = []


# ---------------------------------------------------------------------------
# Sidebar: Trace Viewer
# ---------------------------------------------------------------------------

def _render_trace_sidebar():
    """Render the trace viewer in the sidebar."""
    st.sidebar.title("🔍 Trace Viewer")

    if st.sidebar.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.traces = []
        st.rerun()

    if not st.session_state.traces:
        st.sidebar.info("Ask a question to see the agent's reasoning trace.")
        return

    # Show traces in reverse order (newest first)
    for i, trace in enumerate(reversed(st.session_state.traces)):
        trace_label = trace.get("query", "")[:50] + "…"
        degraded = trace.get("degraded", False)
        label = f"{'⚠️ ' if degraded else ''}{trace_label}"

        with st.sidebar.expander(label, expanded=(i == 0)):
            _render_trace_detail(trace)


def _render_trace_detail(trace: dict):
    """Render a single trace's steps."""
    steps = trace.get("steps", [])
    tools_used = trace.get("tools_used", [])
    total_iters = trace.get("total_iterations", 0)

    st.caption(f"Trace ID: `{trace.get('trace_id', 'N/A')}`")
    st.caption(f"Tools used: {', '.join(tools_used) if tools_used else 'none'} · "
               f"Tool calls: {total_iters}")

    if trace.get("degraded"):
        st.warning("⚠️ Agent hit iteration cap — answer may be incomplete.")

    for step in steps:
        step_type = step.get("type", "unknown")
        step_num = step.get("step_num", "?")

        if step_type == "thinking":
            thinking = step.get("thinking", {})
            with st.container():
                st.markdown(f"**Step {step_num}: 🧠 Thinking**")
                st.markdown(f"- **Evidence:** {thinking.get('evidence_gathered', 'N/A')}")
                st.markdown(f"- **Missing:** {thinking.get('missing_information', 'N/A')}")
                st.markdown(f"- **Next:** {thinking.get('next_step', 'N/A')}")

        elif step_type == "tool_call":
            tool_name = step.get("tool_name", "unknown")
            tool_input = step.get("tool_input", {})
            tool_output = step.get("tool_output_truncated", "")
            parallel = step.get("parallel_group", [])

            icon = {"search_docs": "📚", "query_tickets": "🎫", "web_search": "🌐"}.get(tool_name, "🔧")

            with st.container():
                parallel_tag = " *(parallel)*" if parallel and len(parallel) > 1 else ""
                st.markdown(f"**Step {step_num}: {icon} {tool_name}**{parallel_tag}")
                st.code(json.dumps(tool_input, indent=2), language="json")

                # Show output in a collapsed section
                with st.expander("Tool output", expanded=False):
                    try:
                        parsed = json.loads(tool_output)
                        st.json(parsed)
                    except (json.JSONDecodeError, TypeError):
                        st.text(str(tool_output)[:1000])

        elif step_type == "final_answer":
            with st.container():
                st.markdown(f"**Step {step_num}: ✅ Final Answer**")
                if step.get("degraded"):
                    st.warning("Degraded — synthesized from partial evidence")

        elif step_type == "error":
            with st.container():
                st.markdown(f"**Step {step_num}: ❌ Error**")
                st.error(step.get("error", "Unknown error"))

        st.divider()


# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------

def main():
    st.title("🔧 Engineering Ops Assistant")
    st.caption("Ask debugging, operational, or documentation questions. "
               "Traces appear in the sidebar.")

    # Render trace sidebar
    _render_trace_sidebar()

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question…"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call backend
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    # Build history payload from prior turns (excluding current prompt and errors)
                    history_payload = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                        if m.get("content") and not m["content"].startswith("⚠️")
                    ]

                    response = httpx.post(
                        f"{BACKEND_URL}/chat",
                        json={"query": prompt, "history": history_payload},
                        timeout=120.0,
                    )
                    response.raise_for_status()
                    data = response.json()

                    answer = data.get("answer", "No answer received.")
                    trace = data.get("trace", {})

                    # Store trace
                    st.session_state.traces.append(trace)

                    # Display answer
                    st.markdown(answer)

                    # Store assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                    })

                except httpx.ConnectError:
                    error_msg = (
                        "⚠️ Cannot connect to the backend. "
                        f"Make sure the FastAPI server is running at `{BACKEND_URL}`.\n\n"
                        "Start it with: `uvicorn main:app --reload --port 8000`"
                    )
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })

                except Exception as exc:
                    error_msg = f"⚠️ Error: {exc}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })

        # Rerun to update the sidebar with the new trace
        st.rerun()


if __name__ == "__main__":
    main()
