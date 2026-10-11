"""
Core agentic RAG loop with:
  1. <thinking> block reasoning (not injected sufficiency prompts)
  2. Parallel tool calling
  3. Graceful degradation at iteration cap
  4. Structured error returns (no Python exceptions)
  5. Context truncation middleware
"""
from __future__ import annotations

import json
import logging

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq

from config import GROQ_API_KEY, GROQ_MODEL, MAX_AGENT_ITERATIONS
from tools import ALL_TOOLS, TOOLS_BY_NAME
from trace_logger import (
    TraceLogger,
    parse_thinking_block,
    strip_thinking_block,
    truncate_tool_output,
)

logger = logging.getLogger(__name__)

# =========================================================================
# System Prompt
# =========================================================================

SYSTEM_PROMPT = """\
You are an **Internal Engineering & Operations Assistant**, an expert AI \
that helps engineers debug production issues, find documentation, and \
track ticket statuses.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## MANDATORY REASONING PROTOCOL
Before EVERY action — whether calling a tool or delivering a final answer \
— you MUST emit a <thinking> block:

<thinking>
i. Evidence gathered: [summary of what tools have returned so far]
ii. Missing information: [specific gap, or "None — ready to answer"]
iii. Next step: [Call tool X with query Y / Synthesize final answer]
</thinking>

Do NOT skip this block. It is required for traceability.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PARALLEL TOOL CALLING
When a query spans multiple domains, call multiple tools **simultaneously \
in a single turn** to minimize latency. Examples:
  • Internal service issue + ticket status → call `search_docs` AND \
`query_tickets` together.
  • Internal context + external error code → call `search_docs` AND \
`web_search` together.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## TOOL ROUTING STRATEGY

### USE `search_docs` FOR:
- Internal service names, proprietary microservices, custom CLI tools
- Incident response protocols, on-call schedules, internal playbooks
- Company-specific deployment pipelines, CI/CD configs, infra topologies
- Historical architectural decisions ("Why did we choose X?")
- Post-mortem reports and lessons learned

### USE `query_tickets` FOR:
- Bug reports, task statuses, assignee lookups
- Filtering tickets by component, priority, or status
- Finding related issues or tracking work items

### USE `web_search` FOR:
- OSS documentation, API breaking changes, version-specific bugs
- Stack traces or error codes from **third-party** libraries
- Community workarounds (Stack Overflow, Reddit, GitHub issues)
- Security advisories and CVEs

### HYBRID SCENARIOS:
If an internal service fails due to a third-party dependency error:
1. Call `search_docs` to understand the internal integration
2. Call `web_search` to find the community fix
Call BOTH in parallel when possible.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## EXTERNAL SEARCH CONSTRAINTS (CRITICAL)
When using `web_search`:
1. **DATA LOSS PREVENTION:** NEVER include internal hostnames, private IPs, \
proprietary repo names, credentials, or internal jargon. Sanitize to \
generic technical terms only.
2. **QUERY OPTIMIZATION:** Keep queries concise (<1500 chars).
3. **SUB-QUERIES:** Break complex debugging into focused requests \
(e.g., one for the error, one for the migration guide).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SYNTHESIS & GROUNDING RULES
1. **Internal Precedence:** If internal docs and web results contradict, \
internal documentation ALWAYS wins.
2. **Quarantine External Code:** If a web-sourced fix differs from internal \
policy, label it:
   > **Community Workaround (DO NOT EXECUTE WITHOUT REVIEW)**
3. **Temporal Awareness:** Check the `last_updated` field of internal docs. \
If a runbook is severely outdated compared to a recent external patch or \
CVE, explicitly FLAG the discrepancy instead of blindly enforcing the \
stale runbook.
4. **Source Attribution:** Always cite sources:
   - `[Internal: <Document Title / Source File>]`
   - `[Web: <Domain>]`
5. **Safety Warnings:** If external advice suggests state-modifying \
commands (DROP TABLE, rm -rf, disabling auth, etc.), explicitly warn \
the engineer to verify against internal compliance rules.
6. **Draw Boundaries:** Distinguish internal policies from external \
advice clearly in your response.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## ERROR HANDLING
If a tool returns an error or empty results:
1. Analyze the error message
2. Try reformulating the query with different keywords or a different doc_type
3. Try a different tool if appropriate
4. If all tools fail, acknowledge the limitation honestly
"""

SYNTHESIS_PROMPT = (
    "You have reached the maximum number of tool calls. "
    "Synthesize the best possible answer using ONLY the context received so far. "
    "Acknowledge any missing information explicitly. "
    "Do NOT attempt to call any tools."
)


# =========================================================================
# Agent Loop
# =========================================================================

async def run_agent(query: str) -> dict:
    """
    Execute the agentic RAG loop for a single user query.

    Returns:
        {"answer": str, "trace": dict}
    """
    tracer = TraceLogger(query)

    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    messages: list = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    for iteration in range(MAX_AGENT_ITERATIONS):
        logger.info("Agent iteration %d/%d", iteration + 1, MAX_AGENT_ITERATIONS)

        # ── Iteration cap: strip tools & force synthesis ──────────────
        at_cap = iteration == MAX_AGENT_ITERATIONS - 1
        has_tool_results = any(isinstance(m, ToolMessage) for m in messages)

        if at_cap and has_tool_results:
            messages.append(SystemMessage(content=SYNTHESIS_PROMPT))
            ai_msg: AIMessage = llm.invoke(messages)       # no tools bound
            messages.append(ai_msg)

            thinking = parse_thinking_block(ai_msg.content or "")
            if thinking:
                tracer.log_thinking(thinking)

            final = strip_thinking_block(ai_msg.content or "")
            tracer.log_final_answer(final, degraded=True)
            tracer.save()
            return {"answer": final, "trace": tracer.to_dict()}

        # ── Normal iteration with tools ───────────────────────────────
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        # Parse <thinking> block
        thinking = parse_thinking_block(ai_msg.content or "")
        if thinking:
            tracer.log_thinking(thinking)

        # ── No tool calls → final answer ──────────────────────────────
        if not ai_msg.tool_calls:
            final = strip_thinking_block(ai_msg.content or "")
            tracer.log_final_answer(final)
            tracer.save()
            return {"answer": final, "trace": tracer.to_dict()}

        # ── Execute all tool calls (parallel within this turn) ────────
        parallel_group = [tc["name"] for tc in ai_msg.tool_calls]

        for tool_call in ai_msg.tool_calls:
            tool_name: str = tool_call["name"]
            tool_args: dict = tool_call["args"]
            tool_id: str = tool_call["id"]

            # Structured error return — never raise
            try:
                tool_fn = TOOLS_BY_NAME.get(tool_name)
                if tool_fn is None:
                    raw_output = json.dumps({
                        "status": "error",
                        "tool": tool_name,
                        "error_type": "unknown_tool",
                        "message": f"Tool '{tool_name}' is not available",
                        "suggestion": "Use one of: search_docs, query_tickets, web_search",
                    })
                else:
                    raw_output = tool_fn.invoke(tool_args)
            except Exception as exc:
                raw_output = json.dumps({
                    "status": "error",
                    "tool": tool_name,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "suggestion": "Tool execution failed. Try a different approach.",
                })

            # Truncate context before it enters history & trace
            truncated = truncate_tool_output(tool_name, raw_output)

            tracer.log_tool_call(tool_name, tool_args, truncated, parallel_group)

            messages.append(
                ToolMessage(content=truncated, tool_call_id=tool_id)
            )

    # ── Fallback (should rarely reach here) ───────────────────────────
    final = (
        "I was unable to find a complete answer within the allowed number of "
        "iterations. Please try rephrasing your question or narrowing the scope."
    )
    tracer.log_final_answer(final, degraded=True)
    tracer.save()
    return {"answer": final, "trace": tracer.to_dict()}
