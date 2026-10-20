"""
Structured trace logging & context-truncation middleware.

Every agent query produces a JSON trace file in TRACE_DIR.
The truncation middleware strips noise from tool outputs before they
reach the LLM context window or the trace log.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from config import TRACE_DIR


# =========================================================================
# Context Truncation Middleware
# =========================================================================

_MAX_CONTENT_CHARS = 500  # per individual search result


def truncate_tool_output(tool_name: str, raw_output: str) -> str:
    """
    Strip noise from raw tool JSON before it enters the message history.

    • search_docs  → keep page_content, source_file, last_updated, doc_type
    • web_search   → keep content (≤500 chars), source domain, title
    • query_tickets → pass through (already compact)
    """
    try:
        data = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
    except (json.JSONDecodeError, TypeError):
        # Not JSON — return as-is (probably a plain error string)
        return str(raw_output)[:2000]

    if not isinstance(data, dict):
        return json.dumps(data)[:2000]

    # Errors pass through untouched — LLM needs the full message
    if data.get("status") == "error":
        return json.dumps(data)

    results = data.get("results", [])

    if tool_name == "search_docs":
        cleaned = []
        for r in results:
            cleaned.append({
                "content": (r.get("content") or r.get("page_content") or "")[:_MAX_CONTENT_CHARS],
                "source_file": r.get("source_file", "unknown"),
                "last_updated": r.get("last_updated", "unknown"),
                "doc_type": r.get("doc_type", "unknown"),
            })
        return json.dumps({"status": "success", "results": cleaned, "count": len(cleaned)})

    if tool_name == "web_search":
        cleaned = []
        for r in results:
            url = r.get("url", "")
            domain = urlparse(url).netloc if url else "unknown"
            cleaned.append({
                "title": r.get("title", "")[:200],
                "source": domain,
                "content": _strip_html(r.get("content", ""))[:_MAX_CONTENT_CHARS],
            })
        return json.dumps({"status": "success", "results": cleaned, "count": len(cleaned)})

    # query_tickets — pass through
    return json.dumps(data)


def _strip_html(text: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================================
# Thinking-Block Parser
# =========================================================================

def parse_thinking_block(content: str) -> dict | None:
    """
    Extract a <thinking> block from LLM output.

    Expected format inside the block:
        i.  Evidence gathered: …
        ii. Missing information: …
        iii.Next step: …
    """
    if not content:
        return None
    match = re.search(r"<thinking>(.*?)</thinking>", content, re.DOTALL)
    if not match:
        return None

    raw = match.group(1).strip()
    thinking: dict[str, str] = {
        "evidence_gathered": "",
        "missing_information": "",
        "next_step": "",
    }

    for line in raw.split("\n"):
        line = line.strip()
        low = line.lower()
        if low.startswith("i.") or "evidence gathered" in low:
            thinking["evidence_gathered"] = _after_colon(line)
        elif low.startswith("ii.") or "missing information" in low:
            thinking["missing_information"] = _after_colon(line)
        elif low.startswith("iii.") or "next step" in low:
            thinking["next_step"] = _after_colon(line)

    return thinking


def strip_thinking_block(content: str) -> str:
    """Remove <thinking>…</thinking> from user-facing text."""
    if not content:
        return ""
    return re.sub(r"<thinking>.*?</thinking>", "", content, flags=re.DOTALL).strip()


def _after_colon(line: str) -> str:
    """Return everything after the first colon, stripped."""
    parts = line.split(":", 1)
    return parts[1].strip() if len(parts) > 1 else line


# =========================================================================
# Trace Logger
# =========================================================================

class TraceLogger:
    """
    Accumulates structured trace data for a single agent query.
    Call `.save()` to persist to disk.
    """

    def __init__(self, query: str):
        self.trace_id: str = str(uuid.uuid4())
        self.timestamp: str = datetime.now(timezone.utc).isoformat()
        self.query: str = query
        self.steps: list[dict[str, Any]] = []
        self.final_answer: str | None = None
        self.tools_used: set[str] = set()
        self.degraded: bool = False
        self._step_counter: int = 0

    # ----- logging helpers ------------------------------------------------

    def log_thinking(self, thinking: dict) -> None:
        self._step_counter += 1
        self.steps.append({
            "step_num": self._step_counter,
            "type": "thinking",
            "thinking": thinking,
        })

    def log_tool_call(
        self,
        tool_name: str,
        tool_input: dict,
        tool_output_truncated: str,
        parallel_group: list[str] | None = None,
    ) -> None:
        self._step_counter += 1
        self.tools_used.add(tool_name)
        self.steps.append({
            "step_num": self._step_counter,
            "type": "tool_call",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_output_truncated": tool_output_truncated,
            "parallel_group": parallel_group,
        })

    def log_error(self, tool_name: str, error_info: str) -> None:
        self._step_counter += 1
        self.steps.append({
            "step_num": self._step_counter,
            "type": "error",
            "tool_name": tool_name,
            "error": error_info,
        })

    def log_final_answer(self, answer: str, degraded: bool = False) -> None:
        self._step_counter += 1
        self.final_answer = answer
        self.degraded = degraded
        self.steps.append({
            "step_num": self._step_counter,
            "type": "final_answer",
            "answer": answer[:5000],
            "degraded": degraded,
        })

    # ----- serialization ---------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "query": self.query,
            "steps": self.steps,
            "final_answer": self.final_answer,
            "total_iterations": sum(1 for s in self.steps if s["type"] == "tool_call"),
            "tools_used": sorted(self.tools_used),
            "degraded": self.degraded,
        }

    def save(self) -> str:
        """Write trace JSON to disk and return the file path."""
        os.makedirs(TRACE_DIR, exist_ok=True)
        path = os.path.join(TRACE_DIR, f"{self.trace_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path
