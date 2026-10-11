"""
FastAPI backend — chat endpoint, webhook receivers, trace viewer, and health check.

Start with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from agent import run_agent
from config import TRACE_DIR, WEBHOOK_SECRET
from ingestion import run_full_ingestion, run_partial_ingestion

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Internal Engineering & Ops Assistant",
    description="Agentic RAG backend with structured trace logging",
    version="1.0.0",
)


# =========================================================================
# Request / Response Schemas
# =========================================================================

class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    trace_id: str
    trace: dict


# =========================================================================
# Chat Endpoint
# =========================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint — runs the agent loop and returns answer + trace."""
    result = await run_agent(request.query)
    return ChatResponse(
        answer=result["answer"],
        trace_id=result["trace"]["trace_id"],
        trace=result["trace"],
    )


# =========================================================================
# Trace Endpoints
# =========================================================================

@app.get("/traces")
async def list_traces():
    """List all available trace logs, newest first."""
    if not os.path.exists(TRACE_DIR):
        return {"traces": []}

    traces = []
    for fname in sorted(os.listdir(TRACE_DIR), reverse=True):
        if fname.endswith(".json"):
            fpath = os.path.join(TRACE_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                traces.append({
                    "trace_id": data.get("trace_id"),
                    "timestamp": data.get("timestamp"),
                    "query": data.get("query", "")[:100],
                    "tools_used": data.get("tools_used", []),
                    "degraded": data.get("degraded", False),
                })
            except Exception:
                continue

    return {"traces": traces}


@app.get("/traces/{trace_id}")
async def get_trace(trace_id: str):
    """Get full trace detail by ID."""
    fpath = os.path.join(TRACE_DIR, f"{trace_id}.json")
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="Trace not found")

    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================================
# Trace Viewer (HTML)
# =========================================================================

@app.get("/trace-viewer", response_class=HTMLResponse)
async def trace_viewer():
    """Interactive HTML trace viewer that renders agent reasoning steps."""
    return HTMLResponse(content=_TRACE_VIEWER_HTML)


_TRACE_VIEWER_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agent Trace Viewer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0d1117; color: #c9d1d9; padding: 24px; }
  h1 { color: #58a6ff; margin-bottom: 16px; }
  .trace-list { margin-bottom: 24px; }
  .trace-item { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                padding: 12px 16px; margin-bottom: 8px; cursor: pointer; }
  .trace-item:hover { border-color: #58a6ff; }
  .trace-item .query { font-weight: 600; color: #f0f6fc; }
  .trace-item .meta { font-size: 13px; color: #8b949e; margin-top: 4px; }
  .detail { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
            padding: 20px; display: none; }
  .step { margin: 12px 0; padding: 12px; border-radius: 6px; border-left: 4px solid; }
  .step.tool_call  { border-color: #58a6ff; background: #0d1117; }
  .step.thinking   { border-color: #d29922; background: #1c1608; }
  .step.final_answer { border-color: #3fb950; background: #0a1f0d; }
  .step.error      { border-color: #f85149; background: #1f0a0a; }
  .step-type { font-size: 11px; text-transform: uppercase; font-weight: 700;
               letter-spacing: 1px; margin-bottom: 6px; }
  .step.tool_call .step-type  { color: #58a6ff; }
  .step.thinking .step-type   { color: #d29922; }
  .step.final_answer .step-type { color: #3fb950; }
  .step.error .step-type      { color: #f85149; }
  pre { background: #0d1117; padding: 8px; border-radius: 4px; overflow-x: auto;
        font-size: 13px; margin-top: 6px; white-space: pre-wrap; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px;
           font-size: 11px; font-weight: 600; margin-left: 6px; }
  .badge.degraded { background: #f8514920; color: #f85149; }
  #loading { color: #8b949e; padding: 20px; text-align: center; }
</style>
</head>
<body>
<h1>🔍 Agent Trace Viewer</h1>
<div id="trace-list" class="trace-list"><div id="loading">Loading traces…</div></div>
<div id="detail" class="detail"></div>
<script>
async function loadTraces() {
  const res = await fetch('/traces');
  const data = await res.json();
  const list = document.getElementById('trace-list');
  list.innerHTML = '';
  if (!data.traces.length) { list.innerHTML = '<p>No traces yet.</p>'; return; }
  data.traces.forEach(t => {
    const div = document.createElement('div');
    div.className = 'trace-item';
    div.innerHTML = `<div class="query">${esc(t.query)}${t.degraded
      ? '<span class="badge degraded">DEGRADED</span>' : ''}</div>
      <div class="meta">${t.timestamp} · Tools: ${t.tools_used.join(', ') || 'none'}</div>`;
    div.onclick = () => loadDetail(t.trace_id);
    list.appendChild(div);
  });
}
async function loadDetail(id) {
  const res = await fetch('/traces/' + id);
  const t = await res.json();
  const el = document.getElementById('detail');
  let html = `<h2 style="margin-bottom:12px">Query: ${esc(t.query)}</h2>`;
  t.steps.forEach(s => {
    html += `<div class="step ${s.type}"><div class="step-type">Step ${s.step_num}: ${s.type}</div>`;
    if (s.type === 'tool_call')
      html += `<div><strong>${s.tool_name}</strong>(${esc(JSON.stringify(s.tool_input))})</div>
               <pre>${esc(s.tool_output_truncated||'')}</pre>`;
    if (s.type === 'thinking' && s.thinking)
      html += `<pre>${esc(JSON.stringify(s.thinking, null, 2))}</pre>`;
    if (s.type === 'final_answer')
      html += `<pre>${esc(s.answer||'')}</pre>`;
    if (s.type === 'error')
      html += `<pre style="color:#f85149">${esc(s.error||'')}</pre>`;
    html += '</div>';
  });
  el.innerHTML = html; el.style.display = 'block';
}
function esc(s) { const d = document.createElement('div'); d.textContent = s||''; return d.innerHTML; }
loadTraces();
</script>
</body>
</html>
"""


# =========================================================================
# Webhook Endpoints
# =========================================================================

def _verify_github_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify GitHub HMAC-SHA256 webhook signature."""
    expected = "sha256=" + hmac.new(
        secret.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def _process_github_webhook(payload: dict):
    """Background worker: parse GitHub webhook and run partial ingestion."""
    changed_files = []
    repo_name = payload.get("repository", {}).get("name", "unknown")

    for commit in payload.get("commits", []):
        for path in commit.get("added", []) + commit.get("modified", []):
            if path.endswith(".md"):
                changed_files.append({
                    "path": path,
                    "action": "modified",
                    "content": "",  # Would fetch via GitHub API in production
                    "repo": repo_name,
                })
        for path in commit.get("removed", []):
            if path.endswith(".md"):
                changed_files.append({
                    "path": path,
                    "action": "removed",
                    "content": "",
                    "repo": repo_name,
                })

    if changed_files:
        logger.info("Processing %d changed files from webhook", len(changed_files))
        stats = run_partial_ingestion(changed_files)
        logger.info("Webhook ingestion stats: %s", stats)


def _process_gitlab_webhook(payload: dict):
    """Background worker: parse GitLab webhook and run partial ingestion."""
    changed_files = []
    repo_name = payload.get("project", {}).get("name", "unknown")

    for commit in payload.get("commits", []):
        for path in commit.get("added", []) + commit.get("modified", []):
            if path.endswith(".md"):
                changed_files.append({
                    "path": path,
                    "action": "modified",
                    "content": "",
                    "repo": repo_name,
                })
        for path in commit.get("removed", []):
            if path.endswith(".md"):
                changed_files.append({
                    "path": path,
                    "action": "removed",
                    "content": "",
                    "repo": repo_name,
                })

    if changed_files:
        logger.info("Processing %d changed files from GitLab webhook", len(changed_files))
        stats = run_partial_ingestion(changed_files)
        logger.info("GitLab webhook ingestion stats: %s", stats)


@app.post("/webhook/github")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: Optional[str] = Header(None),
):
    """
    GitHub webhook receiver.
    Verifies HMAC-SHA256 signature, returns 200 immediately,
    processes the payload in a background task.
    """
    body = await request.body()

    # Verify signature
    if WEBHOOK_SECRET:
        if not x_hub_signature_256:
            raise HTTPException(status_code=401, detail="Missing signature header")
        if not _verify_github_signature(body, x_hub_signature_256, WEBHOOK_SECRET):
            raise HTTPException(status_code=401, detail="Invalid signature")

    payload = json.loads(body)
    background_tasks.add_task(_process_github_webhook, payload)
    return JSONResponse({"status": "accepted"}, status_code=200)


@app.post("/webhook/gitlab")
async def gitlab_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_gitlab_token: Optional[str] = Header(None),
):
    """
    GitLab webhook receiver.
    Verifies shared secret token, returns 200 immediately,
    processes the payload in a background task.
    """
    if WEBHOOK_SECRET:
        if x_gitlab_token != WEBHOOK_SECRET:
            raise HTTPException(status_code=401, detail="Invalid token")

    body = await request.body()
    payload = json.loads(body)
    background_tasks.add_task(_process_gitlab_webhook, payload)
    return JSONResponse({"status": "accepted"}, status_code=200)


# =========================================================================
# Manual Ingestion Trigger
# =========================================================================

@app.post("/ingest")
async def trigger_ingestion(background_tasks: BackgroundTasks):
    """Manually trigger a full re-ingestion of all repos."""
    background_tasks.add_task(run_full_ingestion)
    return {"status": "ingestion_started", "message": "Full re-ingestion running in background"}

# =========================================================================
# Health Check
# =========================================================================

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "engineering-ops-assistant"}
