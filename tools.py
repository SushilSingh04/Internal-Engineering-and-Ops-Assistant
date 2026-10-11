"""
Agent tools — search_docs, query_tickets, web_search.

All tools return JSON strings (never raise exceptions).
Errors are returned as structured JSON so the LLM can reason about
failures and retry with a reformulated query.
"""
from __future__ import annotations

import json

from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilySearch
from pinecone import Pinecone
from sqlalchemy import or_

from config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    TAVILY_API_KEY,
)
from setup_db import Ticket, get_session


# =========================================================================
# Shared clients (initialized once at module load)
# =========================================================================

_embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GOOGLE_API_KEY,
    output_dimensionality=EMBEDDING_DIMENSIONS
)

_pc = Pinecone(api_key=PINECONE_API_KEY)
_index = _pc.Index(PINECONE_INDEX_NAME)
_vector_store = PineconeVectorStore(index=_index, embedding=_embeddings)

_tavily = TavilySearch(
    max_results=5,
    search_depth="advanced",
    tavily_api_key=TAVILY_API_KEY,
)


# =========================================================================
# Tool: search_docs
# =========================================================================

@tool
def search_docs(query: str, doc_type: str = "any") -> str:
    """Search internal documentation, runbooks, architecture decision records, and post-mortems.

    Args:
        query: Natural-language search query.
        doc_type: Filter by document category.
                  One of "runbook", "architecture", "onboarding", "any".
                  Use "any" to search across all categories.

    Returns:
        JSON string with matching document chunks including content,
        source_file, last_updated, and doc_type metadata.
    """
    try:
        filter_dict: dict | None = None
        if doc_type and doc_type != "any":
            filter_dict = {"doc_type": {"$eq": doc_type}}

        results = _vector_store.similarity_search(
            query=query,
            k=5,
            filter=filter_dict,
        )

        if not results:
            return json.dumps({
                "status": "error",
                "tool": "search_docs",
                "error_type": "empty_results",
                "message": f"No documents matched query '{query}' with doc_type='{doc_type}'",
                "suggestion": "Try broadening the query or changing doc_type to 'any'",
            })

        output = []
        for doc in results:
            meta = doc.metadata or {}
            output.append({
                "content": doc.page_content,
                "source_file": meta.get("source_file", "unknown"),
                "last_updated": meta.get("last_updated", "unknown"),
                "doc_type": meta.get("doc_type", "unknown"),
            })

        return json.dumps({"status": "success", "results": output})

    except Exception as exc:
        return json.dumps({
            "status": "error",
            "tool": "search_docs",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "suggestion": "Check Pinecone connection and API key. Try a simpler query.",
        })


# =========================================================================
# Tool: query_tickets
# =========================================================================

@tool
def query_tickets(
    status: str | None = None,
    component: str | None = None,
    assignee: str | None = None,
    text_search: str | None = None,
) -> str:
    """Query the engineering ticketing system for bugs, tasks, and incidents.

    All parameters are optional — provide any combination to filter.

    Args:
        status: Filter by ticket status (open, in_progress, resolved, closed).
        component: Filter by service or component name.
        assignee: Filter by assignee email (partial match supported).
        text_search: Free-text search across title and description.

    Returns:
        JSON string with matching tickets including id, title, description,
        status, assignee, priority, component, and timestamps.
    """
    try:
        session = get_session()
        try:
            q = session.query(Ticket)

            if status:
                q = q.filter(Ticket.status == status)
            if component:
                q = q.filter(Ticket.component.ilike(f"%{component}%"))
            if assignee:
                q = q.filter(Ticket.assignee.ilike(f"%{assignee}%"))
            if text_search:
                pattern = f"%{text_search}%"
                q = q.filter(
                    or_(
                        Ticket.title.ilike(pattern),
                        Ticket.description.ilike(pattern),
                    )
                )

            tickets = q.order_by(Ticket.updated_at.desc()).limit(20).all()

            if not tickets:
                return json.dumps({
                    "status": "error",
                    "tool": "query_tickets",
                    "error_type": "empty_results",
                    "message": "No tickets found matching the given filters",
                    "suggestion": "Try relaxing filters or using text_search with broader terms",
                })

            return json.dumps({
                "status": "success",
                "results": [t.to_dict() for t in tickets],
                "count": len(tickets),
            })
        finally:
            session.close()

    except Exception as exc:
        return json.dumps({
            "status": "error",
            "tool": "query_tickets",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "suggestion": "Check Supabase database connection",
        })


# =========================================================================
# Tool: web_search
# =========================================================================

@tool
def web_search(query: str) -> str:
    """Search the public web for third-party library docs, Stack Overflow answers, GitHub issues, or CVEs.

    IMPORTANT: Never include internal hostnames, private IPs, proprietary repo
    names, or credentials in the query. Sanitize to generic technical terms only.

    Args:
        query: Concise search query (under 1500 characters).

    Returns:
        JSON string with web results including title, URL, and content snippet.
    """
    try:
        results = _tavily.invoke({"query": query})

        if not results:
            return json.dumps({
                "status": "error",
                "tool": "web_search",
                "error_type": "empty_results",
                "message": f"No web results found for '{query}'",
                "suggestion": "Try different keywords or a more specific query",
            })

        output = []
        for r in results:
            output.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:800],
            })

        return json.dumps({"status": "success", "results": output})

    except Exception as exc:
        return json.dumps({
            "status": "error",
            "tool": "web_search",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "suggestion": "Check Tavily API key and internet connectivity",
        })


# =========================================================================
# Convenience list for agent.py
# =========================================================================

ALL_TOOLS = [search_docs, query_tickets, web_search]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
