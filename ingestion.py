"""
Document ingestion pipeline — clone repos, split markdown, sync to Pinecone.

Supports:
  • Full ingestion (clone all repos, split, index)
  • Incremental sync via LangChain SQLRecordManager + Supabase
  • Webhook-triggered partial updates for specific changed files
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from langchain_classic.indexes import SQLRecordManager, index
from langchain_community.document_loaders import GitLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pinecone import Pinecone

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CLONE_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    GOOGLE_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    RECORD_MANAGER_NAMESPACE,
    REPOS,
    SUPABASE_DB_URL,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared clients
# ---------------------------------------------------------------------------
_embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL, 
    google_api_key=GOOGLE_API_KEY, 
    output_dimensionality=EMBEDDING_DIMENSIONS
    )

_pc = Pinecone(api_key=PINECONE_API_KEY)
_index = _pc.Index(PINECONE_INDEX_NAME)
_vector_store = PineconeVectorStore(index=_index, embedding=_embeddings)

# Record manager backed by Supabase Postgres (transaction-mode pooler)
_record_manager = SQLRecordManager(
    RECORD_MANAGER_NAMESPACE,
    db_url=SUPABASE_DB_URL,
)


# =========================================================================
# File-level helpers
# =========================================================================

_EXCLUDED_DIRS = {"node_modules", "vendor", ".git", "__pycache__", ".github", ".gitlab"}
_ALLOWED_EXTENSIONS = {".md"}
_EXCLUDED_EXTENSIONS = {".json", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
                        ".lock", ".toml", ".yaml", ".yml", ".cfg", ".ini",
                        ".pyc", ".class", ".jar", ".exe", ".bin", ".wasm"}


def _md_file_filter(file_path: str) -> bool:
    """Only accept .md files, excluding dependency / binary directories."""
    p = Path(file_path)

    # Exclude files inside unwanted directories
    for part in p.parts:
        if part in _EXCLUDED_DIRS:
            return False

    return p.suffix.lower() in _ALLOWED_EXTENSIONS

# =========================================================================
# Splitting
# =========================================================================

# Split on markdown headers first
_MD_HEADERS = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
]
_md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_MD_HEADERS)

# Then fixed-size chunks
_char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)


def _split_document(doc: Document, doc_type: str) -> list[Document]:
    """
    Two-stage split:
      1. By markdown headers (semantic boundaries)
      2. By fixed character size (~400 tokens)
    Attach enriched metadata to each chunk.
    """
    # Stage 1: header-based split
    header_chunks = _md_splitter.split_text(doc.page_content)

    # Stage 2: fixed-size sub-chunks
    final_chunks: list[Document] = []
    source_file = doc.metadata.get("file_path", doc.metadata.get("source", "unknown"))
    now_iso = datetime.now(timezone.utc).isoformat()

    for hc in header_chunks:
        sub_chunks = _char_splitter.split_text(hc.page_content)
        for sc in sub_chunks:
            # Build a stable source ID for the record manager
            content_hash = hashlib.md5(sc.encode()).hexdigest()[:8]
            final_chunks.append(
                Document(
                    page_content=sc,
                    metadata={
                        "source": f"{source_file}::{content_hash}",
                        "source_file": source_file,
                        "doc_type": doc_type,
                        "last_updated": now_iso,
                        # Carry forward header metadata if available
                        **{k: v for k, v in hc.metadata.items() if k.startswith("header")},
                    },
                )
            )

    return final_chunks


# =========================================================================
# Clone & Load
# =========================================================================

def _clone_and_load_repo(repo_config: dict) -> list[Document]:
    """Clone a single repo and return raw Document objects (.md only)."""
    clone_url = repo_config["clone_url"]
    doc_type = repo_config["doc_type"]
    branch = repo_config.get("branch", "main")

    # Derive a local directory name from the URL
    repo_name = clone_url.rstrip("/").split("/")[-1].replace(".git", "")
    repo_path = Path(CLONE_DIR) / repo_name
    abs_repo_path = str(repo_path.resolve())
    repo_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean previous clone (ephemeral storage on Render)
    if os.path.exists(abs_repo_path):
        shutil.rmtree(abs_repo_path)

    logger.info("Cloning %s (branch=%s) → %s", clone_url, branch, abs_repo_path)

    loader = GitLoader(
        clone_url=clone_url,
        repo_path=abs_repo_path,
        branch=branch,
        file_filter=_md_file_filter,
    )

    raw_docs = loader.load()
    logger.info("Loaded %d .md files from %s", len(raw_docs), repo_name)

    # Safely limit docs to avoid 429 Quota/Rate Limit errors on free tiers
    # MAX_DOCS_PER_REPO = 50
    # if len(raw_docs) > MAX_DOCS_PER_REPO:
    #     logger.info("Limiting %s to first %d docs to prevent API rate limits...", repo_name, MAX_DOCS_PER_REPO)
    #     raw_docs = raw_docs[:MAX_DOCS_PER_REPO]

    # Attach doc_type to each raw document
    for d in raw_docs:
        d.metadata["doc_type"] = doc_type

    return raw_docs


# =========================================================================
# Public API
# =========================================================================

def run_full_ingestion() -> dict:
    """
    Clone all configured repos, split documents, and run incremental
    sync against Pinecone via the Supabase-backed record manager.

    Returns indexing stats dict.
    """
    # Ensure the record manager schema exists in Supabase
    _record_manager.create_schema()

    all_chunks: list[Document] = []

    for repo_cfg in REPOS:
        try:
            raw_docs = _clone_and_load_repo(repo_cfg)
            for doc in raw_docs:
                chunks = _split_document(doc, repo_cfg["doc_type"])
                all_chunks.extend(chunks)
        except Exception as exc:
            logger.error("Failed to process repo %s: %s", repo_cfg["clone_url"], exc)
            continue

    logger.info("Total chunks to index: %d", len(all_chunks))

    if not all_chunks:
        return {"num_added": 0, "num_updated": 0, "num_skipped": 0, "num_deleted": 0}

    # Incremental sync: diffs hashes in Supabase, updates Pinecone accordingly
    stats = index(
        docs_source=all_chunks,
        record_manager=_record_manager,
        vector_store=_vector_store,
        cleanup="incremental",
        source_id_key="source",
    )

    logger.info("Indexing stats: %s", stats)
    return stats


def run_partial_ingestion(changed_files: list[dict]) -> dict:
    """
    Process a webhook payload of changed files.

    Each entry in changed_files should have:
        {"path": "docs/runbook.md", "action": "added|modified|removed",
         "content": "raw markdown text", "repo": "runbooks"}

    For removed files, we re-run a full incremental sync so the record
    manager detects the missing source and deletes from Pinecone.
    For added/modified, we process just those documents.
    """
    _record_manager.create_schema()

    has_removals = any(f.get("action") == "removed" for f in changed_files)

    if has_removals:
        # Full sync is the safest way to handle deletions with incremental mode
        logger.info("Detected file removals — running full re-sync")
        return run_full_ingestion()

    # Process only added/modified files
    chunks: list[Document] = []
    for file_info in changed_files:
        if not file_info.get("path", "").endswith(".md"):
            continue

        content = file_info.get("content", "")
        if not content:
            continue

        # Determine doc_type from repo name
        repo = file_info.get("repo", "").lower()
        if "runbook" in repo or "post-mortem" in repo:
            doc_type = "runbook"
        elif "architecture" in repo or "adr" in repo:
            doc_type = "architecture"
        else:
            doc_type = "runbook"

        doc = Document(
            page_content=content,
            metadata={
                "source": file_info["path"],
                "file_path": file_info["path"],
                "doc_type": doc_type,
            },
        )
        chunks.extend(_split_document(doc, doc_type))

    if not chunks:
        return {"num_added": 0, "num_updated": 0, "num_skipped": 0, "num_deleted": 0}

    stats = index(
        docs_source=chunks,
        record_manager=_record_manager,
        vector_store=_vector_store,
        cleanup="incremental",
        source_id_key="source",
    )

    logger.info("Partial indexing stats: %s", stats)
    return stats
