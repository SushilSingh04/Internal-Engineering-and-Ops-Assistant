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
import stat
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import git

from langchain_classic.indexes import SQLRecordManager, index

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
    INGESTION_BATCH_DELAY,
    INGESTION_BATCH_SIZE,
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
    """
    Only accept .md files, excluding dependency / binary directories.
    Within 'locales/', strictly only allow 'en-001'.
    """
    p = Path(file_path)

    # Exclude files inside unwanted directories
    for part in p.parts:
        if part in _EXCLUDED_DIRS:
            return False

    # Check extension 
    if p.suffix.lower() not in _ALLOWED_EXTENSIONS:
        return False

    # Filter locales: inside locales/, only include en-001 or top-level files (e.g. locales/README.md)
    parts = [part.lower() for part in p.parts]
    if "locales" in parts:
        loc_idx = parts.index("locales")
        if loc_idx + 1 < len(parts):
            sub_folder = parts[loc_idx + 1]
            if sub_folder != "en-001" and not sub_folder.endswith(".md"):
                return False

    return True

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
    last_updated = doc.metadata.get("last_updated") or "2026-10-14"

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
                        "last_updated": last_updated,
                        # Carry forward header metadata if available
                        **{k: v for k, v in hc.metadata.items() if k.startswith("header")},
                    },
                )
            )

    return final_chunks


# =========================================================================
# Clone & Load
# =========================================================================

def _safe_rmtree(path: str) -> None:
    """Safely remove a directory tree, clearing Windows read-only flags on git pack files."""
    if not os.path.exists(path):
        return

    def _remove_readonly(func, file_path, exc_info):
        try:
            os.chmod(file_path, stat.S_IWRITE)
            func(file_path)
        except Exception:
            pass

    shutil.rmtree(path, onerror=_remove_readonly)


def _clone_and_load_repo(repo_config: dict) -> list[Document]:
    """
    Clone a repository without checking out files to disk (no_checkout=True).
    Reads .md files directly from the Git object tree.
    
    This avoids Windows OS filename errors (e.g. reserved names like 'aux.*',
    'con.*', or max path limits) and avoids downloading/storing unnecessary non-MD files.
    """
    clone_url = repo_config["clone_url"]
    doc_type = repo_config["doc_type"]
    branch = repo_config.get("branch", "main")

    # Derive a local directory name from the URL
    repo_name = clone_url.rstrip("/").split("/")[-1].replace(".git", "")
    repo_path = Path(CLONE_DIR) / repo_name
    abs_repo_path = str(repo_path.resolve())
    repo_path.parent.mkdir(parents=True, exist_ok=True)

    git_dir = Path(abs_repo_path) / ".git"
    repo: git.Repo | None = None

    # Check if a valid clone already exists
    if git_dir.exists():
        logger.info("Found existing clone at %s — loading directly without re-cloning", abs_repo_path)
        try:
            repo = git.Repo(abs_repo_path)
        except Exception as exc:
            logger.warning("Existing clone at %s corrupted (%s), re-cloning...", abs_repo_path, exc)
            _safe_rmtree(abs_repo_path)
            repo = None
    else:
        _safe_rmtree(abs_repo_path)

    # Perform shallow, no-checkout clone
    if repo is None:
        logger.info("Cloning %s (branch=%s, no-checkout, depth=1) → %s", clone_url, branch, abs_repo_path)
        try:
            repo = git.Repo.clone_from(
                clone_url,
                abs_repo_path,
                branch=branch,
                no_checkout=True,
                depth=1,
            )
        except Exception as shallow_err:
            logger.warning("Shallow clone failed (%s), attempting full clone with no_checkout...", shallow_err)
            _safe_rmtree(abs_repo_path)
            repo = git.Repo.clone_from(
                clone_url,
                abs_repo_path,
                no_checkout=True,
            )
            try:
                repo.git.checkout(branch)
            except Exception:
                pass

    # Extract all .md files directly from git blob tree
    raw_docs: list[Document] = []
    tree = repo.head.commit.tree
    commit_date = datetime.fromtimestamp(repo.head.commit.committed_date, tz=timezone.utc).strftime("%Y-%m-%d")

    for item in tree.traverse():
        if item.type != "blob":
            continue

        file_path = item.path  # Relative path in repo (e.g. 'incident/p1.md')

        if not _md_file_filter(file_path):
            continue

        try:
            content_bytes = item.data_stream.read()
            text_content = content_bytes.decode("utf-8", errors="ignore")
        except Exception as read_err:
            logger.warning("Could not read git blob %s: %s", file_path, read_err)
            continue

        metadata = {
            "source": file_path,
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "doc_type": doc_type,
            "last_updated": commit_date,
        }
        raw_docs.append(Document(page_content=text_content, metadata=metadata))

    logger.info("Loaded %d .md files from %s", len(raw_docs), repo_name)
    return raw_docs


def _index_in_batches(
    chunks: list[Document],
    batch_size: int = INGESTION_BATCH_SIZE,
    delay_seconds: float = INGESTION_BATCH_DELAY,
    max_retries: int = 5,
    max_tpm: int = 24000,  # Strict cap well under 30,000 TPM limit
    max_rpm: int = 80,     # Strict cap well under 100 RPM limit
) -> dict:
    """
    Index documents in batches with a sliding-window rate limiter
    that strictly bounds both Tokens-Per-Minute (TPM) and Requests-Per-Minute (RPM).
    Includes automatic exponential backoff on unexpected rate limits.
    """
    total_stats = {
        "num_added": 0,
        "num_updated": 0,
        "num_skipped": 0,
        "num_deleted": 0,
    }
    total_chunks = len(chunks)
    total_batches = (total_chunks + batch_size - 1) // batch_size

    # Sliding window history of (timestamp, token_count)
    request_history: list[tuple[float, int]] = []

    logger.info(
        "Starting batched indexing: %d chunks across %d batches (batch_size=%d, target_tpm=%d, delay=%.1fs)",
        total_chunks,
        total_batches,
        batch_size,
        max_tpm,
        delay_seconds,
    )

    for b_idx in range(total_batches):
        start_i = b_idx * batch_size
        end_i = min(start_i + batch_size, total_chunks)
        batch = chunks[start_i:end_i]

        # Estimate tokens in this batch (~4 chars per token)
        batch_tokens = sum(max(1, len(d.page_content) // 4) for d in batch)

        # ── Sliding Window Throttler (Strictly protects 30k TPM & 100 RPM) ──
        while True:
            now = time.time()
            # Retain only requests within the last 60 seconds
            request_history = [(t, tok) for t, tok in request_history if now - t < 60.0]
            current_window_tokens = sum(tok for _, tok in request_history)
            current_window_reqs = len(request_history)

            exceeds_tpm = (current_window_tokens + batch_tokens) > max_tpm
            exceeds_rpm = current_window_reqs >= max_rpm

            if (exceeds_tpm or exceeds_rpm) and request_history:
                oldest_timestamp = request_history[0][0]
                sleep_needed = max(1.0, 60.0 - (now - oldest_timestamp) + 1.0)
                logger.info(
                    "Pacing rate: window at %d/%d tokens and %d/%d reqs. Pausing %.1fs for quota window to slide...",
                    current_window_tokens,
                    max_tpm,
                    current_window_reqs,
                    max_rpm,
                    sleep_needed,
                )
                time.sleep(sleep_needed)
            else:
                break

        logger.info(
            "Indexing batch %d/%d (chunks %d-%d of %d, ~%d tokens)...",
            b_idx + 1,
            total_batches,
            start_i + 1,
            end_i,
            total_chunks,
            batch_tokens,
        )

        success = False
        for attempt in range(max_retries):
            try:
                def _document_key_encoder(doc: Document) -> str:
                    source_file = doc.metadata.get("source_file", doc.metadata.get("source", ""))
                    doc_type = doc.metadata.get("doc_type", "")
                    raw = f"{source_file}:{doc_type}:{doc.page_content}"
                    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

                stats = index(
                    docs_source=batch,
                    record_manager=_record_manager,
                    vector_store=_vector_store,
                    cleanup=None,
                    source_id_key="source",
                    key_encoder=_document_key_encoder,
                )
                for k in total_stats:
                    total_stats[k] += stats.get(k, 0)

                # Record successful call in sliding window
                request_history.append((time.time(), batch_tokens))

                logger.info(
                    "Batch %d/%d completed: added=%d, skipped=%d, updated=%d",
                    b_idx + 1,
                    total_batches,
                    stats.get("num_added", 0),
                    stats.get("num_skipped", 0),
                    stats.get("num_updated", 0),
                )
                success = True
                break
            except Exception as exc:
                err_str = str(exc)
                # is_daily_quota = (
                #     or "limit: 1000" in err_str.lower()
                # )
                # if is_daily_quota:
                #     logger.warning(
                #         "Daily Google API quota reached (1,000 requests). Gracefully saving progress. "
                #         "%d new chunks were successfully indexed in this run. Run again tomorrow to resume!",
                #         total_stats["num_added"],
                #     )
                #     return total_stats

                is_rate_limit = (
                    "429" in err_str
                    or "RESOURCE_EXHAUSTED" in err_str
                    or "ResourceExhausted" in err_str
                    or "quota" in err_str.lower()
                    or "rate limit" in err_str.lower()
                )
                if is_rate_limit and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 60  # 35s, 70s, 105s...
                    logger.warning(
                        "Rate limit hit on batch %d/%d (attempt %d/%d). Sleeping for %ds before retry. Error: %s",
                        b_idx + 1,
                        total_batches,
                        attempt + 1,
                        max_retries,
                        wait_time,
                        err_str[:200],
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        "Failed batch %d/%d after %d attempts: %s",
                        b_idx + 1,
                        total_batches,
                        attempt + 1,
                        exc,
                    )
                    raise

        # Respect inter-batch delay
        if success and b_idx < total_batches - 1 and delay_seconds > 0:
            time.sleep(delay_seconds)

    return total_stats


def _get_indexed_source_ids() -> set[str]:
    """Fetch all chunk source IDs already indexed in Supabase."""
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(SUPABASE_DB_URL)
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT DISTINCT group_id FROM upsertion_record WHERE namespace = :ns AND group_id IS NOT NULL"),
                {"ns": RECORD_MANAGER_NAMESPACE}
            ).fetchall()
            return {r[0] for r in rows if r[0]}
    except Exception as exc:
        logger.warning("Could not fetch indexed source IDs from Supabase: %s", exc)
        return set()


def run_full_ingestion() -> dict:
    """
    Clone all configured repos, split documents, skip chunks already indexed,
    and upsert only new chunks to Pinecone in batches across multiple runs.

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

    total_chunks_count = len(all_chunks)
    logger.info("Total chunks across all repositories: %d", total_chunks_count)

    if not all_chunks:
        return {"num_added": 0, "num_updated": 0, "num_skipped": 0, "num_deleted": 0}

    # Pre-filter: skip chunks that already exist in Supabase/Pinecone
    already_indexed = _get_indexed_source_ids()
    unindexed_chunks = [
        c for c in all_chunks 
        if c.metadata.get("source") not in already_indexed
    ]

    logger.info(
        "Ingestion progress: %d already indexed in Pinecone | %d remaining to index (out of %d total chunks)",
        len(already_indexed),
        len(unindexed_chunks),
        total_chunks_count,
    )

    if not unindexed_chunks:
        logger.info("All %d documents are already indexed in Pinecone! Nothing new to embed.", total_chunks_count)
        return {
            "num_added": 0,
            "num_updated": 0,
            "num_skipped": total_chunks_count,
            "num_deleted": 0,
        }

    # Only process chunks that have never been indexed yet
    stats = _index_in_batches(unindexed_chunks)
    stats["num_skipped"] = stats.get("num_skipped", 0) + len(already_indexed)

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
        if not _md_file_filter(file_info.get("path", "")):
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

    stats = _index_in_batches(chunks)

    logger.info("Partial indexing stats: %s", stats)
    return stats
