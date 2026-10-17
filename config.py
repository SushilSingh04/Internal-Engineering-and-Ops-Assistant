"""
Centralized configuration — all env vars and constants loaded here.
"""
import os
import tempfile
from dotenv import load_dotenv

# Loads local env file and read all key value pairs
load_dotenv()

# ---------------------------------------------------------------------------
# Groq LLM
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_AGENT_ITERATIONS: int = int(os.getenv("MAX_AGENT_ITERATIONS", "3"))

# ---------------------------------------------------------------------------
# Google Embeddings
# ---------------------------------------------------------------------------
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

# ---------------------------------------------------------------------------
# Pinecone
# ---------------------------------------------------------------------------
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "engineering-docs")

# ---------------------------------------------------------------------------
# Supabase (Postgres) — transaction-mode pooler, port 6543
# Used for BOTH the SQLRecordManager and the tickets table.
# ---------------------------------------------------------------------------
SUPABASE_DB_URL: str = os.getenv("SUPABASE_DB_URL", "")

# ---------------------------------------------------------------------------
# Tavily Web Search
# ---------------------------------------------------------------------------
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# ---------------------------------------------------------------------------
# Webhook Security
# ---------------------------------------------------------------------------
WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET", "")

# ---------------------------------------------------------------------------
# Repository Sources
# ---------------------------------------------------------------------------
REPOS = [
    {
        "clone_url": "https://gitlab.com/sushilksingh2001/runbooks",
        "doc_type": "runbook",
        "branch": "master",
    },
    {
        "clone_url": "https://github.com/SushilSingh04/architecture-decision-record",
        "doc_type": "architecture",
        "branch": "main",
    },
    {
        "clone_url": "https://github.com/SushilSingh04/post-mortems",
        "doc_type": "runbook",  # post-mortems mapped to runbook category
        "branch": "master",
    },
]

# ---------------------------------------------------------------------------
# Document Ingestion
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1600"))       # ~400 tokens
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
CLONE_DIR: str = os.getenv("CLONE_DIR", os.path.join(tempfile.gettempdir(), "repos"))
INGESTION_BATCH_SIZE: int = int(os.getenv("INGESTION_BATCH_SIZE", "40"))
INGESTION_BATCH_DELAY: float = float(os.getenv("INGESTION_BATCH_DELAY", "3.0"))

# ---------------------------------------------------------------------------
# Record Manager (LangChain indexing)
# ---------------------------------------------------------------------------
RECORD_MANAGER_NAMESPACE: str = "pinecone/engineering-docs"

# ---------------------------------------------------------------------------
# Trace Logging
# ---------------------------------------------------------------------------
TRACE_DIR: str = os.getenv("TRACE_DIR", "traces")

# ---------------------------------------------------------------------------
# API Server
# ---------------------------------------------------------------------------
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")
