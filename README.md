# Internal Engineering & Operations Assistant

An enterprise-grade, agentic Retrieval-Augmented Generation (RAG) assistant designed to help developers, site reliability engineers (SREs), and on-call teams troubleshoot production incidents, explore architectural decision records (ADRs), and inspect live ticketing workflows.

---

## Architecture Overview

```mermaid
flowchart TD
    User["Developer / SRE"] --> UI["Streamlit Frontend (app.py)"]
    UI --> API["FastAPI Backend (main.py)"]
    
    subgraph "Core Agent Loop (agent.py)"
        API --> Agent["ChatGroq LLM (openai/gpt-oss-120b)"]
        Agent -->|"1. Mandatory <thinking> protocol"| Reason["Chain-of-Thought Reasoning"]
        Reason -->|"2. Parallel Tool Dispatch"| ToolExec["Tool Execution Engine"]
        ToolExec -->|"Context Truncation"| Middleware["trace_logger.py Middleware"]
        Middleware -->|"Feed evidence back"| Agent
        Agent -->|"3. Synthesize or degrade gracefully"| Final["Final Answer + Reasoning Trace"]
    end

    subgraph "Knowledge & Data Layer"
        ToolExec -->|"search_docs"| Pinecone[("Pinecone Vector Store (1536 dims)")]
        ToolExec -->|"query_tickets"| Postgres[("Supabase Postgres (Tickets Table)")]
        ToolExec -->|"web_search"| Tavily["Tavily Public Web Search"]
    end

    subgraph "Embeddings Layer (embeddings.py)"
        Pinecone <--> EmbSwitch{"Environment Switcher"}
        EmbSwitch -->|"Local / Dev"| LocalEmb["Local 1536-dim Model (Orange/orange-nomic-v1.5-1536)"]
        EmbSwitch -->|"Deployed Service"| CloudEmb["Cloud Provider (Google GenAI 1536-dim)"]
        LocalEmb -.->|"Fallback"| CloudEmb
        CloudEmb -.->|"Fallback"| LocalEmb
    end

    subgraph "Ingestion Pipeline (ingestion.py)"
        GitRepos["Git Repositories (Runbooks, ADRs, Incidents)"] --> DocSplitter["Markdown & Recursive Splitter"]
        DocSplitter --> RecMgr["SQLRecordManager (Supabase)"]
        RecMgr --> Pinecone
    end
```

---

## Key Features

### 1. Transparent Agentic Reasoning
- **Mandatory `<thinking>` Blocks**: Before invoking tools or synthesizing the final answer, the agent explicitly breaks down:
  - *Evidence gathered so far*
  - *Missing information*
  - *Next logical step*
- **Parallel Tool Execution**: Invokes multiple tools simultaneously when a query spans multiple domains (e.g., querying open tickets while searching architectural decisions).
- **Graceful Degradation**: Enforces a strict iteration cap (`MAX_AGENT_ITERATIONS`). If the cap is reached, tools are stripped and the model is forced into a best-effort synthesis turn to guarantee a fast, deterministic response.

### 2. Hybrid Data Retrieval
- **Unstructured Documentation (`search_docs`)**:
  - Semantic vector search powered by Pinecone.
  - Metadata filtering by document category (`runbook`, `architecture`, `onboarding`, or `any`).
- **Structured Operations Database (`query_tickets`)**:
  - Live query engine backed by Supabase PostgreSQL and SQLAlchemy.
  - Multi-parameter filtering across status (`open`, `in_progress`, `resolved`), priority, component, assignee, and free-text search.
- **Sanitized Web Search (`web_search`)**:
  - Real-time search via Tavily for external library docs, GitHub issues, and CVE advisories.
  - Built-in guardrails against querying proprietary hostnames, private IPs, or internal identifiers.

### 3. Dual-Mode 1536-Dimensional Embeddings
- **Pinecone Vector Space Consistency**: The vector index enforces a strict `1536` dimensionality.
- **Local Provider**: Employs `Orange/orange-nomic-v1.5-1536` via `sentence-transformers` for local offline development.
- **Cloud Provider**: Connects to Google Generative AI (`gemini-embedding-001` with 1536 dimensions) for deployed environments.
- **Automated Environment Detection**:
  - Running locally? The app defaults to the local model.
  - Running on a cloud host (Render, Railway, Fly.io, Kubernetes, AWS, etc.)? Automatically switches to the cloud provider.
  - Seamless, bidirectional failover if either provider encounters network or quota errors.

### 4. Incremental Git Ingestion Pipeline
- Tracks document state using LangChain's `SQLRecordManager` on Postgres, preventing duplicate chunk upserts and infinite re-embedding cycles.
- Resilient tree traversal to avoid reserved Windows file naming conflicts (such as `aux.*` DOS names).
- Hierarchical chunking via `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter`.

### 5. Automated RAG Evaluation (Ragas)
- End-to-end evaluation suite (`evaluate.py`) computing standard industry metrics:
  - **Faithfulness**: Verifies claims are grounded in retrieved context (hallucination detection).
  - **Answer Relevancy**: Assesses direct query alignment and penalizes non-committal answers.
  - **Context Precision**: Evaluates whether relevant chunks are ranked at the top.
  - **Context Recall**: Measures whether all ground-truth facts were captured by retrieval.
- Sequential batch pacing (`RunConfig(max_workers=1)`) to eliminate TPM/RPM 429 rate limit errors from LLM providers.

---

## Project Structure

```text
.
├── agent.py               # Core ReAct loop, thinking block parser, tool binding
├── app.py                 # Streamlit UI with expandable reasoning & multi-turn memory
├── config.py              # Centralized environment variable loader & constants
├── embeddings.py          # Dual-mode (Local 1536-dim / Cloud 1536-dim) embedding engine
├── evaluate.py            # Ragas evaluation pipeline with batch rate-limiting
├── ingestion.py           # Git document loader, chunking, and incremental vector indexing
├── main.py                # FastAPI backend service & HMAC-authenticated CI/CD webhooks
├── requirements.txt       # Project dependencies with pinned compatibility bounds
├── setup_db.py            # SQLAlchemy database schema and idempotent seed data
├── tools.py               # Tool implementations: search_docs, query_tickets, web_search
├── trace_logger.py        # Structured JSON trace persistence & truncation middleware
├── traces/                # Stored runtime execution traces for analysis & auditing
└── generated_test_data.json # Saved evaluation test cases and ground truths
```

---

## Environment Variables

Create a `.env` file in the project root based on the following reference:

```ini
# LLM Provider (Groq)
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-120b
MAX_AGENT_ITERATIONS=5

# Cloud Embedding Provider (Google GenAI)
GOOGLE_API_KEY=AIzaSy...
EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIMENSIONS=1536

# Local Embedding Configuration
LOCAL_EMBEDDING_MODEL=Orange/orange-nomic-v1.5-1536
EMBEDDING_PROVIDER=auto              # "auto" (default), "local", or "cloud"
ENVIRONMENT=development             # "development" or "production"
HF_TOKEN=hf_...                     # Hugging Face token (optional, speeds downloads)
HF_HUB_DISABLE_SYMLINKS_WARNING=1   # Suppress Windows symlink warning

# Vector Database (Pinecone)
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=engineering-docs

# Relational Database (Supabase PostgreSQL)
SUPABASE_DB_URL=postgresql://postgres.xxx:password@aws-0-region.pooler.supabase.com:6543/postgres

# External Web Search (Tavily)
TAVILY_API_KEY=tvly-...

# Security & Server
WEBHOOK_SECRET=your_secure_webhook_secret_here
API_HOST=0.0.0.0
API_PORT=8000
BACKEND_URL=http://localhost:8000
```

---

## Quick Start Guide

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/your-org/internal-engineering-assistant.git
cd internal-engineering-assistant

# Create virtual environment (Python 3.10+ recommended)
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize the Database & Seed Sample Tickets

Run the database setup script to create the `tickets` table and populate test tickets:

```bash
python setup_db.py
```

### 3. Ingest Engineering Documentation

Clone and index the runbooks, architecture decision records, and post-mortems into Pinecone:

```bash
python ingestion.py
```

### 4. Launch the FastAPI Backend

```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
- Interactive API documentation will be available at: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### 5. Launch the Streamlit Frontend

In a separate terminal (with virtual environment activated):

```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## Evaluating the Assistant (Ragas)

### Generate & Evaluate New Test Data
To run the assistant against benchmark questions, record traces, and evaluate performance:

```bash
python evaluate.py --generate
```

### Re-evaluate Existing Test Data
To evaluate saved questions without incurring agent inference time or re-running tools:

```bash
python evaluate.py --dataset generated_test_data.json
```

Results and aggregate score breakdowns are saved to `eval_results.csv`.

---

## Continuous Integration & Ingestion Webhook

The backend exposes an HMAC SHA-256 authenticated webhook endpoint to trigger automated documentation syncs from GitHub / GitLab CI pipelines:

```bash
POST /webhook/ingest
Headers:
  X-Hub-Signature-256: sha256=<hmac_sha256_hex_digest>
  Content-Type: application/json

Body:
{
  "repository": {
    "clone_url": "https://github.com/SushilSingh04/architecture-decision-record"
  },
  "ref": "refs/heads/main"
}
```

When triggered, the service pulls latest commits, re-chunks altered files, and updates Pinecone incrementally without dropping existing index data.

---

## License

This project is licensed under the Apache 2.0 License.
