"""
RAG evaluation pipeline using the Ragas framework.

Computes four metrics:
  1. Faithfulness — is the answer grounded in context? (no hallucinations)
  2. Answer Relevancy — does the answer address the original query?
  3. Context Precision — are relevant chunks ranked higher?
  4. Context Recall — did retrieval capture all necessary information?

Usage:
    # With a pre-existing test dataset:
    python evaluate.py --dataset test_data.json

    # Generate test data by running the agent on sample questions:
    python evaluate.py --generate

Output:
    eval_results.csv + console summary
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import warnings

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
from datasets import Dataset
from langchain_groq import ChatGroq
from embeddings import get_embeddings
from ragas import EvaluationDataset, evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
)

# ---------------------------------------------------------------------------
# Sample evaluation questions (used with --generate)
# ---------------------------------------------------------------------------

_SAMPLE_QUESTIONS = [
    {
        "question": "How do we handle Redis connection pool exhaustion in the auth service?",
        "ground_truth": "The auth service runbook describes increasing the Redis connection pool "
                        "max_connections parameter and implementing connection timeout retries.",
    },
    {
        "question": "What is the status of the Kubernetes 1.29 upgrade?",
        "ground_truth": "The Kubernetes 1.29 upgrade is currently in progress, assigned to bob@company.com. "
                        "It requires testing CRDs and validating Istio sidecar injection.",
    },
    {
        "question": "What are the best practices for writing an Architecture Decision Record?",
        "ground_truth": "ADRs should include a title, status, context, decision, and consequences. "
                        "They should be numbered sequentially and stored in a docs/adr directory.",
    },
    {
        "question": "How do we implement circuit breakers between microservices?",
        "ground_truth": "Circuit breakers should use the Resilience4j pattern with configurable "
                        "failure thresholds, timeout durations, and half-open state probing.",
    },
    {
        "question": "What critical tickets are currently open for the notification service?",
        "ground_truth": "There are two open tickets for notification-service: a flaky test issue "
                        "and a Sentry spike with NoneType AttributeError.",
    },
]


def _extract_contexts_from_trace(trace: dict) -> list[str]:
    """Extract and format context chunks from search_docs, web_search, and query_tickets."""
    contexts: list[str] = []
    seen: set[str] = set()

    for step in trace.get("steps", []):
        if step.get("type") != "tool_call":
            continue

        raw_output = step.get("tool_output_truncated", "")
        if not raw_output:
            continue

        try:
            parsed = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
        except (json.JSONDecodeError, TypeError):
            # Fallback for truncated/malformed JSON: extract content or description strings safely
            matches = re.findall(r'"(?:content|description)"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', str(raw_output))
            for m in matches:
                clean = re.sub(r'\\n', '\n', m).strip()
                if clean and clean not in seen:
                    seen.add(clean)
                    contexts.append(clean)
            continue

        if not isinstance(parsed, dict) or parsed.get("status") != "success":
            continue

        items = parsed.get("results", []) or parsed.get("tickets", [])
        for item in items:
            if not isinstance(item, dict):
                continue

            # Case A: Document chunk or web result (has 'content')
            if item.get("content"):
                src = item.get("source_file") or item.get("url") or ""
                prefix = f"[{src}] " if src else ""
                chunk = f"{prefix}{item['content'].strip()}"
                if chunk not in seen:
                    seen.add(chunk)
                    contexts.append(chunk)

            # Case B: Ticket (has 'title' or 'description')
            elif item.get("title") or item.get("description"):
                ticket_str = (
                    f"Ticket #{item.get('id', '?')} [{item.get('priority', 'normal')}] "
                    f"{item.get('title', '')} (Component: {item.get('component', 'N/A')}, "
                    f"Status: {item.get('status', 'open')}, Assignee: {item.get('assignee', 'unassigned')}): "
                    f"{item.get('description', '')}"
                )
                if ticket_str not in seen:
                    seen.add(ticket_str)
                    contexts.append(ticket_str)

    return contexts if contexts else ["No context retrieved"]


# ---------------------------------------------------------------------------
# Generate test data by running the agent
# ---------------------------------------------------------------------------

async def _generate_test_data() -> list[dict]:
    """Run the agent on sample questions and collect contexts + answers."""
    # Import here to avoid circular deps when just loading the module
    from agent import run_agent

    test_data = []

    for sample in _SAMPLE_QUESTIONS:
        question = sample["question"]
        ground_truth = sample["ground_truth"]

        print(f"  Running agent for: {question[:60]}...")

        try:
            result = await run_agent(question)
            answer = result["answer"]
            trace = result["trace"]
            contexts = _extract_contexts_from_trace(trace)

            test_data.append({
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": ground_truth,
            })

        except Exception as exc:
            print(f"  [WARN] Failed: {exc}")
            test_data.append({
                "user_input": question,
                "response": f"Error: {exc}",
                "retrieved_contexts": ["No context retrieved"],
                "reference": ground_truth,
            })

        # Pacing: pause 5 seconds between queries to avoid Groq TPM rate limits
        await asyncio.sleep(5)

    return test_data


# ---------------------------------------------------------------------------
# Load existing test dataset
# ---------------------------------------------------------------------------

def _load_test_dataset(path: str) -> list[dict]:
    """
    Load a JSON test dataset.

    Expected format (array of objects):
    [
      {
        "user_input": "…",
        "response": "…",
        "retrieved_contexts": ["…", "…"],
        "reference": "…"
      }
    ]

    Also accepts legacy field names (question/answer/contexts/ground_truth)
    and maps them to Ragas 0.2+ column names.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Map legacy field names if present
    mapped = []
    for entry in raw:
        mapped.append({
            "user_input": entry.get("user_input", entry.get("question", "")),
            "response": entry.get("response", entry.get("answer", "")),
            "retrieved_contexts": entry.get("retrieved_contexts", entry.get("contexts", [])),
            "reference": entry.get("reference", entry.get("ground_truth", "")),
        })

    return mapped


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

def run_evaluation(test_data: list[dict], output_path: str = "eval_results.csv") -> pd.DataFrame:
    """
    Run Ragas evaluation on test data and save results to CSV.

    Returns a DataFrame with per-query scores.
    """
    print(f"\nEvaluating {len(test_data)} samples …\n")

    # Build Hugging Face Dataset
    hf_data = {
        "user_input": [d["user_input"] for d in test_data],
        "response": [d["response"] for d in test_data],
        "retrieved_contexts": [d["retrieved_contexts"] for d in test_data],
        "reference": [d["reference"] for d in test_data],
    }
    hf_dataset = Dataset.from_dict(hf_data)
    eval_dataset = EvaluationDataset.from_hf_dataset(hf_dataset)

    eval_llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
        max_retries=10,
        request_timeout=120,
    )

    eval_embeddings = get_embeddings()
    # Instantiate metrics
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ]

    # Rate-limit pacing configuration: run sequentially with exponential retries to prevent Groq TPM limit
    run_config = RunConfig(
        max_workers=1,
        timeout=180,
        max_retries=10,
        max_wait=60,
    )

    # Run evaluation
    results = evaluate(
        dataset=eval_dataset, 
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=run_config,
        batch_size=1,
    )

    # Convert to DataFrame
    df = results.to_pandas()

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("AGGREGATE SCORES")
    print("=" * 60)

    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for col in metric_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            min_val = df[col].min()
            print(f"  {col:<25s}  mean={mean_val:.3f}  min={min_val:.3f}")

    # Flag failing queries (any metric < 0.5)
    print("\n" + "=" * 60)
    print("FAILING QUERIES (any metric < 0.5)")
    print("=" * 60)

    available_cols = [c for c in metric_cols if c in df.columns]
    if available_cols:
        failing = df[df[available_cols].min(axis=1) < 0.5]
        if failing.empty:
            print("  ✅ All queries passed!")
        else:
            for _, row in failing.iterrows():
                q = row.get("user_input", "")[:60]
                scores = {c: f"{row.get(c, 0):.2f}" for c in available_cols}
                print(f"  ❌ {q}…  {scores}")

    return df


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Pipeline (Ragas)")
    parser.add_argument("--dataset", type=str, help="Path to test dataset JSON file")
    parser.add_argument("--generate", action="store_true",
                        help="Generate test data by running the agent on sample questions")
    parser.add_argument("--output", type=str, default="eval_results.csv",
                        help="Output CSV file path (default: eval_results.csv)")
    args = parser.parse_args()

    if args.generate:
        print("Generating test data by running the agent …")
        test_data = asyncio.run(_generate_test_data())

        # Also save the generated test data for future runs
        gen_path = "generated_test_data.json"
        with open(gen_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        print(f"Test data saved to {gen_path}")

    elif args.dataset:
        if not os.path.exists(args.dataset):
            print(f"Error: {args.dataset} not found")
            sys.exit(1)
        test_data = _load_test_dataset(args.dataset)

    else:
        print("Error: Specify --dataset <path> or --generate")
        parser.print_help()
        sys.exit(1)

    run_evaluation(test_data, output_path=args.output)


if __name__ == "__main__":
    main()
