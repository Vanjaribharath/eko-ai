"""
audit.py
========
Every question asked, every set of sources returned, and the
timestamp are written here. This directly demonstrates the audit
logging principle from the EKO-AI security design — in production
this would write to a real database; here it writes to a simple local
JSON-lines file so you can open it and read it directly.

Why JSON Lines (one JSON object per line) instead of a single JSON
array: it's append-only and crash-safe — if the process stops
mid-write, every previous complete line is still valid and readable.
This is a small but real production pattern, not just a prototype
shortcut.
"""

import json
import os
from datetime import datetime, timezone

LOG_PATH = os.path.join(os.path.dirname(__file__), "data", "audit_log.jsonl")


def log_query(query: str, answer_found: bool, confidence: str, source_doc_ids: list, latency_ms: float):
    """
    Appends one audit record. Called once per question, right after
    an answer is produced (whether or not a source was found — the
    refusal cases are exactly the ones a security/content team most
    wants visibility into, since repeated refusals on the same topic
    signal a real knowledge gap).
    """
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "answer_found": answer_found,
        "confidence": confidence,
        "source_doc_ids": source_doc_ids,
        "latency_ms": round(latency_ms, 2),
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_recent_logs(limit: int = 50) -> list:
    """Reads the most recent audit records, newest first. Used by the
    (optional) /audit endpoint so you can see the log without leaving
    the browser."""
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    records = [json.loads(line) for line in lines if line.strip()]
    return list(reversed(records))[:limit]
