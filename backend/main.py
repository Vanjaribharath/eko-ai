"""
main.py  (v2 — with query understanding, spell correction, conversational responses)
=======
What changed from v1:
1. All queries now go through understand.parse_query() first, which
   handles intent detection, spelling correction, and abbreviation
   expansion before anything reaches the retrieval engine.
2. Greetings, farewells, thanks, and meta-questions about EKO-AI
   are answered directly without touching the document index.
3. The /ask response now includes: correction_note, intent,
   corrected_query, and answer_parts (structured bullet list).
4. /document/{doc_id} endpoint added — returns the full readable
   content of a specific document so the frontend can show it.
5. /history endpoint added — returns recent questions + answers
   for a session (keyed by a simple session_id the frontend sends).

HOW TO RUN (unchanged):
    cd backend
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000
"""

import time
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from ingest import build_index
from synthesize import build_answer, build_conversational_response
from understand import parse_query, set_vocabulary, INTENT_KNOWLEDGE
import audit

SEARCH_INDEX = None
INDEX_STATS  = None

# In-memory session history (cleared on server restart — fine for prototype)
# Maps session_id -> list of {question, answer, sources, timestamp}
SESSION_HISTORY: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global SEARCH_INDEX, INDEX_STATS
    print("EKO-AI starting — building search index...")
    SEARCH_INDEX, INDEX_STATS = build_index(verbose=True)
    # Give the spell corrector the document vocabulary
    set_vocabulary(list(SEARCH_INDEX.vocabulary.keys()))
    print(f"Index ready: {INDEX_STATS['total_chunks']} chunks, "
          f"{INDEX_STATS['vocabulary_size']} vocabulary words.")
    yield


app = FastAPI(
    title="EKO-AI v2",
    description="Internal knowledge assistant — retrievalonly, spell-aware, conversational.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    session_id: Optional[str] = "default"


class SourceOut(BaseModel):
    doc_id: str
    doc_title: str
    section: str
    source_file: str
    relevance: float


class AskResponse(BaseModel):
    found: bool
    intent: str
    corrected_query: Optional[str] = None
    correction_note: Optional[str] = None
    answer: str
    answer_parts: List[str]       # individual numbered points for frontend rendering
    confidence: str
    sources: List[SourceOut]
    latency_ms: float


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    if SEARCH_INDEX is None:
        return {"status": "starting"}
    return {"status": "ready", "stats": INDEX_STATS}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Main Q&A endpoint. Steps:
    1. Parse intent + correct spelling
    2. If conversational intent → return scripted natural response
    3. If knowledge intent → search index → build structured answer
    4. Log everything to audit trail
    5. Append to session history
    """
    start = time.time()

    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if SEARCH_INDEX is None:
        raise HTTPException(status_code=503, detail="Index is still building.")

    # Step 1 — understand the query
    parsed = parse_query(request.question)

    # Step 2 — route by intent
    if parsed.intent != INTENT_KNOWLEDGE:
        # Conversational response — no retrieval needed
        conv_answer = build_conversational_response(parsed.intent, request.question)
        latency_ms  = (time.time() - start) * 1000
        audit.log_query(
            query=request.question,
            answer_found=True,
            confidence="high",
            source_doc_ids=[],
            latency_ms=latency_ms,
        )
        return AskResponse(
            found=True,
            intent=parsed.intent,
            corrected_query=None,
            correction_note=None,
            answer=conv_answer.answer_text,
            answer_parts=conv_answer.answer_parts,
            confidence="high",
            sources=[],
            latency_ms=round(latency_ms, 2),
        )

    # Step 3 — knowledge query: search and synthesize
    results = SEARCH_INDEX.search(
        parsed.search_query, top_k=request.top_k, only_approved=True
    )

    correction_note = ""
    if parsed.corrections_made:
        pairs = ", ".join(f'"{a}" → "{b}"' for a, b in parsed.corrections_made)
        correction_note = f"I noticed a possible typo and searched for: {parsed.corrected}. (Corrections: {pairs})"

    answer = build_answer(
        query=parsed.search_query,
        results=results,
        correction_note=correction_note,
    )

    latency_ms = (time.time() - start) * 1000

    # Step 4 — audit log
    audit.log_query(
        query=request.question,
        answer_found=answer.found,
        confidence=answer.confidence,
        source_doc_ids=[s.doc_id for s in answer.sources],
        latency_ms=latency_ms,
    )

    # Step 5 — session history
    session_id = request.session_id or "default"
    if session_id not in SESSION_HISTORY:
        SESSION_HISTORY[session_id] = []
    SESSION_HISTORY[session_id].append({
        "question": request.question,
        "corrected": parsed.corrected if parsed.corrections_made else None,
        "answer": answer.answer_text,
        "sources": [s.doc_id for s in answer.sources],
        "confidence": answer.confidence,
        "timestamp": time.time(),
    })
    # Keep only last 20 exchanges per session
    SESSION_HISTORY[session_id] = SESSION_HISTORY[session_id][-20:]

    return AskResponse(
        found=answer.found,
        intent=parsed.intent,
        corrected_query=parsed.corrected if parsed.corrections_made else None,
        correction_note=correction_note or None,
        answer=answer.answer_text,
        answer_parts=answer.answer_parts,
        confidence=answer.confidence,
        sources=[SourceOut(**vars(s)) for s in answer.sources],
        latency_ms=round(latency_ms, 2),
    )


@app.get("/document/{doc_id}")
def get_document(doc_id: str):
    """
    Returns the full readable content of a specific document so the
    frontend can show it when the user clicks a source citation.
    This replaces "sources you can't read" with "click to read the full policy."
    """
    if SEARCH_INDEX is None:
        raise HTTPException(status_code=503, detail="Index not ready.")

    # Collect all chunks belonging to this document, in order
    chunks = [c for c in SEARCH_INDEX.chunks if c.doc_id == doc_id]
    if not chunks:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found in index.")

    chunks.sort(key=lambda c: c.chunk_index)

    # Build a readable, section-organised view of the document
    sections = []
    current_section = None
    section_text    = []

    for chunk in chunks:
        if chunk.section != current_section:
            if current_section is not None:
                sections.append({"heading": current_section, "text": " ".join(section_text)})
            current_section = chunk.section
            section_text    = [chunk.text]
        else:
            section_text.append(chunk.text)

    if current_section is not None:
        sections.append({"heading": current_section, "text": " ".join(section_text)})

    return {
        "doc_id":     chunks[0].doc_id,
        "doc_title":  chunks[0].doc_title,
        "source_file":chunks[0].source_file,
        "status":     chunks[0].status,
        "metadata":   chunks[0].metadata,
        "sections":   sections,
        "total_chunks": len(chunks),
    }


@app.get("/history")
def get_history(session_id: str = "default", limit: int = 20):
    """Returns recent Q&A history for a session."""
    entries = SESSION_HISTORY.get(session_id, [])
    return {"session_id": session_id, "entries": list(reversed(entries))[:limit]}


@app.get("/audit")
def get_audit_log(limit: int = 50):
    return {"entries": audit.read_recent_logs(limit=limit)}


@app.get("/documents")
def list_documents():
    if SEARCH_INDEX is None:
        return {"documents": []}
    seen = {}
    for chunk in SEARCH_INDEX.chunks:
        if chunk.doc_id not in seen:
            seen[chunk.doc_id] = {
                "doc_id":      chunk.doc_id,
                "doc_title":   chunk.doc_title,
                "source_file": chunk.source_file,
                "status":      chunk.status,
                "chunk_count": 0,
            }
        seen[chunk.doc_id]["chunk_count"] += 1
    return {"documents": sorted(seen.values(), key=lambda d: d["doc_id"])}


@app.post("/reindex")
def reindex():
    global SEARCH_INDEX, INDEX_STATS
    SEARCH_INDEX, INDEX_STATS = build_index(verbose=True)
    set_vocabulary(list(SEARCH_INDEX.vocabulary.keys()))
    return {"status": "reindexed", "stats": INDEX_STATS}
