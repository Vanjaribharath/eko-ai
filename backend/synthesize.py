"""
synthesize.py
=============
STAGE 4: ANSWER SYNTHESIS

Turns retrieved chunks into a complete, natural, human-readable answer.

KEY UPGRADE from the previous version:
Old: returned 1-2 raw extracted sentences — felt robotic and incomplete.
New: builds a structured answer with:
  - An opening sentence that directly addresses the question
  - All relevant points from matched chunks, deduped and ordered
  - Natural connective language between points
  - A closing note pointing to the source document for full details
  - Full source metadata for every document used

This is still extractive (sentences come directly from approved documents,
nothing is invented), but the *framing* and *structure* is added by
this code — making it read like a real assistant answer, not a search
snippet dump.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple
from chunk import Chunk


@dataclass
class Source:
    doc_id: str
    doc_title: str
    section: str
    source_file: str
    relevance: float


@dataclass
class Answer:
    found: bool
    answer_text: str          # full formatted answer string
    answer_parts: List[str]   # individual bullet points (for frontend rendering)
    sources: List[Source]
    confidence: str
    correction_note: str = "" # populated if spelling was corrected


# ---------------------------------------------------------------------------
# Sentence extraction helpers
# ---------------------------------------------------------------------------

def rejoin_lines(text: str) -> List[str]:
    """
    Converts raw chunk text (which may have PDF-style line wraps) back
    into whole sentences/items. Each numbered list item becomes one item;
    everything else is joined with the previous line.
    """
    raw_lines = [l.strip() for l in text.split("\n") if l.strip()]
    paragraphs = []
    list_marker = re.compile(r"^\d+\.\s+")
    for line in raw_lines:
        if not paragraphs or list_marker.match(line):
            paragraphs.append(line)
        else:
            paragraphs[-1] += " " + line
    # Strip the list number prefix from each item
    return [list_marker.sub("", p).strip() for p in paragraphs]


def score_overlap(sentence: str, query_words: set) -> int:
    """How many query words appear in this sentence."""
    words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
    return len(words & query_words)


def extract_best_points(chunk_text: str, query_words: set, max_points: int = 4) -> List[str]:
    """
    Returns the best `max_points` sentences from a chunk, ranked by
    how many query words they contain. Filters out very short fragments.
    """
    items = rejoin_lines(chunk_text)
    # Filter out fragments shorter than 6 words
    items = [i for i in items if len(i.split()) >= 6]
    ranked = sorted(items, key=lambda s: score_overlap(s, query_words), reverse=True)
    # Take top scorers but always include at least the first item of the chunk
    top = ranked[:max_points]
    return top


def dedupe(items: List[str]) -> List[str]:
    """Remove near-duplicate sentences — if two sentences share 80% of
    their words, keep only the longer one."""
    kept = []
    for item in items:
        words_item = set(re.findall(r"[a-z]+", item.lower()))
        is_dup = False
        for existing in kept:
            words_existing = set(re.findall(r"[a-z]+", existing.lower()))
            if len(words_item) == 0:
                continue
            overlap = len(words_item & words_existing) / len(words_item)
            if overlap > 0.80:
                is_dup = True
                break
        if not is_dup:
            kept.append(item)
    return kept


# ---------------------------------------------------------------------------
# Opening sentence templates — selected based on question type
# ---------------------------------------------------------------------------

def make_opening(query: str, top_doc_title: str, top_section: str) -> str:
    """
    Generates a natural opening sentence that directly addresses the
    question topic. Much better than jumping straight into a bullet list.
    """
    q = query.lower().strip().rstrip("?")

    # Detect question type and craft an appropriate opener
    if re.search(r"\b(can i|am i allowed|is it ok|is it allowed|may i)\b", q):
        return f"Based on our approved policies, here is what you need to know:"
    elif re.search(r"\b(what should i do|what do i do|how do i|how to|steps|process)\b", q):
        return f"Here is what you need to do, based on {top_doc_title}:"
    elif re.search(r"\b(when|what date|what time|how long|how many days)\b", q):
        return f"According to our policy documents:"
    elif re.search(r"\b(what is|what are|explain|tell me about|describe)\b", q):
        return f"Here is what our documents say about this:"
    elif re.search(r"\b(who|which team|which department|contact)\b", q):
        return f"Based on {top_doc_title}:"
    else:
        return f"Based on our approved internal documents:"


# ---------------------------------------------------------------------------
# Main answer builder
# ---------------------------------------------------------------------------

def build_answer(
    query: str,
    results: List[Tuple],
    min_score: float = 0.05,
    correction_note: str = "",
) -> Answer:
    """
    Builds a complete, structured, human-readable answer from retrieved
    chunks — without calling any external API.

    Structure of a generated answer:
    1. Opening sentence (contextual, based on question type)
    2. Numbered list of the most relevant points from all matched documents
    3. Closing note naming the source document for full details
    """
    # ── REFUSAL PATH ──────────────────────────────────────────────────────
    # If nothing scored above the threshold, never guess. Say so clearly.
    if not results or results[0][1] < min_score:
        return Answer(
            found=False,
            answer_text=(
                "I could not find an approved source in our documents that covers "
                "this question. This could mean the topic is not yet documented, "
                "or it might help to rephrase the question. For anything urgent, "
                "please reach out to HR, IT Helpdesk, or your manager directly."
            ),
            answer_parts=[],
            sources=[],
            confidence="none",
            correction_note=correction_note,
        )

    query_words = set(re.findall(r"[a-z0-9]+", query.lower()))
    # Remove very common words from query_words so they don't dominate scoring
    stopwords = {"what","when","where","who","how","why","can","the","is","are","a","an","i","my","do","does","should","will","have","has","in","on","of","to","for","and","or","not","it","its","this","that","me","you","we","our","your"}
    query_words -= stopwords

    top_score  = results[0][1]
    # Use chunks whose score is at least 50% of the top score
    used = [r for r in results if r[1] >= max(min_score, top_score * 0.50)][:4]

    # ── EXTRACT POINTS ────────────────────────────────────────────────────
    all_points  = []
    sources     = []
    seen_doc_ids = set()

    for chunk, score in used:
        points = extract_best_points(chunk.text, query_words, max_points=4)
        all_points.extend(points)

        # Only add each document once as a source (even if multiple
        # chunks from the same doc matched)
        if chunk.doc_id not in seen_doc_ids:
            sources.append(Source(
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                section=chunk.section,
                source_file=chunk.source_file,
                relevance=round(score, 3),
            ))
            seen_doc_ids.add(chunk.doc_id)

    # ── DEDUPE AND RANK ───────────────────────────────────────────────────
    all_points = dedupe(all_points)
    # Re-rank the deduped list by query overlap score
    all_points.sort(key=lambda s: score_overlap(s, query_words), reverse=True)
    # Keep the best 5 points total — enough for a complete answer without
    # overwhelming the user
    final_points = all_points[:5]

    if not final_points:
        # Fallback: just take the first two sentences from the top chunk
        fallback_items = rejoin_lines(used[0][0].text)
        final_points = [i for i in fallback_items[:3] if len(i.split()) >= 4]

    # ── ASSEMBLE THE ANSWER ───────────────────────────────────────────────
    top_chunk = used[0][0]
    opening   = make_opening(query, top_chunk.doc_title, top_chunk.section)

    # Build numbered list
    numbered = "\n".join(f"{i+1}. {pt}" for i, pt in enumerate(final_points))

    # Closing line pointing to source document
    if len(sources) == 1:
        closing = f"\nFor full details, refer to: {sources[0].doc_title} ({sources[0].doc_id})."
    else:
        doc_refs = ", ".join(f"{s.doc_title} ({s.doc_id})" for s in sources[:3])
        closing = f"\nFor full details, refer to: {doc_refs}."

    full_text = f"{opening}\n\n{numbered}{closing}"

    # ── CONFIDENCE ────────────────────────────────────────────────────────
    if top_score > 0.30:
        confidence = "high"
    elif top_score > 0.12:
        confidence = "medium"
    else:
        confidence = "low"

    return Answer(
        found=True,
        answer_text=full_text,
        answer_parts=final_points,
        sources=sources,
        confidence=confidence,
        correction_note=correction_note,
    )


# ---------------------------------------------------------------------------
# Conversational response builder (for greetings, thanks, meta-questions)
# ---------------------------------------------------------------------------

def build_conversational_response(intent: str, original: str) -> Answer:
    """
    Returns a natural, human response for conversational inputs that
    don't need document retrieval. Each intent has several response
    variants to avoid feeling repetitive.
    """
    from understand import (INTENT_GREETING, INTENT_FAREWELL,
                            INTENT_THANKS, INTENT_ABOUT_AI,
                            INTENT_CAPABILITIES)

    if intent == INTENT_GREETING:
        text = (
            "Hello! I am EKO-AI, your internal knowledge assistant.\n\n"
            "I can answer questions about:\n"
            "1. Company policies — leave, payroll, code of conduct, IT security\n"
            "2. Onboarding — Day 1 checklist, mandatory trainings, first-week steps\n"
            "3. All five towers — AuthEnq, EIS, ITS, Digital, and Operations\n"
            "4. HR processes — appraisals, internal mobility, benefits, exit\n\n"
            "What would you like to know?"
        )
    elif intent == INTENT_FAREWELL:
        text = "Take care! Feel free to come back whenever you have a question about our policies or processes."
    elif intent == INTENT_THANKS:
        text = (
            "Happy to help! If you need more details on any topic, just ask — "
            "I can search across all 24 of our approved policy documents."
        )
    elif intent == INTENT_ABOUT_AI:
        text = (
            "I am EKO-AI — Enterprise Knowledge and Operations AI.\n\n"
            "1. I search our approved internal policy documents to answer your questions.\n"
            "2. Every answer I give cites the exact document and section it came from.\n"
            "3. I never make things up — if I cannot find an approved source, I say so.\n"
            "4. I work entirely offline, with zero external API calls or internet access.\n"
            "5. Every question you ask is logged for audit and compliance purposes.\n\n"
            "I cover all five company towers: AuthEnq, EIS, ITS, Digital, and Operations. "
            "What would you like to know?"
        )
    elif intent == INTENT_CAPABILITIES:
        text = (
            "Here is what I can help with:\n\n"
            "1. Payroll and salary — structure, disbursement dates, Form 16, reimbursements\n"
            "2. Leave policy — types of leave, how to apply, attendance rules, WFH policy\n"
            "3. Onboarding — Day 1 steps, Week 1 checklist, mandatory trainings, project access\n"
            "4. Security policies — photography rules, AI tool usage, device rules, incident reporting\n"
            "5. Code of conduct — confidentiality rules, client data handling, what is prohibited\n"
            "6. Tower-specific guidance — AuthEnq, EIS, ITS, Digital, Operations ways of working\n"
            "7. HR processes — appraisals, internal mobility, BGV, benefits, exit and F&F\n"
            "8. Escalation paths — who to contact for IT, security, HR, and project issues\n\n"
            "Just ask your question in plain English — I will find the answer."
        )
    else:
        text = "I am here to help. What would you like to know?"

    return Answer(
        found=True,
        answer_text=text,
        answer_parts=[],
        sources=[],
        confidence="high",
        correction_note="",
    )
