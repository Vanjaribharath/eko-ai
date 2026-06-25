"""
understand.py
=============
Query understanding layer — sits between the user's raw input
and the retrieval engine.

Handles:
1. Intent detection — is this a greeting, a farewell, a thanks, a
   meta-question about EKO-AI itself, or an actual knowledge question?
2. Spelling correction — fixes common misspellings using the
   document vocabulary as the reference dictionary. No external library.
3. Query normalization — strips filler words, expands abbreviations
   common in this document set.

WHY THIS EXISTS:
Without this layer the user has to think about how the search engine
works. With it, the user can type naturally ("hii whats my leeve
balance") and the system understands them. This is the single biggest
difference between "feels like a real assistant" and "feels like a
search bar".

ZERO EXTERNAL CALLS: everything here uses Python stdlib only.
"""

import re
import difflib
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Intent categories
# ---------------------------------------------------------------------------
INTENT_GREETING    = "greeting"
INTENT_FAREWELL    = "farewell"
INTENT_THANKS      = "thanks"
INTENT_ABOUT_AI    = "about_ai"
INTENT_CAPABILITIES= "capabilities"
INTENT_KNOWLEDGE   = "knowledge"   # a real question to search the docs for


# Keyword patterns for each conversational intent.
# Ordered from most-specific to least-specific so the first match wins.
INTENT_PATTERNS = [
    (INTENT_GREETING,     r"\b(hi+|hello+|hey+|helo|howdy|good\s*(morning|afternoon|evening)|namaste|hii+|hiiii*)\b"),
    (INTENT_FAREWELL,     r"\b(bye+|goodbye|see\s*you|take\s*care|cya|ttyl|good\s*night)\b"),
    (INTENT_THANKS,       r"\b(thanks?|thank\s*you|thx|ty|appreciate|cheers)\b"),
    # capabilities before about_ai so "what can you do" hits capabilities, not about_ai
    (INTENT_CAPABILITIES, r"\b(what\s+can\s+you|what\s+topics|what\s+do\s+you\s+know|help\s+me\s+with|what\s+questions|your\s+capabilities|your\s+features)\b"),
    (INTENT_ABOUT_AI,     r"\b(what\s+are\s+you|who\s+are\s+you|what\s+is\s+eko|about\s+eko|eko.?ai|what\s+do\s+you\s+do)\b"),
]


@dataclass
class ParsedQuery:
    original: str           # exactly what the user typed
    corrected: str          # after spelling fix
    normalized: str         # after abbreviation expansion
    intent: str             # one of the INTENT_* constants above
    search_query: str       # what actually gets sent to retrieval
    corrections_made: list  # list of (original_word, corrected_word) pairs


# ---------------------------------------------------------------------------
# Abbreviation / shorthand expansion
# Common short forms people actually type in workplace chat
# ---------------------------------------------------------------------------
EXPANSIONS = {
    r"\bpf\b":        "provident fund",
    r"\bhrd\b":       "human resources department",
    r"\bhr\b":        "human resources",
    r"\bkra\b":       "key result area",
    r"\bkpi\b":       "key performance indicator",
    r"\bitil\b":      "ITIL service management",
    r"\bvpn\b":       "VPN virtual private network",
    r"\bsso\b":       "SSO single sign on",
    r"\bmfa\b":       "MFA multi factor authentication",
    r"\bmfr\b":       "multi factor authentication",
    r"\bwfh\b":       "work from home",
    r"\bel\b":        "earned leave",
    r"\bcl\b":        "casual leave",
    r"\bsl\b":        "sick leave",
    r"\bleeve\b":     "leave",
    r"\bleavs\b":     "leave",
    r"\bleave\b":     "leave",
    r"\bdyas\b":      "days",
    r"\bphotu\b":     "photograph photo",
    r"\bclent\b":     "client",
    r"\bwfrom\b":     "work from home",
    r"\blop\b":       "loss of pay",
    r"\bff\b":        "full and final settlement",
    r"\bf&f\b":       "full and final settlement",
    r"\bijp\b":       "internal job posting",
    r"\bkt\b":        "knowledge transfer",
    r"\bsow\b":       "statement of work",
    r"\bnda\b":       "non disclosure agreement",
    r"\bbgv\b":       "background verification",
    r"\beis\b":       "enterprise information systems",
    r"\bits\b":       "IT services",
    r"\bops\b":       "operations",
    r"\bdig\b":       "digital",
    r"\bauth\b":      "authentication",
    r"\bpm\b":        "project manager",
    r"\btl\b":        "team lead",
    r"\bhrbp\b":      "HR business partner",
    r"\brm\b":        "resource manager",
    r"\bpip\b":       "performance improvement plan",
    r"\bposh\b":      "prevention of sexual harassment",
    r"\bgdpr\b":      "data privacy GDPR",
    r"\btds\b":       "tax deducted at source",
    r"\bctc\b":       "cost to company salary",
    r"\bhra\b":       "house rent allowance",
}


def expand_abbreviations(text: str) -> str:
    """Replace known abbreviations with their full forms so the
    search engine can match them against document vocabulary."""
    result = text.lower()
    for pattern, expansion in EXPANSIONS.items():
        result = re.sub(pattern, expansion, result, flags=re.IGNORECASE)
    return result


# ---------------------------------------------------------------------------
# Spelling correction using the document vocabulary as the dictionary.
# This is populated at startup by the retrieval index.
# ---------------------------------------------------------------------------
_vocab_words: list = []   # set from outside by ingest.py after index is built


def set_vocabulary(words: list):
    """Called once after the TF-IDF index is built, so the spellchecker
    knows what words exist in the approved documents."""
    global _vocab_words
    # Keep only words of 4+ chars — short words have too many false matches
    _vocab_words = [w for w in words if len(w) >= 4]


def correct_spelling(text: str) -> tuple:
    """
    Corrects misspelled words in the query by finding the closest
    match in the document vocabulary using difflib's SequenceMatcher.

    Returns (corrected_text, list_of_corrections).

    Why difflib instead of a spelling library:
    1. No external dependency.
    2. The document vocabulary IS the right reference — we want to
       correct toward words that actually appear in approved documents,
       not toward generic English words.
    3. Fast enough for short query strings.
    """
    if not _vocab_words:
        return text, []

    words = text.split()
    corrected = []
    corrections_made = []

    for word in words:
        clean = re.sub(r"[^a-z]", "", word.lower())
        if len(clean) < 4:          # don't try to correct short words
            corrected.append(word)
            continue
        if clean in _vocab_words:   # already correct
            corrected.append(word)
            continue

        # Find closest matches — cutoff 0.78 means "pretty similar"
        matches = difflib.get_close_matches(clean, _vocab_words, n=1, cutoff=0.78)
        if matches and matches[0] != clean:
            corrections_made.append((word, matches[0]))
            corrected.append(matches[0])
        else:
            corrected.append(word)

    return " ".join(corrected), corrections_made


def detect_intent(text: str) -> str:
    """Checks the user's message against the intent patterns and returns
    the first match, defaulting to INTENT_KNOWLEDGE if nothing matches."""
    lower = text.lower().strip()
    for intent, pattern in INTENT_PATTERNS:
        if re.search(pattern, lower):
            return intent
    return INTENT_KNOWLEDGE


def parse_query(raw: str) -> ParsedQuery:
    """
    Main entry point. Takes whatever the user typed and returns a
    ParsedQuery with everything the rest of the system needs.
    """
    # 1. Detect intent on the raw text first (before corrections change words)
    intent = detect_intent(raw)

    # 2. Spelling correction
    corrected, corrections = correct_spelling(raw)

    # 3. Abbreviation expansion (only for knowledge queries — expanding
    #    "hi" → nothing useful if the intent is already a greeting)
    if intent == INTENT_KNOWLEDGE:
        normalized = expand_abbreviations(corrected)
    else:
        normalized = corrected

    return ParsedQuery(
        original=raw,
        corrected=corrected,
        normalized=normalized,
        intent=intent,
        search_query=normalized,
        corrections_made=corrections,
    )
