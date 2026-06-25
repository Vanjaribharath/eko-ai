"""
chunk.py
========
STAGE 2 of the pipeline: CHUNK.

A full document (sometimes thousands of words) is too big and too
unfocused to search well or to hand to an answer-builder directly.
This file cuts a long piece of text into small, overlapping pieces
("chunks") of a few hundred words each.

Why overlap matters:
Imagine a document is cut exactly at the sentence "Employees must
report incidents within 24 hours of discovery, except for..." — if
the next sentence (which contains the actual exception) lands in the
NEXT chunk with no overlap, a search for "incident reporting
exception" might only find one half of the relevant text. A small
overlap between consecutive chunks makes this much less likely.

Why chunk by paragraph/heading first, not just raw word count:
Our sample documents are structured with Markdown-style numbered
sections ("## 1. Section Title"). Splitting at these natural
boundaries first, then sub-splitting only if a section is too long,
keeps each chunk focused on one coherent idea — exactly what makes
retrieval accurate later.
"""

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    """
    A single searchable piece of a document.

    Fields:
      text         - the clean chunk text, shown to the user in answers
      search_text  - a version of the text used ONLY for search scoring,
                     with the document title and section heading repeated
                     a few extra times. This is a standard, well-understood
                     information-retrieval technique called "field boosting":
                     words that appear in a title are usually a much
                     stronger signal of what a passage is about than words
                     buried in the body, so we give them more weight in
                     the TF-IDF math without changing what's displayed.
      doc_id       - which source document this came from (e.g. "POL-COC-001")
      doc_title    - the human-readable title of that document
      section      - which section/heading this chunk came from, if known
      source_file  - the original filename, for citation display
      chunk_index  - this chunk's position within the document (0, 1, 2, ...)
      status       - "approved", "draft", or "retired" — controls whether this
                     chunk is ever allowed to be returned in a search result.
                     This is the technical enforcement of "approved-only retrieval"
                     described in the EKO-AI security design.
    """
    text: str
    doc_id: str
    doc_title: str
    section: str
    source_file: str
    chunk_index: int
    status: str = "approved"
    metadata: dict = field(default_factory=dict)
    search_text: str = ""

    def __post_init__(self):
        if not self.search_text:
            # Repeat the title and section 3 times each — enough to
            # meaningfully shift TF-IDF scoring toward title-relevant
            # chunks, without so much repetition that it drowns out the
            # actual body content for longer chunks.
            boost = f"{self.doc_title} {self.doc_title} {self.doc_title} {self.section} {self.section} {self.section} "
            self.search_text = boost + self.text


def split_into_sections(text: str):
    """
    Splits a document's raw text into (heading, body) pairs.

    Real documents come in two shapes once extracted:
    1. Markdown source files still have "## Heading" markers intact.
    2. PDF-extracted text has NO markdown syntax at all (PDF doesn't
       store "##" — that was only ever a Markdown-file convention).
       Instead, what survives is a line like "1. Device and Access
       Security" sitting on its own line, immediately followed by
       paragraph text. This function checks for Markdown headings
       first, and falls back to detecting these short, numbered,
       title-case lines if no "##" markers are found — which is
       exactly the shape real extracted PDF text takes.

    If neither pattern is found, the whole document is treated as one
    section, which still works fine for short documents like FAQs.
    """
    # First, try real Markdown "## Heading" syntax.
    md_pattern = re.compile(r"^##\s+(.*)$", re.MULTILINE)
    matches = list(md_pattern.finditer(text))
    if matches:
        sections = []
        for i, match in enumerate(matches):
            heading = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                sections.append((heading, body))
        return sections

    # Fallback: detect short lines that look like "1. Some Heading Text"
    # sitting alone on their own line (a strong PDF-extracted heading
    # signal) — these are short (under ~80 chars), start with a number
    # and a period, and are immediately followed by a newline.
    # Require at least two words after the number — this filters out
    # PDF-extraction artifacts where a numbered list marker ("1.", "2.")
    # gets separated from its own text onto its own line (a known quirk
    # of how some PDF renderers lay out <ol> list markup). A real section
    # heading like "2. Confidentiality of Client Information" always has
    # multiple words; a stray list marker on its own line does not.
    heading_pattern = re.compile(r"^(\d+\.[ \t]+[A-Z][A-Za-z]+(?:[ \t]+[A-Za-z][A-Za-z\-]*){1,12})[ \t]*$", re.MULTILINE)
    matches = list(heading_pattern.finditer(text))
    if matches:
        sections = []
        # Capture any text before the very first detected heading (this
        # is usually the document title/metadata block, but for some
        # PDF-extracted documents a numbered heading can fail to match
        # due to line-wrap quirks — keeping this text rather than
        # silently dropping it is the safer default).
        preamble = text[:matches[0].start()].strip()
        if preamble:
            sections.append(("Introduction", preamble))
        for i, match in enumerate(matches):
            heading = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                sections.append((heading, body))
        return sections

    return [("General", text.strip())]


def split_long_text(text: str, max_words: int = 220, overlap_words: int = 40) -> List[str]:
    """
    Splits a single block of text into overlapping word-count-based pieces.

    This only kicks in if a section is longer than `max_words` — most
    of our sample policy sections are already a good chunk size on
    their own, so this mostly matters for long sections like the FAQ
    document.

    How the overlap works, step by step:
    1. Split the text into a flat list of words.
    2. Take the first `max_words` words as chunk 1.
    3. Move forward by (max_words - overlap_words) words, so the next
       chunk starts a bit before the previous one ended — that's the
       overlap.
    4. Repeat until we've covered the whole text.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    pieces = []
    step = max_words - overlap_words
    for start in range(0, len(words), step):
        piece_words = words[start:start + max_words]
        pieces.append(" ".join(piece_words))
        if start + max_words >= len(words):
            break
    return pieces


def extract_doc_metadata(raw_text: str) -> dict:
    """
    Our sample documents all start with a small metadata block like:

        Document ID: POL-COC-001
        Owner: Corporate Compliance
        Status: Approved
        Version: 3.2
        Last Updated: 2026-02-10

    This function pulls those fields out with a simple line-by-line
    scan. In a real system, this metadata would more likely come from
    a proper document management system's database rather than being
    parsed from the text itself — this is a reasonable simplification
    for a prototype.
    """
    meta = {
        "doc_id": "UNKNOWN",
        "owner": "Unknown",
        "status": "approved",  # default to approved for our trusted sample set
        "version": "1.0",
        "last_updated": "Unknown",
    }
    field_map = {
        "Document ID:": "doc_id",
        "Owner:": "owner",
        "Status:": "status",
        "Version:": "version",
        "Last Updated:": "last_updated",
    }
    for line in raw_text.split("\n")[:10]:  # metadata is always near the top
        for prefix, key in field_map.items():
            if line.strip().startswith(prefix):
                meta[key] = line.split(":", 1)[1].strip()
    meta["status"] = meta["status"].lower()
    return meta


def chunk_document(raw_text: str, source_file: str) -> List[Chunk]:
    """
    The main entry point for this file. Takes the raw extracted text
    of one document and returns a list of Chunk objects ready to be
    embedded and stored.

    Steps:
    1. Pull the document title from the first line (assumes a Markdown
       "# Title" on line 1, which all our sample docs have).
    2. Pull metadata (doc_id, status, etc.) from the header block.
    3. Split into sections by "##" headings.
    4. For any section that's still too long, split it further with overlap.
    5. Wrap every resulting piece in a Chunk object with full metadata attached.
    """
    lines = raw_text.strip().split("\n")
    title_line = lines[0] if lines else "Untitled Document"
    doc_title = title_line.lstrip("#").strip()

    meta = extract_doc_metadata(raw_text)
    sections = split_into_sections(raw_text)

    chunks = []
    idx = 0
    for heading, body in sections:
        pieces = split_long_text(body)
        for piece in pieces:
            # `text` is kept as pure body content (no title/section prefix) so
            # it displays cleanly in answers. Search boosting for the title
            # and section name happens separately via `search_text`, built
            # automatically in the Chunk class itself.
            chunks.append(Chunk(
                text=piece,
                doc_id=meta["doc_id"],
                doc_title=doc_title,
                section=heading,
                source_file=source_file,
                chunk_index=idx,
                status=meta["status"],
                metadata=meta,
            ))
            idx += 1
    return chunks
