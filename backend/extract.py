"""
extract.py
==========
STAGE 1 of the pipeline: EXTRACT.

This file's only job is: given a file on disk (a PDF, a Word document,
a Markdown file, or an image), pull the readable text out of it.

Nothing in this file ever calls the internet. Everything here runs
using libraries already installed on your laptop.

Why this file exists separately from chunking/embedding:
Keeping "read the file" separate from "process the text" means you can
swap out how a PDF is read without touching anything else downstream.
This is a basic but important software design principle: each stage
of the pipeline should do exactly one job.
"""

import os
from pypdf import PdfReader
import docx  # this is the python-docx library, imported as "docx"
from PIL import Image


def extract_text_from_pdf(filepath: str) -> str:
    """
    Reads a real PDF file and returns all the text inside it as one string.

    How this works internally:
    1. PdfReader opens the PDF and finds every page.
    2. For each page, .extract_text() pulls out any text that was stored
       as actual text in the PDF (not an image of text).
    3. We join all pages together with double newlines so paragraph
       boundaries are roughly preserved.

    Note: this does NOT handle scanned/image-only PDFs (a PDF that is
    really just a photo of a document). That would require OCR
    (Optical Character Recognition), which is a separate, heavier step
    not included in this prototype — see the companion HTML guide for
    how to add it later using Tesseract OCR.
    """
    reader = PdfReader(filepath)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""  # extract_text() can return None for blank pages
        pages_text.append(text)
    return "\n\n".join(pages_text)


def extract_text_from_docx(filepath: str) -> str:
    """
    Reads a real Word (.docx) file and returns all paragraph text as one string.

    How this works internally:
    A .docx file is actually a structured XML document under the hood.
    The python-docx library parses that XML and exposes it as a list
    of Paragraph objects, each with a .text property. We just walk
    through every paragraph in order and join them.
    """
    document = docx.Document(filepath)
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def extract_text_from_markdown(filepath: str) -> str:
    """
    Reads a plain Markdown (.md) or text (.txt) file.

    Markdown is already plain text, so there's no real "extraction"
    needed — we just read the file directly. We keep this as its own
    function (rather than just calling open() everywhere) so every
    file type goes through the same kind of named, documented step.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def describe_image(filepath: str) -> str:
    """
    For image files (PNG, JPG), this prototype does NOT run real OCR
    or image captioning (that needs extra heavy libraries). Instead,
    it creates a basic, honest description using the image's metadata
    and filename, so the image can still be indexed and found by a
    relevant search query.

    In a production version, this function would be replaced with:
    1. OCR (Tesseract) to read any text inside the image (e.g. a
       diagram with labels), and/or
    2. A local, self-hosted image-captioning model to describe what
       the image visually shows.

    Both of those are "drop-in" replacements for this function — the
    rest of the pipeline doesn't need to change.
    """
    img = Image.open(filepath)
    filename = os.path.basename(filepath)
    # Turn a filename like "escalation-matrix-diagram.png" into readable words
    readable_name = filename.rsplit(".", 1)[0].replace("-", " ").replace("_", " ")
    return (
        f"Image titled '{readable_name}'. "
        f"Format: {img.format}, Size: {img.width}x{img.height} pixels. "
        f"This image is a company diagram related to: {readable_name}."
    )


# This dictionary maps a file extension to the function that knows how
# to read it. Adding support for a new file type later (e.g. .pptx)
# just means writing one new function and adding one new line here.
EXTRACTORS = {
    ".pdf": extract_text_from_pdf,
    ".docx": extract_text_from_docx,
    ".md": extract_text_from_markdown,
    ".txt": extract_text_from_markdown,
    ".png": describe_image,
    ".jpg": describe_image,
    ".jpeg": describe_image,
}


def extract_text(filepath: str) -> str:
    """
    The single entry point the rest of the pipeline calls.
    Given any supported file, this figures out its type from the file
    extension and routes it to the right extractor function above.
    """
    ext = os.path.splitext(filepath)[1].lower()
    extractor = EXTRACTORS.get(ext)
    if extractor is None:
        raise ValueError(f"Unsupported file type: {ext}. Supported types: {list(EXTRACTORS.keys())}")
    return extractor(filepath)
