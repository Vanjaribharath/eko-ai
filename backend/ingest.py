"""
ingest.py
=========
This is the script that actually RUNS Stages 1-3 of the pipeline
(extract -> chunk -> index) over every document in the sample-docs
folder, and builds the in-memory search index that main.py serves
from.

You can run this file directly to test ingestion on its own:
    python ingest.py

Or import build_index() from main.py, which is what actually happens
when the server starts up.

WHAT HAPPENS WHEN A NEW DOCUMENT IS ADDED LATER:
This script currently re-reads the entire sample-docs folder from
scratch every time it runs (this is sometimes called a "full
reindex"). For a document set this small (dozens of files), a full
reindex takes well under a second, so this is the right level of
complexity for a prototype — there's no need for incremental
update logic yet. The companion HTML guide explains how this would
evolve into an incremental "just add the new file" pipeline once the
document set grows much larger.
"""

import os
import time
from extract import extract_text
from chunk import chunk_document
from retrieval import TfidfIndex

SAMPLE_DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample-docs")
SAMPLE_IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample-images")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt", ".png", ".jpg", ".jpeg"}


def discover_files(directory: str) -> list:
    """Lists every supported file in a directory. Skips hidden files
    (anything starting with a dot) and anything with an unsupported
    extension, so a stray .DS_Store or .gitkeep file doesn't break
    ingestion."""
    if not os.path.isdir(directory):
        return []
    files = []
    for name in sorted(os.listdir(directory)):
        if name.startswith("."):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            files.append(os.path.join(directory, name))
    return files


def build_index(verbose: bool = True) -> tuple:
    """
    Runs the full pipeline and returns a ready-to-query TfidfIndex.

    Returns:
        (index, stats) where stats is a small dict describing what
        was ingested — used by the /health endpoint and the frontend's
        "system status" display.
    """
    start_time = time.time()
    all_files = discover_files(SAMPLE_DOCS_DIR) + discover_files(SAMPLE_IMAGES_DIR)

    all_chunks = []
    file_count = {"pdf": 0, "docx": 0, "md": 0, "image": 0}
    failed_files = []

    for filepath in all_files:
        ext = os.path.splitext(filepath)[1].lower()
        try:
            raw_text = extract_text(filepath)

            if ext in (".png", ".jpg", ".jpeg"):
                # Images don't have the same "## Heading" structure as our
                # policy docs, so we wrap the description directly as one chunk
                # rather than running it through the full section-splitter.
                from chunk import Chunk
                filename = os.path.basename(filepath)
                chunks = [Chunk(
                    text=raw_text,
                    doc_id=f"IMG-{filename}",
                    doc_title=filename,
                    section="Image",
                    source_file=filename,
                    chunk_index=0,
                    status="approved",
                )]
                file_count["image"] += 1
            else:
                chunks = chunk_document(raw_text, os.path.basename(filepath))
                if ext == ".pdf":
                    file_count["pdf"] += 1
                elif ext == ".docx":
                    file_count["docx"] += 1
                else:
                    file_count["md"] += 1

            all_chunks.extend(chunks)
            if verbose:
                print(f"  Ingested {os.path.basename(filepath)} -> {len(chunks)} chunks")

        except Exception as e:
            failed_files.append((os.path.basename(filepath), str(e)))
            if verbose:
                print(f"  FAILED on {os.path.basename(filepath)}: {e}")

    index = TfidfIndex()
    index.build(all_chunks)

    elapsed_ms = (time.time() - start_time) * 1000

    stats = {
        "total_documents": len(all_files),
        "total_chunks": len(all_chunks),
        "files_by_type": file_count,
        "failed_files": failed_files,
        "vocabulary_size": len(index.vocabulary),
        "index_build_time_ms": round(elapsed_ms, 2),
    }

    if verbose:
        print(f"\nIndex built: {stats['total_chunks']} chunks from {stats['total_documents']} documents")
        print(f"Vocabulary size: {stats['vocabulary_size']} unique words")
        print(f"Build time: {stats['index_build_time_ms']} ms")

    return index, stats


if __name__ == "__main__":
    # Lets you run "python ingest.py" directly to sanity-check the
    # pipeline without starting the full web server.
    index, stats = build_index(verbose=True)
    print("\n--- Quick test search ---")
    test_query = "can I take a photo of client code"
    results = index.search(test_query, top_k=3)
    for chunk, score in results:
        print(f"\n[{score:.3f}] {chunk.doc_title} — {chunk.section}")
        print(chunk.text[:200] + "...")
