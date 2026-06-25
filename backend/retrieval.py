"""
retrieval.py
============
STAGE 3 of the pipeline: EMBED + SEARCH.

This is the heart of the prototype, and the most important design
decision in the whole system, so this comment block explains the
"why" in full before the "how".

WHY THIS FILE DOES NOT CALL ANY EXTERNAL API OR DOWNLOAD ANY MODEL:
You asked for zero API calls, maximum speed, and a system that
mirrors how a security-conscious internal team (one that already
avoids API keys for exactly this reason) would actually build this.
A full neural embedding model (the kind used in big production RAG
systems) is more powerful at matching meaning across very different
wordings, but it either needs a model download from the internet
or a GPU to run fast — neither of which fits your constraints today.

So this file implements TF-IDF (Term Frequency – Inverse Document
Frequency) search instead, with cosine similarity for ranking. This
is a real, well-established, decades-old information retrieval
technique — it's what early search engines were built on, and it is
still what numerous enterprise search tools quietly use under the
hood, especially for exactly this scenario: small-to-medium private
document sets, on-premises, with hard speed and security
requirements. It needs nothing but Python's standard library plus
numpy, runs in milliseconds, and has zero external network surface
area at all — there is no possible way for this code to leak data
externally, because it has no code path that makes a network call.

WHAT TF-IDF ACTUALLY MEANS, IN PLAIN ENGLISH:
- "Term Frequency" = how often a word appears in a given chunk.
  A chunk that says "leave" five times is probably about leave policy.
- "Inverse Document Frequency" = how RARE a word is across the whole
  document set. Common words like "the", "policy", "employee" appear
  everywhere and tell you very little. Rare words like "gratuity" or
  "tailgating" are much more useful signals of what a chunk is about.
- TF-IDF combines both: a word counts more if it's frequent in THIS
  chunk but rare across all chunks. This is exactly the intuition a
  human uses when skimming for relevant text.

HOW SEARCH WORKS, STEP BY STEP:
1. At startup, every chunk's text gets converted into a TF-IDF vector
   (a long list of numbers, one per unique word in the whole corpus).
2. When a question comes in, it gets converted into a TF-IDF vector
   using the exact same vocabulary.
3. Cosine similarity measures the "angle" between the question's
   vector and every chunk's vector — chunks pointing in a similar
   direction (using similar important words) score higher.
4. The top-scoring chunks are returned.

This is a genuinely fast, genuinely private approach — and it is
also completely upgradeable later: see the bottom of this file for
exactly where a neural embedding model would slot in if you later
decide GPU/local-LLM resources are available.
"""

import re
import math
import numpy as np
from collections import Counter
from typing import List, Tuple
from chunk import Chunk

# A small set of extremely common English words that carry almost no
# topical meaning ("the", "and", "is"...). Removing these before
# scoring keeps the important, rare words from being drowned out.
STOPWORDS = set("""
a an the of to in on for and or is are was were be been being
this that these those it its as at by from with without within
your you yours we our ours they their them he she his her i
not no can may will would should could must shall do does did
how what when where who whom which why if than then so such
""".split())


def tokenize(text: str) -> List[str]:
    """
    Turns raw text into a clean list of lowercase words, with simple
    punctuation stripped and stopwords removed.

    Example: "Employees must report incidents within 24 hours!"
          -> ["employees", "must", "report", "incidents", "within", "24", "hours"]
          -> (after stopword removal) ["employees", "report", "incidents", "24", "hours"]
    """
    text = text.lower()
    words = re.findall(r"[a-z0-9]+", text)  # keeps letters and numbers, drops punctuation
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


class TfidfIndex:
    """
    Holds the full searchable index: every chunk, its metadata, and
    the math needed to score a new question against all of them.
    """

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.vocabulary: dict = {}          # maps each unique word -> a column index
        self.idf: np.ndarray = None          # one IDF weight per word in the vocabulary
        self.chunk_vectors: np.ndarray = None  # one TF-IDF vector per chunk (rows = chunks)

    def build(self, chunks: List[Chunk]):
        """
        Builds the full index from a list of Chunks. Called once at
        startup (and again any time new documents are ingested — see
        the "Pushing New Data" section of the companion HTML guide).
        """
        self.chunks = chunks
        tokenized_chunks = [tokenize(c.search_text) for c in chunks]

        # Step 1: build the vocabulary — every unique word across all chunks,
        # each assigned a fixed column position in our vectors.
        vocab_set = set()
        for tokens in tokenized_chunks:
            vocab_set.update(tokens)
        self.vocabulary = {word: i for i, word in enumerate(sorted(vocab_set))}
        vocab_size = len(self.vocabulary)

        # Step 2: compute IDF (Inverse Document Frequency) for every word.
        # df = "document frequency" = in how many chunks does this word appear at all?
        doc_freq = np.zeros(vocab_size)
        for tokens in tokenized_chunks:
            seen = set(tokens)
            for word in seen:
                doc_freq[self.vocabulary[word]] += 1

        n_chunks = len(chunks)
        # The "+1" in numerator and denominator (Laplace smoothing) avoids
        # division-by-zero and avoids a word that's in every chunk getting
        # a weight of exactly zero.
        self.idf = np.log((n_chunks + 1) / (doc_freq + 1)) + 1

        # Step 3: build a TF-IDF vector for every chunk.
        self.chunk_vectors = np.zeros((n_chunks, vocab_size))
        for row, tokens in enumerate(tokenized_chunks):
            counts = Counter(tokens)
            total_words = max(len(tokens), 1)
            for word, count in counts.items():
                col = self.vocabulary[word]
                term_frequency = count / total_words
                self.chunk_vectors[row, col] = term_frequency * self.idf[col]

        # Step 4: normalize every chunk vector to unit length. This makes
        # cosine similarity reduce to a simple dot product later, which
        # is both simpler and faster to compute.
        norms = np.linalg.norm(self.chunk_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid divide-by-zero for any empty chunk
        self.chunk_vectors = self.chunk_vectors / norms

    def _vectorize_query(self, query: str) -> np.ndarray:
        """Converts a search query into a vector using the SAME vocabulary
        and IDF weights computed from the document set. Any word in the
        query that was never seen in the documents is simply ignored —
        there's no column for it, so it can't contribute to the score."""
        tokens = tokenize(query)
        vec = np.zeros(len(self.vocabulary))
        counts = Counter(tokens)
        total = max(len(tokens), 1)
        for word, count in counts.items():
            if word in self.vocabulary:
                col = self.vocabulary[word]
                vec[col] = (count / total) * self.idf[col]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def search(self, query: str, top_k: int = 5, only_approved: bool = True) -> List[Tuple[Chunk, float]]:
        """
        The main search function. Given a question, returns the top_k
        most relevant chunks, each paired with its similarity score
        (0.0 = totally unrelated, 1.0 = perfect match).

        `only_approved=True` enforces the single most important
        security rule in this whole system: chunks whose status is not
        "approved" are filtered out BEFORE scoring, not after. This
        means a draft or retired document can never accidentally
        surface in an answer, no matter how well it would have scored.
        """
        if self.chunk_vectors is None or len(self.chunks) == 0:
            return []

        query_vec = self._vectorize_query(query)

        # Because every vector is already normalized to unit length,
        # cosine similarity is just a dot product — very fast in numpy.
        scores = self.chunk_vectors @ query_vec  # this is a single fast matrix-vector multiply

        # Pair each chunk with its score, filter by approval status,
        # then sort by score descending and take the top_k.
        scored_chunks = []
        for i, chunk in enumerate(self.chunks):
            if only_approved and chunk.status != "approved":
                continue
            scored_chunks.append((chunk, float(scores[i])))

        scored_chunks.sort(key=lambda pair: pair[1], reverse=True)
        return scored_chunks[:top_k]


# ---------------------------------------------------------------------------
# UPGRADE PATH (read this if you later get GPU access or approve outbound
# calls to a private, self-hosted embedding model):
#
# To swap in a real neural embedding model later, you would only need to
# change two methods in this class:
#   1. build()           — instead of computing TF-IDF vectors, call your
#                           embedding model once per chunk and store the
#                           resulting vectors in self.chunk_vectors.
#   2. _vectorize_query() — call the same embedding model on the incoming
#                           query string instead of doing TF-IDF math.
# Everything else (search(), the approval filter, the ranking logic)
# stays exactly the same, because it only depends on having SOME vector
# per chunk and SOME vector per query — it doesn't care how they were made.
# This is exactly why separating "how we vectorize" from "how we search"
# matters: it makes future upgrades a small, contained change.
# ---------------------------------------------------------------------------
