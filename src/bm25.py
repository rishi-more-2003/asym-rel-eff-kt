"""Lightweight BM25 similarity for deduplication.

Dependency-free implementation using only stdlib — deliberately avoids
vector databases and embedding models per the proposal (Section 3.2).
"""

from __future__ import annotations

import math
import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_score(
    query: str,
    document: str,
    k1: float = 1.5,
    b: float = 0.75,
    avg_dl: float = 15.0,
) -> float:
    """Compute a normalised BM25-ish similarity between two short texts.

    Returns a value in [0, 1] where 1 means all query terms appear in
    the document with high frequency. This is a simplified single-document
    variant suitable for pairwise duplicate detection on short goal strings.
    """
    q_tokens = _tokenize(query)
    d_tokens = _tokenize(document)
    if not q_tokens or not d_tokens:
        return 0.0

    dl = len(d_tokens)
    d_freq = Counter(d_tokens)

    score = 0.0
    for term in set(q_tokens):
        tf = d_freq.get(term, 0)
        if tf == 0:
            continue
        idf = 1.0  # single-doc comparison, treat all terms as equally informative
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / avg_dl)
        score += idf * numerator / denominator

    max_possible = len(set(q_tokens)) * (k1 + 1)
    return min(score / max_possible, 1.0) if max_possible > 0 else 0.0


def pairwise_max_similarity(
    goals: list[str],
) -> list[tuple[int, int, float]]:
    """Return all pairs with similarity above a low threshold, sorted descending."""
    pairs = []
    for i in range(len(goals)):
        for j in range(i + 1, len(goals)):
            s = bm25_score(goals[i], goals[j])
            if s > 0.3:
                pairs.append((i, j, s))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs
