"""
search.py — Query interface for the inverted index.

Provides two operations:

print_word(word)
    Returns the raw inverted index entry for a single word, showing
    every page URL, its word frequency, and all positions.

find(query)
    Accepts one or more space-separated words and returns the set of
    page URLs that contain *all* of the query terms (AND semantics).
    Results are ranked by TF-IDF score (descending) when page_lengths
    is available, otherwise by combined raw frequency.

Ranking algorithm — TF-IDF
---------------------------
TF  (term frequency)  = occurrences(t, d) / total_tokens(d)
IDF (inverse doc freq) = log( N / df(t) )  where N = corpus size,
                         df(t) = number of documents containing t.

score(d, query) = Σ TF(t, d) × IDF(t)  for each query term t

This down-weights terms that appear on almost every page (e.g. "the")
and up-weights terms that are rare but prominent in a document,
producing more relevant rankings than raw frequency alone.

Complexity
----------
find(query with k terms):
  - Intersection: O(k × |posting_list|)  — we intersect sets, each
    set lookup is O(1) average.
  - Scoring: O(|results| × k)  — one multiply-add per (doc, term) pair.
  - Sorting: O(|results| × log|results|)
  Overall: O(k × D + R log R) where D = avg posting list size,
           R = number of matching results.
"""

from __future__ import annotations

import math
import string

from src.indexer import Index

_STRIP_CHARS = string.punctuation + "\u2018\u2019\u201c\u201d"


class Search:
    """Query engine over an inverted index built by Indexer."""

    def __init__(
        self,
        index: Index,
        page_lengths: dict[str, int] | None = None,
    ) -> None:
        self.index = index
        # Optional companion dict mapping URL → total token count.
        # When provided, find() uses TF-IDF ranking; otherwise it
        # falls back to summed raw frequency.
        self.page_lengths: dict[str, int] = page_lengths or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def print_word(self, word: str) -> dict | None:
        """Return the index entry for *word*, or None if not found.

        The returned dict maps each URL to its frequency/positions stats:
            {
                "https://…/page": {"frequency": 3, "positions": [1, 5, 9]},
                …
            }
        """
        normalised = self._normalise(word)
        return self.index.get(normalised)

    def find(self, query: str) -> list[str]:
        """Return URLs that contain every word in *query* (AND semantics).

        Results are sorted by the sum of per-word frequencies on each
        matching page, descending — so pages that use the query terms
        most often come first.

        Args:
            query: A whitespace-separated string of one or more words.

        Returns:
            A (possibly empty) list of URL strings.
        """
        words = self._parse_query(query)
        if not words:
            return []

        # Start with the set of pages that contain the first word, then
        # intersect with pages for every subsequent word.
        matching_urls: set[str] | None = None
        for word in words:
            pages = set(self.index.get(word, {}).keys())
            if matching_urls is None:
                matching_urls = pages
            else:
                matching_urls &= pages

        if not matching_urls:
            return []

        # Rank by TF-IDF when page lengths are available, else by raw frequency.
        if self.page_lengths:
            key_fn = lambda url: self._tfidf_score(words, url)  # noqa: E731
        else:
            key_fn = lambda url: sum(self.index[w][url]["frequency"] for w in words)  # noqa: E731

        return sorted(matching_urls, key=key_fn, reverse=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tfidf_score(self, words: list[str], url: str) -> float:
        """Return the TF-IDF score for *url* across all *words*.

        TF  = occurrences(t, d) / total_tokens(d)
        IDF = log( N / df(t) )   (natural log; 0 when df == 0)

        The score is the sum of TF-IDF contributions from each word.
        A score of 0.0 is returned for an unknown URL.
        """
        # Total number of documents in the corpus.
        N = len(self.page_lengths) or 1
        doc_len = self.page_lengths.get(url) or 1
        score = 0.0
        for word in words:
            if word not in self.index or url not in self.index[word]:
                continue
            tf = self.index[word][url]["frequency"] / doc_len
            df = len(self.index[word])
            idf = math.log(N / df) if df > 0 else 0.0
            score += tf * idf
        return score

    def _normalise(self, word: str) -> str:
        """Lowercase and strip punctuation from *word*, matching index keys."""
        return word.lower().strip(_STRIP_CHARS)

    def _parse_query(self, query: str) -> list[str]:
        """Split the query string into a list of normalised, non-empty tokens."""
        return [self._normalise(t) for t in query.split() if self._normalise(t)]
