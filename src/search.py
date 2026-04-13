"""
search.py — Query interface for the inverted index.

Provides two operations:

print_word(word)
    Returns the raw inverted index entry for a single word, showing
    every page URL, its word frequency, and all positions.

find(query)
    Accepts one or more space-separated words and returns the set of
    page URLs that contain *all* of the query terms (AND semantics).
    Results are sorted by combined frequency (descending) so the most
    relevant pages appear first.
"""

from __future__ import annotations

import string

from src.indexer import Index

_STRIP_CHARS = string.punctuation + "\u2018\u2019\u201c\u201d"


class Search:
    """Query engine over an inverted index built by Indexer."""

    def __init__(self, index: Index) -> None:
        self.index = index

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

        # Rank by combined frequency across all query words.
        def combined_freq(url: str) -> int:
            return sum(self.index[w][url]["frequency"] for w in words)

        return sorted(matching_urls, key=combined_freq, reverse=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalise(self, word: str) -> str:
        """Lowercase and strip punctuation from *word*, matching index keys."""
        return word.lower().strip(_STRIP_CHARS)

    def _parse_query(self, query: str) -> list[str]:
        """Split the query string into a list of normalised, non-empty tokens."""
        return [self._normalise(t) for t in query.split() if self._normalise(t)]
