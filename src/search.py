"""
search.py — Query interface for the inverted index.

Public API
----------
print_word(word)
    Returns the raw inverted index entry for a single word, showing
    every page URL, its word frequency, and all positions.

find(query)
    AND search: pages containing *all* words in *query*, ranked by TF-IDF.

find_phrase(phrase)
    Exact-phrase search: pages where all words appear consecutively in
    the given order, exploiting the position lists stored in the index.

find_wildcard(query)
    Wildcard search: tokens may contain '*' (matches any character
    sequence).  Each wildcard token is expanded against the vocabulary
    and the resulting page sets are AND-intersected across tokens.

suggest(word)
    Spelling suggestions from the index vocabulary using fuzzy matching.

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
find(k terms):          O(k·D + R log R)  D = avg posting list, R = results
find_phrase(k terms):   O(k·D + C·P)      C = AND-candidates, P = positions of word[0]
find_wildcard(k terms): O(k·W + R log R)  W = vocabulary size
suggest(word):          O(W·L)            L = average word length
"""

from __future__ import annotations

import difflib
import fnmatch
import math
import string

from src.indexer import Index

_STRIP_CHARS = string.punctuation + "‘’“”"
# Identical to _STRIP_CHARS but preserves '*' so wildcard tokens survive cleaning.
_WILDCARD_STRIP_CHARS = _STRIP_CHARS.replace("*", "")


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

        Results are ranked by TF-IDF when page_lengths is available,
        otherwise by combined raw frequency (descending).

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

    def find_phrase(self, phrase: str) -> list[str]:
        """Return URLs where all words in *phrase* appear as a consecutive sequence.

        Uses the position lists already stored in the index — no re-fetching
        needed.  For a k-word phrase, we check that word[i] appears at
        position (start + i) for each i, where *start* iterates over the
        positions of word[0] on each candidate page.

        Single-word phrases delegate to find() since ordering is irrelevant.

        Args:
            phrase: A whitespace-separated string of words forming the phrase.

        Returns:
            A (possibly empty) list of URL strings ranked by TF-IDF / frequency.
        """
        words = self._parse_query(phrase)
        if not words:
            return []
        if len(words) == 1:
            return self.find(phrase)

        # AND-intersect first to prune the candidate set cheaply.
        candidates: set[str] | None = None
        for word in words:
            pages = set(self.index.get(word, {}).keys())
            candidates = pages if candidates is None else candidates & pages

        if not candidates:
            return []

        results: list[str] = []
        for url in candidates:
            positions_first = self.index[words[0]][url]["positions"]
            for start in positions_first:
                if all(
                    (start + i) in set(self.index[words[i]][url]["positions"])
                    for i in range(1, len(words))
                ):
                    results.append(url)
                    break

        if self.page_lengths:
            return sorted(results, key=lambda url: self._tfidf_score(words, url), reverse=True)
        return sorted(results, key=lambda url: sum(self.index[w][url]["frequency"] for w in words), reverse=True)

    def find_wildcard(self, query: str) -> list[str]:
        """Return URLs matching *query* where tokens may contain '*' wildcards.

        Each token containing '*' is expanded to all vocabulary words that
        match the pattern (via fnmatch), and the union of their page sets is
        computed.  The per-token page sets are then AND-intersected so that
        every token must match at least one word on the returned page.

        Tokens without '*' behave like a standard AND-find lookup.

        Example:
            ``find_wildcard("cour* fri*")``  →  pages containing any word
            starting with "cour" AND any word starting with "fri".

        Args:
            query: A whitespace-separated string; tokens may contain '*'.

        Returns:
            A (possibly empty) sorted list of URL strings.
        """
        raw_tokens = [t.lower().strip(_WILDCARD_STRIP_CHARS) for t in query.split()]
        tokens = [t for t in raw_tokens if t]
        if not tokens:
            return []

        matching_urls: set[str] | None = None
        for token in tokens:
            if "*" in token:
                matched_words = [w for w in self.index if fnmatch.fnmatch(w, token)]
                pages: set[str] = set()
                for word in matched_words:
                    pages |= set(self.index[word].keys())
            else:
                pages = set(self.index.get(token, {}).keys())

            matching_urls = pages if matching_urls is None else matching_urls & pages

        return sorted(matching_urls or [])

    def suggest(self, word: str, n: int = 3, cutoff: float = 0.6) -> list[str]:
        """Return up to *n* close matches for *word* from the index vocabulary.

        Uses difflib.get_close_matches (SequenceMatcher under the hood) with a
        similarity *cutoff* in [0, 1].  Returns an empty list when *word* is
        already in the index or has no close match above the cutoff.

        Complexity: O(W × L) where W = vocabulary size, L = average word length.

        Args:
            word:   The raw (possibly misspelled) word to look up.
            n:      Maximum number of suggestions to return.
            cutoff: Minimum similarity ratio required (0 = everything, 1 = exact).

        Returns:
            A list of vocabulary words ordered by decreasing similarity.
        """
        normalised = self._normalise(word)
        if not normalised:
            return []
        return difflib.get_close_matches(normalised, self.index.keys(), n=n, cutoff=cutoff)

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
