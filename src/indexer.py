"""
indexer.py — Builds and stores an inverted index from crawled HTML pages.

Index structure
---------------
The inverted index is a nested dict:

    {
        "word": {
            "url1": {"frequency": 3, "positions": [4, 17, 42]},
            "url2": {"frequency": 1, "positions": [8]},
        },
        ...
    }

- **word** is lowercased and stripped of leading/trailing punctuation.
- **frequency** is how many times the word appears in that page.
- **positions** lists the 0-based word positions (token index) in the
  page's visible text, useful for phrase matching or proximity queries.
"""

import json
import re
import string
from pathlib import Path

from bs4 import BeautifulSoup

# Characters stripped from both ends of a raw token before indexing.
_STRIP_CHARS = string.punctuation + "\u2018\u2019\u201c\u201d"

Index = dict[str, dict[str, dict]]


class Indexer:
    """Parses HTML pages and maintains an inverted index."""

    def __init__(self) -> None:
        self.index: Index = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, pages: dict[str, str]) -> None:
        """Populate the index from a URL→HTML mapping produced by the Crawler.

        Calling build() replaces any previously held index data.
        """
        self.index = {}
        for url, html in pages.items():
            self._index_page(url, html)

    def save(self, path: str | Path) -> None:
        """Serialise the index to a JSON file at *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.index, fh, ensure_ascii=False, indent=2)

    def load(self, path: str | Path) -> None:
        """Load a previously saved index from a JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            self.index = json.load(fh)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _index_page(self, url: str, html: str) -> None:
        """Extract visible text from *html* and update the index for *url*."""
        tokens = self._tokenise(html)
        for position, word in enumerate(tokens):
            entry = self.index.setdefault(word, {})
            if url not in entry:
                entry[url] = {"frequency": 0, "positions": []}
            entry[url]["frequency"] += 1
            entry[url]["positions"].append(position)

    def _tokenise(self, html: str) -> list[str]:
        """Return a list of normalised word tokens from the visible page text.

        Steps:
        1. Parse HTML and extract visible text (strips tags, scripts, styles).
        2. Lowercase the text.
        3. Split on whitespace.
        4. Strip punctuation from both ends of each token.
        5. Discard empty tokens and tokens that contain no letter.
        """
        soup = BeautifulSoup(html, "lxml")

        # Remove script and style blocks — their content is not user-facing
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        text = text.lower()

        raw_tokens = text.split()
        tokens: list[str] = []
        for tok in raw_tokens:
            tok = tok.strip(_STRIP_CHARS)
            # Keep only tokens that have at least one letter
            if tok and re.search(r"[a-z]", tok):
                tokens.append(tok)

        return tokens
