"""
crawler.py — Fetches pages from quotes.toscrape.com.

Discovers all internal links starting from the base URL and returns
the raw HTML for each page visited. A politeness window of at least
6 seconds is enforced between successive HTTP requests.
"""

import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://quotes.toscrape.com/"
POLITENESS_DELAY = 6  # seconds between requests


class Crawler:
    """BFS web crawler restricted to the target domain."""

    def __init__(self, base_url: str = BASE_URL, delay: float = POLITENESS_DELAY):
        self.base_url = base_url
        self.delay = delay
        self._domain = urlparse(base_url).netloc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def crawl(self) -> dict[str, str]:
        """Crawl all reachable pages and return a mapping of URL → HTML.

        Returns:
            A dict where each key is a page URL and the value is the
            raw HTML string for that page.
        """
        visited: dict[str, str] = {}
        queue: list[str] = [self.base_url]
        seen: set[str] = {self.base_url}

        while queue:
            url = queue.pop(0)
            html = self._fetch(url)

            if html is None:
                continue

            visited[url] = html

            for link in self._extract_links(html, url):
                if link not in seen:
                    seen.add(link)
                    queue.append(link)

            # Politeness: wait before the next request (skip delay after
            # the very last page to avoid unnecessary blocking).
            if queue:
                time.sleep(self.delay)

        return visited

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch(self, url: str) -> str | None:
        """Perform an HTTP GET and return the response body, or None on error."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            print(f"[Crawler] Could not fetch {url}: {exc}")
            return None

    def _extract_links(self, html: str, base_url: str) -> list[str]:
        """Parse all internal <a href> links from a page.

        Only links on the same domain as the base URL are returned.
        Fragment-only hrefs (e.g. '#top') and external links are ignored.
        """
        soup = BeautifulSoup(html, "lxml")
        links: list[str] = []

        for tag in soup.find_all("a", href=True):
            href: str = tag["href"].strip()

            # Skip javascript, mailto, and fragment-only links
            if href.startswith(("javascript:", "mailto:", "#")):
                continue

            absolute = urljoin(base_url, href)
            parsed = urlparse(absolute)

            # Strip fragments and keep only http/https same-domain URLs
            if parsed.scheme not in ("http", "https"):
                continue
            if parsed.netloc != self._domain:
                continue

            clean = parsed._replace(fragment="").geturl()
            links.append(clean)

        return links
