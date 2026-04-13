"""
Tests for the Crawler module.

All network calls are mocked so the test suite runs offline and quickly.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.crawler import Crawler


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SIMPLE_HTML = """
<html>
<body>
  <a href="/page2">Page 2</a>
  <a href="/page3">Page 3</a>
</body>
</html>
"""

PAGE2_HTML = "<html><body><a href='/'>Home</a></body></html>"
PAGE3_HTML = "<html><body><p>No links here</p></body></html>"


def make_response(text: str, status_code: int = 200) -> MagicMock:
    """Return a mock requests.Response."""
    resp = MagicMock()
    resp.text = text
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# _extract_links
# ---------------------------------------------------------------------------


class TestExtractLinks:
    def test_returns_absolute_internal_links(self):
        crawler = Crawler(base_url="https://quotes.toscrape.com/", delay=0)
        links = crawler._extract_links(SIMPLE_HTML, "https://quotes.toscrape.com/")
        assert "https://quotes.toscrape.com/page2" in links
        assert "https://quotes.toscrape.com/page3" in links

    def test_ignores_external_links(self):
        html = '<html><body><a href="https://example.com/other">ext</a></body></html>'
        crawler = Crawler(base_url="https://quotes.toscrape.com/", delay=0)
        links = crawler._extract_links(html, "https://quotes.toscrape.com/")
        assert links == []

    def test_ignores_fragment_only_hrefs(self):
        html = '<html><body><a href="#section">Jump</a></body></html>'
        crawler = Crawler(base_url="https://quotes.toscrape.com/", delay=0)
        links = crawler._extract_links(html, "https://quotes.toscrape.com/")
        assert links == []

    def test_ignores_javascript_and_mailto(self):
        html = """
        <html><body>
          <a href="javascript:void(0)">JS</a>
          <a href="mailto:foo@bar.com">Email</a>
        </body></html>
        """
        crawler = Crawler(base_url="https://quotes.toscrape.com/", delay=0)
        links = crawler._extract_links(html, "https://quotes.toscrape.com/")
        assert links == []

    def test_strips_fragments_from_links(self):
        html = '<html><body><a href="/page#section">Link</a></body></html>'
        crawler = Crawler(base_url="https://quotes.toscrape.com/", delay=0)
        links = crawler._extract_links(html, "https://quotes.toscrape.com/")
        assert "https://quotes.toscrape.com/page" in links
        # Fragment version should not appear
        assert "https://quotes.toscrape.com/page#section" not in links

    def test_returns_empty_list_for_page_with_no_links(self):
        crawler = Crawler(base_url="https://quotes.toscrape.com/", delay=0)
        links = crawler._extract_links(PAGE3_HTML, "https://quotes.toscrape.com/page3")
        assert links == []


# ---------------------------------------------------------------------------
# _fetch
# ---------------------------------------------------------------------------


class TestFetch:
    @patch("src.crawler.requests.get")
    def test_returns_html_on_success(self, mock_get):
        mock_get.return_value = make_response("<html></html>")
        crawler = Crawler(delay=0)
        result = crawler._fetch("https://quotes.toscrape.com/")
        assert result == "<html></html>"

    @patch("src.crawler.requests.get")
    def test_returns_none_on_http_error(self, mock_get):
        import requests as req

        mock_get.side_effect = req.RequestException("timeout")
        crawler = Crawler(delay=0)
        result = crawler._fetch("https://quotes.toscrape.com/bad")
        assert result is None

    @patch("src.crawler.requests.get")
    def test_prints_error_on_failure(self, mock_get, capsys):
        import requests as req

        mock_get.side_effect = req.RequestException("network error")
        crawler = Crawler(delay=0)
        crawler._fetch("https://quotes.toscrape.com/fail")
        captured = capsys.readouterr()
        assert "Could not fetch" in captured.out


# ---------------------------------------------------------------------------
# crawl (integration of _fetch + _extract_links + BFS)
# ---------------------------------------------------------------------------


class TestCrawl:
    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.get")
    def test_crawls_all_reachable_pages(self, mock_get, mock_sleep):
        responses = {
            "https://quotes.toscrape.com/": make_response(SIMPLE_HTML),
            "https://quotes.toscrape.com/page2": make_response(PAGE2_HTML),
            "https://quotes.toscrape.com/page3": make_response(PAGE3_HTML),
        }
        mock_get.side_effect = lambda url, **kw: responses[url]

        crawler = Crawler(base_url="https://quotes.toscrape.com/", delay=0)
        result = crawler.crawl()

        assert set(result.keys()) == {
            "https://quotes.toscrape.com/",
            "https://quotes.toscrape.com/page2",
            "https://quotes.toscrape.com/page3",
        }

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.get")
    def test_does_not_visit_same_url_twice(self, mock_get, mock_sleep):
        # PAGE2 links back to home, home links to page2 — no infinite loop
        responses = {
            "https://quotes.toscrape.com/": make_response(
                '<html><body><a href="/page2">P2</a></body></html>'
            ),
            "https://quotes.toscrape.com/page2": make_response(
                '<html><body><a href="/">Home</a></body></html>'
            ),
        }
        mock_get.side_effect = lambda url, **kw: responses[url]

        crawler = Crawler(base_url="https://quotes.toscrape.com/", delay=0)
        result = crawler.crawl()

        assert len(result) == 2

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.get")
    def test_skips_failed_pages_and_continues(self, mock_get, mock_sleep):
        import requests as req

        call_count = [0]

        def side_effect(url, **kw):
            call_count[0] += 1
            if "page2" in url:
                raise req.RequestException("fail")
            html = '<html><body><a href="/page2">P2</a></body></html>'
            return make_response(html)

        mock_get.side_effect = side_effect

        crawler = Crawler(base_url="https://quotes.toscrape.com/", delay=0)
        result = crawler.crawl()

        # Only the base URL should be in the result; page2 failed
        assert "https://quotes.toscrape.com/" in result
        assert "https://quotes.toscrape.com/page2" not in result

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.get")
    def test_politeness_delay_called_between_requests(self, mock_get, mock_sleep):
        responses = {
            "https://quotes.toscrape.com/": make_response(
                '<html><body><a href="/p2">P2</a></body></html>'
            ),
            "https://quotes.toscrape.com/p2": make_response(PAGE3_HTML),
        }
        mock_get.side_effect = lambda url, **kw: responses[url]

        crawler = Crawler(base_url="https://quotes.toscrape.com/", delay=6)
        crawler.crawl()

        mock_sleep.assert_called_with(6)
