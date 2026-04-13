"""
Tests for the main.py shell command handlers.

The actual shell loop (run_shell) is not tested here — instead, each
internal command helper is tested directly with a pre-built index.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.indexer import Indexer
from src.main import (
    INDEX_PATH,
    _cmd_build,
    _cmd_find,
    _cmd_load,
    _cmd_print,
)
from src.search import Search


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def index_and_search():
    """Return an Indexer already loaded with a tiny index and its Search."""
    indexer = Indexer()
    indexer.index = {
        "hello": {
            "http://example.com/": {"frequency": 2, "positions": [0, 3]},
        },
        "world": {
            "http://example.com/": {"frequency": 1, "positions": [1]},
            "http://example.com/page2": {"frequency": 1, "positions": [0]},
        },
    }
    return indexer, Search(indexer.index)


# ---------------------------------------------------------------------------
# _cmd_print
# ---------------------------------------------------------------------------


class TestCmdPrint:
    def test_prints_entry_for_known_word(self, index_and_search, capsys):
        _, search = index_and_search
        _cmd_print(search, "hello")
        out = capsys.readouterr().out
        assert "http://example.com/" in out
        assert "frequency" in out

    def test_case_insensitive(self, index_and_search, capsys):
        _, search = index_and_search
        _cmd_print(search, "HELLO")
        out = capsys.readouterr().out
        assert "http://example.com/" in out

    def test_unknown_word_reports_not_found(self, index_and_search, capsys):
        _, search = index_and_search
        _cmd_print(search, "nonsense")
        out = capsys.readouterr().out
        assert "not found" in out

    def test_no_index_loaded_warns_user(self, capsys):
        _cmd_print(None, "hello")
        out = capsys.readouterr().out
        assert "No index loaded" in out

    def test_missing_word_arg_shows_usage(self, index_and_search, capsys):
        _, search = index_and_search
        _cmd_print(search, "")
        out = capsys.readouterr().out
        assert "Usage" in out


# ---------------------------------------------------------------------------
# _cmd_find
# ---------------------------------------------------------------------------


class TestCmdFind:
    def test_finds_pages_for_known_word(self, index_and_search, capsys):
        _, search = index_and_search
        _cmd_find(search, "hello")
        out = capsys.readouterr().out
        assert "http://example.com/" in out

    def test_multi_word_find(self, index_and_search, capsys):
        _, search = index_and_search
        _cmd_find(search, "hello world")
        out = capsys.readouterr().out
        # Only page that has both "hello" and "world"
        assert "http://example.com/" in out

    def test_unknown_query_reports_no_pages(self, index_and_search, capsys):
        _, search = index_and_search
        _cmd_find(search, "zzz")
        out = capsys.readouterr().out
        assert "No pages found" in out

    def test_empty_query_shows_usage(self, index_and_search, capsys):
        _, search = index_and_search
        _cmd_find(search, "")
        out = capsys.readouterr().out
        assert "Usage" in out

    def test_no_index_loaded_warns_user(self, capsys):
        _cmd_find(None, "hello")
        out = capsys.readouterr().out
        assert "No index loaded" in out


# ---------------------------------------------------------------------------
# _cmd_load
# ---------------------------------------------------------------------------


class TestCmdLoad:
    def test_loads_index_from_file(self, capsys):
        sample = {
            "test": {
                "http://example.com/": {"frequency": 1, "positions": [0]}
            }
        }
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            json.dump(sample, tmp)
            tmp_path = tmp.name

        try:
            with patch("src.main.INDEX_PATH", Path(tmp_path)):
                indexer = Indexer()
                result = _cmd_load(indexer)
            assert result is not None
            assert "test" in indexer.index
        finally:
            os.unlink(tmp_path)

    def test_returns_none_when_file_missing(self, capsys):
        with patch("src.main.INDEX_PATH", Path("/nonexistent/path/index.json")):
            indexer = Indexer()
            result = _cmd_load(indexer)
        assert result is None
        assert "Run 'build' first" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _cmd_build
# ---------------------------------------------------------------------------


class TestCmdBuild:
    @patch("src.main.Crawler")
    def test_build_returns_search_object(self, MockCrawler, capsys):
        instance = MockCrawler.return_value
        instance.crawl.return_value = {
            "http://example.com/": "<html><body>hello world</body></html>"
        }
        indexer = Indexer()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_index = Path(tmpdir) / "index.json"
            with patch("src.main.INDEX_PATH", tmp_index):
                result = _cmd_build(indexer)

        assert isinstance(result, Search)
        assert "hello" in indexer.index

    @patch("src.main.Crawler")
    def test_build_saves_index_file(self, MockCrawler, capsys):
        instance = MockCrawler.return_value
        instance.crawl.return_value = {
            "http://example.com/": "<html><body>saved</body></html>"
        }
        indexer = Indexer()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_index = Path(tmpdir) / "index.json"
            with patch("src.main.INDEX_PATH", tmp_index):
                _cmd_build(indexer)
            assert tmp_index.exists()
