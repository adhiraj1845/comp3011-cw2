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
    _cmd_stats,
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

    def test_unknown_word_reports_not_found(self, index_and_search, capsys):
        # "zzz" is not in the index and has no close matches → "not found"
        _, search = index_and_search
        _cmd_find(search, "zzz")
        out = capsys.readouterr().out
        assert "not found" in out

    def test_all_words_in_index_no_intersection_shows_no_pages(self, capsys):
        # Both words exist in the index but on separate pages → no intersection
        index = {
            "alpha": {"http://example.com/1": {"frequency": 1, "positions": [0]}},
            "beta":  {"http://example.com/2": {"frequency": 1, "positions": [0]}},
        }
        search = Search(index)
        _cmd_find(search, "alpha beta")
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

    def test_misspelled_word_shows_suggestion_in_find(self, capsys):
        # "helo" is close to "hello" — suggestion should appear
        index = {
            "hello": {"http://example.com/": {"frequency": 1, "positions": [0]}},
        }
        search = Search(index)
        _cmd_find(search, "helo")
        out = capsys.readouterr().out
        assert "did you mean" in out.lower()
        assert "hello" in out

    def test_misspelled_word_shows_suggestion_in_print(self, capsys):
        index = {
            "hello": {"http://example.com/": {"frequency": 1, "positions": [0]}},
        }
        search = Search(index)
        _cmd_print(search, "helo")
        out = capsys.readouterr().out
        assert "did you mean" in out.lower()
        assert "hello" in out


# ---------------------------------------------------------------------------
# _cmd_find — phrase and wildcard modes
# ---------------------------------------------------------------------------


class TestCmdFindPhrase:
    def test_phrase_search_finds_consecutive_words(self, capsys):
        index = {
            "good": {"http://example.com/1": {"frequency": 1, "positions": [0]}},
            "friends": {"http://example.com/1": {"frequency": 1, "positions": [1]}},
        }
        search = Search(index)
        _cmd_find(search, '"good friends"')
        out = capsys.readouterr().out
        assert "http://example.com/1" in out

    def test_phrase_search_no_result(self, capsys):
        index = {
            "good": {"http://example.com/1": {"frequency": 1, "positions": [0]}},
            "friends": {"http://example.com/1": {"frequency": 1, "positions": [5]}},
        }
        search = Search(index)
        _cmd_find(search, '"good friends"')
        out = capsys.readouterr().out
        assert "No pages found" in out

    def test_phrase_search_no_index(self, capsys):
        _cmd_find(None, '"good friends"')
        out = capsys.readouterr().out
        assert "No index loaded" in out


class TestCmdFindWildcard:
    def test_wildcard_find_returns_matches(self, capsys):
        index = {
            "courage": {"http://example.com/1": {"frequency": 1, "positions": [0]}},
            "course":  {"http://example.com/2": {"frequency": 1, "positions": [0]}},
        }
        search = Search(index)
        _cmd_find(search, "cour*")
        out = capsys.readouterr().out
        assert "http://example.com/1" in out
        assert "http://example.com/2" in out

    def test_wildcard_no_match(self, capsys):
        index = {
            "hello": {"http://example.com/1": {"frequency": 1, "positions": [0]}},
        }
        search = Search(index)
        _cmd_find(search, "zzz*")
        out = capsys.readouterr().out
        assert "No pages found" in out

    def test_wildcard_no_index(self, capsys):
        _cmd_find(None, "cour*")
        out = capsys.readouterr().out
        assert "No index loaded" in out


# ---------------------------------------------------------------------------
# _cmd_stats
# ---------------------------------------------------------------------------


class TestCmdStats:
    def test_stats_shows_page_count(self, index_and_search, capsys):
        _, search = index_and_search
        _cmd_stats(search)
        out = capsys.readouterr().out
        assert "Pages indexed" in out

    def test_stats_shows_vocab_size(self, index_and_search, capsys):
        _, search = index_and_search
        _cmd_stats(search)
        out = capsys.readouterr().out
        assert "Unique words" in out

    def test_stats_shows_top_words(self, index_and_search, capsys):
        _, search = index_and_search
        _cmd_stats(search)
        out = capsys.readouterr().out
        assert "Top 10" in out

    def test_stats_no_index_warns(self, capsys):
        _cmd_stats(None)
        out = capsys.readouterr().out
        assert "No index loaded" in out

    def test_stats_correct_page_count(self, capsys):
        index = {
            "hello": {
                "http://example.com/1": {"frequency": 1, "positions": [0]},
                "http://example.com/2": {"frequency": 1, "positions": [0]},
            }
        }
        page_lengths = {"http://example.com/1": 10, "http://example.com/2": 20}
        search = Search(index, page_lengths)
        _cmd_stats(search)
        out = capsys.readouterr().out
        assert "2" in out  # 2 pages


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

    @patch("src.main.Crawler")
    def test_build_passes_page_lengths_to_search(self, MockCrawler):
        """Search returned by _cmd_build must carry page_lengths for TF-IDF."""
        instance = MockCrawler.return_value
        instance.crawl.return_value = {
            "http://example.com/": "<html><body>hello world</body></html>"
        }
        indexer = Indexer()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_index = Path(tmpdir) / "index.json"
            with patch("src.main.INDEX_PATH", tmp_index):
                result = _cmd_build(indexer)
        assert result.page_lengths != {}

    def test_load_passes_page_lengths_to_search(self):
        """Search returned by _cmd_load must carry page_lengths for TF-IDF."""
        indexer = Indexer()
        indexer.build({"http://example.com/": "<html><body>hello</body></html>"})
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_index = Path(tmpdir) / "index.json"
            indexer.save(tmp_index)
            with patch("src.main.INDEX_PATH", tmp_index):
                result = _cmd_load(Indexer())
        assert result is not None
        assert result.page_lengths != {}


# ---------------------------------------------------------------------------
# run_shell — REPL loop
# ---------------------------------------------------------------------------


from src.main import run_shell  # noqa: E402


class TestRunShell:
    """Integration tests for the interactive REPL in run_shell()."""

    @patch("builtins.input", side_effect=["quit"])
    def test_quit_exits_with_goodbye(self, _mock, capsys):
        run_shell()
        assert "Goodbye" in capsys.readouterr().out

    @patch("builtins.input", side_effect=["exit"])
    def test_exit_exits_with_goodbye(self, _mock, capsys):
        run_shell()
        assert "Goodbye" in capsys.readouterr().out

    @patch("builtins.input", side_effect=EOFError)
    def test_eof_exits_gracefully(self, _mock, capsys):
        run_shell()
        assert "Exiting" in capsys.readouterr().out

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    def test_keyboard_interrupt_exits_gracefully(self, _mock, capsys):
        run_shell()
        assert "Exiting" in capsys.readouterr().out

    @patch("builtins.input", side_effect=["", "  ", "quit"])
    def test_empty_input_lines_are_skipped(self, _mock, capsys):
        run_shell()
        out = capsys.readouterr().out
        assert "Goodbye" in out

    @patch("builtins.input", side_effect=["help", "quit"])
    def test_help_prints_docstring(self, _mock, capsys):
        run_shell()
        out = capsys.readouterr().out
        # The module-level docstring lists all commands
        assert "build" in out

    @patch("builtins.input", side_effect=["foobar", "quit"])
    def test_unknown_command_reports_error(self, _mock, capsys):
        run_shell()
        assert "Unknown command" in capsys.readouterr().out

    @patch("src.main._cmd_build")
    @patch("builtins.input", side_effect=["build", "quit"])
    def test_build_command_calls_handler(self, _mock_input, mock_build, capsys):
        mock_build.return_value = MagicMock()
        run_shell()
        mock_build.assert_called_once()

    @patch("src.main._cmd_load")
    @patch("builtins.input", side_effect=["load", "quit"])
    def test_load_command_calls_handler(self, _mock_input, mock_load, capsys):
        mock_load.return_value = None
        run_shell()
        mock_load.assert_called_once()

    @patch("builtins.input", side_effect=["print hello", "quit"])
    def test_print_command_with_no_index_warns(self, _mock, capsys):
        run_shell()
        assert "No index loaded" in capsys.readouterr().out

    @patch("builtins.input", side_effect=["find world", "quit"])
    def test_find_command_with_no_index_warns(self, _mock, capsys):
        run_shell()
        assert "No index loaded" in capsys.readouterr().out

    @patch("src.main._cmd_build")
    @patch("builtins.input", side_effect=["build", "print hello", "quit"])
    def test_print_command_after_build(self, _mock_input, mock_build, capsys):
        """After a successful build, print dispatches to the search object."""
        fake_search = MagicMock()
        fake_search.print_word.return_value = None
        mock_build.return_value = fake_search
        run_shell()
        fake_search.print_word.assert_called_once_with("hello")

    @patch("src.main._cmd_build")
    @patch("builtins.input", side_effect=["build", "find good friends", "quit"])
    def test_find_command_after_build(self, _mock_input, mock_build, capsys):
        """After a successful build, find dispatches to the search object."""
        fake_search = MagicMock()
        fake_search.find.return_value = []
        mock_build.return_value = fake_search
        run_shell()
        fake_search.find.assert_called_once_with("good friends")

    @patch("builtins.input", side_effect=["stats", "quit"])
    def test_stats_command_with_no_index_warns(self, _mock, capsys):
        run_shell()
        assert "No index loaded" in capsys.readouterr().out
