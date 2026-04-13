"""
Tests for the Indexer module.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.indexer import Indexer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_html(text: str) -> str:
    return f"<html><body><p>{text}</p></body></html>"


# ---------------------------------------------------------------------------
# _tokenise
# ---------------------------------------------------------------------------


class TestTokenise:
    def test_lowercases_tokens(self):
        indexer = Indexer()
        tokens = indexer._tokenise(make_html("Hello World"))
        assert "hello" in tokens
        assert "world" in tokens

    def test_strips_punctuation_from_ends(self):
        indexer = Indexer()
        tokens = indexer._tokenise(make_html("it's a test, really."))
        # "it's" should survive (apostrophe inside), trailing comma/period stripped
        assert "really" in tokens
        assert "test" in tokens

    def test_discards_pure_punctuation_tokens(self):
        indexer = Indexer()
        tokens = indexer._tokenise(make_html("hello -- world"))
        assert "--" not in tokens

    def test_strips_script_and_style_content(self):
        html = """
        <html>
        <head><style>body { color: red; }</style></head>
        <body>
          <script>var x = 1;</script>
          <p>visible text</p>
        </body>
        </html>
        """
        indexer = Indexer()
        tokens = indexer._tokenise(html)
        assert "visible" in tokens
        assert "var" not in tokens
        assert "color" not in tokens

    def test_empty_html_returns_empty_list(self):
        indexer = Indexer()
        tokens = indexer._tokenise("<html></html>")
        assert tokens == []

    def test_handles_unicode_quotes(self):
        indexer = Indexer()
        # Curly quotes should be stripped from token ends
        tokens = indexer._tokenise(make_html("\u201cHello\u201d"))
        assert "hello" in tokens


# ---------------------------------------------------------------------------
# _index_page
# ---------------------------------------------------------------------------


class TestIndexPage:
    def test_frequency_counts_correct(self):
        indexer = Indexer()
        indexer._index_page("http://example.com/", make_html("the cat sat on the mat the"))
        assert indexer.index["the"]["http://example.com/"]["frequency"] == 3

    def test_positions_recorded(self):
        indexer = Indexer()
        indexer._index_page("http://example.com/", make_html("a b a"))
        positions = indexer.index["a"]["http://example.com/"]["positions"]
        assert len(positions) == 2
        assert positions[0] < positions[1]

    def test_multiple_pages_tracked_separately(self):
        indexer = Indexer()
        indexer._index_page("http://example.com/1", make_html("hello world"))
        indexer._index_page("http://example.com/2", make_html("hello there"))
        assert "http://example.com/1" in indexer.index["hello"]
        assert "http://example.com/2" in indexer.index["hello"]
        assert "http://example.com/2" not in indexer.index["world"]


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


class TestBuild:
    def test_build_populates_index(self):
        indexer = Indexer()
        pages = {
            "http://example.com/": make_html("foo bar"),
            "http://example.com/p2": make_html("bar baz"),
        }
        indexer.build(pages)
        assert "foo" in indexer.index
        assert "bar" in indexer.index
        assert "baz" in indexer.index

    def test_build_replaces_previous_index(self):
        indexer = Indexer()
        indexer.build({"http://a.com/": make_html("old")})
        indexer.build({"http://b.com/": make_html("new content")})
        assert "old" not in indexer.index
        assert "new" in indexer.index

    def test_build_empty_pages_gives_empty_index(self):
        indexer = Indexer()
        indexer.build({})
        assert indexer.index == {}


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_creates_valid_json_file(self):
        indexer = Indexer()
        indexer.build({"http://example.com/": make_html("hello world")})

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            indexer.save(tmp_path)
            with open(tmp_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            assert "hello" in data
        finally:
            os.unlink(tmp_path)

    def test_load_restores_index(self):
        indexer = Indexer()
        indexer.build({"http://example.com/": make_html("restore me")})

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
            json.dump(indexer.index, tmp)
            tmp_path = tmp.name

        try:
            fresh = Indexer()
            fresh.load(tmp_path)
            assert "restore" in fresh.index
        finally:
            os.unlink(tmp_path)

    def test_save_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "index.json"
            indexer = Indexer()
            indexer.build({"http://example.com/": make_html("hi")})
            indexer.save(path)
            assert path.exists()

    def test_roundtrip_preserves_frequency_and_positions(self):
        indexer = Indexer()
        html = make_html("the quick brown fox the")
        indexer.build({"http://example.com/": html})

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            indexer.save(tmp_path)
            fresh = Indexer()
            fresh.load(tmp_path)
            entry = fresh.index["the"]["http://example.com/"]
            assert entry["frequency"] == 2
            assert len(entry["positions"]) == 2
        finally:
            os.unlink(tmp_path)
