"""
Tests for the Search module.
"""

import pytest

from src.search import Search


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_index():
    """A small hand-crafted inverted index used across multiple tests."""
    return {
        "the": {
            "http://example.com/1": {"frequency": 3, "positions": [0, 5, 10]},
            "http://example.com/2": {"frequency": 1, "positions": [2]},
        },
        "cat": {
            "http://example.com/1": {"frequency": 2, "positions": [1, 6]},
        },
        "sat": {
            "http://example.com/1": {"frequency": 1, "positions": [2]},
            "http://example.com/3": {"frequency": 2, "positions": [0, 4]},
        },
        "indifference": {
            "http://example.com/3": {"frequency": 1, "positions": [7]},
        },
        "good": {
            "http://example.com/2": {"frequency": 2, "positions": [0, 3]},
            "http://example.com/3": {"frequency": 1, "positions": [1]},
        },
        "friends": {
            "http://example.com/2": {"frequency": 1, "positions": [5]},
        },
    }


# ---------------------------------------------------------------------------
# _normalise
# ---------------------------------------------------------------------------


class TestNormalise:
    def test_lowercases(self):
        s = Search({})
        assert s._normalise("Hello") == "hello"

    def test_strips_trailing_punctuation(self):
        s = Search({})
        assert s._normalise("word.") == "word"
        assert s._normalise(",word,") == "word"

    def test_strips_curly_quotes(self):
        s = Search({})
        assert s._normalise("\u201cword\u201d") == "word"

    def test_empty_string_stays_empty(self):
        s = Search({})
        assert s._normalise("") == ""


# ---------------------------------------------------------------------------
# _parse_query
# ---------------------------------------------------------------------------


class TestParseQuery:
    def test_single_word(self):
        s = Search({})
        assert s._parse_query("hello") == ["hello"]

    def test_multiple_words(self):
        s = Search({})
        assert s._parse_query("good friends") == ["good", "friends"]

    def test_extra_whitespace_ignored(self):
        s = Search({})
        assert s._parse_query("  good   friends  ") == ["good", "friends"]

    def test_empty_query_returns_empty(self):
        s = Search({})
        assert s._parse_query("") == []

    def test_punctuation_stripped_from_tokens(self):
        s = Search({})
        assert s._parse_query("hello,") == ["hello"]


# ---------------------------------------------------------------------------
# print_word
# ---------------------------------------------------------------------------


class TestPrintWord:
    def test_returns_entry_for_known_word(self, sample_index):
        s = Search(sample_index)
        result = s.print_word("cat")
        assert result is not None
        assert "http://example.com/1" in result

    def test_case_insensitive(self, sample_index):
        s = Search(sample_index)
        result = s.print_word("CAT")
        assert result is not None

    def test_returns_none_for_unknown_word(self, sample_index):
        s = Search(sample_index)
        assert s.print_word("nonsense") is None

    def test_returns_none_for_empty_string(self, sample_index):
        s = Search(sample_index)
        assert s.print_word("") is None

    def test_returns_frequency_and_positions(self, sample_index):
        s = Search(sample_index)
        entry = s.print_word("the")
        assert entry["http://example.com/1"]["frequency"] == 3
        assert entry["http://example.com/1"]["positions"] == [0, 5, 10]


# ---------------------------------------------------------------------------
# find — single word
# ---------------------------------------------------------------------------


class TestFindSingleWord:
    def test_finds_pages_containing_word(self, sample_index):
        s = Search(sample_index)
        results = s.find("cat")
        assert "http://example.com/1" in results

    def test_case_insensitive(self, sample_index):
        s = Search(sample_index)
        assert s.find("CAT") == s.find("cat")

    def test_returns_empty_for_unknown_word(self, sample_index):
        s = Search(sample_index)
        assert s.find("zzz") == []

    def test_returns_empty_for_empty_query(self, sample_index):
        s = Search(sample_index)
        assert s.find("") == []

    def test_results_sorted_by_frequency_descending(self, sample_index):
        # "sat" appears in page/1 (freq 1) and page/3 (freq 2)
        # page/3 should come first
        s = Search(sample_index)
        results = s.find("sat")
        assert results[0] == "http://example.com/3"


# ---------------------------------------------------------------------------
# find — multi-word (AND semantics)
# ---------------------------------------------------------------------------


class TestFindMultiWord:
    def test_returns_pages_with_all_words(self, sample_index):
        # "good" and "friends" both appear only on page/2
        s = Search(sample_index)
        results = s.find("good friends")
        assert results == ["http://example.com/2"]

    def test_returns_empty_when_no_page_has_all_words(self, sample_index):
        # "cat" is only on page/1; "indifference" only on page/3 — no overlap
        s = Search(sample_index)
        assert s.find("cat indifference") == []

    def test_intersection_across_three_words(self, sample_index):
        # "the" is on page/1 and page/2; "cat" only on page/1; "sat" on page/1 and page/3
        # intersection of all three: only page/1
        s = Search(sample_index)
        results = s.find("the cat sat")
        assert results == ["http://example.com/1"]

    def test_whitespace_only_query_returns_empty(self, sample_index):
        s = Search(sample_index)
        assert s.find("   ") == []

    def test_duplicate_words_handled(self, sample_index):
        # "cat cat" should behave the same as "cat" for page matching
        s = Search(sample_index)
        results = s.find("cat cat")
        assert "http://example.com/1" in results


# ---------------------------------------------------------------------------
# TF-IDF scoring
# ---------------------------------------------------------------------------


class TestTFIDFScore:
    """Tests for the _tfidf_score helper and TF-IDF-based ranking in find()."""

    @pytest.fixture()
    def tfidf_index(self):
        """Three-page index designed to exercise TF-IDF ranking.

        'rare' appears in page/1 and page/2 only (df=2, N=3 → IDF > 0).
        page/2 is much shorter than page/1, so its TF is higher and its
        TF-IDF score should therefore be higher.

        'common' appears in all three pages (df=N → IDF=0), which tests
        the IDF down-weighting of ubiquitous terms.
        """
        index = {
            "rare": {
                "http://example.com/1": {"frequency": 1, "positions": [0]},
                "http://example.com/2": {"frequency": 1, "positions": [0]},
            },
            "common": {
                "http://example.com/1": {"frequency": 5, "positions": [1, 2, 3, 4, 5]},
                "http://example.com/2": {"frequency": 1, "positions": [1]},
                "http://example.com/3": {"frequency": 2, "positions": [0, 1]},
            },
        }
        # N=3 total pages; 'rare' is absent from page/3, so IDF = log(3/2) > 0
        page_lengths = {
            "http://example.com/1": 100,
            "http://example.com/2": 5,
            "http://example.com/3": 20,
        }
        return index, page_lengths

    def test_tfidf_score_higher_for_shorter_doc(self, tfidf_index):
        index, page_lengths = tfidf_index
        s = Search(index, page_lengths)
        score1 = s._tfidf_score(["rare"], "http://example.com/1")
        score2 = s._tfidf_score(["rare"], "http://example.com/2")
        # page/2 is shorter → higher TF → higher TF-IDF
        assert score2 > score1

    def test_tfidf_score_zero_for_unknown_url(self, tfidf_index):
        index, page_lengths = tfidf_index
        s = Search(index, page_lengths)
        assert s._tfidf_score(["rare"], "http://example.com/unknown") == 0.0

    def test_tfidf_score_zero_for_unknown_word(self, tfidf_index):
        index, page_lengths = tfidf_index
        s = Search(index, page_lengths)
        assert s._tfidf_score(["zzz"], "http://example.com/1") == 0.0

    def test_find_uses_tfidf_when_page_lengths_provided(self, tfidf_index):
        index, page_lengths = tfidf_index
        s = Search(index, page_lengths)
        results = s.find("rare")
        # page/2 should rank first because of its higher TF-IDF score
        assert results[0] == "http://example.com/2"

    def test_find_falls_back_to_frequency_without_page_lengths(self, tfidf_index):
        index, _ = tfidf_index
        # Inject page_lengths for page/1 only so page/2 has no length data
        # and the fallback (raw frequency) path is exercised.
        s = Search(index)  # no page_lengths
        results = s.find("rare")
        # Both pages have the same frequency (1), so order is arbitrary,
        # but the call must succeed and return both pages.
        assert set(results) == {"http://example.com/1", "http://example.com/2"}

    def test_tfidf_idf_downweights_common_terms(self, tfidf_index):
        """A word appearing in every document has IDF = log(N/N) = 0."""
        index, page_lengths = tfidf_index
        s = Search(index, page_lengths)
        # 'common' appears in both pages → df == N → IDF == 0
        assert s._tfidf_score(["common"], "http://example.com/1") == 0.0
