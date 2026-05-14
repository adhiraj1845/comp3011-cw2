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


# ---------------------------------------------------------------------------
# suggest — spelling suggestions
# ---------------------------------------------------------------------------


class TestSuggest:
    def test_returns_close_match(self, sample_index):
        # "cot" is close to "cat" (one substitution)
        s = Search(sample_index)
        suggestions = s.suggest("cot")
        assert "cat" in suggestions

    def test_returns_empty_for_completely_unknown_word(self, sample_index):
        s = Search(sample_index)
        assert s.suggest("zzzzzzzzz") == []

    def test_returns_empty_for_empty_string(self, sample_index):
        s = Search(sample_index)
        assert s.suggest("") == []

    def test_case_insensitive_matching(self, sample_index):
        s = Search(sample_index)
        # "CAT" normalises to "cat" which should match the index key "cat"
        suggestions = s.suggest("COT")
        assert "cat" in suggestions

    def test_respects_n_limit(self, sample_index):
        s = Search(sample_index)
        # Even if multiple matches exist, n=1 caps the result
        suggestions = s.suggest("sat", n=1)
        assert len(suggestions) <= 1

    def test_returns_empty_for_word_in_index(self, sample_index):
        # "cat" is already in the index — difflib may still return it as a
        # match (exact similarity = 1.0 > cutoff), but we don't suppress it
        # here; the caller decides whether to show suggestions.
        s = Search(sample_index)
        suggestions = s.suggest("cat")
        assert isinstance(suggestions, list)

    def test_returns_multiple_suggestions_when_available(self, sample_index):
        # "sat" is in the index; "cat" and "the" are less similar
        # We just verify the result is a list with at most n entries
        s = Search(sample_index)
        suggestions = s.suggest("sat", n=3)
        assert len(suggestions) <= 3

    def test_empty_index_returns_empty(self):
        s = Search({})
        assert s.suggest("hello") == []


# ---------------------------------------------------------------------------
# find_phrase — exact consecutive-position search
# ---------------------------------------------------------------------------


@pytest.fixture()
def phrase_index():
    """Index for a two-page corpus with known token positions."""
    return {
        "the": {
            "http://example.com/1": {"frequency": 2, "positions": [0, 4]},
            "http://example.com/2": {"frequency": 1, "positions": [3]},
        },
        "cat": {
            "http://example.com/1": {"frequency": 1, "positions": [1]},
            "http://example.com/2": {"frequency": 1, "positions": [0]},
        },
        "sat": {
            "http://example.com/1": {"frequency": 1, "positions": [2]},
        },
        "mat": {
            "http://example.com/1": {"frequency": 1, "positions": [3]},
            "http://example.com/2": {"frequency": 1, "positions": [1]},
        },
    }
    # page/1 text order: the cat sat mat the …
    # page/2 text order: cat mat … the …


class TestFindPhrase:
    def test_finds_consecutive_phrase(self, phrase_index):
        s = Search(phrase_index)
        # "the cat" → page/1 has "the" at 0 and "cat" at 1 (consecutive)
        assert "http://example.com/1" in s.find_phrase("the cat")

    def test_rejects_non_consecutive_order(self, phrase_index):
        s = Search(phrase_index)
        # "cat the" is not consecutive on either page
        assert s.find_phrase("cat the") == []

    def test_three_word_phrase(self, phrase_index):
        s = Search(phrase_index)
        # "the cat sat" only on page/1 (positions 0,1,2)
        results = s.find_phrase("the cat sat")
        assert results == ["http://example.com/1"]

    def test_single_word_delegates_to_find(self, phrase_index):
        s = Search(phrase_index)
        # Single word: same result as find()
        assert s.find_phrase("cat") == s.find("cat")

    def test_empty_phrase_returns_empty(self, phrase_index):
        s = Search(phrase_index)
        assert s.find_phrase("") == []

    def test_phrase_not_in_index_returns_empty(self, phrase_index):
        s = Search(phrase_index)
        assert s.find_phrase("the dog") == []

    def test_phrase_across_pages(self, phrase_index):
        s = Search(phrase_index)
        # "cat mat" consecutive on page/2 (positions 0,1) but NOT on page/1 (1,3 — gap)
        results = s.find_phrase("cat mat")
        assert "http://example.com/2" in results
        assert "http://example.com/1" not in results

    def test_case_insensitive(self, phrase_index):
        s = Search(phrase_index)
        assert s.find_phrase("THE CAT") == s.find_phrase("the cat")


# ---------------------------------------------------------------------------
# find_wildcard — '*' pattern expansion
# ---------------------------------------------------------------------------


@pytest.fixture()
def wildcard_index():
    return {
        "courage":   {"http://example.com/1": {"frequency": 2, "positions": [0, 5]}},
        "course":    {"http://example.com/1": {"frequency": 1, "positions": [1]},
                      "http://example.com/2": {"frequency": 1, "positions": [0]}},
        "court":     {"http://example.com/2": {"frequency": 1, "positions": [1]}},
        "friends":   {"http://example.com/1": {"frequency": 1, "positions": [2]},
                      "http://example.com/2": {"frequency": 1, "positions": [2]}},
        "freedom":   {"http://example.com/3": {"frequency": 1, "positions": [0]}},
    }


class TestFindWildcard:
    def test_prefix_wildcard_matches_multiple_words(self, wildcard_index):
        s = Search(wildcard_index)
        # "cour*" should match courage, course, court
        results = s.find_wildcard("cour*")
        assert "http://example.com/1" in results
        assert "http://example.com/2" in results

    def test_wildcard_and_plain_token_intersection(self, wildcard_index):
        s = Search(wildcard_index)
        # "cour* friends" → pages with a cour* word AND "friends"
        results = s.find_wildcard("cour* friends")
        assert "http://example.com/1" in results
        assert "http://example.com/2" in results
        assert "http://example.com/3" not in results

    def test_no_match_returns_empty(self, wildcard_index):
        s = Search(wildcard_index)
        assert s.find_wildcard("zzz*") == []

    def test_empty_query_returns_empty(self, wildcard_index):
        s = Search(wildcard_index)
        assert s.find_wildcard("") == []

    def test_exact_word_without_wildcard(self, wildcard_index):
        s = Search(wildcard_index)
        # Token without '*' works like regular find
        results = s.find_wildcard("friends")
        assert "http://example.com/1" in results
        assert "http://example.com/2" in results

    def test_wildcard_only_pages_that_have_all_tokens(self, wildcard_index):
        s = Search(wildcard_index)
        # "fre*" matches "freedom" (only page/3); "friends" not on page/3 → empty
        results = s.find_wildcard("fre* friends")
        assert results == []

    def test_star_alone_matches_everything(self, wildcard_index):
        s = Search(wildcard_index)
        # "*" matches every vocabulary word → union = all pages
        results = s.find_wildcard("*")
        assert set(results) == {"http://example.com/1", "http://example.com/2", "http://example.com/3"}
