# COMP3011 Coursework 2 — Search Engine Tool

A command-line search engine that crawls [quotes.toscrape.com](https://quotes.toscrape.com/), builds an inverted index, and retrieves pages matching user queries.  Results are ranked by **TF-IDF** — a standard information-retrieval metric that measures how important a term is within a document relative to the whole corpus.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Design Decisions](#design-decisions)
5. [Testing](#testing)
6. [Dependencies](#dependencies)

---

## Architecture

```
comp3011-cw2/
├── src/
│   ├── crawler.py   — BFS web crawler with politeness window
│   ├── indexer.py   — HTML parser and inverted index builder
│   ├── search.py    — TF-IDF query engine
│   └── main.py      — Interactive CLI shell
├── tests/
│   ├── test_crawler.py
│   ├── test_indexer.py
│   ├── test_search.py
│   └── test_main.py
├── data/            — Index file output (index.json)
├── requirements.txt
└── README.md
```

### Module responsibilities

| Module | Responsibility |
|---|---|
| `Crawler` | BFS traversal of the target domain; enforces ≥ 6 s politeness delay between requests; returns `{url: html}` mapping |
| `Indexer` | Tokenises visible page text (lowercased, punctuation-stripped); builds the inverted index; tracks document lengths; serialises to JSON |
| `Search` | Intersects posting lists for AND queries; ranks results by TF-IDF; falls back to raw frequency when document lengths are unavailable |
| `main.py` | REPL shell wiring the three modules together; exposes `build`, `load`, `print`, and `find` commands |

### Inverted index schema

```json
{
  "__page_lengths__": {
    "https://quotes.toscrape.com/": 312
  },
  "courage": {
    "https://quotes.toscrape.com/": {
      "frequency": 3,
      "positions": [14, 87, 201]
    }
  }
}
```

`__page_lengths__` is a reserved key that travels with the index file so TF-IDF ranking can be restored after a `load` without re-crawling.

---

## Installation

```bash
git clone <repo-url>
cd comp3011-cw2
pip install -r requirements.txt
```

Python 3.12+ is required (uses `X | Y` union type hints).

---

## Usage

Start the interactive shell:

```bash
python src/main.py
```

### Commands

| Command | Description |
|---|---|
| `build` | Crawl quotes.toscrape.com, build the inverted index, save to `data/index.json` |
| `load` | Load a previously built index from `data/index.json` |
| `print <word>` | Print the full inverted index entry for a word (all URLs, frequencies, positions) |
| `find <query>` | Return all pages containing **every** word in the query, ranked by TF-IDF |
| `help` | Display the command reference |
| `quit` / `exit` | Exit the shell |

### Example session

```
> load
Index loaded from data/index.json  (2847 unique words).

> print nonsense
Index entry for 'nonsense':
  https://quotes.toscrape.com/page/3/
    frequency : 1
    positions : [42]

> find indifference
Found 1 page(s) containing word 'indifference':
  https://quotes.toscrape.com/page/4/

> find good friends
Found 2 page(s) containing words 'good friends':
  https://quotes.toscrape.com/page/1/
  https://quotes.toscrape.com/page/7/
```

---

## Design Decisions

### Data structure — why a nested dict?

The inverted index is `dict[word → dict[url → {frequency, positions}]]`.

- **O(1) average-case lookup** for any term (Python `dict` is a hash map).
- Storing **positions** alongside frequency enables future phrase-search or
  proximity-ranking extensions without re-crawling.
- The entire structure serialises trivially to JSON.

Alternative considered: a flat list of `(word, url, freq)` tuples.  Rejected
because intersection queries would degrade from O(1) to O(n).

### Ranking — TF-IDF over raw frequency

Raw frequency alone favours long documents regardless of actual relevance.
TF-IDF corrects for document length (TF) and term commonness (IDF):

```
TF(t, d)  = occurrences(t, d) / total_tokens(d)
IDF(t)    = log( N / df(t) )
score(d)  = Σ TF(t, d) × IDF(t)   for each query term t
```

A term appearing on every page gets IDF = log(1) = 0 and contributes
nothing to the score — down-weighting stop-word-like terms automatically.

### Crawling strategy — BFS with a seen-set

Breadth-first search ensures shallow pages are indexed first and prevents
the crawler from going arbitrarily deep before finding common pages.  A
`seen` set prevents duplicate fetches on sites with many cross-links.

### Politeness

`time.sleep(delay)` is called between requests but **not** after the final
page to avoid blocking unnecessarily.  The delay defaults to 6 seconds as
required; it is configurable in tests (set to 0) so the test suite is fast.

### Tokenisation

1. BeautifulSoup strips `<script>` / `<style>` blocks before text extraction.
2. Text is lowercased.
3. Tokens are split on whitespace.
4. Leading/trailing ASCII and Unicode punctuation (including curly quotes) is
   stripped from each token.
5. Tokens containing no alphabetic character are discarded.

This means `"it's"` survives (apostrophe is internal) while `"--"` or `"42"`
do not appear in the index.

### Algorithmic complexity

| Operation | Time complexity | Notes |
|---|---|---|
| `Indexer.build()` | O(T) | T = total tokens across all pages |
| `Indexer.save()` / `load()` | O(W) | W = unique words in index |
| `Search.find()` k-word query | O(k·D + R log R) | D = avg posting list length, R = result count |
| `Search.print_word()` | O(1) | Single dict lookup |

---

## Testing

Run the full test suite with coverage:

```bash
pytest
```

Or explicitly:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Test strategy

- **Unit tests** for every public and private method (tokeniser, BFS logic,
  TF-IDF scorer, REPL command handlers).
- **All network calls are mocked** — the suite runs offline in under 10 s.
- **Edge cases** covered: empty queries, unknown words, missing index file,
  circular links, pages that 404, curly-quote tokens, whitespace-only input.
- **Coverage**: 99 % line coverage across all source modules.

---

## Dependencies

| Package | Purpose |
|---|---|
| `requests` | HTTP GET requests during crawl |
| `beautifulsoup4` + `lxml` | HTML parsing and text extraction |
| `pytest` + `pytest-cov` | Test runner and coverage reporting |

Install all dependencies:

```bash
pip install -r requirements.txt
```
