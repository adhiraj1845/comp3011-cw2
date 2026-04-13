# COMP3011 Coursework 2 — Search Engine Tool

A command-line search engine that crawls [quotes.toscrape.com](https://quotes.toscrape.com/), builds an inverted index, and allows querying across all crawled pages.

## Overview

The tool is split into three core modules:

- **Crawler** — fetches pages from the target website, respecting a 6-second politeness window
- **Indexer** — parses page content and builds an inverted index storing word frequency and position data
- **Search** — queries the inverted index for single or multi-word terms

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the interactive shell:

```bash
python src/main.py
```

### Commands

| Command | Description |
|---|---|
| `build` | Crawl the website, build the index, and save it to `data/index.json` |
| `load` | Load a previously built index from `data/index.json` |
| `print <word>` | Print the inverted index entry for a specific word |
| `find <query>` | Find all pages containing one or more words |
| `quit` | Exit the shell |

### Examples

```
> build
> load
> print nonsense
> find indifference
> find good friends
```

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

## Dependencies

- `requests` — HTTP requests
- `beautifulsoup4` + `lxml` — HTML parsing
- `pytest` + `pytest-cov` — testing and coverage
