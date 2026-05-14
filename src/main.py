"""
main.py — Interactive command-line shell for the search engine.

Commands
--------
build                  Crawl quotes.toscrape.com, build the inverted index,
                       and save it to data/index.json.
load                   Load a previously built index from data/index.json.
print <word>           Show the inverted index entry for a single word.
find <query>           AND search — pages containing every word in the query,
                       ranked by TF-IDF.
find "<phrase>"        Exact-phrase search — pages where the words appear
                       consecutively (wrap the phrase in double quotes).
find <pat*ern>         Wildcard search — expand '*' against the vocabulary
                       before intersecting page sets.
stats                  Show corpus statistics (pages, vocabulary, top words).
help                   Show this help message.
quit / exit            Exit the shell.
"""

import json
from pathlib import Path

from src.crawler import Crawler
from src.indexer import Indexer
from src.search import Search

INDEX_PATH = Path(__file__).parent.parent / "data" / "index.json"


def _cmd_build(indexer: Indexer) -> Search:
    """Crawl the website, build the index, persist it, and return a Search."""
    print("Starting crawl — this will take several minutes due to the politeness window.")
    crawler = Crawler()
    pages = crawler.crawl()
    print(f"Crawled {len(pages)} page(s). Building index…")
    indexer.build(pages)
    indexer.save(INDEX_PATH)
    print(f"Index saved to {INDEX_PATH}  ({len(indexer.index)} unique words).")
    return Search(indexer.index, indexer.page_lengths)


def _cmd_load(indexer: Indexer) -> Search | None:
    """Load the index from disk and return a Search, or None on failure."""
    if not INDEX_PATH.exists():
        print(f"No index file found at {INDEX_PATH}. Run 'build' first.")
        return None
    indexer.load(INDEX_PATH)
    print(f"Index loaded from {INDEX_PATH}  ({len(indexer.index)} unique words).")
    return Search(indexer.index, indexer.page_lengths)


def _cmd_print(search: Search | None, word: str) -> None:
    """Print the inverted index entry for *word*."""
    if search is None:
        print("No index loaded. Use 'build' or 'load' first.")
        return
    if not word:
        print("Usage: print <word>")
        return

    entry = search.print_word(word)
    if entry is None:
        suggestions = search.suggest(word)
        if suggestions:
            print(f"'{word}' not found. Did you mean: {', '.join(suggestions)}?")
        else:
            print(f"'{word}' not found in the index.")
        return

    print(f"\nIndex entry for '{word.lower()}':")
    for url, stats in entry.items():
        freq = stats["frequency"]
        positions = stats["positions"]
        print(f"  {url}")
        print(f"    frequency : {freq}")
        print(f"    positions : {positions}")


def _cmd_find(search: Search | None, query: str) -> None:
    """Dispatch to phrase, wildcard, or AND search based on query syntax."""
    if search is None:
        print("No index loaded. Use 'build' or 'load' first.")
        return
    if not query.strip():
        print('Usage: find <word> [word …]  |  find "<exact phrase>"  |  find <pat*ern>')
        return

    # --- Exact-phrase search (query wrapped in double quotes) ---
    if query.startswith('"') and query.endswith('"') and len(query) > 2:
        phrase = query[1:-1].strip()
        results = search.find_phrase(phrase)
        if not results:
            print(f"No pages found for phrase: {query}")
        else:
            print(f"\nFound {len(results)} page(s) containing phrase {query}:")
            for url in results:
                print(f"  {url}")
        return

    # --- Wildcard search (any token contains '*') ---
    if "*" in query:
        results = search.find_wildcard(query)
        if not results:
            print(f"No pages found for wildcard query: '{query}'")
        else:
            print(f"\nFound {len(results)} page(s) matching wildcard '{query}':")
            for url in results:
                print(f"  {url}")
        return

    # --- Standard AND search ---
    results = search.find(query)
    if not results:
        # Check whether any individual words are absent from the index.
        # If so, offer spelling suggestions; otherwise all words exist but
        # no page contains every one of them simultaneously.
        absent = [w for w in query.split() if search._normalise(w) not in search.index]
        if absent:
            for w in absent:
                suggestions = search.suggest(w)
                if suggestions:
                    print(f"  '{w}' not found — did you mean: {', '.join(suggestions)}?")
                else:
                    print(f"  '{w}' not found in the index.")
        else:
            print(f"No pages found for query: '{query}'")
        return

    words = query.strip().split()
    label = "word" if len(words) == 1 else "words"
    print(f"\nFound {len(results)} page(s) containing {label} '{query.strip()}':")
    for url in results:
        print(f"  {url}")


def _cmd_stats(search: Search | None) -> None:
    """Print corpus statistics derived from the loaded index."""
    if search is None:
        print("No index loaded. Use 'build' or 'load' first.")
        return

    vocab_size = len(search.index)
    page_count = len(search.page_lengths)
    total_tokens = sum(search.page_lengths.values())
    avg_len = total_tokens / page_count if page_count else 0

    # Total occurrences of each word across all pages.
    word_totals: dict[str, int] = {
        word: sum(v["frequency"] for v in pages.values())
        for word, pages in search.index.items()
    }
    top_words = sorted(word_totals, key=word_totals.__getitem__, reverse=True)[:10]

    print(f"\nCorpus statistics")
    print(f"  Pages indexed   : {page_count}")
    print(f"  Unique words    : {vocab_size:,}")
    print(f"  Total tokens    : {total_tokens:,}")
    print(f"  Avg page length : {avg_len:.0f} tokens")
    print(f"\n  Top 10 most frequent words:")
    for word in top_words:
        print(f"    {word:<20} {word_totals[word]:>6} occurrences")


def run_shell() -> None:
    """Start the interactive REPL."""
    indexer = Indexer()
    search: Search | None = None

    print("Search Engine — type 'help' for available commands.")

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue

        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "build":
            search = _cmd_build(indexer)

        elif cmd == "load":
            search = _cmd_load(indexer)

        elif cmd == "print":
            _cmd_print(search, arg)

        elif cmd == "find":
            _cmd_find(search, arg)

        elif cmd == "stats":
            _cmd_stats(search)

        elif cmd == "help":
            print(__doc__)

        elif cmd in ("quit", "exit"):
            print("Goodbye.")
            break

        else:
            print(f"Unknown command '{cmd}'. Type 'help' for available commands.")


if __name__ == "__main__":
    run_shell()
