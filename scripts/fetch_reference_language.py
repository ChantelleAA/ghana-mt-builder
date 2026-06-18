"""
fetch_reference_language.py  —  maintainer tool (not user-facing)
=================================================================
Caches the full Bible of one or more *reference* languages so they can be used
as parallel-corpus candidates against the Ghanaian languages.

Every cache is keyed by `verse_key` (BOOK.chapter.verse), exactly like the
Ghanaian language CSVs and english_cache.csv.  Because everything shares that
key, the retrieval library (ghana_corpus.py) can align ANY two languages by a
simple join — no re-scraping needed at retrieval time.

Reference languages live in `reference_languages.csv`:

    code,name,version_id,abbr,cache_file
    fr,French,93,LSG,reference_caches/fr.csv
    ...

ADDING A NEW REFERENCE LANGUAGE
-------------------------------
  1. Find its YouVersion numeric version id (a full-Bible version works best).
  2. Add a row to reference_languages.csv.
  3. Run:  python scripts/fetch_reference_language.py <code>
  4. It is now selectable as a parallel candidate in ghana_corpus.py — no
     other code changes required.

USAGE
-----
  python scripts/fetch_reference_language.py            # fetch every uncached language
  python scripts/fetch_reference_language.py fr es de   # fetch specific codes
  python scripts/fetch_reference_language.py --force fr  # re-fetch even if cached

Source: the verse text comes from public Bible translations on YouVersion
(bible.com), retrieved via its chapter JSON API.  No Chrome / Selenium needed.
Requires: requests, beautifulsoup4, lxml
"""

import csv
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

import requests
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# Resolve paths relative to the repo root (this file lives in scripts/).
REPO_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_CSV   = os.path.join(REPO_ROOT, "reference_languages.csv")
OUTPUT_ROOT     = os.path.join(REPO_ROOT, "bible_parallel_text_datasets")

CHAPTER_API     = "https://nodejs.bible.com/api/bible/chapter/3.1"

NUM_WORKERS     = 16
REQUEST_DELAY   = 1      # seconds between requests per worker
REQUEST_TIMEOUT = 20
MAX_RETRIES     = 3
RETRY_WAIT      = 5

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}

ALL_BOOK_CODES = [
    "GEN","EXO","LEV","NUM","DEU","JOS","JDG","RUT","1SA","2SA",
    "1KI","2KI","1CH","2CH","EZR","NEH","EST","JOB","PSA","PRO",
    "ECC","SNG","ISA","JER","LAM","EZK","DAN","HOS","JOL","AMO",
    "OBA","JON","MIC","NAM","HAB","ZEP","HAG","ZEC","MAL",
    "MAT","MRK","LUK","JHN","ACT","ROM","1CO","2CO","GAL","EPH",
    "PHP","COL","1TH","2TH","1TI","2TI","TIT","PHM","HEB","JAS",
    "1PE","2PE","1JN","2JN","3JN","JUD","REV",
]

BOOK_CHAPTERS = {
    "GEN":50,"EXO":40,"LEV":27,"NUM":36,"DEU":34,"JOS":24,"JDG":21,
    "RUT":4,"1SA":31,"2SA":24,"1KI":22,"2KI":25,"1CH":29,"2CH":36,
    "EZR":10,"NEH":13,"EST":10,"JOB":42,"PSA":150,"PRO":31,"ECC":12,
    "SNG":8,"ISA":66,"JER":52,"LAM":5,"EZK":48,"DAN":12,"HOS":14,
    "JOL":3,"AMO":9,"OBA":1,"JON":4,"MIC":7,"NAM":3,"HAB":3,"ZEP":3,
    "HAG":2,"ZEC":14,"MAL":4,
    "MAT":28,"MRK":16,"LUK":24,"JHN":21,"ACT":28,"ROM":16,"1CO":16,
    "2CO":13,"GAL":6,"EPH":6,"PHP":4,"COL":4,"1TH":5,"2TH":3,"1TI":6,
    "2TI":4,"TIT":3,"PHM":1,"HEB":13,"JAS":5,"1PE":5,"2PE":3,"1JN":5,
    "2JN":1,"3JN":1,"JUD":1,"REV":22,
}


# ─────────────────────────────────────────────
# TEXT CLEANING  (identical to the main builder so all columns are consistent)
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\d+', '', text)
    lines = text.splitlines()
    processed = []
    for line in lines:
        line = line.strip()
        if line:
            if line[-1] not in ['.', '!', '?', ':', ';']:
                line += '.'
            processed.append(line)
    text = ' '.join(processed)
    text = re.sub(r'[\"“”‘’\(\)\[\]\{\}]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[,.]{2,}', '.', text)
    text = re.sub(r'([,.!?;:])\.', '.', text)
    if text and not text.endswith('.'):
        text += '.'
    return text


# ─────────────────────────────────────────────
# CHAPTER FETCHING  (YouVersion JSON API)
# ─────────────────────────────────────────────

def _parse_chapter_content(content_html: str, book: str, chapter: int) -> dict[int, str]:
    soup = BeautifulSoup(content_html, "lxml")
    prefix = f"{book}.{chapter}."
    parts: dict[int, list[str]] = {}
    for span in soup.find_all("span", attrs={"data-usfm": True}):
        usfm = span["data-usfm"]
        if not usfm.startswith(prefix):
            continue
        tail = usfm[len(prefix):]
        try:
            verse_num = int(re.split(r"[-+]", tail)[0])
        except ValueError:
            continue
        content_spans = span.select("span.content")
        if content_spans:
            t = " ".join(c.get_text(" ", strip=True) for c in content_spans)
        else:
            t = span.get_text(" ", strip=True)
        t = t.strip()
        if t:
            parts.setdefault(verse_num, []).append(t)
    return {n: " ".join(chunks) for n, chunks in parts.items()}


def get_chapter_verses(session, version_num, book, chapter) -> dict[int, str] | None:
    params = {"id": version_num, "reference": f"{book}.{chapter}"}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(REQUEST_DELAY)
            resp = session.get(CHAPTER_API, params=params,
                               headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            content = resp.json().get("content", "")
            if not content:
                return None
            verses = _parse_chapter_content(content, book, chapter)
            return verses if verses else None
        except Exception:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT)
            else:
                return None
    return None


# ─────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────

def load_reference_registry() -> list[dict]:
    if not os.path.exists(REFERENCE_CSV):
        sys.exit(f"Reference registry not found: {REFERENCE_CSV}")
    with open(REFERENCE_CSV, newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f)]


def resolve_cache_path(cache_file: str) -> str:
    return os.path.join(OUTPUT_ROOT, cache_file)


# ─────────────────────────────────────────────
# CACHE WRITER  (thread-safe, resume-safe)
# ─────────────────────────────────────────────

def load_existing_keys(cache_path: str) -> set[str]:
    """Return verse_keys already present so re-runs skip completed work."""
    keys: set[str] = set()
    if os.path.exists(cache_path):
        with open(cache_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                keys.add(row["verse_key"])
    return keys


def fetch_one_language(code: str, name: str, version_num: int,
                       cache_path: str, force: bool):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if force and os.path.exists(cache_path):
        os.remove(cache_path)

    existing = load_existing_keys(cache_path)
    print(f"\n{'='*60}")
    print(f"  {name} ({code})  —  YouVersion version {version_num}")
    print(f"  cache: {cache_path}")
    if existing:
        print(f"  resuming — {len(existing):,} verses already cached")
    print(f"{'='*60}")

    # Build chapter task list, skipping chapters that are already fully present.
    tasks = []
    for book in ALL_BOOK_CODES:
        for chapter in range(1, BOOK_CHAPTERS.get(book, 0) + 1):
            tasks.append((book, chapter))

    write_lock = threading.Lock()
    write_header = not os.path.exists(cache_path)
    out = open(cache_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        out, fieldnames=["verse_key", "version_id", "lang_code", "text"])
    if write_header:
        writer.writeheader()

    # Session pool
    sessions: Queue = Queue()
    for _ in range(NUM_WORKERS):
        s = requests.Session()
        s.headers.update(REQUEST_HEADERS)
        sessions.put(s)

    saved = {"n": 0}

    def work(book, chapter):
        session = sessions.get()
        try:
            verses = get_chapter_verses(session, version_num, book, chapter)
        finally:
            sessions.put(session)
        if not verses:
            return 0
        rows = []
        for verse_num, raw in verses.items():
            key = f"{book}.{chapter}.{verse_num}"
            if key in existing:
                continue
            cleaned = clean_text(raw) if raw.strip() else ""
            if cleaned:
                rows.append({"verse_key": key, "version_id": version_num,
                             "lang_code": code, "text": cleaned})
        if rows:
            with write_lock:
                writer.writerows(rows)
                out.flush()
                saved["n"] += len(rows)
        return len(rows)

    try:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
            futures = {pool.submit(work, b, c): (b, c) for b, c in tasks}
            done = 0
            for fut in as_completed(futures):
                b, c = futures[fut]
                done += 1
                try:
                    n = fut.result()
                except Exception as e:
                    print(f"  {b}.{c} failed: {e}")
                    n = 0
                if done % 100 == 0 or n:
                    print(f"  [{done}/{len(tasks)} chapters] {b}.{c} +{n} "
                          f"(total saved {saved['n']:,})")
    finally:
        out.close()

    print(f"  ✅ {name}: {saved['n']:,} new verses written to {cache_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    argv = sys.argv[1:]
    force = "--force" in argv
    codes = [a for a in argv if not a.startswith("--")]

    registry = load_reference_registry()
    by_code = {r["code"].strip(): r for r in registry}

    if codes:
        selected = []
        for c in codes:
            if c not in by_code:
                sys.exit(f"Unknown reference code '{c}'. "
                         f"Known: {', '.join(by_code)}")
            selected.append(by_code[c])
    else:
        # Default: every language whose cache file does not yet exist.
        selected = [r for r in registry
                    if not os.path.exists(resolve_cache_path(r["cache_file"]))]
        if not selected:
            print("All reference languages are already cached. "
                  "Use --force <code> to re-fetch.")
            return

    print(f"Fetching {len(selected)} reference language(s): "
          f"{', '.join(r['code'] for r in selected)}")

    for r in selected:
        fetch_one_language(
            code=r["code"].strip(),
            name=r["name"].strip(),
            version_num=int(r["version_id"]),
            cache_path=resolve_cache_path(r["cache_file"].strip()),
            force=force,
        )

    print("\nDone. New reference languages are now available in ghana_corpus.py.")


if __name__ == "__main__":
    main()
