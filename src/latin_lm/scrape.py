"""Scrape The Latin Library into a JSONL corpus of works.

Politeness: checks robots.txt, sends a descriptive User-Agent, and sleeps
``delay_seconds`` between network requests. Every fetched page is cached on
disk (including 404s, as markers), so a rerun never refetches a page it
already has.

Structure: each author config entry points at an index page. Links on it are
grouped into works: links whose text looks like a part ("Liber I",
"Praefatio", a bare numeral) are merged under the nearest preceding
<h2> heading (e.g. METAMORPHOSES); links whose text is a title stand alone.
A linked page that turns out to be another index of links (e.g. Cicero's
Philippics) is expanded one level, its pages becoming parts of that work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.robotparser
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urldefrag, urljoin

import requests
from bs4 import BeautifulSoup

from .clean import clean_html, extract_links
from .config import Config, load_config
from .normalize import normalize_text

# Link text that marks a *part* of a multi-page work rather than a work title.
_PART_RE = re.compile(
    r"^(liber\b|libri\b|pars\b|actio\b|praefatio$|prologus$|fragmenta?$"
    r"|[ivxlcdm]+\b[.\s]|[ivxlcdm]+$|[0-9]+$)",
    re.IGNORECASE,
)

_BLOCK_STATUSES = {401, 403, 429}


class ScrapeBlockedError(RuntimeError):
    pass


@dataclass
class Work:
    author: str
    title: str
    part_urls: list[str] = field(default_factory=list)
    part_titles: list[str] = field(default_factory=list)


class CachingFetcher:
    """Fetch pages with on-disk caching, politeness delay, and robots.txt."""

    def __init__(self, base_url: str, user_agent: str, delay: float, cache_dir: Path):
        self.base_url = base_url.rstrip("/")
        self.delay = delay
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = cache_dir / "index.json"
        self.index: dict[str, dict] = {}
        if self.index_path.exists():
            self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self._last_request = 0.0
        self.fetched = 0
        self.cache_hits = 0
        self._robots = urllib.robotparser.RobotFileParser()
        self._robots_checked = False

    def _check_robots(self) -> None:
        if self._robots_checked:
            return
        robots_url = f"{self.base_url}/robots.txt"
        try:
            resp = self.session.get(robots_url, timeout=20)
            if resp.status_code == 200:
                self._robots.parse(resp.text.splitlines())
            else:
                self._robots.parse([])  # no robots.txt -> everything allowed
        except requests.RequestException:
            self._robots.parse([])
        self._robots_checked = True

    def _cache_file(self, url: str) -> Path:
        path = url.split("//", 1)[-1]
        safe = re.sub(r"[^A-Za-z0-9._-]", "__", path)
        if len(safe) > 150:
            safe = safe[:120] + "_" + hashlib.sha1(url.encode()).hexdigest()[:12]
        return self.cache_dir / safe

    def get(self, url: str) -> bytes | None:
        """Return page bytes, or None if the page is missing (e.g. 404)."""
        url, _ = urldefrag(url)
        entry = self.index.get(url)
        if entry is not None:
            self.cache_hits += 1
            if entry["status"] != 200:
                return None
            return self._cache_file(url).read_bytes()

        self._check_robots()
        ua = self.session.headers["User-Agent"]
        if not self._robots.can_fetch(ua, url):
            raise ScrapeBlockedError(f"robots.txt disallows fetching {url}")

        resp = None
        for attempt, backoff in enumerate((5, 20, 0)):
            wait = self.delay - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            try:
                resp = self.session.get(url, timeout=30)
                break
            except requests.RequestException as exc:
                if backoff == 0:
                    # Transient network failure, not a block: skip this page and
                    # leave it uncached so a rerun retries it.
                    print(f"    warning: giving up on {url} after 3 attempts ({exc})",
                          file=sys.stderr)
                    return None
                print(f"    warning: {exc.__class__.__name__} for {url}; "
                      f"retrying in {backoff}s", file=sys.stderr)
                time.sleep(backoff)
        if resp.status_code in _BLOCK_STATUSES:
            raise ScrapeBlockedError(
                f"got HTTP {resp.status_code} for {url}; the site may be blocking scraping"
            )
        self.fetched += 1
        entry = {"status": resp.status_code, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if resp.status_code == 200:
            self._cache_file(url).write_bytes(resp.content)
        self.index[url] = entry
        self._save_index()
        return resp.content if resp.status_code == 200 else None

    def _save_index(self) -> None:
        self.index_path.write_text(json.dumps(self.index, indent=1), encoding="utf-8")


def _nearest_heading(a_tag) -> str | None:
    for h2 in a_tag.find_all_previous("h2"):
        classes = set(h2.get("class") or [])
        if "date" in classes:
            continue
        text = " ".join(h2.get_text(" ", strip=True).split())
        if text:
            return text
    return None


def parse_author_index(html: bytes, author: str) -> list[Work]:
    """Group an author index page's links into works."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(True, class_="footer"):
        tag.decompose()

    works: list[Work] = []
    by_title: dict[str, Work] = {}
    for a in soup.find_all("a", href=True):
        pairs = extract_links(BeautifulSoup(str(a), "lxml"))
        if not pairs:
            continue
        href, text = pairs[0]
        if not text:
            continue
        if _PART_RE.match(text):
            heading = _nearest_heading(a) or f"{author} (ungrouped)"
            title = heading.title() if heading.isupper() else heading
        else:
            title = text
        work = by_title.get(title)
        if work is None:
            work = Work(author=author, title=title)
            by_title[title] = work
            works.append(work)
        work.part_urls.append(href)
        work.part_titles.append(text)
    return works


def _apply_rewrites(url: str, rewrites: dict[str, str]) -> str:
    for old, new in rewrites.items():
        marker = f"thelatinlibrary.com/{old}"
        if marker in url:
            url = url.replace(marker, f"thelatinlibrary.com/{new}")
    return url


def _work_id(author: str, title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", f"{author}-{title}".lower()).strip("-")
    return slug


def discover_authors(fetcher: CachingFetcher, cfg: Config) -> list[tuple[str, str, str]]:
    """Find (author, index_url, register) triples from the register index pages."""
    found: dict[str, tuple[str, str, str]] = {}  # keyed by resolved URL
    for register, page in cfg.scrape.site_indexes.items():
        index_url = urljoin(cfg.scrape.base_url + "/", page)
        html = fetcher.get(index_url)
        if html is None:
            print(f"    warning: register index missing: {index_url}", file=sys.stderr)
            continue
        page_links = clean_html(html).links
        for href, text in page_links:
            if not text:
                continue
            if any(href.lstrip("/").startswith(p) for p in cfg.scrape.discover_exclude):
                continue
            # Directory links ("/caesar") must end in "/" so relative links resolve.
            if not href.rsplit("/", 1)[-1].count("."):
                href = href.rstrip("/") + "/"
            url, _ = urldefrag(urljoin(index_url, href))
            if url not in found:
                found[url] = (text, url, register)
    return list(found.values())


def scrape_author(
    author: str,
    index_url: str,
    fetcher: CachingFetcher,
    cfg: Config,
    assigned_urls: set[str],
    seen_hashes: set[str],
    register: str = "classical",
) -> list[dict]:
    html = fetcher.get(index_url)
    if html is None:
        print(f"    warning: author index missing: {index_url}", file=sys.stderr)
        return []
    works = parse_author_index(html, author)

    records: list[dict] = []
    for work in works:
        texts: list[str] = []
        urls: list[str] = []
        titles: list[str] = []
        queue: list[tuple[str, str, int]] = [
            (urljoin(index_url, href), ptitle, 0)
            for href, ptitle in zip(work.part_urls, work.part_titles)
        ]
        while queue:
            url, ptitle, depth = queue.pop(0)
            url, _ = urldefrag(_apply_rewrites(url, cfg.scrape.url_rewrites))
            if url in assigned_urls:
                continue
            page_bytes = fetcher.get(url)
            if page_bytes is None:
                print(f"    warning: missing page {url}", file=sys.stderr)
                continue
            assigned_urls.add(url)
            page = clean_html(page_bytes)
            if len(page.text) >= cfg.scrape.min_text_chars:
                digest = hashlib.sha1(page.text.encode()).hexdigest()
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                texts.append(page.text)
                urls.append(url)
                titles.append(ptitle or page.title)
            elif page.links and depth < 2:
                # A sub-index: its pages become parts of this work.
                queue = [(urljoin(url, h), t, depth + 1) for h, t in page.links] + queue
            elif page.text:
                texts.append(page.text)
                urls.append(url)
                titles.append(ptitle or page.title)
            else:
                print(f"    warning: empty page {url}", file=sys.stderr)

        if not texts:
            continue
        text = normalize_text("\n\n".join(texts), cfg.normalize)
        if not text:
            continue
        records.append(
            {
                "work_id": _work_id(author, work.title),
                "author": author,
                "register": register,
                "title": work.title,
                "source_url": urls[0],
                "source_urls": urls,
                "part_titles": titles,
                "text": text,
                "n_chars": len(text),
                "n_words": len(text.split()),
            }
        )

    # Some "author pages" are themselves the complete text (e.g. Catullus).
    if not records:
        page = clean_html(html)
        if len(page.text) >= cfg.scrape.min_text_chars:
            digest = hashlib.sha1(page.text.encode()).hexdigest()
            if digest not in seen_hashes:
                seen_hashes.add(digest)
                text = normalize_text(page.text, cfg.normalize)
                if text:
                    records.append(
                        {
                            "work_id": _work_id(author, page.title or author),
                            "author": author,
                            "register": register,
                            "title": page.title or author,
                            "source_url": index_url,
                            "source_urls": [index_url],
                            "part_titles": [page.title or author],
                            "text": text,
                            "n_chars": len(text),
                            "n_words": len(text.split()),
                        }
                    )
    return records


def _agg(records: list[dict], key: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in records:
        s = out.setdefault(r.get(key, "?"), {"works": 0, "words": 0, "chars": 0})
        s["works"] += 1
        s["words"] += r["n_words"]
        s["chars"] += r["n_chars"]
    return out


def _table(groups: dict[str, dict], label: str) -> list[str]:
    lines = [f"{label:<28}{'works':>7}{'words':>12}{'chars':>14}", "-" * 61]
    for name, s in groups.items():
        lines.append(f"{name[:27]:<28}{s['works']:>7}{s['words']:>12,}{s['chars']:>14,}")
    lines.append("-" * 61)
    total = {k: sum(s[k] for s in groups.values()) for k in ("works", "words", "chars")}
    lines.append(f"{'TOTAL':<28}{total['works']:>7}{total['words']:>12,}{total['chars']:>14,}")
    return lines


def report_stats(records: list[dict]) -> str:
    """Full per-author table (for stats.txt); register summary first."""
    lines = _table(_agg(records, "register"), "register")
    lines.append("")
    lines += _table(_agg(records, "author"), "author")
    return "\n".join(lines)


def report_summary(records: list[dict]) -> str:
    """Register table plus the largest authors, for the console."""
    lines = _table(_agg(records, "register"), "register")
    authors = _agg(records, "author")
    top = sorted(authors.items(), key=lambda kv: -kv[1]["words"])[:15]
    lines.append("")
    lines += _table(dict(top), f"top 15 of {len(authors)} authors")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scrape The Latin Library into a JSONL corpus")
    ap.add_argument("--config", default="configs/default.toml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    fetcher = CachingFetcher(
        base_url=cfg.scrape.base_url,
        user_agent=cfg.scrape.user_agent,
        delay=cfg.scrape.delay_seconds,
        cache_dir=Path(cfg.scrape.cache_dir),
    )

    all_records: list[dict] = []
    assigned: set[str] = set()
    seen_hashes: set[str] = set()
    try:
        targets: list[tuple[str, str, str]] = [
            (author, urljoin(cfg.scrape.base_url + "/", page), "classical")
            for author, page in cfg.scrape.authors.items()
        ]
        if cfg.scrape.discover:
            known = {url for _, url, _ in targets}
            discovered = discover_authors(fetcher, cfg)
            targets += [t for t in discovered if t[1] not in known]
            print(f"discovered {len(targets)} author pages", flush=True)
        for author, index_url, register in targets:
            records = scrape_author(author, index_url, fetcher, cfg, assigned,
                                    seen_hashes, register)
            words = sum(r["n_words"] for r in records)
            print(f"[{register}/{author}] {len(records)} works, {words:,} words", flush=True)
            all_records.extend(records)
    except ScrapeBlockedError as exc:
        print(f"\nSTOPPING: {exc}", file=sys.stderr)
        print(
            "Fallback: the Perseus canonical-latinLit repository on GitHub "
            "(TEI XML) has the same authors.",
            file=sys.stderr,
        )
        return 2

    out_path = Path(cfg.scrape.corpus_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + report_summary(all_records))
    (out_path.parent / "stats.txt").write_text(report_stats(all_records), encoding="utf-8")
    print(
        f"\nwrote {len(all_records)} works to {out_path} "
        f"({fetcher.fetched} pages fetched, {fetcher.cache_hits} cache hits)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
