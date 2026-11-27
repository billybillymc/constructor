"""Turn raw Latin Library HTML into clean Latin text.

This module is deliberately self-contained and side-effect free so the
cleaning decisions can be unit-tested and revisited without touching the
scraper. The Latin Library's pages follow one template:

    <h1>WORK TITLE</h1>
    <h2 class="date">(dates)</h2>
    <p class="internal_navigation">1 2 3 ...</p>
    <p>[<a name="1">1</a>] <font size=2>1</font> Gallia est omnis ...</p>
    ...
    <div class="footer"> links </div>

Cleaning removes: navigation/footer/heading chrome, `[N]` section anchors,
inline <font> chapter/sentence numbers, trailing verse line numbers, and
leftover English boilerplate lines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag

# Classes whose elements are site chrome, never text.
_CHROME_CLASSES = {"internal_navigation", "pagehead", "footer", "date", "navigation"}
_CHROME_TAGS = {"script", "style", "head", "title"}

# Lines that are English site boilerplate if they survive tag removal.
_BOILERPLATE_RE = re.compile(
    r"^\s*(the\s+latin\s+library|the\s+classics\s+page|classics\s+homepage"
    r"|the\s+miscellany|christian\s+latin|medieval\s+latin|neo-latin"
    r"|latin\s+library\s+homepage)\s*$",
    re.IGNORECASE,
)

# "[1]" style section markers left behind after <a name=...> anchors are removed,
# and bare "[12]" markers that were plain text to begin with.
_BRACKET_NUM_RE = re.compile(r"\[\s*([0-9]{0,4}|[ivxlcdm]{0,8})\s*\]", re.IGNORECASE)

# A verse line number: 2+ spaces (incl. &nbsp;) or a tab, then a short number at line end.
_VERSE_LINENO_RE = re.compile(r"(?:[ \t ]{2,}|\t)\d{1,4}[ \t ]*$")

# A line that is nothing but a number (chapter headings in some texts).
_BARE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$")

# Function words that occur in English but not in Latin (careful: "in", "is",
# "his", "has", "a" are all real Latin words). A line with 3+ hits is English
# editorial apparatus, not text.
_ENGLISH_MARKERS_RE = re.compile(
    r"\b(the|of|and|which|with|from|this|been|were|was|are|by|to|on"
    r"|translated|edited|copyright|conjectured)\b",
    re.IGNORECASE,
)

_WS_RE = re.compile(r"[ \t ]+")


@dataclass
class CleanedPage:
    title: str
    text: str
    links: list[tuple[str, str]] = field(default_factory=list)  # (href, link text)

    @property
    def is_text_page(self) -> bool:
        # decided by the caller via a length threshold; kept for clarity
        return bool(self.text)


def _is_chrome(tag: Tag) -> bool:
    classes = set(tag.get("class") or [])
    return bool(classes & _CHROME_CLASSES)


def _font_is_number(tag: Tag) -> bool:
    """Inline <font size=2>7</font> chapter/sentence numbers."""
    text = tag.get_text(strip=True)
    return bool(re.fullmatch(r"[0-9]{1,4}|[ivxlcdmIVXLCDM]{1,8}", text)) and len(text) <= 8


def extract_links(soup: BeautifulSoup) -> list[tuple[str, str]]:
    """All same-site content links (href, text), in document order.

    Footer/navigation links are excluded because those elements are removed
    before this runs.
    """
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("mailto:"):
            continue
        low = href.lower()
        if low.endswith((".css", ".ico", ".jpg", ".png", ".gif")):
            continue
        if low.startswith(("http://", "https://")) and "thelatinlibrary.com" not in low:
            continue
        # Site-wide nav targets, not content.
        tail = low.split("/")[-1]
        if tail in {"index.html", "classics.html", "christian.html", "medieval.html",
                    "neo.html", "misc.html", "ius.html"}:
            continue
        text = " ".join(a.get_text(" ", strip=True).split())
        links.append((href, text))
    return links


def clean_html(html: bytes | str) -> CleanedPage:
    """Parse one Latin Library page into (title, clean text, content links)."""
    soup = BeautifulSoup(html, "lxml")

    title = ""
    if soup.title is not None:
        title = " ".join(soup.title.get_text(" ", strip=True).split())

    for tag in soup.find_all(list(_CHROME_TAGS)):
        tag.decompose()
    for tag in soup.find_all(True, class_=lambda c: c and (set([c] if isinstance(c, str) else c) & _CHROME_CLASSES)):
        tag.decompose()

    # Collect links *after* chrome removal so footer links don't count.
    links = extract_links(soup)

    # Paragraphs/cells that are nothing but links are navigation (e.g. a bare
    # "Seneca the Younger" link between letters), not text.
    for block in soup.find_all(["p", "td", "li"]):
        if block.attrs is None:
            continue
        anchors = block.find_all("a", href=True)
        if anchors:
            link_text = "".join(a.get_text(strip=True) for a in anchors)
            if len(block.get_text(strip=True)) <= len(link_text) + 5:
                block.decompose()

    # Headings are metadata (the title is kept separately), not running text.
    for tag in soup.find_all(["h1", "h2", "h3"]):
        tag.decompose()

    # Section anchors [<a name="1">1</a>] -> drop the anchor, brackets cleaned later.
    for a in soup.find_all("a"):
        if a.attrs is None:  # already destroyed along with a decomposed ancestor
            continue
        if a.get("name") and not a.get("href"):
            a.decompose()
        else:
            a.unwrap()  # keep link text (rare inline links), drop the tag

    # Inline chapter/sentence numbers.
    for f in soup.find_all("font"):
        if f.attrs is None:
            continue
        if _font_is_number(f):
            f.decompose()
        else:
            f.unwrap()
    for sup in soup.find_all(["sup", "sub"]):
        sup.decompose()

    # <br> separates verse lines; make that explicit before text extraction.
    for br in soup.find_all("br"):
        br.replace_with("\n")

    body = soup.body if soup.body is not None else soup
    blocks: list[str] = []
    paragraphs = body.find_all(["p", "blockquote", "td", "li"]) or [body]
    seen_block_text: set[int] = set()
    for p in paragraphs:
        # Skip containers whose text we'd double-count via nested blocks.
        if p.find(["p", "blockquote", "td", "li"]) is not None:
            continue
        if id(p) in seen_block_text:
            continue
        seen_block_text.add(id(p))
        raw = p.get_text()
        block = _clean_block(raw)
        if block:
            blocks.append(block)

    text = "\n\n".join(blocks)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return CleanedPage(title=title, text=text, links=links)


def _clean_block(raw: str) -> str:
    """Clean one paragraph/verse block of extracted text."""
    # Undo double-escaped entities left in the source (&amp;lt; parsed to &lt;).
    raw = raw.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    raw = raw.replace("&amp;", "&")
    # Hand-typed broken entities that occur in the source ("stultos&t;").
    raw = raw.replace("&gtr;", ">").replace("&t;", ">")
    lines_out: list[str] = []
    for line in raw.split("\n"):
        line = _BRACKET_NUM_RE.sub(" ", line)
        line = _VERSE_LINENO_RE.sub("", line)
        line = _WS_RE.sub(" ", line).strip()
        if not line:
            continue
        if _BOILERPLATE_RE.match(line):
            continue
        if _BARE_NUM_RE.match(line):
            continue
        if len(_ENGLISH_MARKERS_RE.findall(line)) >= 3:
            continue  # an English editorial/translation line, not Latin text
        lines_out.append(line)
    return "\n".join(lines_out)
