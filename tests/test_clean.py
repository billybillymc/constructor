from latin_lm.clean import clean_html

PROSE_PAGE = b"""
<html><head><title> Caesar: Bellum Gallicum I </title>
<link rel="StyleSheet" href="latinlibrary.css"></head>
<body>
<h1>C. IVLI CAESARIS COMMENTARIORVM DE BELLO GALLICO LIBER PRIMVS</h1>
<h2 class="date">(100 &#8211; 44 B.C.)</h2>
<p class="internal_navigation">
<a href="#1">1</a> <a href="#2">2</a> <a href="#3">3</a>
</p>
<p>[<a name="1">1</a>] <font size=2>1</font> Gallia est omnis divisa in partes tres,
quarum unam incolunt Belgae. <font size=2>2</font> Hi omnes lingua inter se differunt.</p>
<p>[<a name="2">2</a>] Apud Helvetios longe nobilissimus fuit Orgetorix.</p>
<div class="footer">
<p><a href="/caes.html">Caesar</a> <a href="/index.html">The Latin Library</a>
<a href="/classics.html">The Classics Page</a></p>
</div>
</body></html>
"""

VERSE_PAGE = b"""
<html><head><title>Vergil: Aeneid I</title></head><body>
<h1>AENEIDOS LIBER PRIMVS</h1>
<p>Arma virumque cano, Troiae qui primus ab oris<br>
Italiam, fato profugus, Laviniaque venit<br>
litora, multum ille et terris iactatus et alto&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5<br>
vi superum saevae memorem Iunonis ob iram;</p>
<div class="footer"><p><a href="/verg.html">Vergil</a></p></div>
</body></html>
"""

INDEX_PAGE = b"""
<html><head><title>Cicero: In Verrem</title></head><body>
<h1>IN VERREM</h1>
<p><a href="verres.1.shtml">Actio Prima</a></p>
<p><a href="verres.2.1.shtml">Actio Secunda Liber I</a></p>
<div class="footer"><p><a href="/index.html">The Latin Library</a></p></div>
</body></html>
"""


def test_prose_cleaning():
    page = clean_html(PROSE_PAGE)
    assert page.title == "Caesar: Bellum Gallicum I"
    assert "Gallia est omnis divisa" in page.text
    assert "Orgetorix" in page.text


def test_section_and_sentence_numbers_removed():
    text = clean_html(PROSE_PAGE).text
    assert "[1]" not in text and "[2]" not in text
    # inline <font size=2> sentence numbers gone
    assert "1 Gallia" not in text
    assert "2 Hi omnes" not in text


def test_navigation_and_footer_removed():
    text = clean_html(PROSE_PAGE).text
    assert "The Latin Library" not in text
    assert "Classics" not in text
    # the internal navigation digit row is gone
    assert not text.startswith("1 2 3")


def test_headings_not_in_text():
    text = clean_html(PROSE_PAGE).text
    assert "COMMENTARIORVM" not in text
    assert "B.C." not in text


def test_verse_line_numbers_removed():
    text = clean_html(VERSE_PAGE).text
    assert "Arma virumque cano" in text
    assert "alto" in text
    # the trailing line number 5 is stripped
    assert "alto 5" not in text and "alto 5" not in text
    # verse structure (one line per verse) survives
    assert "oris\nItaliam" in text


NAV_LINK_PAGE = b"""
<html><head><title>Seneca: Epistulae I</title></head><body>
<p>SENECA LVCILIO SVO SALVTEM. Ita fac, mi Lucili: vindica te tibi,
et tempus quod adhuc aut auferebatur aut subripiebatur aut excidebat
collige et serva.</p>
<p><a href="/sen.html">Seneca the Younger</a></p>
<p>Interrogabis fortasse quid ego faciam qui tibi ista praecipio
(this word in the text is corrupt: it has been conjectured by the editors).</p>
</body></html>
"""


def test_link_only_paragraphs_removed():
    text = clean_html(NAV_LINK_PAGE).text
    assert "vindica te tibi" in text
    assert "Seneca the Younger" not in text


def test_english_editorial_lines_removed():
    text = clean_html(NAV_LINK_PAGE).text
    # the Latin line around the dropped English parenthetical is also one line;
    # the all-English editorial line in it trips the 3-marker filter
    assert "conjectured" not in text


def test_double_escaped_entities_unescaped():
    page = b"<html><body><p>susceperat, &amp;lt;cum&amp;gt; minus etiam praecipitur</p></body></html>"
    text = clean_html(page).text
    assert "<cum>" in text
    assert "&lt;" not in text


def test_index_page_links_extracted_and_footer_links_excluded():
    page = clean_html(INDEX_PAGE)
    hrefs = [h for h, _ in page.links]
    assert "verres.1.shtml" in hrefs
    assert "verres.2.1.shtml" in hrefs
    assert all("index.html" not in h for h in hrefs)
    # an index page has very little text
    assert len(page.text) < 100
