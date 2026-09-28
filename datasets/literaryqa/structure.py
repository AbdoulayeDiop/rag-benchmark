"""Structure-preserving variant of the vendored LiteraryQA text extraction.

`vendor/literaryqa/clean.py` flattens a Gutenberg book into plain lines, which
loses the heading levels needed to compare structure-aware chunking against flat
text. `extract_blocks` below is that function's `extract_raw_text` with a single
change: it yields (tag name, text) pairs instead of joining the text away. Every
soup transformation and regex is copied unchanged, and `load.py` asserts that
re-joining the blocks reproduces the vendored output byte for byte, so the plain
`.txt` rendering cannot drift from upstream.

Keeping this separate from load.py makes the adapted code easy to diff against
vendor/literaryqa/clean.py when the pin is updated.
"""

import re

from bs4 import BeautifulSoup
from bs4.element import NavigableString

from literaryqa.clean import (
    GUTENBERG_PRODUCTION_PATTERNS,
    _keep_alt_img_text,
    _keep_songs,
    _keep_span_margin_left,
    _remove_sidebar,
)

ALLOWED_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "pre"}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


def extract_blocks(html_content, **kwargs):
    """Extract (tag, text) blocks from Gutenberg HTML.

    Mirrors `literaryqa.clean.extract_raw_text`; see this module's docstring.
    """
    table = {
        '<div class="stage-direction center">': '<div class="stage-direction">',
    }
    html_content = html_content.translate(table)
    soup = BeautifulSoup(html_content, "html5lib")

    if kwargs.get("keep_alt_img_text", True):
        _keep_alt_img_text(soup)

    if kwargs.get("remove_img", True):
        for class_name in ["tnote", "transnote", "covernote"]:
            for div in soup.find_all("div", class_=class_name):
                div.decompose()

    if kwargs.get("remove_tn", True):
        for class_name in ["footnote", "footnotes"]:
            for div in soup.find_all("div", class_=class_name):
                div.decompose()

    if kwargs.get("remove_pagenum", True):
        for class_name in ["pagenum", "ns", "pageno"]:
            for span in soup.find_all("span", class_=class_name):
                span.decompose()

    if kwargs.get("remove_citation", True):
        for a_tag in soup.find_all("a", class_="citation"):
            a_tag.decompose()

    if kwargs.get("remove_links", True):
        for a_tag in soup.find_all("a", href=True):
            a_tag.decompose()

    if kwargs.get("remove_sidebar", True):
        _remove_sidebar(soup)

    if kwargs.get("keep_dropcap", True):
        for div in soup.find_all("div", class_="drop-cap"):
            div.name = "p"
        for tag in soup.find_all("div", class_="center"):
            tag.unwrap()

    if kwargs.get("keep_span_margin_left", True):
        _keep_span_margin_left(soup)

    if kwargs.get("keep_poem", True):
        for poem_div in soup.find_all("div", class_="poem"):
            for stanza_div in poem_div.find_all("div", class_="stanza"):
                for span in stanza_div.find_all("span"):
                    span.unwrap()
                for br in stanza_div.find_all("br"):
                    br.replace_with("\n")
                stanza_div.unwrap()

            for p in poem_div.find_all("p"):
                p.insert_after(soup.new_tag("br"))
                p.unwrap()

            poem_div.name = "pre"
            del poem_div["class"]

    if kwargs.get("keep_stage_dir", True):
        for div in soup.find_all("div", class_="stage-direction"):
            div.name = "p"

    if kwargs.get("keep_scene_desc", True):
        for div in soup.find_all("div", class_="scene-description"):
            div.name = "p"

    if kwargs.get("keep_songs", True):
        _keep_songs(soup)

    for div_id in ["notes", "footnotes", "linenotes"]:
        for tag in soup.find_all("div", id=div_id):
            tag.decompose()

    for p in soup.find_all("p", class_="hang"):
        p.decompose()

    blocks = []
    for tag in soup.find_all(ALLOWED_TAGS):
        if tag.name == "pre":
            text = tag.get_text(separator="\n", strip=True)
        else:
            text = tag.get_text(separator=" ", strip=True)

        if kwargs.get("remove_pagenum", True):
            text = re.sub(r"\[(Pg|Page)\s*\d+\]", " ", text, flags=re.IGNORECASE)
            text = re.sub(r"\[p\s*\d+\s*\]", " ", text, flags=re.IGNORECASE)
            text = re.sub(
                r"^p\.\s+\d+:.*$", " ", text,
                flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
            )

        if kwargs.get("remove_citation", True):
            text = re.sub(r"\[\d+\]", " ", text)

        if kwargs.get("remove_footnotes", True):
            text = re.sub(r"\[[ivxlcm]+\]", " ", text)  # only lowercase!

        if kwargs.get("remove_transnotes", True):
            text = re.sub(
                r"\[transcriber.*s note.*\]", " ", text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            text = re.sub(
                r"^transcriber.*s note[s]?:?", " ", text,
                flags=re.IGNORECASE | re.MULTILINE,
            )

        if kwargs.get("normalize_whitespace", True):
            text = re.sub(r"\s+([?!.,:;])", r"\1", text)
            text = re.sub(r"\(\s+", "(", text)
            text = re.sub(r"\s+\)", ")", text)
            text = text.replace(" ( ) ", "")
            if tag.name != "pre":
                text = text.replace("\n", " ")
                text = " ".join(text.split())

        if not text or (
            kwargs.get("remove_pagenum", True)
            and re.search(r"^p. \d+:", text, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
        ):
            continue

        if kwargs.get("remove_gutenberg_preface", True):
            for pattern in GUTENBERG_PRODUCTION_PATTERNS:
                text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

        blocks.append((tag.name, text))

    return blocks
