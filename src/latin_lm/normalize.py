"""Configurable text normalization.

Normalization is a linguistic choice, so every step is controlled by
``[normalize]`` in the config. Defaults: Unicode NFC, macrons/breves
stripped, case kept, u/v and i/j left as the source has them.
"""

from __future__ import annotations

import re
import unicodedata

from .config import NormalizeConfig

_COMBINING_MACRON = "̄"
_COMBINING_BREVE = "̆"

_SPACES_RE = re.compile(r"[ \t ]+")
_BLANKS_RE = re.compile(r"\n{3,}")

# Some source pages are Windows-1252 mislabeled as UTF-8/Latin-1, leaving C1
# control codepoints (U+0080-U+009F) where em-dashes, quotes etc. belong.
_C1_MAP = {}
for _b in range(0x80, 0xA0):
    try:
        _C1_MAP[_b] = bytes([_b]).decode("cp1252")
    except UnicodeDecodeError:
        _C1_MAP[_b] = None  # undefined in cp1252: drop


def _fix_mojibake(text: str) -> str:
    return text.translate(_C1_MAP)


def normalize_text(text: str, cfg: NormalizeConfig) -> str:
    text = _fix_mojibake(text)
    # Work in NFD so precomposed characters (ā, ĕ, ...) expose their diacritics.
    text = unicodedata.normalize("NFD", text)

    if cfg.strip_macrons:
        text = text.replace(_COMBINING_MACRON, "").replace(_COMBINING_BREVE, "")

    if cfg.fold_uv:
        text = text.replace("v", "u").replace("V", "U")
    if cfg.fold_ij:
        text = text.replace("j", "i").replace("J", "I")
    if cfg.lowercase:
        text = text.lower()

    if cfg.collapse_whitespace:
        text = _SPACES_RE.sub(" ", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
        text = _BLANKS_RE.sub("\n\n", text)

    if cfg.unicode_form.lower() != "none":
        text = unicodedata.normalize(cfg.unicode_form, text)
    return text.strip()
