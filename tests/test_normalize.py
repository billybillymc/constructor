import unicodedata

from latin_lm.config import NormalizeConfig
from latin_lm.normalize import normalize_text


def test_macrons_stripped_by_default():
    cfg = NormalizeConfig()
    assert normalize_text("Rōma āra", cfg) == "Roma ara"  # Rōma āra
    # decomposed input too
    assert normalize_text("Rōma", cfg) == "Roma"
    # breves as well
    assert normalize_text("fĕrō", cfg) == "fero"


def test_case_kept_by_default():
    cfg = NormalizeConfig()
    assert normalize_text("Gallia EST omnis", cfg) == "Gallia EST omnis"


def test_uv_ij_left_alone_by_default():
    cfg = NormalizeConfig()
    assert normalize_text("veni vidi vici, Iulius", cfg) == "veni vidi vici, Iulius"


def test_uv_folding_when_enabled():
    cfg = NormalizeConfig(fold_uv=True)
    assert normalize_text("veni Vidi", cfg) == "ueni Uidi"


def test_ij_folding_when_enabled():
    cfg = NormalizeConfig(fold_ij=True)
    assert normalize_text("Julius jam", cfg) == "Iulius iam"


def test_lowercase_when_enabled():
    cfg = NormalizeConfig(lowercase=True)
    assert normalize_text("Gallia EST", cfg) == "gallia est"


def test_output_is_nfc():
    cfg = NormalizeConfig(strip_macrons=False)
    out = normalize_text("Rōma", cfg)  # decomposed macron, kept
    assert out == unicodedata.normalize("NFC", out)
    assert out == "Rōma"


def test_cp1252_mojibake_repaired():
    cfg = NormalizeConfig()
    # U+0097 is a C1 control standing in for a cp1252 em-dash
    assert normalize_text("credetur?\u0097Sed", cfg) == "credetur?—Sed"
    # undefined cp1252 bytes are dropped
    assert normalize_text("ab\u0081cd", cfg) == "abcd"


def test_whitespace_collapsed_but_lines_kept():
    cfg = NormalizeConfig()
    out = normalize_text("arma  virumque\t cano\nItaliam   fato\n\n\n\nlitora", cfg)
    assert out == "arma virumque cano\nItaliam fato\n\nlitora"
