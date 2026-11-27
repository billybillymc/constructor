import pytest

from latin_lm.config import TokenizerConfig
from latin_lm.tokenizer import (
    BPETokenizer,
    CharTokenizer,
    load_tokenizer,
    train_tokenizer,
)

LATIN = [
    "Gallia est omnis divisa in partes tres, quarum unam incolunt Belgae.",
    "Arma virumque cano, Troiae qui primus ab oris\nItaliam fato profugus venit.",
    "Ceterum censeo Carthaginem esse delendam. Quousque tandem abutere, Catilina?",
]


@pytest.fixture(scope="module")
def bpe():
    return train_tokenizer(LATIN * 50, TokenizerConfig(kind="bpe", vocab_size=500))


@pytest.fixture(scope="module")
def char():
    return train_tokenizer(LATIN, TokenizerConfig(kind="char"))


def test_bpe_roundtrip(bpe):
    for text in LATIN:
        assert bpe.decode(bpe.encode(text)) == text


def test_bpe_roundtrip_unseen_chars(bpe):
    # byte-level BPE round-trips text outside the training distribution too
    text = "zephyrus (Graece) — «mirum»"
    assert bpe.decode(bpe.encode(text)) == text


def test_bpe_has_eot(bpe):
    assert bpe.eot_id is not None
    assert bpe.decode([bpe.eot_id]) == bpe.eot_token


def test_char_roundtrip(char):
    for text in LATIN:
        assert char.decode(char.encode(text)) == text


def test_char_unseen_maps_to_replacement(char):
    out = char.decode(char.encode("Gallia 中"))
    assert out == "Gallia �"


def test_save_load_roundtrip(tmp_path, bpe, char):
    for tok, cls in ((bpe, BPETokenizer), (char, CharTokenizer)):
        d = tmp_path / cls.__name__
        tok.save(d)
        loaded = load_tokenizer(d)
        assert isinstance(loaded, cls)
        for text in LATIN:
            assert loaded.decode(loaded.encode(text)) == tok.decode(tok.encode(text))
        assert loaded.vocab_size == tok.vocab_size
        assert loaded.eot_id == tok.eot_id
