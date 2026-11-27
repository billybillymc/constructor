"""Swappable tokenizers behind one small interface.

Two implementations:
  * ``BPETokenizer`` — byte-level BPE trained on the *training split only*
    (Hugging Face ``tokenizers``), vocab ~8000. Default.
  * ``CharTokenizer`` — character-level, vocab derived from the training split.

Both guarantee encode->decode round-trips the original text for text drawn
from the training distribution (byte-level BPE round-trips anything; the char
tokenizer maps unseen characters to a visible replacement and is tested on
training text).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .config import TokenizerConfig


class LatinTokenizer(ABC):
    """The interface the rest of the pipeline sees."""

    eot_token: str

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def eot_id(self) -> int: ...

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str: ...

    @abstractmethod
    def save(self, directory: Path) -> None: ...

    @classmethod
    @abstractmethod
    def train(cls, texts: list[str], cfg: TokenizerConfig) -> "LatinTokenizer": ...

    @classmethod
    @abstractmethod
    def load(cls, directory: Path) -> "LatinTokenizer": ...


class BPETokenizer(LatinTokenizer):
    FILENAME = "bpe_tokenizer.json"

    def __init__(self, tok: Tokenizer, eot_token: str):
        self._tok = tok
        self.eot_token = eot_token

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def eot_id(self) -> int:
        return self._tok.token_to_id(self.eot_token)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(directory / self.FILENAME))
        (directory / "tokenizer_meta.json").write_text(
            json.dumps({"kind": "bpe", "eot_token": self.eot_token}), encoding="utf-8"
        )

    @classmethod
    def train(cls, texts: list[str], cfg: TokenizerConfig) -> "BPETokenizer":
        tok = Tokenizer(models.BPE(unk_token=None))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=cfg.vocab_size,
            special_tokens=[cfg.eot_token],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tok.train_from_iterator(texts, trainer=trainer)
        return cls(tok, cfg.eot_token)

    @classmethod
    def load(cls, directory: Path) -> "BPETokenizer":
        meta = json.loads((directory / "tokenizer_meta.json").read_text(encoding="utf-8"))
        tok = Tokenizer.from_file(str(directory / cls.FILENAME))
        return cls(tok, meta["eot_token"])


class CharTokenizer(LatinTokenizer):
    FILENAME = "char_vocab.json"
    UNK = "�"  # unseen characters decode to the replacement character

    def __init__(self, chars: list[str], eot_token: str):
        self.eot_token = eot_token
        # id 0 = EOT, id 1 = UNK, then the character inventory.
        self._itos = [eot_token, self.UNK] + chars
        self._stoi = {ch: i for i, ch in enumerate(self._itos)}

    @property
    def vocab_size(self) -> int:
        return len(self._itos)

    @property
    def eot_id(self) -> int:
        return 0

    def encode(self, text: str) -> list[int]:
        unk = self._stoi[self.UNK]
        return [self._stoi.get(ch, unk) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._itos[i] for i in ids)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / self.FILENAME).write_text(
            json.dumps({"chars": self._itos[2:], "eot_token": self.eot_token}),
            encoding="utf-8",
        )
        (directory / "tokenizer_meta.json").write_text(
            json.dumps({"kind": "char", "eot_token": self.eot_token}), encoding="utf-8"
        )

    @classmethod
    def train(cls, texts: list[str], cfg: TokenizerConfig) -> "CharTokenizer":
        chars = sorted(set().union(*(set(t) for t in texts)))
        chars = [c for c in chars if c != cls.UNK]
        return cls(chars, cfg.eot_token)

    @classmethod
    def load(cls, directory: Path) -> "CharTokenizer":
        data = json.loads((directory / cls.FILENAME).read_text(encoding="utf-8"))
        return cls(data["chars"], data["eot_token"])


_KINDS: dict[str, type[LatinTokenizer]] = {"bpe": BPETokenizer, "char": CharTokenizer}


def train_tokenizer(texts: list[str], cfg: TokenizerConfig) -> LatinTokenizer:
    return _KINDS[cfg.kind].train(texts, cfg)


def load_tokenizer(directory: Path) -> LatinTokenizer:
    meta = json.loads((directory / "tokenizer_meta.json").read_text(encoding="utf-8"))
    return _KINDS[meta["kind"]].load(directory)
