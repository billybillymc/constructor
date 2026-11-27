"""Load the single TOML config file into plain dataclasses.

Every entry point takes ``--config path/to/file.toml`` (default
``configs/default.toml``) so experiments can swap configs wholesale.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class ScrapeConfig:
    base_url: str = "https://thelatinlibrary.com"
    user_agent: str = "LatinCorpusResearchBot/0.1"
    delay_seconds: float = 1.0
    cache_dir: str = "data/raw_html"
    corpus_path: str = "data/corpus/works.jsonl"
    min_text_chars: int = 600
    authors: dict[str, str] = field(default_factory=dict)
    # Path-prefix fixes for stale links on index pages, e.g. "gallic/" -> "caesar/".
    url_rewrites: dict[str, str] = field(default_factory=dict)
    # Site-wide author discovery from register index pages.
    discover: bool = False
    site_indexes: dict[str, str] = field(default_factory=dict)  # register -> page
    discover_exclude: list[str] = field(default_factory=list)   # href prefixes to skip


@dataclass
class NormalizeConfig:
    unicode_form: str = "NFC"
    strip_macrons: bool = True
    lowercase: bool = False
    fold_uv: bool = False
    fold_ij: bool = False
    collapse_whitespace: bool = True


@dataclass
class SplitConfig:
    seed: int = 1457
    train_frac: float = 0.90
    val_frac: float = 0.05
    test_frac: float = 0.05
    big_work_frac: float = 0.02
    manifest_path: str = "data/processed/split_manifest.json"


@dataclass
class TokenizerConfig:
    kind: str = "bpe"
    vocab_size: int = 8000
    out_dir: str = "data/processed"
    eot_token: str = "<|endoftext|>"


@dataclass
class Config:
    """The shared data-pipeline config. Model/training hyperparameters live
    in each experiment's own config.toml under experiments/."""

    scrape: ScrapeConfig = field(default_factory=ScrapeConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    source_path: str = ""  # where this config was loaded from


def _build(cls: type, data: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path = "configs/default.toml") -> Config:
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)
    cfg = Config(
        scrape=_build(ScrapeConfig, raw.get("scrape", {})),
        normalize=_build(NormalizeConfig, raw.get("normalize", {})),
        split=_build(SplitConfig, raw.get("split", {})),
        tokenizer=_build(TokenizerConfig, raw.get("tokenizer", {})),
        source_path=str(path),
    )
    total = cfg.split.train_frac + cfg.split.val_frac + cfg.split.test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions sum to {total}, expected 1.0")
    return cfg
