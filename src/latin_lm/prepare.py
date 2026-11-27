"""Stage 2: split the corpus by work, train the tokenizer, encode to binary.

Splitting is by *work*, never by line or chunk: a work's full text lands in
exactly one split. Works larger than ``big_work_frac`` of the corpus are
forced into train (a single giant work like Livy's Ab Urbe Condita would
otherwise swallow val or test); the remaining works are shuffled with a fixed
seed, then test and val are filled to their target size in characters and the
rest is train. The assignment is written to a manifest on disk.

The tokenizer is trained on the training split only. Each split is encoded to
a flat binary file of token ids (uint16/uint32) that np.memmap can open, with
an <|endoftext|> token between works.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from .config import Config, load_config
from .tokenizer import LatinTokenizer, train_tokenizer


def load_corpus(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise SystemExit(f"no records in {path}; run `scrape` first")
    return records


def split_works(records: list[dict], cfg: Config) -> dict[str, str]:
    """Return {work_id: split}, deterministic for a given seed and corpus."""
    sp = cfg.split
    total_chars = sum(r["n_chars"] for r in records)
    big_cutoff = sp.big_work_frac * total_chars

    forced_train = [r for r in records if r["n_chars"] > big_cutoff]
    rest = [r for r in records if r["n_chars"] <= big_cutoff]
    rng = random.Random(sp.seed)
    rest = sorted(rest, key=lambda r: r["work_id"])  # order independent of scrape order
    rng.shuffle(rest)

    assignment: dict[str, str] = {r["work_id"]: "train" for r in forced_train}
    val_target = sp.val_frac * total_chars
    test_target = sp.test_frac * total_chars
    val_chars = test_chars = 0
    for r in rest:
        if test_chars < test_target:
            assignment[r["work_id"]] = "test"
            test_chars += r["n_chars"]
        elif val_chars < val_target:
            assignment[r["work_id"]] = "val"
            val_chars += r["n_chars"]
        else:
            assignment[r["work_id"]] = "train"
    return assignment


def write_manifest(records: list[dict], assignment: dict[str, str], cfg: Config) -> dict:
    sp = cfg.split
    by_split: dict[str, dict] = {s: {"works": 0, "chars": 0, "words": 0} for s in ("train", "val", "test")}
    entries = []
    for r in records:
        s = assignment[r["work_id"]]
        by_split[s]["works"] += 1
        by_split[s]["chars"] += r["n_chars"]
        by_split[s]["words"] += r["n_words"]
        entries.append(
            {"work_id": r["work_id"], "author": r["author"], "title": r["title"],
             "split": s, "n_chars": r["n_chars"], "n_words": r["n_words"]}
        )
    total_chars = sum(r["n_chars"] for r in records)
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": sp.seed,
        "target_fracs": {"train": sp.train_frac, "val": sp.val_frac, "test": sp.test_frac},
        "big_work_frac": sp.big_work_frac,
        "actual_char_fracs": {s: by_split[s]["chars"] / total_chars for s in by_split},
        "totals": by_split,
        "works": entries,
    }
    path = Path(sp.manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def encode_split(
    records: list[dict], tokenizer: LatinTokenizer, out_path: Path
) -> dict:
    """Encode works (EOT-separated) to a flat binary file of token ids."""
    dtype = np.uint16 if tokenizer.vocab_size < 2**16 else np.uint32
    ids: list[int] = []
    eot = tokenizer.eot_id
    for r in records:
        ids.append(eot)
        ids.extend(tokenizer.encode(r["text"]))
    ids.append(eot)
    arr = np.array(ids, dtype=dtype)
    arr.tofile(out_path)
    return {"path": out_path.name, "dtype": str(np.dtype(dtype)), "n_tokens": int(arr.size)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Split, tokenize, and encode the corpus")
    ap.add_argument("--config", default="configs/default.toml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    records = load_corpus(Path(cfg.scrape.corpus_path))
    assignment = split_works(records, cfg)
    manifest = write_manifest(records, assignment, cfg)
    print("split sizes (chars):", json.dumps(manifest["actual_char_fracs"], indent=0))

    splits = {s: [r for r in records if assignment[r["work_id"]] == s]
              for s in ("train", "val", "test")}
    for name, recs in splits.items():
        if not recs:
            raise SystemExit(f"split '{name}' is empty; adjust fractions or corpus size")

    out_dir = Path(cfg.tokenizer.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"training {cfg.tokenizer.kind} tokenizer on the training split only ...")
    tokenizer = train_tokenizer([r["text"] for r in splits["train"]], cfg.tokenizer)
    tokenizer.save(out_dir)
    print(f"vocab size: {tokenizer.vocab_size}")

    # Round-trip check on real corpus text before encoding anything.
    probe = splits["train"][0]["text"][:5000]
    if tokenizer.decode(tokenizer.encode(probe)) != probe:
        raise SystemExit("tokenizer round-trip failed on training text; aborting")

    meta = {"tokenizer_kind": cfg.tokenizer.kind, "vocab_size": tokenizer.vocab_size,
            "eot_id": tokenizer.eot_id, "splits": {}}
    for name, recs in splits.items():
        info = encode_split(recs, tokenizer, out_dir / f"{name}.bin")
        meta["splits"][name] = info
        print(f"{name}: {info['n_tokens']:,} tokens -> {out_dir / f'{name}.bin'}")
    (out_dir / "encode_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"manifest: {cfg.split.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
