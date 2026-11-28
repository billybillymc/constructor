"""Build the fact-mix pretraining data for 05_fact_mix.

The question this experiment asks: does knowledge bind better when it enters
during pretraining instead of being bolted on by SFT afterwards? So the
pretraining stream here is the ordinary corpus PLUS the chat/fact dialogues
(the same binaries 04_chat_sft trains on) repeated UPSAMPLE times — about 5%
of the stream — in their chat format, special tokens included.

Output (data/processed/fact_mix/):
  train.bin             corpus train + UPSAMPLE x chat train (uint16)
  val.bin, test.bin     copied from the corpus — perplexities stay comparable
  tokenizer files       copied from data/processed/chat (vocab 16002; corpus
                        token ids are unchanged, the two chat tokens are
                        appended at the end)
  encode_meta.json      adjusted vocab/counts

    uv run python experiments/05_fact_mix/make_data.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

UPSAMPLE = 15  # ~15 x 54k = 810k chat tokens in a ~16.4M stream (~5%)


def main() -> int:
    corpus = Path("data/processed")
    chat = corpus / "chat"
    out = corpus / "fact_mix"
    out.mkdir(parents=True, exist_ok=True)

    corpus_train = np.fromfile(corpus / "train.bin", dtype=np.uint16)
    chat_train = np.fromfile(chat / "train.bin", dtype=np.uint16)
    mixed = np.concatenate([corpus_train] + [chat_train] * UPSAMPLE)
    mixed.tofile(out / "train.bin")

    for name in ("val.bin", "test.bin"):
        shutil.copy(corpus / name, out / name)
    for name in ("bpe_tokenizer.json", "tokenizer_meta.json"):
        shutil.copy(chat / name, out / name)

    base_meta = json.loads((corpus / "encode_meta.json").read_text(encoding="utf-8"))
    chat_meta = json.loads((chat / "chat_meta.json").read_text(encoding="utf-8"))
    meta = dict(base_meta)
    meta["vocab_size"] = chat_meta["vocab_size"]
    meta["splits"] = dict(base_meta["splits"])
    meta["splits"]["train"] = {"path": "train.bin", "dtype": "uint16",
                               "n_tokens": int(mixed.size)}
    meta["fact_mix"] = {"upsample": UPSAMPLE,
                        "chat_tokens": int(chat_train.size) * UPSAMPLE,
                        "chat_fraction": int(chat_train.size) * UPSAMPLE / mixed.size}
    (out / "encode_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    print(f"train: {corpus_train.size:,} corpus + {UPSAMPLE} x {chat_train.size:,} chat "
          f"= {mixed.size:,} tokens ({meta['fact_mix']['chat_fraction']:.1%} chat)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
