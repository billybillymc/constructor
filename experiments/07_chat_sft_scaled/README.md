# 07_chat_sft_scaled — does a stronger base make a better chat model?

**Technique under study:** the effect of base-model quality on chat SFT.
This is experiment 04 repeated with one change: the starting weights are the
06_scaled (110M) best checkpoint instead of 03_muon (53M).

Everything else is held fixed: `train.py` is a byte-identical copy of 04's,
`model.py` a byte-identical copy of 06's, the chat dataset is the same
`data/processed/chat` build 04 used (2,012 dialogues / 54k train tokens),
and the SFT recipe is unchanged (AdamW lr 1e-4, 400 steps, masked loss,
vocab grown 16000 → 16002 with mean-initialized rows). Only the per-device
batch shrinks (8 × 2 accumulation = 04's effective 16) to fit 110M params
under full AdamW in 6 GB VRAM.

| | 04_chat_sft | 07_chat_sft_scaled |
|---|---|---|
| base model | 03_muon, 53M, test ppl 133.5 | 06_scaled, 110M, test ppl 132.9 |
| base minimal-pairs overall | 69.9% | 77.4% |
| SFT data, steps, lr | identical | identical |

The bases are nearly tied on perplexity but far apart on grammatical
knowledge (minimal pairs), so this asks which of those the chat model
inherits. Predictions worth writing down beforehand: fluency and format
should improve; entity binding (06 facts: 58.3%, near chance) and
arithmetic probably stay broken — if they don't, that's the surprise.

Masked val loss is comparable to 04's (same data, same masking), with the
same caveat: lowercase casing twins of training dialogues leak into val,
so behavior on held-out prompts is the real measure.

## Run

```
uv run train 07_chat_sft_scaled --run-name sft_v1
```
