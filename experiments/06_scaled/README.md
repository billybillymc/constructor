# 06_scaled — more parameters, more steps

**Technique under study:** scale, on both axes the 6 GB laptop GPU allows.
Same recipe as 03_muon (Muon + AdamW, identical model.py/train.py — only the
config differs):

| | 03_muon | 06_scaled |
|---|---|---|
| layers x width | 14 x 512 | 20 x 640 |
| params | 53M | ~110M |
| dropout | 0.05 | 0.1 (more repetition needs more regularization) |
| steps | 8,000 | 16,000 |
| tokens seen | 131M (~8 epochs) | 262M (~17 epochs) |
| tokens/step | 16,384 (16 x 2 x 512) | 16,384 (8 x 4 x 512) |

The corpus stays fixed at 15.6M training tokens, so this is deliberately deep
into the data-bound regime: ~17 epochs over the same text. The question is
how much more a bigger model can wring out of a fixed small corpus before
val loss turns — best-by-val checkpointing catches the turn wherever it
happens.

Expect several hours of wall clock. Perplexities are directly comparable to
01/02/03 (same tokenizer, same splits).

## Run

```
uv run train 06_scaled --run-name scaled_v1
```
