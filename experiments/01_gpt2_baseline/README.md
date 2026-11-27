# 01 — GPT-2 baseline

The reference point: the architecture and recipe everything else is measured
against.

**Architecture** (`model.py`): learned absolute position embeddings,
pre-LayerNorm blocks, GELU MLP (4x), dropout, weight-tied head, GPT-2 init
with depth-scaled residual projections.

**Recipe** (`train.py`): one AdamW over all parameters (weight decay on
everything), linear warmup -> cosine decay, gradient clipping, bf16 autocast.

This is the setup that produced the first trained model in this repo
(`runs/real_run_v1`: 13.7M params, val ppl 108.5 on the small 10-author
corpus with an 8k-vocab tokenizer; note that perplexities are only
comparable between runs that share a tokenizer and corpus).

```sh
uv run train 01_gpt2_baseline --overfit   # sanity check
uv run train 01_gpt2_baseline --smoke     # 200-step end-to-end test
uv run train 01_gpt2_baseline             # real run
```
