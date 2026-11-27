# 03 — Muon optimizer

Same model as `02_modern_arch` (model.py is a byte-for-byte copy, so any
difference in results is the optimizer). The change: 2D hidden weight
matrices are trained with **Muon** (MomentUm Orthogonalized by Newton-Schulz,
Keller Jordan 2024) instead of AdamW.

Muon keeps a momentum buffer per matrix, then orthogonalizes the update with
a 5-step quintic Newton-Schulz iteration before applying it — every update
has balanced singular values instead of being dominated by a few directions.
It drove most of the modded-nanogpt speedrun records and is used at scale
(e.g. Kimi/Moonshot). Embeddings, the tied head, and RMSNorm gains stay on
AdamW, as in the reference implementation.

Both optimizers share one warmup+cosine schedule applied as a multiplier to
their own base learning rates (`muon_lr = 0.02`, `lr = 5e-4`).

```sh
uv run train 03_muon --overfit | --smoke
uv run train 03_muon
```
