# 02 — Modern architecture

Same training recipe family as the baseline (AdamW, warmup -> cosine), but
the architecture is replaced with the current small-model consensus stack:

| Component | 01 baseline | 02 modern |
|---|---|---|
| Positions | learned absolute embeddings | rotary (RoPE) |
| Norm | LayerNorm | RMSNorm |
| Feed-forward | GELU, 4x | SwiGLU, 2.75x, 3 matrices |
| Attention | — | QK-norm per head |
| Init | GPT-2 (scaled residuals) | zero-init output projections |
| Biases | optional | none |
| Weight decay | all parameters | 2D hidden weights only |

Sources: the Llama/TinyLlama/SmolLM lineage for the core stack; QK-norm and
zero-init projections from the modded-nanogpt speedrun records; the
weight-decay exclusion follows OLMo-2/SmolLM practice.

Default size: 14 layers x 512 dim (depth-over-width, per MiniLingua/YuLan
findings for small models), ~53M params with the 16k tokenizer.

```sh
uv run train 02_modern_arch --overfit | --smoke
uv run train 02_modern_arch
```
