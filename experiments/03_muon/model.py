"""Modern decoder-only transformer, in plain PyTorch.

The 2024-26 small-model consensus stack (Llama/TinyLlama/SmolLM lineage,
plus two modded-nanogpt tricks):

  * rotary position embeddings (RoPE) instead of learned absolute positions
  * RMSNorm, pre-norm placement
  * SwiGLU feed-forward (hidden = 2.75x, three matrices, no biases)
  * QK-norm: RMS-normalize queries and keys per head before attention
  * zero-initialized output projections (residual branches start as identity)
  * no biases anywhere; weight-tied head

Same file layout as 01_gpt2_baseline/model.py so the two diff cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    n_layer: int = 14
    n_head: int = 8
    n_embd: int = 512
    block_size: int = 512
    dropout: float = 0.05
    mlp_hidden: int = 1408      # SwiGLU hidden size (~2.75x n_embd)
    rope_theta: float = 10000.0
    vocab_size: int = 16000     # overwritten from the trained tokenizer at train time


class Rotary(nn.Module):
    """Precomputed RoPE angles; applied to (B, n_head, T, head_dim) tensors."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)                      # (T, head_dim/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(-2)
        cos = self.cos[:T]
        sin = self.sin[:T]
        x1, x2 = x.chunk(2, dim=-1)                           # rotate halves
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim, cfg.block_size, cfg.rope_theta)
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # QK-norm stabilizes attention logits; RoPE rotation preserves it.
        q = self.rotary(F.rms_norm(q, (self.head_dim,)))
        k = self.rotary(F.rms_norm(k, (self.head_dim,)))
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.n_embd, cfg.mlp_hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, cfg.mlp_hidden, bias=False)
        self.proj = nn.Linear(cfg.mlp_hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = nn.RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)
        # Residual branches start at zero: the network begins as the identity.
        for name, p in self.named_parameters():
            if name.endswith(("attn.proj.weight", "mlp.proj.weight")):
                nn.init.zeros_(p)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        return sum(p.numel() for p in self.parameters())  # RoPE adds none

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block size {self.cfg.block_size}"
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
