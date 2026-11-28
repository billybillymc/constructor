"""Train the modern-architecture model with the Muon optimizer.

Identical model to 02_modern_arch; the only change is the optimizer. Muon
(MomentUm Orthogonalized by Newton-Schulz, Keller Jordan 2024, the engine of
the modded-nanogpt speedrun records) orthogonalizes the momentum update of
each 2D hidden weight matrix, so every update has balanced singular values
instead of being dominated by a few directions. Parameters that are not 2D
hidden matrices — the (tied) embedding and the RMSNorm gains — keep AdamW,
as in the reference implementation.

    uv run train 03_muon --overfit | --smoke | (real run)
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from latin_lm import train_utils as tu
from model import GPT, ModelConfig

EXPERIMENT = Path(__file__).resolve().parent.name


@dataclass
class TrainConfig:
    seed: int = 1457
    data_dir: str = "data/processed"
    out_dir: str = "runs"
    batch_size: int = 16
    grad_accum_steps: int = 2
    max_steps: int = 8000
    # Muon, for 2D hidden weight matrices.
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_weight_decay: float = 0.01
    # AdamW, for embeddings and norm gains.
    lr: float = 5e-4
    min_lr_frac: float = 0.1     # both optimizers decay to this fraction of peak
    warmup_steps: int = 300
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_batches: int = 40
    sample_tokens: int = 250
    sample_temperature: float = 0.8
    sample_top_k: int = 50


# --------------------------------------------------------------------------- #
# Muon
# --------------------------------------------------------------------------- #

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Approximate UV^T for G = USV^T via a quintic Newton-Schulz iteration.

    Coefficients from Keller Jordan's reference implementation: tuned to
    maximize convergence slope at zero rather than to converge precisely to 1,
    which is cheap and empirically sufficient.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Momentum -> orthogonalize -> step, for 2D weight matrices only."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                if group["weight_decay"] > 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                # Scale so update RMS is comparable across matrix shapes.
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(g, alpha=-group["lr"] * scale)


def make_optimizers(model: GPT, cfg: TrainConfig, device: str):
    hidden, other = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (hidden if p.ndim == 2 and "tok_emb" not in name else other).append(p)
    muon = Muon(hidden, lr=cfg.muon_lr, momentum=cfg.muon_momentum,
                ns_steps=cfg.muon_ns_steps, weight_decay=cfg.muon_weight_decay)
    adamw = torch.optim.AdamW(other, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                              weight_decay=0.0, fused=(device == "cuda"))
    return muon, adamw


def lr_mult(step: int, cfg: TrainConfig, max_steps: int) -> float:
    """Warmup then cosine, as a multiplier applied to both optimizers."""
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, max_steps - cfg.warmup_steps)
    return cfg.min_lr_frac + 0.5 * (1 - cfg.min_lr_frac) * (1 + math.cos(math.pi * min(progress, 1.0)))


def run_overfit_check(model, ds, cfg: TrainConfig, device, autocast_ctx) -> bool:
    print("overfit check: 300 steps on a single batch (Muon + AdamW) ...")
    x, y = ds.batch(min(8, cfg.batch_size), device)
    muon, adamw = make_optimizers(model, cfg, device)
    first = last = None
    for i in range(300):
        with autocast_ctx():
            _, loss = model(x, y)
        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)
        loss.backward()
        muon.step()
        adamw.step()
        first = loss.item() if i == 0 else first
        last = loss.item()
    print(f"overfit check: loss {first:.3f} -> {last:.4f}")
    return last < 0.3


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"Train {EXPERIMENT}")
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.toml"))
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="override config")
    args = ap.parse_args(argv)

    mcfg, tcfg = tu.load_experiment_config(Path(args.config), ModelConfig, TrainConfig)
    tu.set_seed(tcfg.seed)
    device = tu.detect_device()
    autocast_ctx = tu.autocast_for(device)
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    meta, tokenizer, sets = tu.load_datasets(Path(tcfg.data_dir), mcfg.block_size, tcfg.seed)
    mcfg.vocab_size = meta["vocab_size"]
    model = GPT(mcfg).to(device)
    print(f"model parameters: {model.num_params():,}")

    if args.overfit:
        ok = run_overfit_check(model, sets["train"], tcfg, device, autocast_ctx)
        print("overfit check PASSED" if ok else "overfit check FAILED")
        return 0 if ok else 1

    max_steps = args.max_steps or (200 if args.smoke else tcfg.max_steps)
    eval_interval = 50 if args.smoke else tcfg.eval_interval
    eval_batches = 8 if args.smoke else tcfg.eval_batches

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S") + ("_smoke" if args.smoke else "")
    run_dir = Path(tcfg.out_dir) / EXPERIMENT / run_name
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.toml")

    muon, adamw = make_optimizers(model, tcfg, device)
    optimizers = [muon, adamw]
    for opt in optimizers:
        for g in opt.param_groups:
            g["base_lr"] = g["lr"]

    start_step, best_val = 0, float("inf")
    if args.resume:
        ckpt = tu.load_checkpoint(Path(args.resume), device)
        model.load_state_dict(ckpt["model"])
        for opt, state in zip(optimizers, ckpt["optimizers"]):
            opt.load_state_dict(state)
        start_step, best_val = ckpt["step"], ckpt["best_val"]
        print(f"resumed from {args.resume} at step {start_step}")

    log = tu.CsvLog(run_dir / "log.csv")
    tokens_per_step = tcfg.batch_size * tcfg.grad_accum_steps * mcfg.block_size
    print(f"training for {max_steps} steps ({tokens_per_step:,} tokens/step) -> {run_dir}")
    model.train()
    t0 = time.time()
    running = None
    for step in range(start_step, max_steps):
        mult = lr_mult(step, tcfg, max_steps)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * mult

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        for _ in range(tcfg.grad_accum_steps):
            x, y = sets["train"].batch(tcfg.batch_size, device)
            with autocast_ctx():
                _, loss = model(x, y)
            (loss / tcfg.grad_accum_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        for opt in optimizers:
            opt.step()

        item = loss.item()
        running = item if running is None else 0.95 * running + 0.05 * item

        if (step + 1) % eval_interval == 0 or step + 1 == max_steps:
            val_loss = tu.estimate_loss(model, sets["val"], eval_batches,
                                        tcfg.batch_size, device, autocast_ctx)
            elapsed = time.time() - t0
            log.row(step + 1, running, val_loss, tcfg.muon_lr * mult, elapsed)
            print(f"step {step + 1:>6} | train {running:.4f} | val {val_loss:.4f} "
                  f"| lr x{mult:.3f} | {elapsed:,.0f}s", flush=True)
            sample = tu.generate_sample(model, tokenizer, device, tcfg.sample_tokens,
                                        tcfg.sample_temperature, tcfg.sample_top_k)
            (run_dir / "samples" / f"step_{step + 1:06d}.txt").write_text(sample, encoding="utf-8")
            tu.save_checkpoint(run_dir / "ckpt_last.pt", model, optimizers,
                               step + 1, best_val, asdict(mcfg), EXPERIMENT)
            if val_loss < best_val:
                best_val = val_loss
                tu.save_checkpoint(run_dir / "ckpt_best.pt", model, optimizers,
                                   step + 1, best_val, asdict(mcfg), EXPERIMENT)
    log.close()

    best = tu.load_checkpoint(run_dir / "ckpt_best.pt", device)
    model.load_state_dict(best["model"])
    tu.final_report(run_dir, f"{EXPERIMENT}/{run_name}", model, sets,
                    max(eval_batches, 40), tcfg.batch_size, device, autocast_ctx,
                    best["step"], max_steps)
    print("\n=== samples ===")
    for i in range(3):
        print(f"--- sample {i + 1} ---")
        print(tu.generate_sample(model, tokenizer, device, tcfg.sample_tokens,
                                 tcfg.sample_temperature, tcfg.sample_top_k)[:600])
    return 0


if __name__ == "__main__":
    sys.exit(main())
