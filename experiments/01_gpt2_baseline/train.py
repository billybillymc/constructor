"""Train the GPT-2-style baseline.

Recipe: single AdamW over all parameters, linear warmup then cosine decay,
gradient clipping, bf16 autocast on CUDA. This is the loop that trained
`runs/real_run_v1`, kept as the reference point for later experiments.

    uv run train 01_gpt2_baseline --overfit
    uv run train 01_gpt2_baseline --smoke
    uv run train 01_gpt2_baseline
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
    batch_size: int = 64
    grad_accum_steps: int = 1
    max_steps: int = 5000
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_batches: int = 40
    sample_tokens: int = 250
    sample_temperature: float = 0.8
    sample_top_k: int = 50


def lr_at(step: int, cfg: TrainConfig, max_steps: int) -> float:
    """Linear warmup to cfg.lr, then cosine decay to cfg.min_lr."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))


def run_overfit_check(model, ds, cfg: TrainConfig, device, autocast_ctx) -> bool:
    print("overfit check: 300 steps on a single batch ...")
    x, y = ds.batch(min(8, cfg.batch_size), device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = last = None
    for i in range(300):
        with autocast_ctx():
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg.lr, betas=(tcfg.beta1, tcfg.beta2),
        weight_decay=tcfg.weight_decay, fused=(device == "cuda"),
    )

    start_step, best_val = 0, float("inf")
    if args.resume:
        ckpt = tu.load_checkpoint(Path(args.resume), device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizers"][0])
        start_step, best_val = ckpt["step"], ckpt["best_val"]
        print(f"resumed from {args.resume} at step {start_step}")

    log = tu.CsvLog(run_dir / "log.csv")
    print(f"training for {max_steps} steps -> {run_dir}")
    model.train()
    t0 = time.time()
    running = None
    for step in range(start_step, max_steps):
        lr = lr_at(step, tcfg, max_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for _ in range(tcfg.grad_accum_steps):
            x, y = sets["train"].batch(tcfg.batch_size, device)
            with autocast_ctx():
                _, loss = model(x, y)
            (loss / tcfg.grad_accum_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()

        item = loss.item()
        running = item if running is None else 0.95 * running + 0.05 * item

        if (step + 1) % eval_interval == 0 or step + 1 == max_steps:
            val_loss = tu.estimate_loss(model, sets["val"], eval_batches,
                                        tcfg.batch_size, device, autocast_ctx)
            elapsed = time.time() - t0
            log.row(step + 1, running, val_loss, lr, elapsed)
            print(f"step {step + 1:>6} | train {running:.4f} | val {val_loss:.4f} "
                  f"| lr {lr:.2e} | {elapsed:,.0f}s", flush=True)
            sample = tu.generate_sample(model, tokenizer, device, tcfg.sample_tokens,
                                        tcfg.sample_temperature, tcfg.sample_top_k)
            (run_dir / "samples" / f"step_{step + 1:06d}.txt").write_text(sample, encoding="utf-8")
            tu.save_checkpoint(run_dir / "ckpt_last.pt", model, [optimizer],
                               step + 1, best_val, asdict(mcfg), EXPERIMENT)
            if val_loss < best_val:
                best_val = val_loss
                tu.save_checkpoint(run_dir / "ckpt_best.pt", model, [optimizer],
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
