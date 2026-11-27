"""Supervised fine-tuning: turn the 03_muon base model into a chat model.

The technique under study is SFT itself, so everything else is held fixed:
the architecture is byte-identical to 02/03 and the starting weights are the
03_muon best checkpoint. Three things change relative to base training:

  * the data is chat-formatted dialogues
    (<|user|> question <|assistant|> answer <|endoftext|>), built by
    make_dataset.py from hand-written seeds, Donatus's Ars Minor, generated
    morphology drills and Latin arithmetic
  * the embedding matrix grows by two rows for the chat special tokens,
    initialized to the mean of the existing embeddings
  * the loss is masked to assistant tokens (answers and the end-of-turn
    token), so the model learns to answer, not to imitate questions

Plain AdamW at a low learning rate — the standard SFT recipe.

    uv run python experiments/04_chat_sft/make_dataset.py   # once, first
    uv run train 04_chat_sft --run-name <name>
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from latin_lm import train_utils as tu
from latin_lm.tokenizer import load_tokenizer
from model import GPT, ModelConfig

EXPERIMENT = Path(__file__).resolve().parent.name

SAMPLE_PROMPTS = [
    "Salve!",
    "Quis fuit Caesar?",
    "Quot sunt duo et tria?",
    "Scribe sententiam de amicitia.",
]


@dataclass
class TrainConfig:
    seed: int = 1457
    data_dir: str = "data/processed/chat"
    out_dir: str = "runs"
    init_from: str = "runs/03_muon/full_corpus_v1/ckpt_best.pt"
    batch_size: int = 16
    grad_accum_steps: int = 1
    max_steps: int = 400
    lr: float = 1e-4
    min_lr_frac: float = 0.1
    warmup_steps: int = 20
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    eval_interval: int = 25
    eval_batches: int = 20
    sample_tokens: int = 90
    sample_temperature: float = 0.7
    sample_top_k: int = 50


class MaskedBinDataset:
    """Token stream + parallel loss mask; serves random (x, y, m) windows."""

    def __init__(self, directory: Path, split: str, block_size: int, seed: int):
        self.data = np.memmap(directory / f"{split}.bin", dtype=np.uint16, mode="r")
        self.mask = np.memmap(directory / f"{split}_mask.bin", dtype=np.uint8, mode="r")
        assert len(self.data) == len(self.mask)
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)
        if len(self.data) <= block_size + 1:
            raise SystemExit(f"{directory / split}.bin has too few tokens ({len(self.data)})")

    def batch(self, batch_size: int, device: str):
        ix = self.rng.integers(0, len(self.data) - self.block_size - 1, size=batch_size)
        x = torch.stack([torch.from_numpy(self.data[i:i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1:i + 1 + self.block_size].astype(np.int64)) for i in ix])
        m = torch.stack([torch.from_numpy(self.mask[i + 1:i + 1 + self.block_size].astype(np.float32)) for i in ix])
        if device == "cuda":
            return (x.pin_memory().to(device, non_blocking=True),
                    y.pin_memory().to(device, non_blocking=True),
                    m.pin_memory().to(device, non_blocking=True))
        return x.to(device), y.to(device), m.to(device)


def masked_loss(model: GPT, x, y, m) -> torch.Tensor:
    logits, _ = model(x)
    per_token = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                                reduction="none")
    m = m.view(-1)
    return (per_token * m).sum() / m.sum().clamp(min=1.0)


@torch.no_grad()
def estimate_masked_loss(model, ds, batches, batch_size, device, autocast_ctx) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(batches):
        x, y, m = ds.batch(batch_size, device)
        with autocast_ctx():
            loss = masked_loss(model, x, y, m)
        losses.append(loss.item())
    if was_training:
        model.train()
    return float(np.mean(losses))


def chat_sample(model, tokenizer, device, prompt: str, cfg: TrainConfig) -> str:
    ids = tokenizer.encode(f"<|user|>{prompt}<|assistant|>")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=cfg.sample_tokens,
                         temperature=cfg.sample_temperature, top_k=cfg.sample_top_k)
    generated = out[0, len(ids):].tolist()
    if tokenizer.eot_id in generated:
        generated = generated[:generated.index(tokenizer.eot_id)]
    return tokenizer.decode(generated).strip()


def write_samples(path: Path, model, tokenizer, device, cfg: TrainConfig) -> str:
    was_training = model.training
    lines = []
    for prompt in SAMPLE_PROMPTS:
        lines.append(f"> {prompt}\n{chat_sample(model, tokenizer, device, prompt, cfg)}\n")
    if was_training:
        model.train()
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text


def resize_embeddings(state: dict, new_vocab: int) -> dict:
    """Grow the tied embedding/head matrix; new rows start at the mean."""
    old = state["tok_emb.weight"]
    if old.size(0) == new_vocab:
        return state
    grown = old.new_empty((new_vocab, old.size(1)))
    grown[:old.size(0)] = old
    grown[old.size(0):] = old.mean(dim=0)
    state = dict(state)
    state["tok_emb.weight"] = grown
    state["head.weight"] = grown
    return state


def lr_mult(step: int, cfg: TrainConfig, max_steps: int) -> float:
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, max_steps - cfg.warmup_steps)
    return cfg.min_lr_frac + 0.5 * (1 - cfg.min_lr_frac) * (1 + math.cos(math.pi * min(progress, 1.0)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"Train {EXPERIMENT}")
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.toml"))
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="override config")
    args = ap.parse_args(argv)

    mcfg, tcfg = tu.load_experiment_config(Path(args.config), ModelConfig, TrainConfig)
    tu.set_seed(tcfg.seed)
    device = tu.detect_device()
    autocast_ctx = tu.autocast_for(device)
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    data_dir = Path(tcfg.data_dir)
    if not (data_dir / "chat_meta.json").exists():
        raise SystemExit(f"{data_dir} missing; run experiments/{EXPERIMENT}/make_dataset.py first")
    meta = json.loads((data_dir / "chat_meta.json").read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(data_dir)
    sets = {name: MaskedBinDataset(data_dir, name, mcfg.block_size, tcfg.seed + i)
            for i, name in enumerate(("train", "val"))}

    mcfg.vocab_size = meta["vocab_size"]
    model = GPT(mcfg).to(device)
    base = tu.load_checkpoint(Path(tcfg.init_from), device)
    model.load_state_dict(resize_embeddings(base["model"], meta["vocab_size"]))
    print(f"initialized from {tcfg.init_from} (experiment {base['experiment']}, "
          f"step {base['step']}); vocab {base['model_config']['vocab_size']} -> {meta['vocab_size']}")
    print(f"model parameters: {model.num_params():,}")

    max_steps = args.max_steps or tcfg.max_steps
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(tcfg.out_dir) / EXPERIMENT / run_name
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.toml")

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr,
                                  betas=(tcfg.beta1, tcfg.beta2),
                                  weight_decay=tcfg.weight_decay,
                                  fused=(device == "cuda"))
    log = tu.CsvLog(run_dir / "log.csv")
    print(f"fine-tuning for {max_steps} steps on {meta['splits']['train']['tokens']:,} "
          f"chat tokens -> {run_dir}")
    model.train()
    t0 = time.time()
    running, best_val = None, float("inf")
    for step in range(max_steps):
        mult = lr_mult(step, tcfg, max_steps)
        for g in optimizer.param_groups:
            g["lr"] = tcfg.lr * mult

        optimizer.zero_grad(set_to_none=True)
        for _ in range(tcfg.grad_accum_steps):
            x, y, m = sets["train"].batch(tcfg.batch_size, device)
            with autocast_ctx():
                loss = masked_loss(model, x, y, m)
            (loss / tcfg.grad_accum_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()

        item = loss.item()
        running = item if running is None else 0.95 * running + 0.05 * item

        if (step + 1) % tcfg.eval_interval == 0 or step + 1 == max_steps:
            val_loss = estimate_masked_loss(model, sets["val"], tcfg.eval_batches,
                                            tcfg.batch_size, device, autocast_ctx)
            elapsed = time.time() - t0
            log.row(step + 1, running, val_loss, tcfg.lr * mult, elapsed)
            print(f"step {step + 1:>6} | train {running:.4f} | val {val_loss:.4f} "
                  f"| lr {tcfg.lr * mult:.2e} | {elapsed:,.0f}s", flush=True)
            write_samples(run_dir / "samples" / f"step_{step + 1:06d}.txt",
                          model, tokenizer, device, tcfg)
            tu.save_checkpoint(run_dir / "ckpt_last.pt", model, [optimizer],
                               step + 1, best_val, asdict(mcfg), EXPERIMENT)
            if val_loss < best_val:
                best_val = val_loss
                tu.save_checkpoint(run_dir / "ckpt_best.pt", model, [optimizer],
                                   step + 1, best_val, asdict(mcfg), EXPERIMENT)
    log.close()

    best = tu.load_checkpoint(run_dir / "ckpt_best.pt", device)
    model.load_state_dict(best["model"])
    val_loss = estimate_masked_loss(model, sets["val"], tcfg.eval_batches,
                                    tcfg.batch_size, device, autocast_ctx)
    report = {
        "run": f"{EXPERIMENT}/{run_name}",
        "best_checkpoint_step": best["step"],
        "val_loss_masked": val_loss,
        "val_perplexity_masked": math.exp(val_loss),
        "note": "masked (assistant-token) loss on chat data; not comparable "
                "to the base-model corpus perplexities",
        "params": model.num_params(),
        "max_steps": max_steps,
        "init_from": tcfg.init_from,
        "dataset": meta["sources"],
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    tu.plot_loss_curve(run_dir / "log.csv", run_dir / "loss_curve.png",
                       f"{EXPERIMENT}/{run_name}")
    print("\n=== final report ===")
    print(f"best checkpoint: step {best['step']}")
    print(f"val masked loss {val_loss:.4f} | perplexity {math.exp(val_loss):8.2f}")
    print("\n=== chat samples (best checkpoint) ===")
    print(write_samples(run_dir / "samples" / "final.txt", model, tokenizer, device, tcfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
