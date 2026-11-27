"""Shared training plumbing used by every experiment.

Each experiment under ``experiments/`` keeps its own model.py and train.py —
the architecture and the optimization recipe ARE the experiment — while the
boring parts (device detection, memory-mapped batches, evaluation, logging,
checkpoints, sampling, the loss plot) live here.
"""

from __future__ import annotations

import contextlib
import csv
import json
import random
import tomllib
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .tokenizer import LatinTokenizer, load_tokenizer


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def autocast_for(device: str):
    """bf16 autocast on CUDA, no-op elsewhere."""
    if device == "cuda":
        return lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext


def build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def load_experiment_config(path: Path, model_cls: type, train_cls: type):
    """Parse an experiment's config.toml into its own dataclasses."""
    with path.open("rb") as f:
        raw = tomllib.load(f)
    return (
        build_dataclass(model_cls, raw.get("model", {})),
        build_dataclass(train_cls, raw.get("train", {})),
    )


class BinDataset:
    """Memory-mapped token file; serves random (x, y) batches."""

    def __init__(self, path: Path, dtype: str, block_size: int, seed: int):
        self.data = np.memmap(path, dtype=np.dtype(dtype), mode="r")
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)
        if len(self.data) <= block_size + 1:
            raise SystemExit(f"{path} has too few tokens ({len(self.data)})")

    def batch(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        ix = self.rng.integers(0, len(self.data) - self.block_size - 1, size=batch_size)
        x = torch.stack([torch.from_numpy(self.data[i:i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1:i + 1 + self.block_size].astype(np.int64)) for i in ix])
        if device == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)


def load_datasets(data_dir: Path, block_size: int, seed: int):
    """Return (meta, tokenizer, {split: BinDataset})."""
    meta = json.loads((data_dir / "encode_meta.json").read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(data_dir)
    sets = {
        name: BinDataset(data_dir / f"{name}.bin", meta["splits"][name]["dtype"],
                         block_size, seed + i)
        for i, name in enumerate(("train", "val", "test"))
    }
    return meta, tokenizer, sets


@torch.no_grad()
def estimate_loss(model, ds: BinDataset, batches: int, batch_size: int,
                  device: str, autocast_ctx) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = ds.batch(batch_size, device)
        with autocast_ctx():
            _, loss = model(x, y)
        losses.append(loss.item())
    if was_training:
        model.train()
    return float(np.mean(losses))


def generate_sample(model, tokenizer: LatinTokenizer, device: str,
                    tokens: int, temperature: float, top_k: int) -> str:
    was_training = model.training
    idx = torch.tensor([[tokenizer.eot_id]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=tokens,
                         temperature=temperature, top_k=top_k)
    if was_training:
        model.train()
    return tokenizer.decode(out[0].tolist()).replace(tokenizer.eot_token, "\n----\n")


class CsvLog:
    COLUMNS = ["step", "train_loss", "val_loss", "lr", "elapsed_s"]

    def __init__(self, path: Path):
        new = not path.exists()
        self._f = path.open("a", newline="")
        self._w = csv.writer(self._f)
        if new:
            self._w.writerow(self.COLUMNS)

    def row(self, step: int, train_loss: float, val_loss: float, lr: float, elapsed: float):
        self._w.writerow([step, f"{train_loss:.4f}", f"{val_loss:.4f}",
                          f"{lr:.2e}", f"{elapsed:.1f}"])
        self._f.flush()

    def close(self):
        self._f.close()


def save_checkpoint(path: Path, model, optimizers: list, step: int,
                    best_val: float, model_cfg_dict: dict, experiment: str) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizers": [opt.state_dict() for opt in optimizers],
            "step": step,
            "best_val": best_val,
            "model_config": model_cfg_dict,
            "experiment": experiment,
        },
        path,
    )


def load_checkpoint(path: Path, device: str) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "optimizers" not in ckpt and "optimizer" in ckpt:  # pre-restructure runs
        ckpt["optimizers"] = [ckpt.pop("optimizer")]
    ckpt.setdefault("experiment", "01_gpt2_baseline")
    return ckpt


def plot_loss_curve(csv_path: Path, out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, tr, va = [], [], []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            tr.append(float(row["train_loss"]))
            va.append(float(row["val_loss"]))
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(steps, tr, color="#3B6FD4", linewidth=2, label="train loss")
    ax.plot(steps, va, color="#E08214", linewidth=2, label="val loss")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.6)
    ax.legend(frameon=False)
    ax.set_title(title, loc="left", color="#333333")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def final_report(run_dir: Path, run_name: str, model, sets, eval_batches: int,
                 batch_size: int, device: str, autocast_ctx, best_step: int,
                 max_steps: int) -> dict:
    import math

    val_loss = estimate_loss(model, sets["val"], eval_batches, batch_size, device, autocast_ctx)
    test_loss = estimate_loss(model, sets["test"], eval_batches, batch_size, device, autocast_ctx)
    report = {
        "run": run_name,
        "best_checkpoint_step": best_step,
        "val_loss": val_loss, "val_perplexity": math.exp(val_loss),
        "test_loss": test_loss, "test_perplexity": math.exp(test_loss),
        "params": model.num_params(),
        "max_steps": max_steps,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    plot_loss_curve(run_dir / "log.csv", run_dir / "loss_curve.png", run_name)
    print("\n=== final report ===")
    print(f"best checkpoint: step {best_step}")
    print(f"val  loss {val_loss:.4f} | perplexity {math.exp(val_loss):8.2f}")
    print(f"test loss {test_loss:.4f} | perplexity {math.exp(test_loss):8.2f}")
    print(f"loss curve: {run_dir / 'loss_curve.png'}")
    return report
