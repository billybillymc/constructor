"""Prompt a trained Latin LM from the command line.

    python -m loquor "Gallia est omnis"      # one-shot completion
    python -m loquor                          # interactive REPL
    python -m loquor --list                   # show available checkpoints

By default the newest ``ckpt_best.pt`` under ``runs/`` is used; pick another
with ``--ckpt``. Each checkpoint records which experiment produced it, and
that experiment's model.py is loaded to rebuild the architecture.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

from latin_lm.tokenizer import load_tokenizer
from latin_lm.train_utils import detect_device, load_checkpoint

EXPERIMENTS_DIR = Path("experiments")


def find_checkpoints(runs_dir: Path) -> list[Path]:
    ckpts = list(runs_dir.glob("*/ckpt_best.pt")) + list(runs_dir.glob("*/*/ckpt_best.pt"))
    return sorted(ckpts, key=lambda p: p.stat().st_mtime)


def import_experiment_model(experiment: str):
    path = EXPERIMENTS_DIR / experiment / "model.py"
    if not path.is_file():
        raise SystemExit(f"model file not found for experiment '{experiment}': {path}")
    spec = importlib.util.spec_from_file_location(f"model_{experiment}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # required for @dataclass to resolve annotations
    spec.loader.exec_module(module)
    return module


def load_model(ckpt_path: Path, device: str):
    ckpt = load_checkpoint(ckpt_path, device)
    mod = import_experiment_model(ckpt["experiment"])
    model = mod.GPT(mod.ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt["experiment"]


def complete(model, tokenizer, prompt: str, device: str,
             tokens: int, temperature: float, top_k: int) -> str:
    ids = [tokenizer.eot_id] + tokenizer.encode(prompt)
    ids = ids[-model.cfg.block_size:]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=tokens,
                         temperature=temperature, top_k=top_k)
    text = tokenizer.decode(out[0].tolist())
    return text.replace(tokenizer.eot_token, "").strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m loquor",
                                 description="Prompt a trained Latin LM")
    ap.add_argument("prompt", nargs="*", help="prompt text; omit for a REPL")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="checkpoint file (default: newest runs/**/ckpt_best.pt)")
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"),
                    help="directory holding the trained tokenizer")
    ap.add_argument("--tokens", type=int, default=200, help="tokens to generate")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for reproducible sampling (default: random)")
    ap.add_argument("--list", action="store_true", help="list checkpoints and exit")
    args = ap.parse_args(argv)

    ckpts = find_checkpoints(Path("runs"))
    if args.list:
        for p in ckpts:
            print(p)
        return 0

    ckpt_path = args.ckpt or (ckpts[-1] if ckpts else None)
    if ckpt_path is None or not ckpt_path.exists():
        print("no checkpoint found; train a model first (uv run train --list)", file=sys.stderr)
        return 1
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = detect_device()
    tokenizer = load_tokenizer(args.data_dir)
    model, experiment = load_model(ckpt_path, device)
    if model.cfg.vocab_size != tokenizer.vocab_size:
        print(f"warning: checkpoint vocab {model.cfg.vocab_size} != tokenizer vocab "
              f"{tokenizer.vocab_size}; this checkpoint was trained with a different "
              f"tokenizer (pass the matching --data-dir)", file=sys.stderr)

    def respond(prompt: str) -> None:
        print(complete(model, tokenizer, prompt, device,
                       args.tokens, args.temperature, args.top_k))

    if args.prompt:
        respond(" ".join(args.prompt))
        return 0

    print(f"loquor — {ckpt_path} [{experiment}] on {device} "
          f"(temp {args.temperature}, top-k {args.top_k}, {args.tokens} tokens)")
    print("empty line or Ctrl+C to quit")
    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt:
            return 0
        respond(prompt)


if __name__ == "__main__":
    raise SystemExit(main())
