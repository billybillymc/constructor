"""Dispatch `uv run train <experiment>` to experiments/<experiment>/train.py.

Each experiment is a self-contained directory (model.py, train.py,
config.toml, README.md). This wrapper just locates it, puts it on sys.path
so `from model import GPT` resolves to that experiment's model, and runs its
train.py. Extra arguments pass through:

    uv run train --list
    uv run train 01_gpt2_baseline --smoke
    uv run train muon --resume runs/03_muon/<run>/ckpt_last.pt
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path("experiments")


def list_experiments() -> list[str]:
    if not EXPERIMENTS_DIR.is_dir():
        return []
    return sorted(p.name for p in EXPERIMENTS_DIR.iterdir() if (p / "train.py").is_file())


def resolve(name: str, available: list[str]) -> str | None:
    if name in available:
        return name
    matches = [e for e in available if name in e]
    return matches[0] if len(matches) == 1 else None


def main() -> int:
    argv = sys.argv[1:]
    available = list_experiments()
    if not argv or argv[0] in ("-h", "--help", "--list"):
        print("usage: train <experiment> [train.py args...]\n\nexperiments:")
        for e in available:
            print(f"  {e}")
        return 0
    name = resolve(argv[0], available)
    if name is None:
        print(f"unknown experiment '{argv[0]}'; available: {', '.join(available)}",
              file=sys.stderr)
        return 1
    exp_dir = (EXPERIMENTS_DIR / name).resolve()
    sys.path.insert(0, str(exp_dir))
    sys.argv = [str(exp_dir / "train.py")] + argv[1:]
    runpy.run_path(str(exp_dir / "train.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
