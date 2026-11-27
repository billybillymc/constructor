"""Prompt a trained Latin LM from the command line.

    python -m loquor "Gallia est omnis"      # one-shot
    python -m loquor                          # interactive REPL
    python -m loquor --list                   # show available checkpoints

By default the newest ``ckpt_best.pt`` under ``runs/`` is used; pick another
with ``--ckpt``. Each checkpoint records which experiment produced it, and
that experiment's model.py is loaded to rebuild the architecture.

A chat-tuned checkpoint (04_chat_sft) is automatically spoken to through its
chat template, with conversation history kept in the REPL. For base-model
checkpoints, ``--fewshot`` wraps your prompt in a short Interrogatio/Responsum
transcript so the model answers instead of free-associating.
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
CHAT_EXPERIMENTS = {"04_chat_sft"}
CHAT_DATA_DIR = Path("data/processed/chat")
USER_TOK, ASSISTANT_TOK = "<|user|>", "<|assistant|>"

FEWSHOT_PREFIX = (
    "Interrogatio: Salve! Quid agis?\n"
    "Responsum: Salve! Bene mecum agitur, gratias tibi.\n\n"
    "Interrogatio: Quis fuit Vergilius?\n"
    "Responsum: Vergilius poeta Romanus fuit, qui Aeneida scripsit.\n\n"
    "Interrogatio: Quid est insula?\n"
    "Responsum: Insula terra est quam undique aqua circumdat.\n\n"
)


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


def generate_ids(model, ids: list[int], device: str,
                 tokens: int, temperature: float, top_k: int) -> list[int]:
    ids = ids[-(model.cfg.block_size - tokens):]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=tokens,
                         temperature=temperature, top_k=top_k)
    return out[0, len(ids):].tolist()


def complete(model, tokenizer, prompt: str, device: str,
             tokens: int, temperature: float, top_k: int) -> str:
    ids = [tokenizer.eot_id] + tokenizer.encode(prompt)
    generated = generate_ids(model, ids, device, tokens, temperature, top_k)
    return tokenizer.decode(generated).replace(tokenizer.eot_token, "").strip()


def chat_complete(model, tokenizer, history: list[tuple[str, str]], prompt: str,
                  device: str, tokens: int, temperature: float, top_k: int) -> str:
    text = "".join(f"{USER_TOK}{u}{ASSISTANT_TOK}{a}{tokenizer.eot_token}"
                   for u, a in history)
    text += f"{USER_TOK}{prompt}{ASSISTANT_TOK}"
    generated = generate_ids(model, tokenizer.encode(text), device,
                             tokens, temperature, top_k)
    if tokenizer.eot_id in generated:
        generated = generated[:generated.index(tokenizer.eot_id)]
    return tokenizer.decode(generated).strip()


def fewshot_complete(model, tokenizer, prompt: str, device: str,
                     tokens: int, temperature: float, top_k: int) -> str:
    text = FEWSHOT_PREFIX + f"Interrogatio: {prompt}\nResponsum:"
    generated = generate_ids(model, [tokenizer.eot_id] + tokenizer.encode(text),
                             device, tokens, temperature, top_k)
    if tokenizer.eot_id in generated:
        generated = generated[:generated.index(tokenizer.eot_id)]
    answer = tokenizer.decode(generated)
    return answer.split("Interrogatio:")[0].strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m loquor",
                                 description="Prompt a trained Latin LM")
    ap.add_argument("prompt", nargs="*", help="prompt text; omit for a REPL")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="checkpoint file (default: newest runs/**/ckpt_best.pt)")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="directory holding the trained tokenizer "
                         "(default: data/processed, or data/processed/chat "
                         "for a chat checkpoint)")
    ap.add_argument("--tokens", type=int, default=200, help="tokens to generate")
    ap.add_argument("--temperature", type=float, default=None,
                    help="sampling temperature (default: 0.6 chat, 0.8 completion)")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for reproducible sampling (default: random)")
    ap.add_argument("--fewshot", action="store_true",
                    help="wrap prompts in a Q/A transcript (base models only)")
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
    model, experiment = load_model(ckpt_path, device)
    chat = experiment in CHAT_EXPERIMENTS
    if args.temperature is None:
        args.temperature = 0.6 if chat else 0.8
    data_dir = args.data_dir or (CHAT_DATA_DIR if chat else Path("data/processed"))
    tokenizer = load_tokenizer(data_dir)
    if model.cfg.vocab_size != tokenizer.vocab_size:
        print(f"warning: checkpoint vocab {model.cfg.vocab_size} != tokenizer vocab "
              f"{tokenizer.vocab_size}; this checkpoint was trained with a different "
              f"tokenizer (pass the matching --data-dir)", file=sys.stderr)

    history: list[tuple[str, str]] = []

    def respond(prompt: str) -> None:
        if chat:
            answer = chat_complete(model, tokenizer, history, prompt, device,
                                   args.tokens, args.temperature, args.top_k)
            history.append((prompt, answer))
        elif args.fewshot:
            answer = fewshot_complete(model, tokenizer, prompt, device,
                                      args.tokens, args.temperature, args.top_k)
        else:
            answer = complete(model, tokenizer, prompt, device,
                              args.tokens, args.temperature, args.top_k)
        print(answer)

    if args.prompt:
        respond(" ".join(args.prompt))
        return 0

    mode = "chat" if chat else ("fewshot" if args.fewshot else "completion")
    print(f"loquor — {ckpt_path} [{experiment}, {mode}] on {device} "
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
