# latin-lm

Language models trained from scratch on Latin, as a series of **experiments
comparing training techniques** on a shared corpus and data pipeline. Built
as a testbed for linguistic experiments (minimal-pairs suites, tokenizer
comparisons, hidden-state probing): the tokenizer is swappable, splits are
by whole work, and every run saves its exact config.

## Layout

```
src/latin_lm/             shared pipeline: scrape, clean, normalize, split,
                          tokenize, encode + shared training utilities
experiments/
  01_gpt2_baseline/       GPT-2-style arch, AdamW + cosine (the reference)
  02_modern_arch/         RoPE + RMSNorm + SwiGLU + QK-norm, AdamW
  03_muon/                same arch as 02, Muon optimizer
src/loquor/               CLI to prompt any trained checkpoint
configs/default.toml      data-pipeline config (corpus, normalization,
                          splits, tokenizer)
data/                     HTML cache, corpus JSONL, encoded binaries
runs/<experiment>/<run>/  logs, samples, checkpoints, config snapshot
```

Each experiment directory is self-contained — its own `model.py`,
`train.py`, `config.toml`, and a README saying what technique it isolates
and why. Experiments 02 and 03 share a byte-identical `model.py`, so their
result difference is attributable to the optimizer alone.

## Setup

```sh
uv sync   # PyTorch installs with CUDA 12.6 wheels on Windows
```

## Pipeline

### 1. Corpus: `uv run scrape`

Politely scrapes The Latin Library (robots.txt check, 1 request/second,
descriptive user agent, on-disk HTML cache — reruns never refetch). With
`discover = true` (default) it crawls every author on the site's five
register indexes (classical, christian, medieval, neo-Latin, misc) and
labels each work's register. Cleaning and normalization are unit-tested
modules; normalization (NFC, macrons stripped, case kept, u/v and i/j as
the source) is configurable per `[normalize]`.

Output: `data/corpus/works.jsonl` (one record per work) and
`data/corpus/stats.txt`.

### 2. Data: `uv run prepare`

Splits by whole work (~90/5/5, seeded; giant works forced into train),
writes the split manifest, trains the BPE tokenizer (16k vocab, training
split only; or `kind = "char"`), and encodes each split to memory-mappable
token binaries under `data/processed/`.

### 3. Train: `uv run train <experiment>`

```sh
uv run train --list
uv run train 01_gpt2_baseline --overfit   # sanity: overfit one batch
uv run train 02_modern_arch --smoke       # 200-step end-to-end test
uv run train 03_muon                      # real run
uv run train 03 --resume runs/03_muon/<run>/ckpt_last.pt
```

Every experiment logs step/train-loss/val-loss/lr to CSV, saves generated
samples and checkpoints (`ckpt_last` + best-by-val `ckpt_best`) each eval
interval, and finishes with val/test perplexity, a loss curve PNG, and
sample generations. Perplexities are comparable *between experiments* when
they share the same prepared data (tokenizer + splits).

### 4. Prompt: `uv run python -m loquor "Gallia est omnis"`

Loads the newest `ckpt_best.pt` (or `--ckpt ...`); each checkpoint records
its experiment, and the matching `model.py` is used to rebuild it. Options:
`--tokens --temperature --top-k --seed --list`.

## Tests

```sh
uv run pytest
```

## Adding an experiment

Copy the closest experiment directory, change one thing, write its README.
`uv run train <name>` picks it up automatically; keep `model.py` exposing
`GPT` and `ModelConfig` so loquor can load its checkpoints.
