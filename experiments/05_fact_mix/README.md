# 05_fact_mix — when does knowledge need to enter?

**Technique under study:** fact-dense data mixed into *pretraining*, instead
of bolted on afterwards by SFT.

04_chat_sft showed the SFT split cleanly: format transplants easily, but
facts don't bind — the model answers "quis fuit caesar?" with Hannibal's
biography. The standard explanation is that a few hundred low-LR SFT steps
cannot build new factual associations; knowledge has to be present during
pretraining, where the model sees it many times while its representations
are still forming.

So this run is **byte-identical to 03_muon** (same model.py, same train.py,
same hyperparameters, same 8000 steps) with exactly one change: the training
stream is the corpus **plus the chat dialogues repeated 15x (~5% of
tokens)**, in their chat format, special tokens included (vocab 16002).
val/test stay pure corpus, so perplexities remain comparable with 02/03 —
expect a small penalty there, since ~5% of capacity goes to the dialogues.

The measurements that matter:

* **facts accuracy** on the minimal-pairs suite (`evals/minimal_pairs.py`)
  vs 03 (same compute, no facts) and 04 (same facts, via SFT)
* whether chat behavior emerges *without any SFT stage* — the chat format
  was seen 15 times over, inline with the corpus

## Run

```
uv run python experiments/05_fact_mix/make_data.py
uv run train 05_fact_mix --run-name mix_v1
python -m loquor --ckpt runs/05_fact_mix/mix_v1/ckpt_best.pt   # speaks chat template
```
