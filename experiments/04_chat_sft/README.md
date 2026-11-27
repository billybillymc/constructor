# 04_chat_sft — supervised fine-tuning into a chat model

**Technique under study:** supervised fine-tuning (SFT). Everything else is
held fixed: the architecture is byte-identical to 02/03 and training starts
from the 03_muon best checkpoint, so whatever conversational behavior appears
is attributable to the fine-tuning stage, not to a different model.

A base LM continues documents; ask it *"Legisne libros?"* and it writes more
poem. SFT teaches a turn format instead: every training example is

```
<|user|> question <|assistant|> answer <|endoftext|>
```

with the loss **masked to the assistant's tokens** (answers plus the
end-of-turn token), so the model learns to produce answers, not to imitate
questions. The two chat markers are new special tokens appended to the 16k
BPE tokenizer; the tied embedding matrix grows by two rows initialized to the
mean embedding.

## Dataset (`make_dataset.py`)

| source | dialogues | what it is |
|---|---|---|
| seed_dialogues.jsonl | ~200 | hand-written Latin exchanges: greetings, identity, grammar, Roman history and myth, definitions, proverbs, fables, composition requests, multi-turn |
| Donatus, *Ars Minor* | ~110 | the original Latin grammar catechism, mined from the corpus — natively Q&A-shaped |
| morphology drills | ~310 | generated from hard-coded paradigm tables: 10 nouns declined, 6 verbs in 4 tenses, single-form questions |
| arithmetic | 200 | addition and subtraction in Latin number words, 1–20 |

~23k training tokens — tiny, so the run makes many passes; best-by-val
checkpointing stops the damage from late overfitting.

## Recipe

Plain AdamW at a low learning rate (1e-4, warmup + cosine), 400 steps,
batch 16 × 512. The optimizer is deliberately *not* Muon: low-LR AdamW is the
standard SFT recipe, and the optimizer comparison already lives in 03.

## What to expect

A 53M-parameter model with 15M pretraining tokens becomes a model that
*converses in format*: it greets back, answers simple questions in plausible
Latin, declines nouns, does small sums. It has very little knowledge to
retrieve, so factual answers are often confidently wrong — a clean
demonstration of what SFT does (format, register, turn-taking) and does not
do (add knowledge).

## Run

```
uv run python experiments/04_chat_sft/make_dataset.py
uv run train 04_chat_sft --run-name sft_v1
python -m loquor            # picks up the chat checkpoint, applies the template
```
