"""Build the chat SFT dataset for 04_chat_sft.

Four sources, in one JSONL-of-dialogues intermediate and one packed binary:

  1. seed_dialogues.jsonl — hand-written Latin exchanges (greetings, facts,
     definitions, proverbs, composition, multi-turn).
  2. Donatus, Ars Minor, mined from the corpus — the original Latin grammar
     catechism, already question/answer shaped.
  3. Generated morphology drills — noun declensions and verb conjugations
     from hard-coded paradigm tables (no stem logic to get wrong).
  4. Generated arithmetic in Latin number words (1–20).

Output (data/processed/chat/):
  bpe_tokenizer.json + tokenizer_meta.json — the base 16k BPE tokenizer with
      <|user|> and <|assistant|> appended as special tokens
  {train,val}.bin       uint16 token streams (dialogues packed back to back)
  {train,val}_mask.bin  uint8, 1 where the loss applies (assistant text + EOT)
  chat_meta.json        counts, special ids, source composition

    uv run python experiments/04_chat_sft/make_dataset.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
USER, ASSISTANT = "<|user|>", "<|assistant|>"
SEED = 1457
VAL_FRAC = 0.05

Dialogue = list[list[str]]  # [[user, assistant], ...]


# --------------------------------------------------------------------------- #
# 1. hand-written seed dialogues
# --------------------------------------------------------------------------- #

def load_seed_dialogues() -> list[Dialogue]:
    out = []
    for path in sorted(HERE.glob("seed_dialogues*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line)["turns"])
    return out


# --------------------------------------------------------------------------- #
# 2. Donatus, Ars Minor (question/answer catechism) from the corpus
# --------------------------------------------------------------------------- #

def mine_donatus(corpus_path: Path) -> list[Dialogue]:
    text = None
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            work = json.loads(line)
            if work["title"] == "Donatus: Ars Minor":
                text = work["text"]
                break
    if text is None:
        print("warning: Donatus: Ars Minor not found in corpus; skipping")
        return []

    # A block is usually "question? answer..." — the question runs up to the
    # first "?", the answer is the rest of the block (or, when the block is
    # only a question, the next block).
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    dialogues: list[Dialogue] = []
    i = 0
    while i < len(blocks):
        block = blocks[i]
        if block.isupper() or "?" not in block[:200]:
            i += 1
            continue
        qpos = block.index("?")
        question = re.sub(r"\s+", " ", block[:qpos + 1]).strip()
        answer = block[qpos + 1:].strip()
        j = i + 1
        if (not answer and j < len(blocks) and not blocks[j].isupper()
                and "?" not in blocks[j][:80]):
            answer = blocks[j]
            j += 1
        answer = re.sub(r"\s+", " ", answer).strip()
        if question and 0 < len(answer) <= 600:
            dialogues.append([[question[0].upper() + question[1:], answer]])
        i = j
    return dialogues


# --------------------------------------------------------------------------- #
# 3. morphology drills from paradigm tables
# --------------------------------------------------------------------------- #

# forms in the traditional order: nom, gen, dat, acc, voc, abl
NOUNS = {
    "rosa": ("primae", "feminini",
             ["rosa", "rosae", "rosae", "rosam", "rosa", "rosa"],
             ["rosae", "rosarum", "rosis", "rosas", "rosae", "rosis"]),
    "puella": ("primae", "feminini",
               ["puella", "puellae", "puellae", "puellam", "puella", "puella"],
               ["puellae", "puellarum", "puellis", "puellas", "puellae", "puellis"]),
    "dominus": ("secundae", "masculini",
                ["dominus", "domini", "domino", "dominum", "domine", "domino"],
                ["domini", "dominorum", "dominis", "dominos", "domini", "dominis"]),
    "bellum": ("secundae", "neutrius",
               ["bellum", "belli", "bello", "bellum", "bellum", "bello"],
               ["bella", "bellorum", "bellis", "bella", "bella", "bellis"]),
    "rex": ("tertiae", "masculini",
            ["rex", "regis", "regi", "regem", "rex", "rege"],
            ["reges", "regum", "regibus", "reges", "reges", "regibus"]),
    "miles": ("tertiae", "masculini",
              ["miles", "militis", "militi", "militem", "miles", "milite"],
              ["milites", "militum", "militibus", "milites", "milites", "militibus"]),
    "urbs": ("tertiae", "feminini",
             ["urbs", "urbis", "urbi", "urbem", "urbs", "urbe"],
             ["urbes", "urbium", "urbibus", "urbes", "urbes", "urbibus"]),
    "corpus": ("tertiae", "neutrius",
               ["corpus", "corporis", "corpori", "corpus", "corpus", "corpore"],
               ["corpora", "corporum", "corporibus", "corpora", "corpora", "corporibus"]),
    "manus": ("quartae", "feminini",
              ["manus", "manus", "manui", "manum", "manus", "manu"],
              ["manus", "manuum", "manibus", "manus", "manus", "manibus"]),
    "dies": ("quintae", "masculini",
             ["dies", "diei", "diei", "diem", "dies", "die"],
             ["dies", "dierum", "diebus", "dies", "dies", "diebus"]),
}

CASE_NOM = ["nominativus", "genetivus", "dativus", "accusativus", "vocativus", "ablativus"]
CASE_ABL = ["nominativo", "genetivo", "dativo", "accusativo", "vocativo", "ablativo"]

# indicative active, persons 1s 2s 3s 1p 2p 3p
VERBS = {
    "amare": {
        "praesenti": ["amo", "amas", "amat", "amamus", "amatis", "amant"],
        "imperfecto": ["amabam", "amabas", "amabat", "amabamus", "amabatis", "amabant"],
        "futuro": ["amabo", "amabis", "amabit", "amabimus", "amabitis", "amabunt"],
        "perfecto": ["amavi", "amavisti", "amavit", "amavimus", "amavistis", "amaverunt"],
    },
    "monere": {
        "praesenti": ["moneo", "mones", "monet", "monemus", "monetis", "monent"],
        "imperfecto": ["monebam", "monebas", "monebat", "monebamus", "monebatis", "monebant"],
        "futuro": ["monebo", "monebis", "monebit", "monebimus", "monebitis", "monebunt"],
        "perfecto": ["monui", "monuisti", "monuit", "monuimus", "monuistis", "monuerunt"],
    },
    "regere": {
        "praesenti": ["rego", "regis", "regit", "regimus", "regitis", "regunt"],
        "imperfecto": ["regebam", "regebas", "regebat", "regebamus", "regebatis", "regebant"],
        "futuro": ["regam", "reges", "reget", "regemus", "regetis", "regent"],
        "perfecto": ["rexi", "rexisti", "rexit", "reximus", "rexistis", "rexerunt"],
    },
    "audire": {
        "praesenti": ["audio", "audis", "audit", "audimus", "auditis", "audiunt"],
        "imperfecto": ["audiebam", "audiebas", "audiebat", "audiebamus", "audiebatis", "audiebant"],
        "futuro": ["audiam", "audies", "audiet", "audiemus", "audietis", "audient"],
        "perfecto": ["audivi", "audivisti", "audivit", "audivimus", "audivistis", "audiverunt"],
    },
    "capere": {
        "praesenti": ["capio", "capis", "capit", "capimus", "capitis", "capiunt"],
        "imperfecto": ["capiebam", "capiebas", "capiebat", "capiebamus", "capiebatis", "capiebant"],
        "futuro": ["capiam", "capies", "capiet", "capiemus", "capietis", "capient"],
        "perfecto": ["cepi", "cepisti", "cepit", "cepimus", "cepistis", "ceperunt"],
    },
    "esse": {
        "praesenti": ["sum", "es", "est", "sumus", "estis", "sunt"],
        "imperfecto": ["eram", "eras", "erat", "eramus", "eratis", "erant"],
        "futuro": ["ero", "eris", "erit", "erimus", "eritis", "erunt"],
        "perfecto": ["fui", "fuisti", "fuit", "fuimus", "fuistis", "fuerunt"],
    },
}

PERSONS = [("prima", "singularis"), ("secunda", "singularis"), ("tertia", "singularis"),
           ("prima", "pluralis"), ("secunda", "pluralis"), ("tertia", "pluralis")]


def morphology_drills(rng: random.Random) -> list[Dialogue]:
    out: list[Dialogue] = []

    decline_q = [
        "Declina nomen '{n}'.",
        "Quomodo declinatur nomen '{n}'?",
        "Declina '{n}' per omnes casus.",
    ]
    for noun, (decl, gender, sg, pl) in NOUNS.items():
        sg_part = ", ".join(f"{c} '{f}'" for c, f in zip(CASE_ABL, sg))
        pl_part = ", ".join(f"{c} '{f}'" for c, f in zip(CASE_ABL, pl))
        answer = (f"Nomen '{noun}' {decl} declinationis est, generis {gender}. "
                  f"Singulariter: {sg_part}. Pluraliter: {pl_part}.")
        out.append([[rng.choice(decline_q).format(n=noun), answer]])
        out.append([[f"Cuius declinationis est nomen '{noun}'?",
                     f"Nomen '{noun}' {decl} declinationis est, generis {gender}."]])
        for num_label, forms in (("singulari", sg), ("plurali", pl)):
            for case_nom, case_abl, form in zip(CASE_NOM, CASE_ABL, forms):
                q = f"Quid est casus {case_nom} nominis '{noun}' numero {num_label}?"
                a = f"Casu {case_abl} numero {num_label} dicitur '{form}'."
                out.append([[q, a]])

    conj_q = [
        "Coniuga verbum '{v}' tempore {t}.",
        "Quomodo coniugatur verbum '{v}' tempore {t}?",
    ]
    for verb, tenses in VERBS.items():
        for tense, forms in tenses.items():
            out.append([[rng.choice(conj_q).format(v=verb, t=tense),
                         ", ".join(forms) + "."]])
            for (person, number), form in zip(PERSONS, forms):
                q = (f"Quid est {person} persona numeri {number} "
                     f"verbi '{verb}' tempore {tense}?")
                a = f"Tempore {tense} {person} persona numeri {number} est '{form}'."
                out.append([[q, a]])
    return out


# --------------------------------------------------------------------------- #
# 4. arithmetic in Latin number words
# --------------------------------------------------------------------------- #

NUM = {1: "unum", 2: "duo", 3: "tria", 4: "quattuor", 5: "quinque", 6: "sex",
       7: "septem", 8: "octo", 9: "novem", 10: "decem", 11: "undecim",
       12: "duodecim", 13: "tredecim", 14: "quattuordecim", 15: "quindecim",
       16: "sedecim", 17: "septendecim", 18: "duodeviginti", 19: "undeviginti",
       20: "viginti"}


def arithmetic_drills(rng: random.Random) -> list[Dialogue]:
    out: list[Dialogue] = []
    add = [(a, b) for a in range(1, 20) for b in range(1, 20) if a + b <= 20]
    for a, b in rng.sample(add, 120):
        q = rng.choice([f"Quot sunt {NUM[a]} et {NUM[b]}?",
                        f"Si {NUM[a]} et {NUM[b]} addis, quot fiunt?"])
        out.append([[q[0].upper() + q[1:],
                     f"{NUM[a].capitalize()} et {NUM[b]} sunt {NUM[a + b]}."]])
    sub = [(a, b) for a in range(2, 21) for b in range(1, 20) if a > b]
    for a, b in rng.sample(sub, 80):
        q = f"Si de {NUM[a]} {NUM[b]} demas, quot restant?"
        out.append([[q, f"Restant {NUM[a - b]}."]])
    return out


# --------------------------------------------------------------------------- #
# augmentation: users type lowercase and drop punctuation; the model should
# not fall apart when they do
# --------------------------------------------------------------------------- #

def augment_casing(dialogues: list[Dialogue], rng: random.Random) -> list[Dialogue]:
    """Every dialogue gets a lowercase variant: people type entirely in
    lowercase (proper nouns included), and often drop the final punctuation.
    Without systematic exposure the model treats 'caesar' as a stranger."""
    variants: list[Dialogue] = []
    for turns in dialogues:
        first_user = turns[0][0]
        varied = first_user.lower()
        if rng.random() < 0.5 and varied.endswith((".", "!", "?")):
            varied = varied[:-1]
        if varied != first_user:
            variants.append([[varied, turns[0][1]]] + [list(t) for t in turns[1:]])
    return variants


# --------------------------------------------------------------------------- #
# tokenizer + packing
# --------------------------------------------------------------------------- #

def encode_dialogue(tok: Tokenizer, uid: int, aid: int, eot: int,
                    turns: Dialogue) -> tuple[list[int], list[int]]:
    ids: list[int] = []
    mask: list[int] = []
    for user_text, assistant_text in turns:
        u = tok.encode(user_text).ids
        a = tok.encode(assistant_text).ids
        ids += [uid] + u + [aid]
        mask += [0] * (len(u) + 2)
        ids += a + [eot]
        mask += [1] * (len(a) + 1)
    return ids, mask


def main() -> int:
    data_dir = Path("data/processed")
    out_dir = data_dir / "chat"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    seed_dialogues = load_seed_dialogues()
    donatus = mine_donatus(Path("data/corpus/works.jsonl"))
    morphology = morphology_drills(rng)
    arithmetic = arithmetic_drills(rng)
    dialogues = seed_dialogues + donatus + morphology + arithmetic
    casing = augment_casing(dialogues, rng)
    sources = {"seed": len(seed_dialogues), "donatus": len(donatus),
               "morphology": len(morphology), "arithmetic": len(arithmetic),
               "casing_variants": len(casing)}
    dialogues += casing
    rng.shuffle(dialogues)
    print("dialogues:", json.dumps(sources), "| total", len(dialogues))

    # Base BPE tokenizer + the two chat special tokens appended at the end.
    base_meta = json.loads((data_dir / "tokenizer_meta.json").read_text(encoding="utf-8"))
    assert base_meta["kind"] == "bpe", "chat SFT assumes the BPE tokenizer"
    tok = Tokenizer.from_file(str(data_dir / "bpe_tokenizer.json"))
    tok.add_special_tokens([USER, ASSISTANT])
    tok.save(str(out_dir / "bpe_tokenizer.json"))
    (out_dir / "tokenizer_meta.json").write_text(
        json.dumps({"kind": "bpe", "eot_token": base_meta["eot_token"]}),
        encoding="utf-8")
    uid = tok.token_to_id(USER)
    aid = tok.token_to_id(ASSISTANT)
    eot = tok.token_to_id(base_meta["eot_token"])
    print(f"vocab {tok.get_vocab_size()} (user={uid}, assistant={aid}, eot={eot})")

    n_val = max(1, int(len(dialogues) * VAL_FRAC))
    splits = {"val": dialogues[:n_val], "train": dialogues[n_val:]}
    meta: dict = {"vocab_size": tok.get_vocab_size(), "user_id": uid,
                  "assistant_id": aid, "eot_id": eot, "sources": sources,
                  "splits": {}}
    for name, items in splits.items():
        ids: list[int] = []
        mask: list[int] = []
        for turns in items:
            i, m = encode_dialogue(tok, uid, aid, eot, turns)
            ids += i
            mask += m
        np.array(ids, dtype=np.uint16).tofile(out_dir / f"{name}.bin")
        np.array(mask, dtype=np.uint8).tofile(out_dir / f"{name}_mask.bin")
        meta["splits"][name] = {"dialogues": len(items), "tokens": len(ids),
                                "loss_tokens": int(sum(mask)), "dtype": "uint16"}
        print(f"{name}: {len(items)} dialogues, {len(ids):,} tokens "
              f"({sum(mask):,} under loss)")
    (out_dir / "chat_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
