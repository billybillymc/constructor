"""Minimal-pairs evaluation: what does each model actually know?

BLiMP-style: each item is a pair of sentences differing in one word. The
model scores both; it "passes" when it assigns the acceptable sentence a
higher mean per-token log-probability. Grammar categories are generated
from paradigm tables; the facts category is hand-written entity-binding
pairs (both sentences grammatical, one historically wrong).

Categories
  sv_number     subject-verb number agreement   (Puella cantat / *cantant)
  sv_person     pronoun-verb person agreement   (Ego canto / *cantat)
  adj_gender    noun-adjective gender agreement (dominus bonus / *bona)
  prep_case     preposition case government     (cum amico / *amicum)
  obj_case      transitive object case          (gladium tenet / *gladius)
  facts         entity binding                  (Aeneida scripsit Vergilius / Ovidius)

Usage
  uv run python evals/minimal_pairs.py              # newest ckpt per experiment
  uv run python evals/minimal_pairs.py --ckpt PATH  # just one checkpoint

Writes evals/minimal_pairs.jsonl (the items) and evals/results.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from latin_lm.tokenizer import load_tokenizer
from latin_lm.train_utils import detect_device
from loquor.__main__ import CHAT_DATA_DIR, CHAT_EXPERIMENTS, find_checkpoints, load_model

EVALS_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# pair generation
# --------------------------------------------------------------------------- #

def make_pairs() -> list[dict]:
    pairs: list[dict] = []

    def add(category: str, good: str, bad: str) -> None:
        pairs.append({"category": category, "good": good, "bad": bad})

    # (nom sg, nom pl, gender) — animate nouns, so any verb below fits
    nouns = [("puella", "puellae", "f"), ("regina", "reginae", "f"),
             ("dominus", "domini", "m"), ("servus", "servi", "m"),
             ("puer", "pueri", "m"), ("rex", "reges", "m"),
             ("miles", "milites", "m"), ("magister", "magistri", "m"),
             ("nauta", "nautae", "m"), ("agricola", "agricolae", "m")]
    verbs = [("cantat", "cantant"), ("laborat", "laborant"),
             ("ambulat", "ambulant"), ("dormit", "dormiunt"),
             ("currit", "currunt"), ("venit", "veniunt"),
             ("sedet", "sedent"), ("ridet", "rident"),
             ("scribit", "scribunt"), ("legit", "legunt")]
    for i, (sg, pl, _g) in enumerate(nouns):
        for v_sg, v_pl in (verbs[i % 10], verbs[(i + 3) % 10], verbs[(i + 6) % 10]):
            add("sv_number", f"{sg.capitalize()} {v_sg}.", f"{sg.capitalize()} {v_pl}.")
            add("sv_number", f"{pl.capitalize()} {v_pl}.", f"{pl.capitalize()} {v_sg}.")

    # pronoun-person: (pronoun, index into [1s,2s,3s,1p,2p,3p], wrong index)
    person_verbs = {
        "cantare": ["canto", "cantas", "cantat", "cantamus", "cantatis", "cantant"],
        "laborare": ["laboro", "laboras", "laborat", "laboramus", "laboratis", "laborant"],
        "dormire": ["dormio", "dormis", "dormit", "dormimus", "dormitis", "dormiunt"],
        "venire": ["venio", "venis", "venit", "venimus", "venitis", "veniunt"],
    }
    pronoun_cases = [("Ego", 0, 2), ("Tu", 1, 0), ("Nos", 3, 5), ("Vos", 4, 2)]
    for forms in person_verbs.values():
        for pron, right, wrong in pronoun_cases:
            add("sv_person", f"{pron} {forms[right]}.", f"{pron} {forms[wrong]}.")

    # noun-adjective gender agreement, predicative frame
    gendered = [("dominus", "m"), ("servus", "m"), ("puer", "m"), ("nauta", "m"),
                ("puella", "f"), ("regina", "f"), ("rosa", "f"), ("via", "f"),
                ("bellum", "n"), ("donum", "n"), ("templum", "n"), ("verbum", "n")]
    adjectives = [("bonus", "bona", "bonum"), ("magnus", "magna", "magnum"),
                  ("parvus", "parva", "parvum"), ("novus", "nova", "novum")]
    gender_idx = {"m": 0, "f": 1, "n": 2}
    for i, (noun, g) in enumerate(gendered):
        for adj in (adjectives[i % 4], adjectives[(i + 1) % 4], adjectives[(i + 2) % 4]):
            right = adj[gender_idx[g]]
            wrong = adj[(gender_idx[g] + 1) % 3]
            add("adj_gender", f"{noun.capitalize()} {right} est.",
                f"{noun.capitalize()} {wrong} est.")

    # preposition case government: (abl, acc) object forms
    abl_objects = [("amico", "amicum"), ("puella", "puellam"),
                   ("servo", "servum"), ("rege", "regem"), ("milite", "militem")]
    for prep in ("Cum", "Sine", "De"):
        for abl, acc in abl_objects:
            add("prep_case", f"{prep} {abl} ambulat.", f"{prep} {acc} ambulat.")
    acc_objects = [("urbem", "urbe"), ("villam", "villa"),
                   ("montem", "monte"), ("regem", "rege"), ("amicum", "amico")]
    for prep in ("Ad", "Per", "Ante"):
        for acc, abl in acc_objects:
            add("prep_case", f"{prep} {acc} venit.", f"{prep} {abl} venit.")

    # transitive object must be accusative, not nominative
    frames = [("Miles", "tenet"), ("Puella", "amat"),
              ("Dominus", "videt"), ("Servus", "habet")]
    objects = [("gladium", "gladius"), ("rosam", "rosa"), ("librum", "liber"),
               ("aquam", "aqua"), ("panem", "panis")]
    for subj, verb in frames:
        for acc, nom in objects:
            add("obj_case", f"{subj} {acc} {verb}.", f"{subj} {nom} {verb}.")

    # entity binding: both grammatical, one wrong about the world
    facts = [
        ("Aeneida scripsit Vergilius.", "Aeneida scripsit Ovidius."),
        ("Metamorphoses scripsit Ovidius.", "Metamorphoses scripsit Caesar."),
        ("Romam condidit Romulus.", "Romam condidit Cicero."),
        ("Galliam subegit Caesar.", "Galliam subegit Seneca."),
        ("Hannibal cum elephantis Alpes transiit.", "Cicero cum elephantis Alpes transiit."),
        ("Seneca philosophus Stoicus fuit.", "Plautus philosophus Stoicus fuit."),
        ("Plautus comoedias scripsit.", "Tacitus comoedias scripsit."),
        ("Tacitus Annales scripsit.", "Plautus Annales scripsit."),
        ("Iuppiter rex deorum est.", "Mercurius rex deorum est."),
        ("Neptunus deus maris est.", "Mars deus maris est."),
        ("Mars deus belli est.", "Neptunus deus belli est."),
        ("Minerva dea sapientiae est.", "Venus dea sapientiae est."),
        ("Venus dea amoris est.", "Minerva dea amoris est."),
        ("Augustus primus imperator Romanus fuit.", "Nero primus imperator Romanus fuit."),
        ("Cicero orator clarissimus fuit.", "Hannibal orator clarissimus fuit."),
        ("Troia equo ligneo capta est.", "Roma equo ligneo capta est."),
        ("Ulixes rex Ithacae fuit.", "Achilles rex Ithacae fuit."),
        ("Caesar Idibus Martiis occisus est.", "Augustus Idibus Martiis occisus est."),
        ("Scipio Hannibalem apud Zamam vicit.", "Brutus Hannibalem apud Zamam vicit."),
        ("Lupa Romulum et Remum nutrivit.", "Aquila Romulum et Remum nutrivit."),
        ("Tiberis per Romam fluit.", "Rhenus per Romam fluit."),
        ("Catullus carmina ad Lesbiam scripsit.", "Livius carmina ad Lesbiam scripsit."),
        ("Ovidius in exsilium missus est.", "Horatius in exsilium missus est."),
        ("Georgica scripsit Vergilius.", "Georgica scripsit Plautus."),
    ]
    for good, bad in facts:
        add("facts", good, bad)
    return pairs


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

@torch.no_grad()
def mean_logprob(model, tokenizer, device: str, text: str) -> float:
    ids = [tokenizer.eot_id] + tokenizer.encode(text)
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    logits, _ = model(x)
    logprobs = F.log_softmax(logits.float(), dim=-1)
    token_lp = logprobs[0, torch.arange(y.size(1), device=device), y[0]]
    return token_lp.mean().item()


def newest_per_experiment(ckpts: list[Path]) -> list[Path]:
    by_exp: dict[str, Path] = {}
    for p in ckpts:  # ckpts are sorted oldest-first, so later wins
        exp = p.parent.parent.name if p.parent.parent.name != "runs" else p.parent.name
        by_exp[exp] = p
    return sorted(by_exp.values())


def evaluate(ckpt_path: Path, pairs: list[dict], device: str) -> dict | None:
    model, experiment = load_model(ckpt_path, device)
    data_dir = CHAT_DATA_DIR if experiment in CHAT_EXPERIMENTS else Path("data/processed")
    tokenizer = load_tokenizer(data_dir)
    if model.cfg.vocab_size != tokenizer.vocab_size:
        print(f"  skipping {ckpt_path}: vocab {model.cfg.vocab_size} != "
              f"tokenizer {tokenizer.vocab_size}")
        return None
    wins: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    for pair in pairs:
        good = mean_logprob(model, tokenizer, device, pair["good"])
        bad = mean_logprob(model, tokenizer, device, pair["bad"])
        totals[pair["category"]] += 1
        if good > bad:
            wins[pair["category"]] += 1
    accuracy = {cat: wins[cat] / totals[cat] for cat in totals}
    accuracy["overall"] = sum(wins.values()) / sum(totals.values())
    return {"checkpoint": str(ckpt_path), "experiment": experiment,
            "accuracy": accuracy}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Minimal-pairs evaluation")
    ap.add_argument("--ckpt", type=Path, action="append", default=None,
                    help="checkpoint(s) to score (default: newest per experiment)")
    args = ap.parse_args(argv)

    pairs = make_pairs()
    with (EVALS_DIR / "minimal_pairs.jsonl").open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    counts = defaultdict(int)
    for pair in pairs:
        counts[pair["category"]] += 1
    print(f"{len(pairs)} pairs:", dict(counts))

    device = detect_device()
    ckpts = args.ckpt or newest_per_experiment(find_checkpoints(Path("runs")))
    results = []
    for ckpt in ckpts:
        print(f"scoring {ckpt} ...", flush=True)
        result = evaluate(ckpt, pairs, device)
        if result:
            results.append(result)

    categories = ["sv_number", "sv_person", "adj_gender", "prep_case",
                  "obj_case", "facts", "overall"]
    width = max(len(r["experiment"]) for r in results) + 2
    header = "model".ljust(width) + "".join(c.rjust(12) for c in categories)
    print("\n" + header)
    print("-" * len(header))
    for r in sorted(results, key=lambda r: r["experiment"]):
        row = r["experiment"].ljust(width)
        for cat in categories:
            value = r["accuracy"].get(cat)
            row += (f"{value:12.1%}" if value is not None else " " * 12)
        print(row)

    (EVALS_DIR / "results.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwrote {EVALS_DIR / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
