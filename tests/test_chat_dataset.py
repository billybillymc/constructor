"""Tests for the 04_chat_sft dataset builder (mask alignment above all)."""

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def md():
    path = ROOT / "experiments" / "04_chat_sft" / "make_dataset.py"
    spec = importlib.util.spec_from_file_location("chat_make_dataset", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tok(md):
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=300, special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(["salve amice quid agis bene mecum agitur"], trainer=trainer)
    tok.add_special_tokens([md.USER, md.ASSISTANT])
    return tok


def test_mask_aligns_with_assistant_tokens(md, tok):
    uid = tok.token_to_id(md.USER)
    aid = tok.token_to_id(md.ASSISTANT)
    eot = tok.token_to_id("<|endoftext|>")
    turns = [["Salve!", "Salve et tu!"], ["Quid agis?", "Bene mecum agitur."]]
    ids, mask = md.encode_dialogue(tok, uid, aid, eot, turns)
    assert len(ids) == len(mask)
    # structural tokens never carry loss; every EOT does
    for i, t in zip(ids, mask):
        if i in (uid, aid):
            assert t == 0
        if i == eot:
            assert t == 1
    assert ids[0] == uid and ids[-1] == eot and mask[-1] == 1
    # decoding exactly the masked tokens reconstructs the answers
    answer_ids = [i for i, t in zip(ids, mask) if t == 1 and i != eot]
    decoded = tok.decode(answer_ids, skip_special_tokens=False)
    assert "Salve et tu!" in decoded and "Bene mecum agitur." in decoded
    # two turns -> two EOTs
    assert sum(1 for i in ids if i == eot) == 2


def test_seed_dialogues_parse(md):
    dialogues = md.load_seed_dialogues()
    assert len(dialogues) > 150
    for turns in dialogues:
        for pair in turns:
            assert len(pair) == 2
            assert pair[0].strip() and pair[1].strip()


def test_morphology_drills_deterministic_and_correct(md):
    a = md.morphology_drills(random.Random(7))
    b = md.morphology_drills(random.Random(7))
    assert a == b
    text = json.dumps(a)
    assert "rosarum" in text          # full declension present
    assert "'regis'" in text          # single-case answer present
    assert "amavimus" in text         # perfect tense conjugated


def test_arithmetic_drills_deterministic(md):
    a = md.arithmetic_drills(random.Random(7))
    b = md.arithmetic_drills(random.Random(7))
    assert a == b
    assert len(a) == 200
