from latin_lm.config import Config, SplitConfig
from latin_lm.prepare import split_works


def make_records(n=100, chars=1000):
    return [
        {"work_id": f"work-{i:03d}", "author": "A", "title": f"W{i}",
         "n_chars": chars, "n_words": chars // 6}
        for i in range(n)
    ]


def make_cfg(**kw):
    cfg = Config()
    cfg.split = SplitConfig(**kw)
    return cfg


def test_every_work_in_exactly_one_split():
    records = make_records()
    assignment = split_works(records, make_cfg())
    assert set(assignment) == {r["work_id"] for r in records}
    assert set(assignment.values()) <= {"train", "val", "test"}


def test_deterministic_given_seed():
    records = make_records()
    a = split_works(records, make_cfg(seed=7))
    b = split_works(records, make_cfg(seed=7))
    assert a == b


def test_different_seed_changes_assignment():
    records = make_records()
    a = split_works(records, make_cfg(seed=7))
    b = split_works(records, make_cfg(seed=8))
    assert a != b


def test_order_of_records_does_not_matter():
    records = make_records()
    a = split_works(records, make_cfg(seed=7))
    b = split_works(list(reversed(records)), make_cfg(seed=7))
    assert a == b


def test_fractions_roughly_met():
    records = make_records(n=200)
    assignment = split_works(records, make_cfg())
    total = sum(r["n_chars"] for r in records)
    by = {"train": 0, "val": 0, "test": 0}
    for r in records:
        by[assignment[r["work_id"]]] += r["n_chars"]
    assert 0.85 <= by["train"] / total <= 0.95
    assert 0.03 <= by["val"] / total <= 0.08
    assert 0.03 <= by["test"] / total <= 0.08


def test_big_work_forced_into_train():
    records = make_records(n=50, chars=1000)
    records.append({"work_id": "zz-huge", "author": "Livy", "title": "AUC",
                    "n_chars": 500_000, "n_words": 80_000})
    assignment = split_works(records, make_cfg())
    assert assignment["zz-huge"] == "train"
