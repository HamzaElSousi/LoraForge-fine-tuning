"""Tests for scripts/prepare_dataset.py - Phase 0 verification.

Runs fully offline against data/sample_50.jsonl: no GPU, no network, no model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import prepare_dataset as pd  # noqa: E402

SAMPLE = REPO / "data" / "sample_50.jsonl"


def test_sample_file_exists_and_is_valid_jsonl():
    rows = list(pd._read_jsonl(SAMPLE))
    assert len(rows) == 50
    for row in rows:
        assert set(row) >= {"instruction", "response"}


def test_extract_pair_tolerates_aliases():
    assert pd.extract_pair({"prompt": "hi", "completion": "hello"}) == ("hi", "hello")
    assert pd.extract_pair({"question": "q", "answer": "a"}) == ("q", "a")
    assert pd.extract_pair({"instruction": "  ", "response": "x"}) is None
    assert pd.extract_pair({"instruction": "x"}) is None


def test_to_messages_shape():
    msg = pd.to_messages("ask", "reply")
    assert msg == {
        "messages": [
            {"role": "user", "content": "ask"},
            {"role": "assistant", "content": "reply"},
        ]
    }


def test_clean_and_dedupe_drops_duplicates_and_empties():
    raw = [
        {"instruction": "Same Q", "response": "a"},
        {"instruction": "same q", "response": "b"},  # dupe (case-insensitive)
        {"instruction": "", "response": "c"},          # empty instruction
        {"instruction": "Other", "response": "d"},
    ]
    pairs = pd.clean_and_dedupe(raw)
    assert [p[0] for p in pairs] == ["Same Q", "Other"]


def test_split_is_ninety_ten_and_deterministic():
    pairs = [(f"q{i}", f"a{i}") for i in range(50)]
    train, test = pd.split(pairs, test_size=0.1, seed=42)
    assert len(train) == 45
    assert len(test) == 5
    # Deterministic for a fixed seed.
    train2, test2 = pd.split(pairs, test_size=0.1, seed=42)
    assert train == train2 and test == test2
    # No leakage between splits.
    assert set(train).isdisjoint(set(test))


def test_split_never_empties_train_on_tiny_input():
    train, test = pd.split([("q", "a"), ("q2", "a2")], test_size=0.5, seed=0)
    assert len(train) >= 1


def test_end_to_end_writes_valid_conversational_jsonl(tmp_path):
    rc = pd.main(["--input", str(SAMPLE), "--outdir", str(tmp_path), "--seed", "42"])
    assert rc == 0

    train_path, test_path = tmp_path / "train.jsonl", tmp_path / "test.jsonl"
    train = [json.loads(l) for l in train_path.read_text(encoding="utf-8").splitlines()]
    test = [json.loads(l) for l in test_path.read_text(encoding="utf-8").splitlines()]

    assert len(train) == 45
    assert len(test) == 5
    for rec in train + test:
        msgs = rec["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert all(m["content"].strip() for m in msgs)


def test_main_rejects_bad_test_size(tmp_path):
    with pytest.raises(SystemExit):
        pd.main(["--input", str(SAMPLE), "--outdir", str(tmp_path), "--test-size", "1.5"])
