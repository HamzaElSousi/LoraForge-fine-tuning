#!/usr/bin/env python3
"""Prepare a raw instruction/response dataset for SFT.

Reads raw rows of ``{"instruction": ..., "response": ...}`` (a local JSONL/CSV file,
or the public Bitext customer-support set via ``--from-hf``), cleans and de-duplicates
them, then writes a 90/10 train/test split as *conversational* JSONL:

    {"messages": [{"role": "user", "content": ...},
                  {"role": "assistant", "content": ...}]}

The conversational ``messages`` format is deliberately model-agnostic. The training
notebook applies the *base model's own* chat template (Qwen, Gemma, ...) with
``tokenizer.apply_chat_template`` so the formatting always matches what the model was
pretrained on. Do not bake a specific template (e.g. ChatML ``<|im_end|>``) in here, or
training and serving will silently disagree and quality will drop.

Usage
-----
    # Smoke set used by the tests and the notebook dry-run:
    python scripts/prepare_dataset.py --input data/sample_50.jsonl --outdir data

    # The full Bitext customer-support dataset (downloads ~27K rows from HF):
    python scripts/prepare_dataset.py --from-hf --outdir data

    # A custom domain (>=500 rows recommended):
    python scripts/prepare_dataset.py --input data/my_domain.jsonl --outdir data
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable, Iterator

BITEXT_HF_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"


def _read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc


def _read_csv(path: Path) -> Iterator[dict]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as fh:
        yield from csv.DictReader(fh)


def _read_from_hf() -> Iterator[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - only when datasets missing
        raise SystemExit(
            "--from-hf needs the `datasets` package: pip install datasets"
        ) from exc
    ds = load_dataset(BITEXT_HF_ID, split="train")
    for row in ds:
        yield dict(row)


def load_raw(args: argparse.Namespace) -> Iterator[dict]:
    if args.from_hf:
        return _read_from_hf()
    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"input file not found: {path}")
    if path.suffix.lower() == ".csv":
        return _read_csv(path)
    return _read_jsonl(path)


def extract_pair(row: dict) -> tuple[str, str] | None:
    """Pull (instruction, response) from a raw row, tolerating common column names.

    Bitext uses ``instruction``/``response``; other sources use ``prompt``/``completion``
    or ``question``/``answer``. Returns None if either side is missing or empty.
    """
    instr = row.get("instruction") or row.get("prompt") or row.get("question") or row.get("input")
    resp = row.get("response") or row.get("completion") or row.get("answer") or row.get("output")
    if not isinstance(instr, str) or not isinstance(resp, str):
        return None
    instr, resp = instr.strip(), resp.strip()
    if not instr or not resp:
        return None
    return instr, resp


def to_messages(instruction: str, response: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
    }


def clean_and_dedupe(rows: Iterable[dict]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for row in rows:
        pair = extract_pair(row)
        if pair is None:
            continue
        key = pair[0].lower()
        if key in seen:
            continue
        seen.add(key)
        pairs.append(pair)
    return pairs


def split(
    pairs: list[tuple[str, str]], test_size: float, seed: int
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * test_size)) if shuffled else 0
    # Never let the test split swallow everything on tiny inputs.
    n_test = min(n_test, max(0, len(shuffled) - 1))
    return shuffled[n_test:], shuffled[:n_test]


def write_jsonl(pairs: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for instruction, response in pairs:
            fh.write(json.dumps(to_messages(instruction, response), ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="raw JSONL or CSV of {instruction, response} rows")
    src.add_argument("--from-hf", action="store_true", help=f"download {BITEXT_HF_ID}")
    parser.add_argument("--outdir", default="data", help="output directory (default: data)")
    parser.add_argument("--test-size", type=float, default=0.1, help="held-out fraction (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="shuffle seed (default: 42)")
    args = parser.parse_args(argv)

    if not 0.0 < args.test_size < 1.0:
        parser.error("--test-size must be between 0 and 1")

    pairs = clean_and_dedupe(load_raw(args))
    if not pairs:
        raise SystemExit("no valid {instruction, response} rows found")

    train, test = split(pairs, args.test_size, args.seed)
    outdir = Path(args.outdir)
    write_jsonl(train, outdir / "train.jsonl")
    write_jsonl(test, outdir / "test.jsonl")

    total = len(pairs)
    print(f"Loaded {total} clean, de-duplicated pairs")
    print(f"  train: {len(train):>6}  -> {outdir / 'train.jsonl'}")
    print(f"  test:  {len(test):>6}  -> {outdir / 'test.jsonl'}")
    print("Format: conversational `messages` (chat template applied at train time).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
