#!/usr/bin/env python3
"""Quantitative eval supplement: ROUGE-L + BERTScore on the held-out test split.

The headline eval is EvalKit (eval/suite-*.yaml, LLM-judge). This script adds reference-
based metrics: it asks an Ollama model each held-out instruction, then scores the answers
against the gold responses. Run it for base and fine-tuned and compare.

    # held-out split produced by scripts/prepare_dataset.py
    python eval/eval.py --model loraforge-ft   --test data/test.jsonl
    python eval/eval.py --model loraforge-base --test data/test.jsonl

Requires requirements-eval.txt (rouge-score, bert-score) and a running Ollama. Network is
local only. Keep the held-out split out of training (see RESEARCH.md pitfall on contamination).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

SYSTEM_PROMPT = (
    "You are a helpful, concise customer-support assistant. Answer only using information "
    "you are confident about. If a question is outside customer support or you do not know, "
    "say so plainly instead of guessing."
)


def load_test(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        msgs = json.loads(line)["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        gold = next(m["content"] for m in msgs if m["role"] == "assistant")
        pairs.append((user, gold))
    return pairs


def ask_ollama(base_url: str, model: str, question: str, timeout: float) -> str:
    body = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    }
    resp = httpx.post(f"{base_url}/api/chat", json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def rouge_l(preds: list[str], refs: list[str]) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
    return sum(scores) / len(scores) if scores else 0.0


def bert_f1(preds: list[str], refs: list[str]) -> float:
    from bert_score import score as bert_score

    _, _, f1 = bert_score(preds, refs, lang="en", rescale_with_baseline=True, verbose=False)
    return float(f1.mean())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Ollama model name (e.g. loraforge-ft or loraforge-base)")
    p.add_argument("--test", default="data/test.jsonl", help="held-out conversational JSONL")
    p.add_argument("--base-url", default="http://localhost:11434")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--limit", type=int, default=0, help="only score the first N examples (0 = all)")
    args = p.parse_args(argv)

    test_path = Path(args.test)
    if not test_path.exists():
        raise SystemExit(f"test split not found: {test_path} (run scripts/prepare_dataset.py first)")

    pairs = load_test(test_path)
    if args.limit:
        pairs = pairs[: args.limit]

    preds, refs = [], []
    for i, (question, gold) in enumerate(pairs, 1):
        print(f"  [{i}/{len(pairs)}] {question[:60]}...", file=sys.stderr)
        preds.append(ask_ollama(args.base_url, args.model, question, args.timeout))
        refs.append(gold)

    rl = rouge_l(preds, refs)
    bf = bert_f1(preds, refs)
    print(f"\nModel: {args.model}   (n={len(pairs)})")
    print(f"  ROUGE-L   (F1): {rl:.4f}")
    print(f"  BERTScore (F1): {bf:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
