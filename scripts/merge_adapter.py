#!/usr/bin/env python3
"""Merge a trained LoRA adapter back into the base model as a single fp16 HF model.

Output is a standard Transformers directory (weights + tokenizer + added tokens) that
``quantize.sh`` then converts to GGUF. Saving the tokenizer alongside the weights is not
optional: if added/special tokens are missing, GGUF conversion misaligns the vocab.

    python scripts/merge_adapter.py \
        --base unsloth/Qwen3-4B-Instruct-2507 \
        --adapter outputs/adapter \
        --output outputs/merged-16bit

Uses Unsloth's fast merge when available and falls back to plain peft so the script also
works on a box without Unsloth installed.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="unsloth/Qwen3-4B-Instruct-2507", help="base model repo")
    p.add_argument("--adapter", default="outputs/adapter", help="trained LoRA adapter dir")
    p.add_argument("--output", default="outputs/merged-16bit", help="merged fp16 output dir")
    p.add_argument("--max-seq-len", type=int, default=2048)
    return p.parse_args()


def merge_with_unsloth(args) -> bool:
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        return False
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,  # Unsloth loads base + adapter from the adapter dir
        max_seq_length=args.max_seq_len,
        load_in_4bit=False,
        dtype=None,
    )
    model.save_pretrained_merged(args.output, tokenizer, save_method="merged_16bit")
    print(f"[unsloth] merged fp16 model written to {args.output}")
    return True


def merge_with_peft(args) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    # Save the tokenizer from the adapter dir so added tokens survive into the merge.
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(str(out))
    print(f"[peft] merged fp16 model written to {out}")


def main() -> None:
    args = parse_args()
    if not merge_with_unsloth(args):
        print("Unsloth not available; merging with peft.")
        merge_with_peft(args)


if __name__ == "__main__":
    main()
