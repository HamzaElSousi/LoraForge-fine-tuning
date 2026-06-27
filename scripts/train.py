#!/usr/bin/env python3
"""Headless QLoRA fine-tune with Unsloth (mirror of notebooks/01-finetune.ipynb).

Runs on a single CUDA GPU (a free T4 is enough for Qwen3 4B in 4-bit).
This is the same logic as the notebook for people who prefer ``python scripts/train.py``
on a rented box; Hamza will normally use the notebook on Kaggle instead.

It expects the conversational JSONL produced by ``scripts/prepare_dataset.py`` and applies
the *base model's own* chat template via ``tokenizer.apply_chat_template`` so training
formatting matches what the model serves later. Saves a LoRA adapter + loss.png; merging
to fp16 and GGUF conversion are separate steps (scripts/merge_adapter.py, quantize.sh).

Example (full run):
    python scripts/train.py \
        --model unsloth/Qwen3-4B-Instruct-2507 \
        --train data/train.jsonl --test data/test.jsonl \
        --output outputs --epochs 3

Example (60-step smoke test on the sample set):
    python scripts/prepare_dataset.py --input data/sample_50.jsonl --outdir data
    python scripts/train.py --train data/train.jsonl --max-steps 60 --output outputs
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Default to the Unsloth Qwen3 4B Instruct (2507) repo (ungated, no HF token needed).
    p.add_argument("--model", default="unsloth/Qwen3-4B-Instruct-2507", help="base model repo")
    p.add_argument("--train", default="data/train.jsonl", help="conversational train JSONL")
    p.add_argument("--test", default=None, help="optional conversational eval JSONL")
    p.add_argument("--output", default="outputs", help="where to write the adapter + loss.png")
    p.add_argument("--max-seq-len", type=int, default=2048, help="cap to avoid T4 OOM")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max-steps", type=int, default=-1, help="override epochs; use 60 for a smoke run")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-4bit", action="store_true", help="use 16-bit LoRA instead of 4-bit QLoRA")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Imported lazily so --help and the unit tests work without a GPU stack installed.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load base in 4-bit (QLoRA). Unsloth picks a sane dtype for the GPU.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=not args.no_4bit,
        dtype=None,
    )

    # 2. Attach LoRA adapters. PRD hyperparameters: r=16, alpha=32, attention+MLP proj.
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",  # extends context + cuts VRAM
        random_state=args.seed,
    )

    # 3. Apply the model's native chat template. Our rows are already {"messages": [...]}.
    tokenizer = get_chat_template(tokenizer)  # uses the tokenizer's built-in template

    def format_chat(batch):
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in batch["messages"]
        ]
        return {"text": texts}

    train_ds = load_dataset("json", data_files=args.train, split="train").map(
        format_chat, batched=True, remove_columns=["messages"]
    )
    eval_ds = None
    if args.test and Path(args.test).exists():
        eval_ds = load_dataset("json", data_files=args.test, split="train").map(
            format_chat, batched=True, remove_columns=["messages"]
        )

    # 4. SFT. Keep PRD defaults; max_steps overrides epochs for the smoke run.
    cfg = SFTConfig(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=5,
        num_train_epochs=args.epochs if args.max_steps < 0 else 1,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=args.seed,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=cfg,
    )

    stats = trainer.train()
    print(f"Training done. final loss: {stats.training_loss:.4f}")

    # 5. Save the adapter + tokenizer (added tokens included) and the loss curve.
    adapter_dir = out / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Adapter saved to {adapter_dir}")

    _plot_loss(trainer, out / "loss.png")


def _plot_loss(trainer, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping loss.png")
        return

    history = trainer.state.log_history
    steps = [h["step"] for h in history if "loss" in h]
    losses = [h["loss"] for h in history if "loss" in h]
    if not losses:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, label="train loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("LoRAForge training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"Loss curve saved to {path}")


if __name__ == "__main__":
    main()
