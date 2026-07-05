# LoRAForge — V2+ Suggestions

**Current state:** complete train→quantize→serve→eval pipeline with published GGUF weights on the HF Hub; thin test coverage (2 test files) and no CI. The pipeline is real — v2 is about hardening it and deepening the training story.

## Prioritized suggestions

1. **CI pipeline** — Effort: S
   *What:* GitHub Actions running the two test files + ruff on push; add smoke tests for `prepare_dataset.py` (schema/split invariants) and the FastAPI proxy (auth, passthrough, error paths).
   *Why:* A fine-tuning repo with a green badge and tested data-prep signals engineering discipline where most fine-tune projects are notebook dumps.

2. **Publish the eval results table in the README** — Effort: S
   *What:* Run the EvalKit base-vs-tuned comparison and commit the scores table + a one-paragraph honest reading (where fine-tuning helped, where it didn't).
   *Why:* "I fine-tuned a model" is common; "here's the measured delta, including where it regressed" is rare and is the interview answer.

3. **Notebook → script parity** — Effort: M
   *What:* Make `scripts/train.py` runnable end-to-end on Kaggle/Colab via CLI args (currently the notebook is the canonical path), with a `make train-cloud` recipe.
   *Why:* Reproducibility-from-CLI turns the README's "one working day" claim into something a reviewer can actually execute.

4. **Second domain adapter** — Effort: M
   *What:* Train one more LoRA (e.g. SQL generation or your own support docs) and serve both behind the proxy with per-route model selection.
   *Why:* Proves the pipeline generalizes beyond the tutorial dataset; multi-adapter serving is a real production pattern.

5. **Quantization comparison** — Effort: M
   *What:* Q4_K_M vs Q5_K_M vs Q8_0: quality (EvalKit) and latency/RAM table on the same hardware.
   *Why:* The quantization trade-off table is the piece infra interviewers actually probe on.

6. **Adapter hot-swap without re-create** — Effort: L
   *What:* Serve fp16 base + runtime LoRA loading (llama.cpp `--lora` or vLLM) instead of baked merged GGUFs.
   *Why:* Demonstrates understanding of adapter mechanics beyond the merge-and-ship happy path.

## Quick wins (do this weekend)
- Add the CI workflow (tests already run offline)
- Run the eval and commit the results table
- Add 3-4 unit tests for dataset prep edge cases (empty rows, long samples, split determinism)
