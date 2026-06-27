# LoRAForge - Research (mid-2026)

> **Update (2026-06-26):** The base model changed from Gemma 3n E4B to **Qwen3-4B-Instruct-2507**.
> Gemma 3n is multimodal and its Unsloth QLoRA support is experimental; training crashed in
> the forward pass (`Gemma3nRMSNorm` patch bug). Qwen3 4B is ungated, text-only, and fully
> supported by Unsloth, so no HF token is needed. Mentions of "Gemma 4 E4B" below are
> historical; see `README.md` and `CONCEPTS.md` for the current setup.

Findings that shaped `PLAN.md`. Current as of June 2026; re-verify versions at build time.

## 1. Trainer: use Unsloth (not raw peft + trl)

Unsloth is the 2026 default for single-GPU QLoRA: roughly **2–5x faster training and
~70% less VRAM** than vanilla HuggingFace, with longer context support. It wraps
`peft` + `trl` under a simpler API and ships ready-made Colab/Kaggle notebooks per model.

Practical notes pulled from current guides:
- A **free T4 (16 GB)** is enough to QLoRA-fine-tune a multi-billion-param model; the gap
  between full fine-tune and QLoRA is ~1–2%.
- Set `use_gradient_checkpointing="unsloth"` (designed to extend context + cut VRAM).
- OOM playbook: reduce `max_seq_length` first, then `per_device_train_batch_size` to 1.
- Avoid 4-bit QLoRA on MoE models; use 16-bit LoRA there. (Our picks are dense, so QLoRA is fine.)

Implication: keep the PRD's hyperparameters (r=16, alpha=32, lr=2e-4, 3 epochs) but run
them through Unsloth. We still document the raw peft/trl path in CONCEPTS.md for the
"I understand what's underneath" interview story.

## 2. Base model: Gemma 4 E4B (primary), Qwen3.5-3B (fallback)

- **Gemma 4 E4B** fits a free T4 with QLoRA and is explicitly recommended as a fast
  starter in current Unsloth guides. Biggest win: Hamza already runs `gemma4:e4b` in
  Ollama, so we can evaluate **fine-tuned vs the exact base** with EvalKit. Cost: Gemma
  weights are gated on Hugging Face (accept license + HF token) - a one-time manual step.
- **Qwen3.5** (Apache-2.0, no gating) is the friction-free fallback if we want to skip HF
  license approval. Slightly less tidy for the base-vs-tuned story since the Ollama base
  differs, but trivial to pull.

## 3. Cloud GPU: Kaggle preferred, Colab backup

- **Kaggle:** guaranteed **30 GPU-hours/week**, **9-hour** sessions, T4 or P100 (16 GB),
  most reliable free option for longer jobs (fewest surprise disconnects).
- **Colab Free:** 15–30 GPU-h/week, up to 12-h sessions, T4 16 GB, but flakier and can
  drop long runs.
- Our QLoRA run on E4B is well under these limits, so either works; Kaggle is the safer
  default for an uninterrupted train → merge → quantize pass.

## 4. GGUF conversion + serving via Ollama

Confirmed current workflow (llama.cpp + Ollama):
1. `python convert_hf_to_gguf.py --outfile model-f16.gguf /path/to/merged-hf-model`
2. `./build/bin/llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M`
3. Ollama `Modelfile`:
   ```
   FROM ./models/model-q4_k_m.gguf
   PARAMETER temperature 0.7
   PARAMETER stop "<|im_end|>"     # match the chat template used in training
   SYSTEM "You are ..."
   ```
   then `ollama create loraforge-ft -f Modelfile`.

Why this beats the PRD's llama.cpp-server + Docker for us: **Ollama already exposes an
OpenAI-compatible `/v1/chat/completions`**, so importing the GGUF is one command and runs
on the stack Hamza already has. We keep a thin FastAPI layer in front only for the two
things Ollama doesn't do: **API-key auth** and a **locked server-side system prompt** -
which are the PRD's actual serving deliverables. Q4_K_M remains the size/quality sweet spot.

## 5. Evaluation: dogfood EvalKit

Serve base and fine-tuned models on Ollama, then run an EvalKit suite (it already supports
`provider: ollama` + `llm_judge`) against each and diff the pass rates / scores. This:
- proves the fine-tune actually improved something (PRD success criterion), and
- creates a cross-project narrative (LoRAForge produces the model, EvalKit grades it).
Keep ROUGE-L / BERTScore in `eval/eval.py` as a quantitative supplement on the held-out
test split, but EvalKit's LLM-judge + out-of-scope rejection cases are the headline.

## 6. Pitfalls to design around

- **Chat-template mismatch** between training and serving silently degrades quality -
  the Ollama `Modelfile` stop tokens / template must match the ChatML used in training.
- **Eval contamination:** hold out the 10% test split before training; never train on it.
- **VRAM creep:** long `max_seq_length` is the usual T4 OOM cause (cap ~2048).
- **`serve/` must not pull torch** - inference is GGUF via Ollama/llama.cpp only (keeps the
  serving image tiny and CPU-only).
- **Tokenizer/added-tokens** must be saved with the merged model or GGUF conversion misaligns.

## Sources
- [Fine-Tuning LLMs with QLoRA and Unsloth: Complete 2026 Guide](https://pockit.tools/blog/fine-tuning-llms-qlora-unsloth-complete-guide/)
- [Unsloth docs - Qwen3.5 fine-tune](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- [Gemma 4 Fine-Tuning on Consumer GPUs (CloudInsight)](https://cloudinsight.cc/en/blog/gemma-4-fine-tuning)
- [llama.cpp quantize README (ggml-org)](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [llama-quantize GGUF guide (2026)](https://knightli.com/en/2026/04/12/llama-quantize-gguf-guide/)
- [Convert a model to GGUF and deploy on Ollama (NVIDIA Brev docs)](https://docs.nvidia.com/brev/latest/ollama-brev.html)
- [Colab vs Kaggle free GPU limits 2026 (Thunder Compute)](https://www.thundercompute.com/blog/colab-alternatives-for-cheap-deep-learning-in-2025)
- [Google Colab Free Tier T4 guide 2026](https://aicreditmart.com/ai-credits-providers/google-colab-free-tier-t4-gpu-access-guide-2026/)
- Dataset: [Bitext Customer Support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
