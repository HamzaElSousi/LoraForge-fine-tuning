# LoRAForge - Build Plan (for next session)

> **Update (2026-06-26):** The base model changed from Gemma 3n E4B to **Qwen3-4B-Instruct-2507**.
> Gemma 3n is multimodal and its Unsloth QLoRA support is experimental; training crashed in
> the forward pass (`Gemma3nRMSNorm` patch bug). Qwen3 4B is ungated, text-only, and fully
> supported by Unsloth, so no HF token is needed. Mentions of "Gemma 4 E4B" below are
> historical; see `README.md` and `CONCEPTS.md` for the current setup.

## Context

LoRAForge is project #5 in the Open-Source AI Engineering Series: fine-tune an open
LLM on a domain, quantize it to GGUF, and serve it as a self-hosted OpenAI-compatible
API. It is the one project that proves "I can train and adapt models," not just call
APIs. PRD is complete at `loraforge/PRD.md`; this plan adapts it to mid-2026 tooling
and wires it into the rest of the series (Ollama + EvalKit). See `RESEARCH.md` for the
evidence behind each decision.

**Hard constraint:** Hamza has no local GPU. Training runs on a **free cloud T4**
(Kaggle preferred). Everything after training (quantize, serve, evaluate) runs locally
on CPU via the Ollama he already uses. See `KICKOFF.md` for the manual todos.

## Key decisions (and where they differ from the PRD)

| Topic | PRD | This build | Why |
|---|---|---|---|
| Trainer | raw `peft` + `trl` SFTTrainer | **Unsloth** (wraps peft/trl) | 2–5x faster, ~70% less VRAM, fits a free T4 comfortably; current 2026 standard |
| Base model | Llama 3 8B (gated, 16 GB) | **Gemma 4 E4B** (primary), Qwen3.5-3B (no-gating fallback) | E4B fits free T4 with QLoRA AND lets us compare against the `gemma4:e4b` already in Ollama. Qwen is Apache-2.0, no license gate, if we want to skip HF approval |
| Cloud GPU | Colab Free | **Kaggle** (30 GPU-h/week, 9-h sessions, reliable), Colab as backup | Kaggle disconnects far less on longer runs |
| GGUF → serve | llama.cpp server + FastAPI + Docker | **Ollama Modelfile** (`ollama create`) + thin FastAPI auth proxy | Ollama already gives `/v1/chat/completions`; importing the GGUF is one command and matches the series. Keep a small FastAPI layer only for API-key auth + server-side system prompt (the PRD's real differentiators) |
| Evaluation | bespoke `eval.py` (ROUGE/BERTScore/LLM-judge) | **EvalKit** for the LLM-judge + base-vs-tuned comparison; keep ROUGE/BERTScore as a supplement | Dogfoods our own eval harness; serve base + fine-tuned on Ollama and run an EvalKit suite against each |

**The series-integration story (the resume gold):** fine-tune Gemma 4 E4B → quantize to
GGUF → `ollama create` the fine-tuned model → run the same EvalKit suite against base
`gemma4:e4b` and the fine-tuned model → show the improvement table. One narrative across
LoRAForge + Ollama + EvalKit.

## Default domain/dataset

Customer-support assistant on the **Bitext Customer Support** dataset (27K examples, free,
no registration) - proven quick-start. Format to ChatML. (Swappable for a custom domain
later; the pipeline is dataset-agnostic.) This pairs naturally with EvalKit support-style
rubrics and the DocChat/Acme theme.

## Split of work: cloud GPU vs local

- **Cloud (Kaggle/Colab notebook - Hamza runs "Run all", I write the cells):** load base in
  4-bit (Unsloth/QLoRA) → format dataset → LoRA train → merge adapter → `convert_hf_to_gguf`
  → quantize Q4_K_M → **download a single ~3 GB `.gguf`** + `loss.png` + the adapter.
- **Local (I drive, CPU only):** `ollama create` from the GGUF, the FastAPI auth proxy,
  the EvalKit base-vs-tuned eval, tests, README, packaging.

Keeping merge+GGUF inside the notebook means only the final `.gguf` crosses machines.

## Phases

### Phase 0 - Scaffold + dataset prep (local)
- Repo skeleton per PRD structure (`scripts/`, `serve/`, `eval/`, `data/`, `notebooks/`,
  `tests/`), `requirements-train.txt` / `requirements-serve.txt` (serve has **no torch**),
  `.env.example`, MIT `LICENSE`, `.gitignore` (models/*.gguf ignored).
- `scripts/prepare_dataset.py`: raw `{instruction,response}` JSONL → ChatML train/test split
  (90/10). `data/sample_50.jsonl` to smoke-test the pipeline without the full dataset.
- **Verify:** prepare the sample set; assert ChatML formatting + split counts.

### Phase 1 - Training notebook + script (runs on cloud T4)
- `notebooks/01-finetune.ipynb` (Kaggle/Colab) using Unsloth: 4-bit load, LoRA
  (r=16, alpha=32, target q/k/v/o_proj), SFT (defaults from PRD; `use_gradient_checkpointing="unsloth"`),
  save adapter + `loss.png`.
- `scripts/train.py`: same logic for headless runs.
- **Verify:** a 60-step smoke run on `sample_50.jsonl` completes on a T4 and loss decreases.
  (Hamza runs this; I review the loss curve + logs he pastes back.)

### Phase 2 - Merge + quantize to GGUF (in the notebook)
- Notebook cells: merge LoRA into base → `convert_hf_to_gguf.py` → `llama-quantize` Q4_K_M
  → download `loraforge-<model>-q4_k_m.gguf`.
- `scripts/merge_adapter.py` + `scripts/quantize.sh` mirror this for local/cloud reuse.
- **Verify:** GGUF loads and answers a prompt (in-notebook `llama-cli` sanity check).

### Phase 3 - Serve on Ollama + FastAPI auth proxy (local)
- `serve/Modelfile` (`FROM ./models/<model>.gguf`, system prompt, stop tokens);
  `ollama create loraforge-ft -f serve/Modelfile`.
- `serve/main.py`: FastAPI `/v1/chat/completions` that checks an API key, injects the
  server-side system prompt, and proxies to Ollama's `/v1`. `serve/Dockerfile` +
  `docker-compose.yml` (no torch). Drop-in for any OpenAI SDK client.
- **Verify:** `curl` with the key returns an OpenAI-shaped response; wrong key → 401;
  OpenAI Python SDK pointed at it works.

### Phase 4 - Evaluate base vs fine-tuned with EvalKit (local)
- `ollama create loraforge-base` from base `gemma4:e4b`; serve both.
- An EvalKit suite (reuse `evalkit`) with support rubrics + out-of-scope rejection cases;
  run it against base and fine-tuned, capture both HTML reports + the score delta.
- Optional supplement: `eval/eval.py` for ROUGE-L / BERTScore on the held-out test split.
- **Verify:** fine-tuned beats base by the PRD targets (≥15% LLM-judge, ≥80% rejection);
  record real numbers in the README + PRD resume bullets.

### Phase 5 - Tests, docs, packaging
- `tests/test_api.py` (FastAPI auth + OpenAI-compat, mocked Ollama - no real calls in CI).
- `CONCEPTS.md` (LoRA/QLoRA/GGUF explained), README with Mermaid diagrams, the eval table,
  and the Ollama-first quickstart. **No em dashes in the README** (per standing preference).
- Provide the pytest command; do not auto-run tests unless asked.

## Success criteria (from PRD, made concrete)
- Fine-tuned ≥ 15% higher than base on the EvalKit LLM-judge suite.
- Out-of-scope rejection ≥ 80% (base typically < 30%).
- Quantized model runs locally via Ollama in < 2s/response on CPU.
- `/v1/chat/completions` passes an OpenAI SDK call with API-key auth.
- Whole pipeline reproducible from the README in ~one working day.

## Open decisions to confirm at kickoff
1. Base model: **Gemma 4 E4B** (ties to existing Ollama base; needs HF license accept) vs
   **Qwen3.5-3B** (no gating, simplest). 
2. Domain/dataset: **Bitext customer support** default vs a custom domain.
3. GPU host: **Kaggle** (recommended) vs Colab.
