# LoRAForge - Next-Session Kickoff

> **Update (2026-06-26):** The base model changed from Gemma 3n E4B to **Qwen3-4B-Instruct-2507**.
> Gemma 3n is multimodal and its Unsloth QLoRA support is experimental; training crashed in
> the forward pass (`Gemma3nRMSNorm` patch bug). Qwen3 4B is ungated, text-only, and fully
> supported by Unsloth, so no HF token is needed. Mentions of "Gemma 4 E4B" below are
> historical; see `README.md` and `CONCEPTS.md` for the current setup.

Everything needed to start the LoRAForge build cold in a new session. Plan is in
`PLAN.md`, research in `RESEARCH.md`.

---

## Kickoff prompt (paste this to start the session)

> Start the LoRAForge build. Read `loraforge/PLAN.md` and `loraforge/RESEARCH.md` first,
> then begin at Phase 0 and work through the phases.
>
> Config: fine-tune **Gemma 4 E4B** (fall back to Qwen3.5-3B if the HF license gate is a
> hassle) on the **Bitext customer-support dataset**, train on **Kaggle** (Colab backup),
> quantize to GGUF, serve it on **Ollama** plus a thin FastAPI auth proxy, and evaluate
> **base vs fine-tuned with EvalKit**.
>
> I've done the account/token setup in `KICKOFF.md`. Build everything that doesn't need a
> GPU. When the training notebook is ready, hand it to me to "Run all" on Kaggle; I'll
> download the resulting `.gguf` into `loraforge/models/` and give you the path. Then take
> it through quantize-side wiring, Ollama serving, EvalKit eval, tests, and docs.
> No em dashes in the README. Don't run pytest automatically; give me the command.

---

## Your manual todos

### A. Before we start (accounts + access - ~15 min)
- [ ] **Kaggle account**, then phone-verify it to unlock GPU. In a new notebook, confirm
      Settings → Accelerator shows **GPU T4**. (Alternative: a Google account for Colab.)
- [ ] **Hugging Face account** + create a **read access token**
      (huggingface.co → Settings → Access Tokens).
- [ ] **If using Gemma 4 E4B:** open the model page on Hugging Face and **accept the
      license** (gated model). Skip this if we go with Qwen3.5 (Apache-2.0, no gate).
- [ ] Decide the three open questions in `PLAN.md` (base model / dataset / GPU host), or
      just tell me "use the defaults" and I'll proceed with Gemma 4 E4B + Bitext + Kaggle.
- [ ] (Already true for you) local **Ollama** running with ~6–8 GB free RAM for serving
      the quantized model.

### B. The one step only you can do - the GPU training run
I'll write the full notebook; you execute it on the cloud GPU:
- [ ] Open `notebooks/01-finetune.ipynb` on Kaggle (or Colab), set runtime to **T4 GPU**.
- [ ] Paste your **HF token** into the secret/first cell.
- [ ] **Run all.** It trains → merges the adapter → converts to GGUF → quantizes Q4_K_M.
- [ ] **Download** `loraforge-<model>-q4_k_m.gguf` and `loss.png`, drop the `.gguf` into
      `loraforge/models/`, and tell me the path. I take over from there (Ollama import,
      serving, eval, docs).

### C. Optional - custom domain instead of Bitext
- [ ] Provide a JSONL of `{"instruction": "...", "response": "..."}` (≥ 500 rows, ideally
      2,000–5,000) and I'll point the pipeline at it instead of the public dataset.

---

## What I'll do (so you don't have to)
Scaffold the repo, dataset prep, the Unsloth training notebook + `train.py`, merge +
quantize scripts, the Ollama `Modelfile`, the FastAPI auth/OpenAI-compat proxy +
Docker, the EvalKit base-vs-fine-tuned eval, tests, `CONCEPTS.md`, and the README.
The only thing I can't do is press "Run all" on a GPU - that's step B.
