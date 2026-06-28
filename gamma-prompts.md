# Gamma prompts for LoRAForge decks

Paste one of these into Gamma (gamma.app -> Create -> "Paste in text" or the prompt box). Each
is self-contained with the real numbers so Gamma does not invent them. Companion markdown decks:
`presentation-technical.md` and `presentation-general.md`.

---

## Prompt A: Technical deck (for engineers)

> Create a 14-slide technical presentation titled "LoRAForge" for a software-engineering and
> ML/AI-engineer audience. Tone: precise, confident, no hype, no em dashes anywhere.
>
> Subject: a from-scratch pipeline that QLoRA fine-tunes the open Qwen3-4B-Instruct model on the
> Bitext customer-support dataset, quantizes it to a 2.5 GB Q4_K_M GGUF, serves it on Ollama
> behind a FastAPI auth proxy, and evaluates base vs fine-tuned. The one GPU step runs on a free
> Kaggle T4; everything after runs on CPU.
>
> Slides, in order:
> 1. Title and a one-line summary.
> 2. System overview as a pipeline: dataset prep (neutral messages JSONL, 90/10 split) to Unsloth
>    QLoRA on a T4, to merge into fp16, to llama.cpp quantize to GGUF Q4_K_M, to Ollama plus a
>    FastAPI proxy, to EvalKit. Note only one 2.5 GB file crosses from GPU to laptop.
> 3. QLoRA mechanics: 4-bit frozen base plus LoRA adapters on the q,k,v,o and gate,up,down
>    projections; r=16, alpha=32; trained 33M of 4.06B params (0.81%); lr 2e-4, 3 epochs,
>    effective batch 8, sequence length 2048; final loss 0.57.
> 4. Chat-template correctness as the silent failure mode: keep the dataset template-agnostic,
>    apply each model's own template at train and serve, and match the Ollama stop token to
>    Qwen3's ChatML token <|im_end|>, or generations run on.
> 5. Quantization: merge to fp16, convert to GGUF, quantize to Q4_K_M; 8 GB down to 2.5 GB,
>    CPU-runnable.
> 6. Serving: a no-torch FastAPI proxy adds API-key auth (401 on a bad or missing key) and a
>    locked server-side system prompt that strips client system messages; a tested prompt-injection
>    attempt was blocked.
> 7. Headless training via the Kaggle API (push, poll status, pull output), with one browser click
>    to force a T4, because the API's enable_gpu flag only provisions a legacy P100.
> 8. Two real failures: Gemma 3n (multimodal) crashed Unsloth's training forward pass with a
>    Gemma3nRMSNorm error, so I switched to text-only Qwen3-4B; and the Kaggle P100 (compute
>    capability sm_60) was incompatible with the PyTorch build (cudaErrorNoKernelImageForDevice),
>    so I forced a T4 (sm_75).
> 9. Evaluation methodology: two EvalKit suites with identical cases (6 support-quality, 4
>    out-of-scope), judged by a local model and fully offline; a 10% split held out before training
>    as a contamination guard; plus ROUGE-L on a held-out subsample.
> 10. Results as a table: support quality 67% to 100%, out-of-scope rejection 50% to 75%, overall
>     pass rate 60% to 90%, ROUGE-L 0.191 to 0.400.
> 11. Honest caveats: out-of-scope rejection missed the 80% target because the data teaches
>     helpfulness not refusal; about 10s CPU latency for a 4B Q4_K_M model; 3 epochs took about 8
>     hours on one T4.
> 12. Engineering lessons: change the dependency before patching it; "GPU on" is not the GPU you
>     assumed; chat-template correctness is the quiet failure mode; report the delta and the misses;
>     verify live.
> 13. Try it: an Ollama one-line install pulling the model from Hugging Face, plus links to the
>     GitHub repo (github.com/HamzaElSousi/LoraForge-fine-tuning) and the HF model
>     (huggingface.co/HamzaElSousi/loraforge-qwen3-4b-gguf).
> 14. Questions.
>
> Use clean, minimal slides with short bullets, and simple tables for the stack and the results.
> No em dashes anywhere.

---

## Prompt B: Outcome-focused deck (for a general audience)

> Create a 10-slide outcome-focused presentation titled "LoRAForge" for a general, non-technical
> audience such as recruiters, managers, and a mixed crowd. Tone: clear, plain language, confident,
> story-driven, no jargon, no code, no em dashes anywhere.
>
> Core message: I built and own a custom AI instead of renting one. I taught a small open AI model
> to be a customer-support agent, shrank it to run on a normal laptop, and proved it improved, all
> on free hardware.
>
> Slides, in order:
> 1. Title and a one-line summary: I built and own a custom AI instead of renting one.
> 2. The big idea: most people rent AI, paying per message for a model they do not control; I
>    wanted to own one.
> 3. What it does, in plain terms: I took a general-purpose AI and trained it on thousands of real
>    customer-support conversations; the result runs on a laptop with no monthly bill and no data
>    leaving the machine. Use the analogy of hiring a generalist and giving them a week of
>    on-the-job training for one role.
> 4. Why it matters: cost (you own it, no per-message fees), privacy (it runs locally), and control
>    (you decide how it behaves).
> 5. The story, because it did not go to plan: two real roadblocks (a model that would not cooperate
>    with the tools, and a free GPU that was not the one promised and could not run the code), both
>    solved by smart choices instead of brute force.
> 6. The results: measured before and after training on questions it had never seen; support-answer
>    quality went from roughly 60% to 90%, and it sounds much more like a real support agent.
> 7. Being honest about the limits: it still answers some off-topic questions it should politely
>    decline (a fixable data gap), and it replies in a few seconds on a laptop rather than instantly
>    (the cost of running locally without a graphics card).
> 8. What this shows: I can build and adapt AI, not just call an API; I make practical trade-offs
>    under real constraints; I measure honestly and report the misses; and I ship end to end.
> 9. It is real and public: anyone can download the model from Hugging Face and the full project
>    from GitHub.
> 10. Thank you and questions.
>
> Keep slides visual and uncluttered, with short phrases not paragraphs. No technical terms, no
> code, no em dashes anywhere.
