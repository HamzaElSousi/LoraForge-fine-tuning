# Concepts behind LoRAForge

A plain-language tour of what the pipeline actually does, so the README can stay short and
this can answer the "explain what is underneath" interview question.

## 1. Fine-tuning vs prompting

Prompting steers a frozen model at inference time. Fine-tuning changes the model's weights
so the behavior is baked in: it answers in your domain's voice without a giant system
prompt, and it learns to decline things outside its scope. The cost is a training run and
the discipline of a clean dataset. LoRAForge fine-tunes on customer-support conversations
so the model behaves like a support agent by default.

## 2. LoRA (Low-Rank Adaptation)

Full fine-tuning updates every weight in the model, which for a multi-billion parameter
model needs far more memory than a free GPU has. LoRA freezes the original weights and
inserts small trainable matrices next to the attention and MLP projections. Instead of
learning a full weight update `W -> W + dW`, it learns `dW = B @ A`, where `A` and `B` are
low-rank (rank `r`, here 16). You train roughly 1 to 2 percent of the parameters, so the
optimizer state is tiny and the adapter you save is only a few megabytes.

Key hyperparameters used here:
- `r = 16`: the rank, how much capacity the adapter has.
- `lora_alpha = 32`: a scaling factor on the adapter output (commonly 2x the rank).
- target modules: `q_proj, k_proj, v_proj, o_proj` (attention) plus `gate/up/down_proj`
  (MLP). Adapting both attention and MLP is the current default for instruction tuning.

## 3. QLoRA (Quantized LoRA)

QLoRA goes one step further: it loads the frozen base model in 4-bit precision (NF4) to cut
memory roughly 4x, then trains the LoRA adapters on top in higher precision. The frozen
weights are only read during the forward pass, so 4-bit is fine for them, while the small
adapters stay accurate. This is what lets a multi-billion parameter model fine-tune on a
single free 16 GB T4. The reported quality gap versus full fine-tuning is about 1 to 2
percent.

## 4. Unsloth

Unsloth is a drop-in layer over Hugging Face `peft` and `trl` with hand-written CUDA
kernels. Same QLoRA math, but roughly 2 to 5x faster and about 70 percent less VRAM, plus
helpers like `get_chat_template`. The raw stack underneath is:
- `transformers` for the model,
- `peft` for the LoRA adapters,
- `trl`'s `SFTTrainer` for supervised fine-tuning,
- `bitsandbytes` for the 4-bit quantization.

`scripts/train.py` and the notebook use Unsloth; the bullet list above is the path you would
write by hand without it.

## 5. Chat templates (the quiet failure mode)

Each instruct model was trained with a specific way of marking who is speaking. Qwen3 (and
other ChatML models) use `<|im_start|>` / `<|im_end|>`; Gemma instead uses
`<start_of_turn>user ... <end_of_turn>`.
If you format training data one way and serve with a different template, the model sees
boundaries it never learned and quality silently drops.

LoRAForge avoids this by never hardcoding a template. `prepare_dataset.py` emits neutral
`{"messages": [...]}` rows, and both training and the Ollama `Modelfile` use the base
model's own template (training via `tokenizer.apply_chat_template`, serving via the
matching `stop` token). Change the base model and the template follows automatically.

## 6. Merging the adapter

After training you have base weights plus a LoRA adapter. For deployment it is simpler to
fold them into one set of weights: `W_merged = W + B @ A`. That gives a standard model
directory with no LoRA dependency at inference. The tokenizer (including any added tokens)
must be saved with the merged model, or the next step misaligns the vocabulary.

## 7. GGUF and quantization for serving

GGUF is the single-file model format used by llama.cpp and Ollama. Converting the merged
fp16 model to GGUF, then quantizing it, shrinks the model so it runs on CPU.

- `convert_hf_to_gguf.py` turns the Hugging Face directory into an fp16 GGUF.
- `llama-quantize ... Q4_K_M` compresses the weights to a mixed 4-bit scheme.

`Q4_K_M` is the common size-versus-quality sweet spot: about a quarter of the fp16 size
with little measurable quality loss, small enough that a multi-billion parameter model fits
in a few gigabytes of RAM and answers in seconds on a laptop CPU.

## 8. Serving: Ollama plus a thin auth proxy

Ollama imports the GGUF with one `ollama create` and already exposes an OpenAI-compatible
`/v1/chat/completions`. What it does not give you is access control or a system prompt the
caller cannot change. The FastAPI proxy in `serve/main.py` adds exactly those two things:
an API key check and a locked server-side system prompt. Everything else is passed straight
through, so any OpenAI SDK client works by pointing `base_url` at the proxy. The serving
image has no torch, because inference happens inside Ollama and llama.cpp.

## 9. Evaluation: prove the fine-tune did something

A fine-tune is only worth shipping if it beats the base model. LoRAForge serves both the
base `loraforge-base` (the un-fine-tuned Qwen3 4B) and the fine-tuned `loraforge-ft` on
Ollama and runs the same EvalKit
suite against each, using an LLM judge plus out-of-scope rejection cases. The headline
numbers are the support-quality pass rate and the out-of-scope rejection rate. `eval/eval.py`
adds reference-based ROUGE-L and BERTScore on the held-out 10 percent test split as a second
signal. The test split is held out before training so the model is never evaluated on data
it saw.
