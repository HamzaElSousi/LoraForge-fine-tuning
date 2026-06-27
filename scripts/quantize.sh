#!/usr/bin/env bash
# Convert a merged fp16 HF model to GGUF and quantize it to Q4_K_M with llama.cpp.
#
# Runs after scripts/merge_adapter.py. In the Kaggle notebook this mirrors the cells in
# notebooks/01-finetune.ipynb; locally it lets you re-quantize without retraining.
#
# Usage:
#   scripts/quantize.sh <merged_hf_dir> <output_basename> [quant]
# Example:
#   scripts/quantize.sh outputs/merged-16bit loraforge-gemma4-e4b Q4_K_M
#
# Requires a built llama.cpp. Point LLAMA_CPP at its directory, or this script will clone
# and build it next to the repo on first run.
set -euo pipefail

MERGED_DIR="${1:?usage: quantize.sh <merged_hf_dir> <output_basename> [quant]}"
OUT_BASE="${2:?usage: quantize.sh <merged_hf_dir> <output_basename> [quant]}"
QUANT="${3:-Q4_K_M}"
OUT_DIR="${OUT_DIR:-models}"

# Locate or build llama.cpp.
LLAMA_CPP="${LLAMA_CPP:-./llama.cpp}"
if [[ ! -d "$LLAMA_CPP" ]]; then
  echo ">> Cloning llama.cpp into $LLAMA_CPP"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP"
fi
if [[ ! -x "$LLAMA_CPP/build/bin/llama-quantize" ]]; then
  echo ">> Building llama.cpp (release)"
  cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$LLAMA_CPP/build" --target llama-quantize --config Release -j"$(nproc)"
fi

mkdir -p "$OUT_DIR"
F16_GGUF="$OUT_DIR/${OUT_BASE}-f16.gguf"
Q_GGUF="$OUT_DIR/${OUT_BASE}-$(echo "$QUANT" | tr '[:upper:]' '[:lower:]').gguf"

echo ">> Converting HF -> GGUF (f16): $F16_GGUF"
python "$LLAMA_CPP/convert_hf_to_gguf.py" "$MERGED_DIR" --outfile "$F16_GGUF" --outtype f16

echo ">> Quantizing -> $QUANT: $Q_GGUF"
"$LLAMA_CPP/build/bin/llama-quantize" "$F16_GGUF" "$Q_GGUF" "$QUANT"

echo ">> Done. Quantized model: $Q_GGUF"
echo ">> (You can delete the f16 intermediate $F16_GGUF to save space.)"
