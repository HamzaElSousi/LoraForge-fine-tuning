# models/

Drop the trained GGUF here. This directory is gitignored for `*.gguf` (the files are ~2-3 GB).

After running `notebooks/01-finetune.ipynb` on Kaggle, download the quantized model and
place it here, e.g.:

    models/loraforge-qwen3-4b-q4_k_m.gguf

`serve/Modelfile` points at this path. If you name it differently, update the `FROM` line
in `serve/Modelfile` to match.
