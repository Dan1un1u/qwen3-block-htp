# Recipe references

Each JSON contains the exact frozen Qwen3 schedule arguments/environment and
explicit neutral switches for unrelated experiments. These are migration
references, not generic model launch configurations. R3 and R4 are dense matrix
multiplications; no butterfly path is selected. W4 uses per-output-channel
signed [-7,7], one FP32 scale per output row. `config/branch.json` determines
which W4A8 variant is default/allowed. Resolve with `python3 tools/recipe.py show`.
