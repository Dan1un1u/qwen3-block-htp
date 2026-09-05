# F16F16 deterministic text generation

EXP-0217 adds QBH_GENERATION_SEQUENCE=10 to the existing Host/DSP generation boundary. The Host uses the same Qwen tokenizer/chat template (thinking disabled) and detokenization as W4F16. The DSP owns embedding, all 28 layers, final RMSNorm, streamed FP16 LM head and strict-greater/lower-token-ID greedy selection. Head weights stay FP16; no W4 reconstruction is used. The head streams batch-eight groups through the existing aligned phase-dead VTCM projection buffers, with unity HMX scale and no logits DDR output.

Current tested scope: exactly 64 prompt tokens and 16 selected tokens (15 continuous feedback decode calls), persistent capacity 80. This is a deterministic research runner, not yet arbitrary-length interactive chat. The preparation script uses the original local checkpoint and an independent Transformers FP16 reference, retaining the expected token IDs separately. Transformer arithmetic and schedules are unchanged. Generation JSON token matches do not imply full-logit equality; placeholder hidden/cache error fields with zero compared elements must not be reported as a new tensor audit.

After Project Memory bootstrap and successful preflight:

```sh
/home/daniuniu/.cache/qwen3-block-htp-py/bin/python scripts/prepare_exp0217.py
scripts/build_exp0152.sh
scripts/deploy_exp0217.sh
scripts/run_exp0217.sh > /mnt/d/llm_exp/results/qwen3-block-htp/exp0217/new_run.jsonl
/home/daniuniu/.cache/qwen3-block-htp-py/bin/python scripts/verify_exp0217_generation.py /mnt/d/llm_exp/results/qwen3-block-htp/exp0217/new_run.jsonl
```

Preparation intentionally refuses to replace existing evidence. Reuse the retained package once it exists. The verification command decodes the device token IDs and validates them against the independent FP16 reference. `scripts/summarize_exp0217.py` regenerates formal ledgers, complete diagnostics and the stable module overview from the retained ten-session logs. All binaries, models and results are outside Git under the Project Contract D-drive roots.
