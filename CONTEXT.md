# Current Llama context

Read contract/status/context/index after bootstrap. No active experiment or
background device job. Qwen3 freeze is unchanged. L32-0001 W16A16, L32-0002
W4A16 and L32-0003 W4A8 OFF functional integrations are complete. Both Llama
branches contain identical common tracked files; only config/branch.json differs.
Full source heads and evidence ledger hashes are in PROJECT_STATUS.yaml.
No quantized quality acceptance, baseline promotion or formal profiling.

W4A16 uses fresh original-derived C64 enhanced GPTQ, signed[-7,7] W4 per-output
channel and one FP32 scale, no groups. EOS8-token output is exact: The capital
of France is Paris. Forced-after-EOS16-token attempt remains failed at step12;
never promote it. Device PPL31.039101 vs original BF16 teacher26.697999 (+16.26%),
EN+15.84%,ZH+16.69%; quantized software31.027335. Main loss is quantization;
retained overall5% and language10% quality criteria FAILED. M64+7 auxiliary
functional speed437.3534 prefill/6.3121 decode tok/s. Evidence sealed in L32-0002.

W4A8 OFF reuses those original-derived Llama W4 weights, with fresh static affine
minmax A8 from65536 Llama calibration tokens. No Qwen tensor/qparam/prefix reuse,
no heldout fitting. Single1/3/16-layer prefill+decode output and KV exactly match
independent SDK19.0.07 actual-arithmetic references. Full16 feedback IDs all42845
and selected U8 codes match exactly: unusable repeated Sleep text. Device PPL
1206603.740108 overall,1309236.426225 EN,1112016.559026 ZH. Same independent128
documents/2048 targets as L32-0002 (1024 tokens per language). NLL first-target
code audit exact and error<1.8e-7; every target passed execution/physical audit.
A8 model quality explicitly has NO threshold; this does not relax arithmetic.

All7 projections per layer and LM head read native packed W4 with no S8 weight
expansion; integer HMX QK/AV, log2 softmax widened differences/exact integer
normalization. Head64 no Q/K norm, Llama SF32 RoPE once before KV publish; GQA4
scratch and FFN8192 metadata fixed. Head-major row U8 KV capacity80 supports M64
plus15 decode; arbitrary-length serving, native segmented KV and dense R3/R4
are unsupported and explicitly rejected. Exact8MiB VTCM, peak7668960B, no timed
intermediate DDR/spill, one HMX owner and one FastRPC per token boundary.
W4A16 full16 shared-code regression retained identical previous error metrics.
Auxiliary single-run M64+15 speed217.3520/2.57149 tok/s. Optimization deferred;
these timings are not formal paired comparisons or acceptance of slowdown.

Original BF16 source read-only: /mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin.
Six hashes in tools/llama_reference.py. Models/results roots in status; L32-0002
owns quant-a01 and frozen calibration/heldout data. L32-0003 owns layers-a01,
stack1-a02,stack3-a01,stack16-a01,frontend-a01; results calibration-a01,
device-stack1-a05,device-stack3-a01,device-stack16-a01,device-frontend-a02 and
validation_summary.json. All failed attempts preserved. See source
 docs/LLAMA32_W4A16.md and docs/LLAMA32_W4A8.md for reproduction and limitations.
Next: report completed chains and W4A16 quality gap, then discuss with user;
do not autonomously start optimization or claim rotated Llama is validated.
