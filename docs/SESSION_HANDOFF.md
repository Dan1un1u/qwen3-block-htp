# L32-0006 final session handoff

No active experiment or background job. Both Llama source branches are clean and
synchronized; common source differs only in config/branch.json. Qwen3 freeze
unchanged. Next experiment L32-0007 requires user direction. Exact source heads
and evidence ledger are in PROJECT_STATUS.yaml.

Round1 reuses exact SOLE templates through dead K carrier scratch and four-row
shuffle, plus three-way prefill SwiGLU using existing workers/scratch. Round2
vector-transposes aligned native head64 KV carriers with bounded scalar edges.
Round3 adapts continuous GateUp/SwiGLU to eight Llama groups, O/Gate prefetch and
native-W4 DMA batches (Gate32,QKV16,Down8,O16). Generic const-score API preserved.
All are scheduling/layout changes; weights, quantization codes and qparams fixed.

A8 independent single/3/16 output and KV checks pass; padding/scalar softmax
audit passes. W4 full16 output remains byte-identical to sealed L32-0002, OPT2
audit passes. Full generated token IDs and selected codes match old baseline.
Exactly 8MiB VTCM, peak A8 7668960 / W4 8330752 bytes; zero timed intermediate
DDR/spill, one HMX owner, native W4 with no expansion. All 220 formal candidate
decode steps show 128 SwiGLU publish/consume, overlap,16 O/Gate prefetch pairs.

Fixed ten rotated ABC/BCA/CAB cycles, M64 plus7 continuous decode:
W4A16 OPT2 1261.16115 / 23.60401 token/s;
L32-0005 A8 baseline 1437.08911 / 39.52282;
L32-0006 candidate 2015.91637 / 45.64662.
New/old Host ratios .712871 [.707863,.719768] prefill and
.865843 [.861942,.869598] decode. Throughput +40.28% / +15.49%.
Both retained 10% slowdown gates pass.

Additional fixed ten AB/BA A8 pairs, M64 plus15 continuous decode:
old 1435.85739 / 40.21846; new 2046.81285 / 46.67570 token/s.
Host ratios .701509 [.690540,.709048] / .861657 [.855672,.867739].
Both gates pass. Frozen Qwen3 OFF historical M64+15 1705.31769 /48.35725:
Llama prefill +20.03%, decode -3.48%. Different models/historical sessions,
NOT a paired cross-model efficiency claim.

560 timed token-boundary additive Host/DSP ledgers reconcile. 64 successful DSP
processes,640 total boundaries; no failed native or build attempts this experiment.
Includes embedding,16layers,final norm,head,greedy and FastRPC; excludes model
load/session setup and external tokenizer. No repeat1 gate, optional stopping or
layer extrapolation. Existing L32-0005 evidence 241 files unchanged.

Remaining decode costs: attention26.35%,GateUp/SwiGLU22.76%,head14.80%,
Down11.57%,Host boundary10.51%. These may guide a future authorized experiment.
A8 text remains unusable (prior PPL1206603.740108); no quality gate. W4 prior
PPL31.039101 vs BF16teacher26.697999 still fails retained PPL criteria.
No PPL rerun or quality-baseline promotion. Tested capacity80,M64+7/15 only.
Llama R3/R4 and arbitrary-length serving remain unsupported.

Evidence /mnt/d/llm_exp/results/llama32-htp/l32-0006; 296 files hashed in
docs/experiments/L32-0006-evidence-sha256.json. Source report
docs/LLAMA32_PIPELINE_ITERATIONS.md and result PROFILE.md. Profiled native
cfd9feeb1d9403185aeffeabfa47bbd174c7b70d; subsequent source changes only reports.
Both branch propagation and binary hashes verified by closure_checks.json.
Profiler/report tools are experiment-specific; future runs require fresh immutable
destinations. No Qwen/model artifact changed; original BF16 remains read-only.
