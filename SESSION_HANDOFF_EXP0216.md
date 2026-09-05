# Session Handoff — EXP-0216

Generated on 2026-09-05 for a model/chat switch. This file is a navigation aid,
not an authority override. On every new session, bootstrap Project Memory first
and follow `PROJECT_CONTRACT.md`, `PROJECT_STATUS.yaml`, `CONTEXT.md`, and
`experiments/index.yaml` when any statement conflicts with this handoff.

## Resume objective

Resume the W4U8 prefill optimization loop at EXP-0216. Decode exploration has
already hit its declared local-optimum stop condition and must not be restarted
as the first action. The selected decode source is EXP-0211 at 45.919316 token/s.
Prefill now eliminates explicit HVX W4-to-S8 expansion and feeds reordered
packed W4 directly to integer HMX. The enabling gate passed, so the prefill loop
continues until the user manually interrupts it.

The current experiment extends the already successful M64 direct-W4 MLP path to
Q, K, V, and O projections across all 28 layers. The primary ranking measure is
directly measured complete-prefill token/s, not local ticks or extrapolation.

## Required bootstrap

Use this source worktree:

`/home/daniuniu/work/qwen3-block-htp`

At the beginning of the new task, before reading source or running anything:

```bash
/home/daniuniu/work/qwen3-block-htp-project-memory/scripts/bootstrap.sh \
  /home/daniuniu/work/qwen3-block-htp
```

Read the four authority files in the order printed by bootstrap, then run:

```bash
python3 /home/daniuniu/work/qwen3-block-htp-project-memory/scripts/project_memory.py \
  preflight --source-worktree /home/daniuniu/work/qwen3-block-htp
```

Any bootstrap, fetch, fast-forward, schema, origin, branch, hash, lock, or
preflight failure must be resolved without using stale memory. Never auto-stash,
reset, clean, clear a lock, force-push, rewrite history, or silently accept a new
artifact hash.

## Exact current state

- Active experiment: `EXP-0216`
- Source branch: `codex/exp-0216-w4u8-prefill-direct-w4-qkvo`
- Source commit: `25b2fdfead7546a981ccb457d9106e081bb6cb34`
- Source and memory branches were clean, pushed, and remote-synchronized at
  handoff time.
- Device: PJZ110 was connected and authorized through Windows ADB.
- Backend: standalone FastRPC/cDSP; QNN and QAIRT runtime paths are forbidden.
- Control selector: mask 31, direct-W4 MLP only for M64 prefill.
- Candidate selector: mask 63, direct-W4 Q/K/V/O plus MLP for M64 prefill.
- Candidate expected physical counts: 196 direct prefill projections and 840
  direct HMX commands. Candidate QKVO and MLP explicit expansion must both be
  exactly zero.
- The code, experiment scripts, static check, build, deploy, and smoke run exist.
  Formal short-gate and ten-pair profiling have not yet been run.

The smoke directory is:

`D:/llm_exp/results/qwen3-block-htp/exp0216/20260905_smoke_worktree`

Smoke was encouraging but is not adoption evidence: complete prefill Host wall
was about 43.817 ms for control and 39.130 ms for candidate. Candidate QKVO
expansion became zero, MLP expansion remained zero, and output/token hashes were
byte exact. Decode counters remained on the unchanged path.

## Immediate next commands

After bootstrap and preflight, run the five-pair short gate:

```bash
cd /home/daniuniu/work/qwen3-block-htp
scripts/run_exp0216_short_gate.sh \
  /mnt/d/llm_exp/results/qwen3-block-htp/exp0216/20260905_short_gate_25b2fdf \
  25b2fdfead7546a981ccb457d9106e081bb6cb34
```

If the short gate passes, run the ten-pair formal comparison:

```bash
scripts/run_exp0216_formal.sh \
  /mnt/d/llm_exp/results/qwen3-block-htp/exp0216/20260905_formal_25b2fdf \
  25b2fdfead7546a981ccb457d9106e081bb6cb34 \
  /mnt/d/llm_exp/results/qwen3-block-htp/exp0216/20260905_short_gate_25b2fdf
```

Do not bypass the scripts' clean-tree and exact-commit checks. Poll long-running
commands rather than starting duplicate device jobs.

## EXP-0216 acceptance checks

Retain the same weights, scales, qparams, output encoding, Attention math,
KV-cache behavior, decode path, LM head, and token sequence. Require complete
prefill hidden/cache/selected-code/token byte equality between control and
candidate. Require exactly 8 MiB VTCM, zero timed intermediate DDR, zero
spill/fill, one FastRPC invocation, one HMX owner, no QNN, equal HMX tile-pair
work, and zero explicit QKVO/MLP prefill expansion. The performance gate is
higher paired full-prefill token/s.

If the candidate fails, record the negative result and continue the prefill
loop with a genuinely different hypothesis; the original stop-and-discuss gate
applied only to the first direct-W4 feasibility experiment, which already
passed in EXP-0214.

## Results that must not be rediscovered

EXP-0211 is the explicitly selected decode baseline and is tagged
`baseline-w4u8-token-generation-exp0211` at
`b111cf7e49ee5a230ed93b54cd5b5b33e8d1ebfa`. Formal decode is 45.919316 token/s.
EXP-0212 and EXP-0213 each lowered the local QKV interval but increased
end-to-end decode latency by moving or serializing waits. They triggered the
declared two-experiment local-optimum stop gate; do not resume those prefetch
hypotheses without a new system-level overlap mechanism.

EXP-0214 proved direct packed-W4 HMX on real M64 projections. Compared with the
optimized explicit-expand path, Gate/Up was 1.198x faster at repeat one and
1.392x at repeat ten; Down was 1.311x and 1.702x faster. Outputs were exact and
explicit expansion was zero.

EXP-0215 integrated direct W4 into all 28 layers' Gate/Up/Down projections.
Formal prefill increased from 1276.612 to 1459.897 token/s (+14.357%), with ten
of ten wins. Complete MLP became 36.160% faster, Down 97.617% faster, and decode
was preserved at 46.026 token/s. Formal evidence:

`D:/llm_exp/results/qwen3-block-htp/exp0215/20260905_formal_0483c29`

## Likely continuation after EXP-0216

First close EXP-0216 in Project Memory with the complete profiling report and
evidence hashes. Only the user may promote it. Register EXP-0217 before further
stateful work. Choose the next hypothesis from the new full-profile critical
path rather than assuming it in advance.

If EXP-0216 confirms the smoke result, the most likely next target is the M64 LM
head, which still uses explicit W4-to-S8 expansion. Reuse the proven decode
direct-W4 carrier only after a bounded M64 gate; do not modify decode behavior
or silently change vocabulary order, argmax semantics, qparams, or output math.

## Stable physical and storage contracts

- Exactly 8 MiB VTCM is requested once per prepared session; plans allocate only
  what they need and report peak use.
- Timed intermediate tensors remain VTCM-resident. Legal Host/DSP input/output
  boundaries do not authorize hidden intermediate DDR.
- One HMX owner; overlap comes from DMA, HVX, and HMX scheduling, not concurrent
  competing HMX owners.
- Git stores source/scripts/small evidence only. Models and retained artifacts:
  `D:/llm_exp/models/qwen3-block-htp`. Formal results:
  `D:/llm_exp/results/qwen3-block-htp`. WSL ext4 stores build trees.
- No QNN execution path. No claim of W4U8 semantic quality; current accuracy
  gate concerns implementation equivalence and local mathematical validity.
- Every completed experiment must retain the complete profiling contract,
  including direct prefill/decode token/s and complete Host wall.

## New-session operating instruction

Continue autonomously after successful bootstrap/preflight. Resolve routine
network, ADB, build, and clean-tree preparation issues safely, but never hide a
state conflict or violate the destructive-action restrictions. Report only
meaningful milestones, final gates, failures that change direction, or required
user decisions. The user wants the prefill loop to continue until manually
stopped.

## Telemetry repair update

Initial f724a22 five-pair collection stopped at summary because generation_profile omitted required QKV ring fields. Raw evidence remains in 20260905_short_gate_f724a22; it is not a passed gate. The user authorized repair. The commands above now target the pushed Host telemetry and strict-summary repair d134995; fresh collection is required.

Additional audit completeness repair 25b2fdf exports all 56 prefill cache buffers and validates all 112 nonzero per-layer hidden hashes. d134995 short/formal passed existing script gates, but did not retain cache bytes; final closure will use freshly collected 25b2fdf evidence.
