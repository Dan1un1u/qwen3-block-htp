# L32-0026 closure: original fast HMX retained

User prioritizes speed and correct implementation on original fast pipeline.
PPL/text quality deferred; exact scalar dense mode3 diagnostic only. Preserve
independent numerical/physical gates and10% speed gate.

Source /home/daniuniu/work/llama32-htp, branch codex/llama32-no-rotation,
closure cb8be962f3f31a1b89c8f8cd2bebb9cceef049a7; clean/pushed. Active none; next L32-0027.
Read docs/LLAMA32_H512_FAST_ROUNDING.md and docs/experiments/L32-0026.md.

Two bounded candidates: A scales H512 matrix by16 and converter1/16, producing
identical six phase/layer outputs/raw/stage1/finalR4 captures to original.
B reverses sixteen K32 tiles using same accumulator and one final conversion.
Both local0/7/15 components, exact actual tails, cosine pass. B first-factor
prefill halfword differences31/27/6 ->27/21/4, but SP2 differences1/1/2 ->2/2/0.
Layer15 output exact; other layers not consistently improved.

B chain3 passes: prefill cosine.9999999999245163,NRMSE1.06697e-7,7927changed;
decode exact, KVexact. Full16 fails: prefill cosine.609073601365145,NRMSE
.2967764437,116736changed; decode.7699031028646253,.6505855918,2048changed.
KV value differences352930/363758, append-structure checks0. No formal/E2E/PPL.
Do not promote, select order by layer, or mistake local pass for full16 pass.

All changed native src/include restored to parent7478516 via
f1162649cdbff56b134dbc8eab9a2517a0c35fec. Fresh layer0 audit identical to original.
Only result-ID tooling and report changes remain. N1 build seal atf116264
differs from docs closure: rebuild after successful preflight before device.
9CLI/18RPC,1exit0/8original ideal-exact exit1 retained,5successful builds,
no crashes.8MiB VTCM,peak8229344,ordinary intermediateDDR/spill0; auditcaptures
untimed.63prior files,5model manifests/876payload entries verified.
Evidence /mnt/d/llm_exp/results/llama32-htp/l32-0026;
ledger 599d86cf654b2b41526690c76ad1c0a251b5f206815a57ce4ae12b0024f94ac9,120files. No processes active.

Next possible bounded fast option: exact binary-lattice decomposition of FP16
inputs into integer byte planes, native HMX +/-1 dense Hadamard dot products,
wide integer merge then single normalization/RNE. Not implemented or timed.
Observed per-row plane counts mostly3; whole-LUT fixed coverage needs4/4/5
planes for0/7/15. Cannot assume SP2 two-plane construction directly applies.
First validate exact representation and isolated cost; no full scalar dense
recompute. Need cover signed digits,overflow,full frozenLUT range and rounding.
Local v79 header exposesFP16 conversion; noFP32 output established. Bundled
v81 PRM37-bit accumulator description is not a v79 guarantee. Offline plain
FP32 double rounding explains only8/68 observedH512errors, not full contract.
Do not repeat H16 placement, exponent-shift or arbitrary order tuning.
Historical fast0022 passes chain3 and singlelayer speed; full160023 fails.
Exact0024full16 passes but enormous cost, retained diagnostic only. No new
quality conclusions or Qwen/otherrecipe work.
