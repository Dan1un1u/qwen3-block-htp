# Active L32-0029: full-model no-rotation factorial

User approved actual full-model paper ablations. Read docs/experiments/L32-0029.md. Fixed selected L32-0018 SP2mode8+FP32residual1, original Llama weights, no rotation. Qwen EXP0274 owns its separate active phase; device execution serialized. Other approved items remain pending, not complete. All-on must reproduce sealed0018 outputs before timing.

# L32-0028 closure: no-rotation paper speed baseline selected

User selects L32-0018 W4A8 SP2mode8+FP32 residual,R3/R4 OFF, historical
2069.7025566101083/42.51010980249135tok/s as primary Llama paper speed baseline.
Supersedes0027 rotated primary selection; rotation acceptance/failures retained
as supplementary evidence. This does not promote fixed-scale text/PPL quality.

Source0ae07a92e4d99fd3fd23c89adc3ce4a2b44f7b4f,codex/llama32-no-rotation,clean/pushed; native src/include unchanged
from0f7d083. New source docs/PAPER_NO_ROTATION_BASELINES.md/json pin original
measured0018e3d065,exact command,evidence hashes and scope. Five hot files match
0018; later changes guarded rotations or restored trials. Rotation branch common
commits patch-equivalent except branch default; no missed validated OFF kernel.
All1810018+1511Qwen0268 evidence entries rehashed. New Qwen kernel changes are
not timed on Llama and do not inherit this historical throughput.

Active none,next29. No new Llama model/hardware/native/PPL. Qwen ownsEXP0269:
FP32 singlelayer reference/KV passes,final short speedgate fails; no formal or
fullmodel expansion. Original Qwen paper2030.24/48.17 retained. Reports and audit
in/mnt/d/llm_exp/results/qwen3-block-htp/exp0269;ledgerSHA25620d3ddb9e1152e35e714f45b159f26312e5a79177cae907c31b7952a8af26ff5. Next discuss Qwen prefill FP32 Norm/O cost or draft
paper from selected historical results; no automatic new Llama experiment.

---
Historical context follows.

# L32-0027 closure: paper revision; original fast path accepted with known rounding

User accepts the original fast HMX implementation for paper preparation and
explicitly pauses further software/hardware numerical alignment. This replaces
the prior next-step suggestion of H512 integer-plane repair. Do not resume it,
run new PPL, build models, or profile based on the old0026 handoff.

Source /home/daniuniu/work/llama32-htp, codex/llama32-no-rotation,
HEAD 0f7d083dc99af7e2983ecd426fca03e9d57a0a07. Native src/include unchanged from cb8be962f3f31a1b89c8f8cd2bebb9cceef049a7.
Only docs/PAPER_STORY_RUNTIME_V2.md and its EVIDENCE.json were added.
Active none; next L32-0028. No hardware/model/build/native work in0027.

Read the V2 report for the current paper story: consumer-native physical
interfaces and bounded-VTCM scheduling; exact SP2 integer decomposition to
native packed-W4; phase-specific decode row packing/prefill overlap; paired
cross-model attribution plus FP32 residual and dense rotation extensions.

Qwen EXP0268 fair optimized-U8 control gives SP2 fullmodel latency +2.7099%
prefill / +0.8185% decode. Llama0012 gives +1.8851% / -0.0234% (decode CI
crosses zero). Llama0018 FP32 residual without rotations fullmodel is
2069.70/42.51token/s, paired latency -3.2309%/+8.1282%. Llama0022 fastR3R4
on FP32-residual/SP2 has only singlelayer speed +6.2074%/-0.9298% (decode
CI crosses zero); no combined fullfrontend E2E or PPL. Do not compose these
separate rates into an unmeasured combined result.

Original fastfull160023 independent cosine .5051155954/.7882501259 FAIL
remains preserved. User acceptance is paper/performance working scope only,
not numerical gate/quality pass. Later0026 rejected candidates remain rejected;
mode3 exact dense is diagnostic only. Do not relabel failures, measured source
heads, binaries, or historical quality/default flags. Broad PPL usable-quality
claims and all-A8 floating-boundary claims remain unsupported.

Report SHA256 02f0b458fd2d79e73c711bfa16b741c40fb151217548da0baa6a8baad91b6904.
Evidence index SHA256 b2444718b14fffcbf57475a5bced441ca3ae2624b8df19030bfcba3069093379.
Checked all local document links, five Qwen frozen report hashes, existing0018
and0026 ledger hashes, and0018 speed calculations. Index records hashes of
consumed authority at registration commit47c5df3; those snapshots are historical,
not assertions that mutable authority files stay byte-identical after closure.
No new PPL/performance result. All frozen source branches remain unchanged.

Next work is paper drafting/figures from existing evidence. Further native or
hardware work requires a newly authorized bounded experiment and preflight.


---
## Historical L32-0026 context (superseded next action; retained findings)

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
