# EXP0269 closure: single-layer FP32 passes, speed gate fails

User selected original Qwen EXP0268 no-rotation SP2 and Llama L32-0018
no-rotation SP2+FP32 residual as paper speed baselines. Keep2030.24/48.17 and
2069.70/42.51tok/s as historical full-model numbers, respectively. No quality
acceptance. Selection/audit manifest is docs/PAPER_NO_ROTATION_BASELINES.json
in both source worktrees. Llama documentation selection is L32-0028.

Qwen source codex/exp-0269-sp2-fp32-residual ata41a7258ed596d810bd4a62d83390fd6944b8967, clean/pushed.
Actual hardware/build headaa55a3ece5a04f40c1cfe135b699433b8f6e4ff2; closure adds documents only.
Future device use requires new approved experiment/preflight/fresh build seal.
Memory active none,next270. All older frozen branches and original evidence
preserved. Llama native unchanged; source documents closure0ae07a92e4d99fd3fd23c89adc3ce4a2b44f7b4f.

Final safe path: native-W4 O/Down raw integer output->FP32 scale/add residual,
ordered FP32 Norm then A8; original SP2 LUT/W4/attention/KV/Qwen QK norm/RoPE.
Host embedding/finalnorm/head glue ported, but Qwen full frontend NOT validated.
Independent exact singlelayers0/14/27 M64+decode1 and physical KV/actual boundary
checks pass; repeat10 deterministic. VTCM8MiB,peak6682752B, tensor DDR/spill0.

Initial rows48..63 corruption fixed by 2048B HMX raw-slot alignment. Scalar
Norm candidate rejected as slower. Named epilogue vectors remove1024B indexed
activation stack staging. Register prefill transpose rejected after assembly
found256B activation spills; original VTCM transpose restored. Keep decode HVX
padding stores and live-lane ordered reduction. No rounding gate relaxation.

Final safe source fixed five-short AB/BA repeat10:
prefill1348.52716->1554.16358us,ratio1.152489639,CI[1.100393355,1.209263461];
decode1053.72296->1102.91984us,ratio1.046688629,CI[.957624975,1.148527582].
Both upper-CI gates fail. Earlier short11.28%/1.94% belongs to rejected spilling
prefill candidate and is NOT a promoted result. Do not resample formal as if
short passed. No formal10/chain3/full28/E2E/PPL. Main extra prefill costs are
inputNorm+57.24us,postNorm+57.94us,O+30.30us. Pause and discuss cost before
fullmodel; the first requested port is not accepted as a paper speed baseline.

67 device CLI,710 token-boundary profiles;2 initial independent numerical
failures preserved. Evidence/mnt/d/llm_exp/results/qwen3-block-htp/exp0269,546 files/213645003B;
ledgerSHA25620d3ddb9e1152e35e714f45b159f26312e5a79177cae907c31b7952a8af26ff5. Read SUMMARY.json,REPORT.md,c3repair/single_gate.json,
c3repair/layer-short.json,assembly_final_gate.json and source_closure.json.

Audited all181 Llama0018 and1511 Qwen0268 sealed files. Llama five common hot
files identical0018->0f7d083; later seven native-file changes only rotation/
audit routes.0025/26 rejected trials restored;0027 docs. Rotation branch
common patches all equivalent except branch-default config (exact count in
baseline_audit.json). No omitted later validated OFF optimization. New Qwen
edits have not been tested on Llama; don't reuse historical speed for them.
