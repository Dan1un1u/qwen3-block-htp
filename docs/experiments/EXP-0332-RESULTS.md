# EXP-0332 R3 + R4 integration and matched quality

Completed 2026-10-10. Qwen3-1.7B, native W4A8, uniform INT16 Down mode8, FP32 residual2. Fixed64-token prefill; denseR3 mode1 OPT2, denseR4 mode5 OPT10 / butterfly mode4 OPT12. No new calibration/training or baseline promotion. Other models and migration are deferred.

## Accuracy

WikiText-2 raw test frozen positions64..8255,8192 targets,191 reset windows with 64-token warmup and 43 targets;22-target tail retained. Effective context64..106. All target IDs, positions and contexts match historical BF16 teacher. This is subset PPL, not complete WT2 PPL or the long2048 experiment.

| Configuration | Mean NLL | PPL |
|---|---:|---:|
| R4 dense, original rounding | 3.952847645 | 52.083471098 |
| R4 dense, projection rounding2 | 3.908981428 | 49.848152189 |
| R3 + dense R4, projection rounding2 | 3.724087477 | 41.433406541 |
| R4 butterfly, projection rounding2 | 3.882254486 | 48.533509964 |
| R3 + butterfly R4, projection rounding2 | 3.733579421 | 41.828562568 |
| Original BF16 teacher, historical matched windows | 3.329912929 | 27.935909199 |

Historical noR4/rounding0 hardware PPL49.62699611 is a non-paired retained reference. Teacher does not use the artificial hardware prefix seed; do not describe its difference as pure weight quantization error.

| Paired comparison | PPL change | 95% window-bootstrap ratio | Better windows |
|---|---:|---|---:|
| D2/D0 | -4.2918% | [0.926376, 0.988712] | 111/191 |
| RD2/D2 | -16.8808% | [0.798994, 0.864849] | 149/191 |
| RB2/B2 | -13.8151% | [0.827255, 0.896401] | 132/191 |
| B2/D2 | -2.6373% | [0.948599, 0.999061] | 105/191 |
| RB2/RD2 | +0.9537% | [0.984671, 1.035261] | 91/191 |

R3 improves both R4 paths under their own frozen contracts. R3+dense and R3+butterfly differ by0.954% in PPL; the paired interval contains1. Do not claim either arithmetic precision superior from this comparison. Best combined PPL remains48.32% above matched teacher. No quality acceptance threshold or checkpoint selection was introduced. R3 comparison includes its frozen Q/K grids, score metadata and once-transformed prefix.

## Text and numerical validation

All5arms passed human inspection of the same two frozen prompts before PPL. Paris and photosynthesis answers are understandable. Generated43-token traces continue after EOS for fixed-workload measurement; readability uses text before the first EOS. Full traces and tokenizer hash are retained. This small screen is not general answer-quality acceptance.

- R3+dense and R3+butterfly: component/1/consecutive3/full28 device checks.4steps per scope,4/12/112 layer R3 checks. Maximum sampled R3 quantizer code discrepancy0 against actual FP16 rotation outputs; dense transform stays within frozen one FP16 ULP plus FP16 minimum-normal bounds.
- R4: FP16 dense two-stage reference bounds retained; FP32 butterfly exact against independent FP32 stage-order reference. Uniform INT16 codes/native byteplanes exact, reconstructed native-W4 Down + FP32 residual exact on the captured last layer0/2/27 for every sampled step.
- All28layers and8heads: first64prefill K tokens and correction bytes equal independent complete-segment repack after inserting matched rotated token0 prefix, for both combinations. Inactive padding and VTCM-only decode tails are not misrepresented as authoritative DDR.
- R4-only original-rounding dense and butterfly controls reproduce112 historical layer hashes and all selected audit tensors from EXP0331.
- Both combinations repeat two identical samples with112 matching layer/step hashes, and captured tensors exactly reproduce prior runs. Histogram repair reproduces560 generation layer hashes across five arms plus all audit files.
- Device statuses/numerics, KV advance, requested/granted8MiB pass. Peak dense8300288, butterfly8365824bytes unchanged by R3. No intermediate DDR/spill on timed/PPL runs. One existing HMX owner, existing joins/threads and VTCM phase arenas retained.
- All40960scored token records pass: exact vocab151936, finite NLL, no nonfinite or saturated head entries, exact target/context and physical checks.215 timed diagnostic profiles with audit disabled and6020layer ledgers close.

These validate the finite implementation contracts, not idealFP32 bit identity. The historical ideal R3 full-model alignment failure and nonzero FP16 rounding remain. DenseR4 FP16 versus butterflyR4 FP32 unification is explicitly deferred by the user. Long-context combinedR3+R4 and other models are not validated here.

## Repairs and preserved attempts

1. Protocol previously prohibited R3 and R4 together. Admit only productionR3 mode1 with R4 modes4/5 on this Qwen17 native uniformDown16/FP32residual path; preserve independent switches and reject unsupported reference combinations.
2. Generation host gate omitted per-layer R3 audit-write bytes. The first device call completed computation but the strict host count check failed. Account for exactly layer_count*983040 diagnostic bytes; keep the original failed log. No arithmetic change.
3. DenseR4 nominal Down aliases normalized after EXP0329. The old quality histogram in Down was overwritten by final norm/head resident-bias load. First PPL step was rejected by vocab count3865524. Move its1KiB histogram to dead Gate, with explicit bias/activation range guard. Retain invalid attempt, accept no NLL from it. Normal generated tensors remain exactly unchanged.
4. Text decoder first used nonexistent generic tokenizer.json; resume decoding completed immutable device records with the retained qwen3-tokenizer.json and record its hash. No rerun or replacement of device evidence.

## Source-build-to-device workflow

Host: AMD Ryzen 9 8945HX with Radeon Graphics; 16logical CPUs,20GiB WSL memory limit,8build jobs. SDK6.6.0.0, Hexagon19.0.07/v79, AndroidNDKr26c, CMake3.28.6. Full28 clean build. PhonePJZ110/SM8750.

| Stage | Seconds |
|---|---:|
| Clean Android + DSP build | 20.832 |
| Binary archive/deployment/hash checks | 1.227 |
| Device launch to model-ready receipt | 2.634 |
| Device launch to first result receipt | 2.673 |
| Entire build-to-first-result including preflights/input preparation | 32.710 |

Small host-only incremental edit:4.144s. Changing layer count or rebuilding the DSP:18.501–19.694s. Composition, local/device model hash checks and cached-package deployment:119.496s, separately charged;114688bytes of prefix blobs pushed, full weights already on phone. Four run binaries total about2.53MB. First-time full-weight transfer, model export, toolchain installation and development/debug time excluded. UTC/monotonic start/receipt events are in build-to-first-inference.json. Total timing is a real continuous workflow, not an inferred sum. All four fresh delivery binaries byte-identical to quality build.

Recommended collaboration default: members submit source, central host builds with pinned SDK/flags and binds artifacts to commits; about20seconds compilation does not justify making binary-only delivery the default. Compiled binaries remain useful as supplementary debugging artifacts with manifest/source revision/ABI/toolchain/flags. No server or repository migration was performed.

## Diagnostic speed only

One64+42fixedtrajectory per arm, repeat1 auxiliary. Full module attribution and Host-wall percentages in FULL_PROFILING_REPORT.md. No formal5short/10formal speed gate, confidence claim, throughput parity or baseline promotion.

| Configuration | Prefill token/s | Decode token/s |
|---|---:|---:|
| R4 dense, original rounding | 1610.909 | 45.496 |
| R4 dense, projection rounding2 | 1610.460 | 45.624 |
| R3 + dense R4, projection rounding2 | 1613.183 | 45.238 |
| R4 butterfly, projection rounding2 | 1572.053 | 45.869 |
| R3 + butterfly R4, projection rounding2 | 1576.571 | 45.380 |

## Reproduction and provenance

Quality measured source 74162460e2fd6b98444f5d190c970b0ee1247abb; delivery source 8271d56ead1318f72873374140882f4cc04eff99 produces the same four binaries. Parent source7fb199b8d84ef1a9827e5c4112659f34db73d2b9. Frozen dataset tokenizer/weights/QK/R4/package/build hashes and commands archived; source-diff.patch contains this experiment code changes. Source/runtime helpers and original attempt logs are retained. Every new evidence file is indexed by EVIDENCE_SHA256.json.

Fetch policy amendment is synced: three bounded attempts, process-local proxy bypass on retries, then clean last-synchronized verified local authority when network refresh remains unavailable, explicitly disclosing remote freshness uncertainty. Identity/hash/origin/lock/divergence errors still fail. Failure injection of fetch transport passed; real bootstrap and all source/memory synchronization this session succeeded.

Next: discuss repository migration with the source quantizer, four compatibleR3/R4presets, per-developer environments and centrally owned device queue; resolve dense/butterfly intermediate precision later.
