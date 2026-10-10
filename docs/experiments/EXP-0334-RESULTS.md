# EXP-0334 — private migration completed

Destination private and synchronized: https://github.com/Dan1un1u/htp-research, main e74c1d4b0d0bf4733206cd7ba4dce83a2b38c76c. Original source f107ccda remains unchanged and synchronized. New canonical agent authority and local collaboration entrypoint are in the destination. Old AGENTS/evidence remain historical. Remote access deployment is deferred.

# EXP-0334 migration acceptance

Migration is published and synchronized; local acceptance is complete. Destination: https://github.com/Dan1un1u/htp-research; GitHub confirms private visibility and owner push access. Current publication, remote synchronization and completion are tracked by project-memory/PROJECT_STATUS.json. Old source and sealed historical results remain unchanged.

## Source, contracts and retained frontier

The source snapshot is Qwen3-1.7B commit f107ccda70288729d065839b77d3b59b761c5fa7, from codex/exp-0333-long-rotation-quality. All 1,652 mapped source files verify byte-for-byte at runtime/qwen3. The old authority entry is replaced by the new root AGENTS.md; inherited source documents and drivers are historical, not a second active control plane. The 796 runtime/compile symbols and nine CMake switches are retained without changing arithmetic, defaults or interfaces. SDKs, weights, binaries and raw logs are excluded from Git.

Presets C0/C2/R2/D0/D2/RD2/B2/RB2 preserve no rotation, R3, dense R4, HVX butterfly R4 and both, projection rounding 0/2, context/capacity/client/native QKV scheduling and historical format/pipeline switches. M64 remains speed development. Primary quality uses full 2048 windows and the fixed 8,192-target WikiText-2 subset with tail. config/qwen17-migration-frontier.json and history/docs/experiments preserve exact historical commits, gates and evidence seals. Dense R4 intermediate FP16 and butterfly FP32 remain separate finite-precision contracts.

## Shared assets and member isolation

One immutable SHA256 blob/package library contains the original checkpoint and four existing executable rotation packages. Initial import verifies 907 unique blobs (6,071,174,681 bytes) with zero new payload copies: verified same-filesystem hardlinks retain old assets. Members can publish distinct quantizer results; identities bind manifests, file hashes and execution contracts. Provenance is separate from content identity. Mutable submissions are copied into immutable blobs; completed private export payloads become verified library views. Aliases are immutable.

A single base-v1 environment references the existing SDK 6.6.0.0, Hexagon tools 19.0.07/v79, NDK r26c, CMake 3.28.6 and shared Python dependency base. Each job owns its detached source checkout, build, optional dependency overlay, cache, temporary files, inputs, results and logs. Preparation is bounded to two workers. Model publication and index locks are short; formal host-wall execution drains and excludes central heavy preparation. One device lease covers deployment, execution, paired campaigns and result collection. Coordinator identity checks reject a second independent queue/lock root targeting the same phone.

Local submit/status/prepare/build/run/reference/quantize/campaign/worker and inspected recovery interfaces are implemented. VPN, SSH, remote authentication, network job API and GitHub runners are deferred. This is a local research service, not a remote security boundary.

## Original-weight export acceptance

The plugin interface starts from a hashed original checkpoint plus an explicitly frozen activation/calibration template. One-layer full6144 R4 export and full28 RTN backbone plus head/embedding/norm export succeeded, including independent W4 pack/unpack and signed partial/reconstruction bounds. Published staging was deduplicated. Package IDs and original shard hashes are in migration/acceptance.json and the outside-Git evidence archive.

Built-in fresh RTN is a separately named exploration recipe; it does not reproduce historical enhanced GPTQ quality or promote new weights. Historical enhanced-GPTQ, calibration, folding and other quantizer functions remain byte-identical and available for member plugin reuse. Old orchestration drivers must not bypass the new service locks or call the device directly.

## Implementation regression

| Scope | Configuration | Verification |
| --- | --- | --- |
| Full28, 64+42 | No R3/R4, rounding2 | 43 historical score rows exactly equal; 1,204 layer output hashes equal |
| Full28, 64+3 | R3 | 112 independent R3 checks, maximum quantized-code error 0 |
| Full28, 64+3 | R3+dense R4 | 112 R3 checks; four FP16-stage/INT16/conditional Down-residual checks pass |
| Full28, 2048+3 | R3+HVX butterfly R4 | 112 R3 checks; four FP32-stage/INT16/conditional Down-residual checks pass; four high-context scores exactly equal historical EXP-0333 |
| One layer, 64+3 | R3+dense R4 | Four R3 and four R4/Down checks pass |
| Three layers, 64+3 | R3+HVX butterfly R4 | Twelve R3 and four R4/Down checks pass |

Device execution enforces ledger closure, requested/granted 8 MiB VTCM, cache guards and zero timed intermediate DDR/spill. Full28 long-path peak plans are 7,013,600 bytes without R4, 6,554,848 dense, and 6,620,384 butterfly. Conditional references use actual hardware inputs and the frozen EXP-0333 finite contracts; they are not ideal FP32 end-to-end equivalence. The short/high-context diagnostic score subsets are not new PPL estimates or formal throughput baselines. Historical full 8,192-target PPL and formal speed evidence remain the quality/performance records.

All 15 service tests pass, covering hash integrity/dedup, incompatible models, commit pinning, transactional claims, isolated queues, preparation capacity, quiet admission/drain, paired ownership, failed lease retention/recovery, build-only tasks and Windows ADB command bounds. A separate local clone passes bootstrap, the 1,652-file audit and all 15 tests.

## Build-to-run timing and retained failures

Full28 configure+compile work totals 42.47–43.50 seconds on this host. Concurrent jobs can spend additional time in resource/queue waits; build_total_seconds therefore includes waiting and must not be presented as compile cost. With the hash cache populated, deployment/verification to run-ready takes 56.18–63.79 seconds in the full28 regressions, plus fixture input preparation/upload before first execution. Initial registry adoption/first cache population are separate cold setup costs. These measurements support source submission with a centrally pinned build environment; prebuilt binaries remain optional sealed artifacts.

Failed attempts are retained: missing Android NDK configure parameter was repaired; Android protected hardlinks required an owned verified phone cache copy; one expanded shell batch exceeded Windows process argument limits before inference. The latter was fixed with eight-file staging batches and a 24,000-byte transport guard. Inspected recovery confirmed dead owner and idle phone before releasing failed leases. No numerical gate, reference, weight hash or historical classification was changed. The phone cache verifies 892 owned blobs and has no dependence on old deployment symlinks.

## Evidence and handoff

Machine-readable acceptance: migration/acceptance.json. Raw task build logs, protocol records, numerical captures, export metadata and failed-attempt/recovery evidence are sealed outside Git at shared/results/exp0334/EVIDENCE_SHA256.json. This archive contains 868 files, 1,474,054,578 bytes, SHA256 ee1b0380ed6a6fb66bf20b25cc40f6b5d726becc7f502ce0c116073cba2acf97. Existing EXP-0333 evidence seal remains 7870592aab22091168a13d312f643077476e8a3c02921474e115b7fa564fffa8.

All acceptance jobs are completed or failed with evidence retained; there is no active phone lease. No runtime optimization, recalibration, precision unification or baseline promotion occurred during migration. Future experiments require their own user-approved registry entry and must use the new bootstrap/preflight and service entrypoint.

Post-publication bootstrap encountered the host proxy CONNECT failure. Direct process-local fetch succeeded against the private origin. Bootstrap now retries fetch at most three times, 30 seconds each, using only a process-local proxy bypass after the first failure; all failures remain a blocking unverified refresh. Configured GitHub CLI credentials are supplied ephemerally, without storing a token in Git. No global proxy setting or runtime arithmetic changed.
