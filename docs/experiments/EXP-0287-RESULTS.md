# EXP0287 — Qwen3-1.7B W16A16 inference DRAM versus W4A8-SP2

## Main result
| Recipe | Peak observed DRAM (MiB) | GiB | KV capacity |
|---|---:|---:|---:|
| Latest W16A16, EXP0283 OPT3 | 3901.422 | 3.809982 | 80 |
| W4A8-SP2, FP32 residual, EXP0286 | 2780.527 | 2.715359 | 128 |

W4A8-SP2 uses **28.7304% less observed DRAM**;
W16A16 / SP2 = **1.403123x**.
Both batch1,64-token prefill+15 decode, full28 layers. Three independent process
launches, ten trajectories per process, unchanged original binaries. W16A16 OPT3
output token sequences and FP16 logits exactly match the sealed W16A16 reference.
All480 W16A16 profiles pass; no new PPL or formal performance claim.

## Theoretical interpretation
The result agrees exactly with the actual runtime allocation layout plus page
rounding and ordinary resident memory. It does **not** give the idealized fourfold
total-memory reduction suggested by counting W16 vs W4 bits alone.

| Arena content | W16A16 MiB | W4A8-SP2 MiB |
|---|---:|---:|
| Backbone base weight representation | 2688.000 | 672.000 |
| FP16 embedding | 593.500 | 593.500 |
| LM-head base weight representation | 593.500 | 148.375 |
| Additional persistent weight layouts/bundles | 0.000 | 1330.500 |
| Live K/V backing | 8.750 | 7.383 |
| Metadata, LUT, reference storage, boundaries and other arena bytes | 9.880 | 21.077 |
| **Unrounded main arena total** | **3893.630** | **2772.835** |

Independent dimension check:
- Backbone parameters =28*(2*2048^2+2*2048*1024+3*2048*6144)=1,409,286,144.
  At16bits:2688MiB; at4bits:672MiB. Exactly4x.
- Embedding:151936*2048*2=622,329,856bytes=593.5MiB in **both** paths.
- LM head base:593.5MiB FP16 versus148.375MiB W4, also exactly4x.
- W4 additionally retains672MiB direct-N backbone layout,510.125MiB Gate/Up+Down
  streaming bundles (including their auxiliary contents), and148.375MiB direct-N
  LM-head layout, totaling**1330.5MiB**. These are distinct physical regions;
  retaining them supports current prefill/decode consumers. These are not an
  inevitable cost of4-bit quantization, and not a cost attributable solely to SP2.
- FP16 embedding and packed FP16 LM-head are separate allocations even when
  original model parameters are tied. Parameter count alone misses that physical
  representation duplication. No storage deduplication was performed.
- SP2 activation intermediates in VTCM are excluded by the requested metric.
  Ordinary host memory is~7.7MiB; most of the peak is the fixed shared arena.

Exact physical closure:
- W16 arena4,082,766,464bytes ->4KiB-rounded4,082,769,920bytes.
  Three4KiB RPC buffers +7,964KiB ordinary RSS yield4,090,937,344bytes.
- SP2 arena2,907,528,448bytes ->4KiB-rounded2,907,529,216bytes.
  Three4KiB RPC buffers +7,864KiB ordinary RSS yield2,915,594,240bytes.
No unexplained multi-GiB gap remains in observable accounting. This closure does
not prove hidden DSP-private/kernel allocations are zero.

## Capacity / trajectory caveats
Original W16 package uses capacity80; EXP0286 SP2 uses128. Attempting128 with the
unchanged W16 package fails vertical-layer package size audit before inference.
Both failed diagnostics preserved; no threshold or manifest hash was changed.
We retain original W16 baseline rather than fabricate its reference/cache files.
F16 live KV bytes scale as28*2*K*1024*2:8.75MiB at80,14MiB at128.
The arena also retains equal-sized cache references, so matching128 would add
**10.5MiB** in arena accounting (projection only, NOT a hardware result).
That is~0.27% of current W16 peak and cannot explain the1.095GiB recipe difference.
No claim of perfectly matched capacities is made.
W16 uses its original greedy trajectory; SP2 uses its frozen fixed trajectory and
prefix policy. Persistent arena is capacity-planned before inference, so token
values do not change these fixed allocation counts. This is a memory comparison,
not a claim of paired quality/speed or identical mathematical recipes.

## Reproducibility and measurement scope
Same external native collector as EXP0286, unchanged SHA256. Deduplicate DMA-BUF
by inode, add ordinary smaps RSS, subtract any DMA RSS already counted.
DMA smaps RSS/PSS is zero here; adding VmRSS alone would miss the principal arena.
Start inference classification after flushed eval_model_ready; exclude loading.
No VTCM, sampler process memory, system-wide before/after delta or disk model sizes.
W16 actual median sample interval~53.6ms; sampled peak, not a certified exact max.
No access to global DMA-BUF/kernel-only or DSP firmware-private accounting; same
coverage limitation as EXP0286. Unmapped reclaimable file cache is not included.
Three W16 measured peaks:
- Run1: 3901.421875MiB, 246 inference snapshots, outputs exact.
- Run2: 3901.363281MiB, 247 inference snapshots, outputs exact.
- Run3: 3901.355469MiB, 251 inference snapshots, outputs exact.

## Allocation attribution diagnostic
layout_probe.so interposes rpcmem_to_fd only in separate diagnostic runs, after
the host fills its arena. It reads descriptors using headers extracted from exact
sealed source81e16c4, matching ABI134/header89168bytes. No tensor content or offsets
are changed. Both diagnostic outputs exactly reproduce their own sealed recipe.
These runs are **not** included in peak memory statistics. All captured ranges
are in bounds and mutually disjoint; summing them plus remainder exactly recovers
shared_bytes. Collector/probe sources, raw profiles, descriptors and SHA ledger
are retained. Runtime source, model tensors and selected baselines remain unchanged.

Next possible memory work (not performed): study redundant DDR weight layouts and
package reference retention; preserve current speed before accepting a lower-memory
variant. Do not market the current implementation as4x lower whole-runtime memory.

Evidence ledger SHA256: 0c0e4aaf230668ea7181215257b077cfe312efb14b201ea391eda2fa0c0718ca
