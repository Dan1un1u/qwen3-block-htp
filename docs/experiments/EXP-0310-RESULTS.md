# EXP-0310 — SM8750 / V79 independent memory bandwidth

Completed 2026-09-23. This measures the highest sustained effective throughput found by the disclosed microbenchmarks, **not an independently verified theoretical bus ceiling**. No LLM is executed and no model baseline is promoted.

## Results

Decimal GB/s (10^9 bytes/s). Each row: 3 independently opened FastRPC sessions × 12 timed rounds; discard the first 2 warm-up rounds in each session, giving 30 formal samples. Median is the primary result; best is explicitly identified.

| Path | Median GB/s | Best GB/s | Range GB/s | Bytes / DSP cycle |
|---|---:|---:|---:|---:|
| VTCM → HVX, 1 context | 135.17 | 135.17 | 135.17–135.17 | 64.00 |
| VTCM → HVX, 2 contexts | 270.32 | 270.33 | 270.32–270.33 | 127.99 |
| VTCM → HVX, 4 contexts | 540.64 | 540.64 | 540.63–540.64 | 255.98 |
| VTCM → HVX, 6 contexts | 810.95 | 810.96 | 810.91–810.96 | 383.97 |
| HVX 6-context read, 2 KiB address skew | 810.95 | 810.96 | 810.90–810.96 | 383.97 |
| HVX → VTCM, 6 contexts | 810.63 | 810.73 | 790.34–810.73 | 383.82 |
| HVX VTCM copy, read + write aggregate | 810.62 | 810.73 | 810.58–810.73 | 383.82 |
| VTCM → HMX, FP16 × FP16, stream32 | 1060.15 | 1060.17 | 1060.14–1060.17 | 501.97 |
| VTCM → HMX, FP16 × FP16, stream4 | 1060.15 | 1060.16 | 1060.15–1060.16 | 501.97 |
| VTCM → HMX, U8 × S8 | 928.97 | 928.98 | 928.96–928.98 | 439.85 |
| VTCM → HMX, U8 × packed S4 | 774.14 | 774.16 | 774.13–774.16 | 366.55 |
| DDR → VTCM, DMA 1 MiB × 8, source bypass | 63.33 | 64.39 | 61.79–64.39 | 29.99 |
| DDR → VTCM, DMA 2 MiB × 4, source bypass | 63.28 | 63.79 | 61.30–63.79 | 29.96 |
| DDR → VTCM, DMA 4 MiB × 2, source bypass | 63.47 | 64.07 | 61.16–64.07 | 30.05 |
| DDR → VTCM, DMA 1 MiB × 8, non-bypass control | 37.61 | 37.72 | 37.29–37.72 | 17.81 |

The HMX maximum measured in this experiment is about 1.06 TB/s of summed activation and weight operands. Six-context HVX read is about 0.811 TB/s. DMA DDR→VTCM is approximately 63–64 GB/s; the single best round is not a guarantee of sustainable service. All formal output validations pass.

## Device and power

- PJZ110 / SM8750, HTP V79, serial 3B15C8007Z300000.
- Architecture-total 8,388,608-byte VTCM grant verified on every invocation.
- `qurt_hvx_get_units()` returns 0x600: six 128-byte HVX units.
- Existing project session requests DCVS performance mode, core and bus target/max TURBO_L3, minimum NOM, HMX power on. Firmware retains clock/arbitration control.
- Observed DSP cycles / qtimer ratio is ~2,112 MHz throughout formal measurement. QTimer is 19.2 MHz; cycle values are not used as wall time.
- Battery 85%; temperature 29.2–29.3 °C in before/after snapshots, USB connected. These are device temperature readings, not a measured DSP die temperature.

## Byte accounting and timing

`GB/s = counted_payload_bytes × 19.2e6 / qtimer_ticks / 1e9`. HVX multi-context wall interval is earliest worker start to latest worker completion, not summed worker time. Thread creation, context acquisition, RPC, memory initialization, correctness checking and host printing are outside timed intervals. DSP barriers finish outstanding work before the end timestamp.

**HVX:** 128-byte aligned VTCM vector loads/stores. Formal workers traverse private 512 KiB regions, repeating 8,192 times (copy: 4,096). A 32-load unrolled hardware-counted read loop has no inner mode branch or reduction. Inline-assembly volatile loads are retained, all 32 loads and pointer advances confirmed in saved disassembly. Last-read register contents are checked against an address-sensitive byte pattern; copy and write destinations are checked byte-for-byte. The copy row counts reads plus writes; its one-way copied-payload rate is half the displayed aggregate (~405 GB/s). The read and write rows each count only one direction. Single-context ~135 GB/s must not be quoted as the chip-wide HVX ceiling.

**HMX:** Native matrix instructions read streamed operands from VTCM. Each FP16 tile reads 2,048 activation + 2,048 weight bytes; U8×S8 reads 2,048 + 1,024; U8×packed-S4 reads 2,048 + 512. Count each explicitly streamed tile once, never multiply by matrix-element reuse or expand S4 to S8 in byte accounting. Formal activation extent is 3 MiB; FP16 weights occupy another 3 MiB, S8 1.5 MiB, S4 0.75 MiB. Repeat 4,096 traversals. Timed loops include accumulator clearing, output conversion/store and stream submission overhead. Output bytes and bias metadata are not added to the reported input bandwidth. HMX bandwidth is **effective operand consumption while executing MACs**, not a pure load-only measurement or a physical-port performance-counter reading. Integer MAC scheduling and format affect the byte rate; the lower packed-W4 GB/s does not mean lower dot-product throughput. U8×S8 and U8×S4 have essentially equal time for the same tile count, with fewer bytes consumed by S4. FP16 accumulations of exact 1/256 inputs are checked against exact expected half outputs; integer positive sums against their conversion/saturation contract. Untimed independent activation-zero and weight-zero perturbations also verify both operands affect results.

**DMA:** RPCmem-allocated uncached 128 MiB host source, with byte pattern dependent on within-page, page and MiB address. Source spans the full allocation in rotation; each formal timing transfers 2 GiB into an 8 MiB VTCM destination. Descriptors link 1 MiB×8, 2 MiB×4 or 4 MiB×2. Descriptor setup, `dmstart`, wait, and completion barrier are included. No DMA source/destination compression enabled. Source cache bypass is set for headline results; destination is VTCM. All 128 MiB of distinct source regions are copied and checked outside timing, in addition to checks of each timed round's final destination. Non-bypass control is ~37.6 GB/s. Large footprint + explicit DSP-cache bypass avoids a small hot-cache copy benchmark, but there are no DRAM-controller counters proving every request's physical external-bus provenance; system-cache/controller effects cannot be separately attributed.

## Search and validation

Swept HVX 1/2/4/6 contexts and 16/64/512 KiB working sets, then address skews from 0 to 16 KiB; HMX stream lengths 1/4/16/32 and activation extents 64 KiB/1 MiB/3 MiB; DMA chunks 4 KiB through 8 MiB and chain depths 1 through 256 within 8 MiB VTCM. Selected adjacent large DMA configurations and HMX stream4/32 agree near their plateaus. The initial HVX read loop left software control overhead: six-context ~765 GB/s rose to ~811 GB/s after hardware-loop unrolling; final result therefore does not simply report an unoptimized memcpy. No throughput is extrapolated from LLM layer/token timing.

180 successful scan/formal RPC invocations are recorded in `sweep.jsonl`, including 45 final-version formal invocations / 450 retained timing samples across 15 configurations. Historical intermediate scans use their corresponding binary seals and are not pooled with formal data. Initial compile API mismatch and smoke timing/bias-layout defects are retained in logs and excluded. A too-large repeat-count scan was rejected by ABI validation before measurement; corrected follow-up is separately logged. These were attributable harness issues repaired under PC037. Final 45 invocations have rc=0, status=0, errors=0; all full VTCM grants, six-unit hardware query, positive timestamps, and complete DMA source coverage pass. Final device executable/shared-library hashes match the seal.

## Evidence and reproduction

Measured source: `3ee9dffcf5159d82f03dbba1e7e2ab5619627a1f`, branch `codex/exp-0310-peak-bandwidth`. Evidence: `/mnt/d/llm_exp/results/qwen3-block-htp/exp0310`.

- `seal_final/manifest.json`: measured source and binary SHA256.
- `seal_final/bandwidth.disasm`: actual HVX and HMX instructions.
- `seal_final/*CMakeCache.txt`, `build_attempt6.log`: build configuration.
- `formal_cases.json`, `formal_*.log`, `summary.json`: exact cases, all raw ticks/cycles, aggregate statistics.
- `formal_thermal_*`, `device_binary_sha256.txt`, `device_getprop.txt`: device evidence.
- `scripts/exp0310_bandwidth_sweep.py` and `src/{dsp,host}/bandwidth_*`: versioned runner and implementations.

Build under the project environment with `build_cmake android BUILD=ReleaseG -j8` and `build_cmake hexagon BUILD=ReleaseG DSP_ARCH=v79 -j8`. Use matching `qwen3_bandwidth_cli`, `libqwen3_probe.so`, `libqwen3_probe_skel.so` from one build. New IDL entry changes method ordinals; do not mix historical stubs/skels. Follow project-memory experiment ownership/preflight before future hardware use. CLI arguments are `mode bytes workers repeats stream depth bypass`; modes 0/1/2 = HVX read/write/copy, 3/4/5 = HMX FP16/U8S8/U8S4, 6 = DMA. For HVX, depth encodes address skew as `(depth−1)×128` bytes. Host arrays expose all 12 rounds; analysis discards rounds0–1.

## Interpretation boundaries

These are separately measured, aligned, contiguous microbenchmark peaks. They cannot be summed to claim simultaneous HVX+HMX+DMA performance, nor assumed for strided/gather access or LLM execution. They do establish a measured on-chip/off-chip bandwidth gap: ~13× for aggregate HVX reading and ~17× for the largest HMX operand rate versus DMA. Calculating a true architectural upper bound would additionally require an authoritative V79 port-width/issue-rate/clock specification; this experiment does not invent those unpublished parameters.
