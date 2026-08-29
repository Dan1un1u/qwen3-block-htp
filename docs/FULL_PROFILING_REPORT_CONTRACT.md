# Complete profiling report contract

Every experiment closure must retain one compact report and reproduce its
tables in the user-facing completion response. A raw JSON link or a list of
only the experiment's target stages is not a complete report.

## Required identity and comparison

The report names the Experiment ID, source branch and commit, formal evidence
directory, artifact directory, execution unit, Project Variant, direct control,
candidate, repeat scopes, paired-round count, and whether the comparison is
paired. If an older Selected Baseline is shown for context but was not measured
in the same paired run, it is labelled non-paired and cannot replace the direct
control.

## Required tables

For repeat one and repeat ten, every numeric row shows the control median,
candidate median, and candidate-versus-control percentage. The report contains:

1. Primary latency: Host wall per block, DSP block total, invocation, runtime
   setup and teardown, named ledger total, and unattributed ledger time.
2. The complete additive Block Timing Ledger: input, metadata, input norm, QKV,
   QK Norm/RoPE, Attention, O, post-Attention residual, post-Attention norm,
   Gate/Up, activation, Down, final residual, and output. Fused stages remain
   present as zero or near-zero rows.
3. Projection diagnostics: DMA, HMX, pack/unpack, ready/wait, experiment-specific
   expansion, prefetch, lifetime, and worker/pool counters.
4. Attention diagnostics: Q/K preparation, packing, QK HMX, requantization,
   Softmax, AV HMX, AV requantization, pipeline waits, and relevant task counts.
5. MLP diagnostics: Gate/Up and Down pipeline time, activation work, weight
   staging and expansion, HMX compute, readiness, slot pressure, and relevant
   publication/consumption counts.
6. Physical contract: weight and boundary DDR bytes, intermediate DDR bytes,
   DMA descriptors, spill/fill, HMX commands and tile pairs, requested/acquired
   and peak VTCM, FastRPC count, HMX ownership, and backend/fallback status.
7. Correctness: final mismatch and maximum LSB, output hash, independent
   implementation-reference results, non-finite or mask violations, and every
   experiment-specific numerical gate.

Block Timing Ledger rows are mutually exclusive and may be summed. Engine-work
and wait counters may overlap one another and are diagnostic only; their table
must say so explicitly. Bytes or setup work amortized by a Prepared Runtime
Session are reported per block using the experiment's declared normalization.

## Missing data and closure state

Rows are never silently omitted. Use `0` for a measured zero and `N/A` only when
the measurement does not exist, followed by the reason. If building, loading,
or execution fails before profiling, the experiment can be closed only with a
report that identifies the precise failure boundary, lists all available static
and runtime evidence, and marks all later sections `N/A`.

The experiment index records `full_profiling_report`. Until that file exists and
the user-facing tables have been delivered, execution may have stopped but the
experiment is not recorded as completed.
