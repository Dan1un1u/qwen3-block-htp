# R3 long-context scheduling (EXP0327)

Qwen3-1.7B W4A8 uniform INT16 Down, FP32 residual2, projection rounding2, dense R3 mode1, frozen EXP0324 package. User provisionally accepts accuracy; this change preserves the deployed integer outputs and does not claim ideal arithmetic equality.

`QBH_R3_OPT=2` reuses EXP0259 Q/K-ready streamed preparation; head workers write disjoint raw FP16 rows into the dead Gate arena. Ring biases use dead attention-concat scratch, avoiding the live R3 rows. HMX matrix jobs remain serialized, with preparation overlapping QKV.

`QBH_LONG_OPT` bit21 (`2097152`) enables existing native W4 QKV on every complete M64 prefill chunk, not only at position zero. The logical tile geometry is unchanged. Without this bit, the old ring rejects OPT2 at the second long-prefill chunk because it enters the expanded-S8 four-slot path. The rejected experiment is retained. Both R3 and OFF cost references receive this shared optimization.

Bits19:20 select a capacity-clamped attention-client limit:0 retains original maximum4;1 caps at2;2 caps at3;3 is reserved. Slot sizes and owners are unchanged. The existing384KiB reserve remains; do not re-enable the EXP0319 failing larger reserve. Runtime computes required storage from current padded KV length, not the initial prompt length.

Numerical checks compare exact deployed output hashes, persistent cache guards, and short R3 raw/carrier captures, before timing. Additional independent-reference/PPL work is intentionally not repeated for these bit-preserving schedule changes. Formal results and recommended flags are recorded in project memory.
