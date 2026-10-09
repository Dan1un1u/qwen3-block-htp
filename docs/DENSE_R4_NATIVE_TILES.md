# Dense R4 tile-preserving H12 implementation

Runtime source: 5feace117f8f816e31b925ab7e1721af13dd8b13. Qwen3-1.7B F=6144=12*512, M=64. Mode5 OPT7 remains old dense reference; OPT8 streams native channel tiles; OPT9 bulk; OPT10 bulk with four output contexts. All are explicit FP16 HMX factors, not butterfly evaluation.

Let A[a,t,c] be the existing H512 output (a=0..11,c=0..511). The second-stage contract remains
Y[t,b,c] = store_FP16(HMX_sum_a(A[a,t,c]*H12[b,a]) * half(1/sqrt(12))).
The first-stage FP16 output and both normalization multipliers are unchanged. Native input is split into16 chunks of32 coordinates. For each chunk the dense matrix T=H12 tensor I32 has shape384*384; its block(b,a) is H12[b,a]I32. HMX consumes row-major activation tiles and N-major/K-minor weight tiles. Confusing the two tile orders was the retained initial correctness failure.

Old H12 pads12 columns to32 for each token/coordinate row, requiring scattered transposition and output gather. New T consumes whole native32-channel tiles. Activation tile(r,a) is a contiguous copy of first-stage tile(2a+r,chunk); each tile2048B. Matrix output has the same native halfword interleave and is inverse-dealt into pairs of contiguous token rows. INT16 conversion preserves FP32 inverse-scale multiply, nearest/ties-away rounding, clipping[-32768,32767], low byte and high-byte+128 decomposition. One128B vector store owns four token rows for each output tile and byte plane.

OPT8 keeps two input/output slot pairs in the dead786432B H512 input arena, with HMX operating on one slot while HVX finishes the previous output and prepares the next input. OPT9/10 copy all first-stage inputs into that arena, then safely reuse the dead first-stage output arena for one32M*12K*12N HMX command. Output conversion dispatches once. OPT9 partitions12 groups across three contexts; OPT10 uses four (three existing persistent pool workers plus main) with disjoint three-group ownership each. No transient threads, extra VTCM allocation or live-input overwrite. Decode retains the prior small-row path.

Per layer first-stage tile pairs6144; old H12 pairs1024, new4608. Total R4 tile pairs7168->10752 (+50%), commands9->2. Input/output padding and scatter/gather vanish. The bulk schedule intentionally trades fine-grained overlap for fewer commands/dispatches; full Host wall decides which schedule is preferable.

Numerical evidence: initial one-layer and consecutive3-layer stages/codes pass independent FP16 bounds and exact conditional integer Down/residual reconstruction. Full64+42 OPT9/10 each match1204 old dense layer-step hashes and every archived audit tensor byte. The mathematical full H12 tensor H512 transform and original-derived rotated W4 Down package remain frozen. Dense FP16 and HVX FP32 butterfly have distinct declared contracts and are not asserted bit-equal.
