# L32-0040 uniform INT16 Down control

Only frozen per-layer middle scale and fused SwiGLU LUT change. Native DSP path is byte-identical to SP2 mode8: uniform reconstruction grid instead of241 SP2 levels, both in the same two-byte physical carrier. Original A8/SP2 controls retained. No precision/quality claim; independent reference before full-model timing. Signed24 raw-store bounds must pass for all frozen Down weight rows, signed32 merged result and modular intermediate arithmetic audited.
