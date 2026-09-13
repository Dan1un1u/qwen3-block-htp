# L32-0010 active

Full-model SP2 Down E2E comparison authorized. Read docs/experiments/L32-0010.md.
L32-0009 completed single-layer evidence retained; no jobs yet.

## L32-0010 full-model SP2 E2E closure

All16 SP2 Down integration exact,3-layer/KV then16-step independent integer greedy gates pass. Fixed10 same-build AB/BA M64+15 pairs: U8/SP2 prefill2026.0344/1787.5134 token/s,latency+13.3437%,CI[1.116943,1.145942],FAIL10%gate. Decode46.2703/45.8830 token/s,+0.8440%,CI[.999387,1.018670],PASS. Single-layer pass did not transfer to full-model prefill. Gate/Up+SwiGLU adds3546.3us,Down1082.8us perM64; exact stall cause not isolated. All320timed ledgers reconcile;VTCMmax8212960B,no intermediateDDR/spill/weight expansion. No PPL,quality promotion or repeated unchanged formal. Next discuss prefill-only optimization separately; default remains originalU8. See L32-0010 record/report for full data.
