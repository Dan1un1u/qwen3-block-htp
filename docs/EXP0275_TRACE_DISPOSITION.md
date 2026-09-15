# Diagnostic trace disposition

All 15 selected-layer diagnostic runs retained and independently byte-exact. Attempt a01 produced no logcat events; a02 enabled SDK-documented process-name .farf routing; a03 added post-join draining delays in trace-only code. a02/a03 retain roughly 250 events per process, only one RPC count, and no complete two-RPC stream. The exact transport truncation mechanism is unresolved; the earlier burst-drain cause was a repair hypothesis, not a verified root cause. No further trace repair in this bounded phase.

Full timeline coverage and engine utilization are N/A. Partial records are retained as raw diagnostics and are not used for timing or overlap percentages. Production binaries were built with QBH_PAPER_TRACE=OFF at 19c9f728dbf2d06a2f6d0ec6488f4d1547627226; all 14000 timing profiles remain eligible. No arithmetic, VTCM, calibration or timing-sample changes were made. The bounded diagnostic event array is DDR metadata, not timed tensor traffic.
