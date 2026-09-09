# EXP0262 checkpoint

Source codex/exp-0262-w4u8-r4-native-layout-pipeline 016cb33213457caf1fa61f1b44655c07bc136c9f; active EXP0262. Native ABI125 built de5a612e8e044449c04bbaf63f7840321492a89d, runtime_l1.json selects exp0262-l1_attempt2. Frozen EXP0261 packages reused, local/remote verified in frozen_package_check.json. No mllm.

Vector pairwise native layout and output gather+quantization exact; OPT2 restores decode FP16 LUT input production on Gate/Up readiness and uses prefill8batch doublebuffer input/output256KiB slots. Same H12/H512 math, no butterfly, oldcontrolOPT0 preserved; oneHMXowner,8MiB,zero intermediateDDR. Numerical_gate.json pass;116 exactfilecomparisons to sealed EXP0261 across4captures. Known idealR3 failure unchanged. Parent scalarR4 oracle audit_a2 is a read-only link to sealed EXP0261.

5short complete, integrity pass, repeat10pre ratio1.43759/decode1.04119. Tenformal next, no further native tuning. Use scripts/device_exp0262.py formal, then report_exp0262.py, provenance/seal/closure. Fullmodel forbidden by prefill>10percent. E2E N/A. Retained runner_parse_failure before staging; no native mismatch. All scripts committed; source/memory clean/pushed required before run.
