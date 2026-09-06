# EXP0234 running checkpoint

PC052 second phase. Source branch codex/exp-0234-w4f16-group128-64k, current committed source 1c934d27dbe2cb142f89a7226e916eb8586c269e. Bootstrap, read four authorities, then preflight before work.

Data and independent group quantizer oracles passed. Input freeze SHA256 a69984650c65d3a1b5d0ffa632be4b9e1c4ae6c5651978281aa694f9be01de80. Same verified EXP233 paired panel; actual calibration is exact C64 512x128 positions. Six F/C64 development/primary/reserve scores are verified byte copies (reused_controls.json); reserve scores only enter the decision if triggered.

CPU G64 exporter is RUNNING from actual start source 6ab684555a47bd8612fc50c038f92ca7e019c9c2, unified exec session 18937. Do not duplicate, interrupt, or edit its quantizer files. Check commands/logs/processes if session is unavailable. Legacy manifest labels source at completion; report separately verifies actual starting source archive and identical quantizer file hashes. Current source changes after start affect evaluation/reporting only.

G8 development completed at PPL27.478140993518856 with exact EXP231 per-token NLL/top1 regression. G8 primary is running in session73233, source 1c934d27dbe2cb142f89a7226e916eb8586c269e. Frozen G8 primary depends on F/C64/G8 development checks only, so it can overlap CPU export; G64 final scoring still requires allfour development checks and reserve still requires fixed summary trigger. This is an orchestration dependency repair, no numerical/protocol change.

After export: G64 development, G64 primary, summary; if triggered, G8/G64 reserve then combined summary. Generate report, independently verify complete file/hash coverage, close EXP234 memory. EXP235 fixed C64 sensitivity remains authorized and must follow under its own registration/preflight. Its protocol was frozen before EXP233 final scores. Other recipes/runtime frozen, no promotion.

## Scoring continuation running

Source78d76eaea19a778d9500afa59428625620bb88da adds finish_exp0234_scoring.py. Unified session48693 waits for the exact successful export and G8 primary command records, then executes G64 development/primary, fixed summary, conditional G8/G64 reserve and combined summary. Do not manually duplicate these pending stages while session48693/process remains active. It stops ready for report; report/independent closure and EXP235 still require continuation. implementation_equivalence.json proves the group source differs only by result import, both core quantizer hashes match G8, and complete CPU per-layer calibration/quantization AST is identical.

## Halfway checkpoint

Current source a37a64d1b928184665a66c3ed1b5e0a6155615e5, clean/pushed. G64 export session18937 completed layers0..13 (14/28), all checks pass, elapsed3610s; keep running. G8 primary session73233 completed: PPL26.77058071194769, independent math.fsum matches. Scoring continuation session48693 still waits on export and will run all remaining fixed/conditional scores automatically. Do not duplicate stages. report_exp0234.py now records all stage actual-start sources versus legacy completion-HEAD fields. verify_exp0234_closure.py is ready to independently check complete evidence/artifact coverage, hashes and36 raw-token PPL reductions after report closure. All three PC052 phases remain authorized; EXP235 implementation/execution follows successful EXP234 closure.

## Three-quarter checkpoint

G64 exporter session18937 completed layers0..20 (21/28), elapsed5284.7s, checks pass. Current source c8e57fd0d308df10f0efa82cbbe0f96c49ed2839 is clean/pushed; added archived audit_exp0234_equivalence.py stage successfully reproduced implementation_equivalence.json without modifying original evidence. Stage session40481 completed. The only active jobs remain export18937 and scoring continuation48693; the latter will launch G64 development/primary and conditional reserve scores after export closes. Do not duplicate them. EXP235 remains pending and authorized after EXP234 closure.

## Export complete, automated scoring active

Export session18937 completed successfully:6983.851180413039s,28layers,196projection checks pass,112forward checks exact NRMSE0. G64 manifest SHA256 fc1a0f9655b71a1e21c15f5272440058b95b3ea59b63dc0b04343dafcfab9ae0. Actual export start source remains6ab684555a47bd8612fc50c038f92ca7e019c9c2; completion labelc8e57fd0d308df10f0efa82cbbe0f96c49ed2839. Quantizer files unchanged and independently verified. Session48693 now runs G64 development, primary and fixed conditional reserve sequence; do not duplicate. After it finishes, run report_exp0234.py with CPUvenv, then verify_exp0234_closure.py; close memory and proceed to EXP235.
