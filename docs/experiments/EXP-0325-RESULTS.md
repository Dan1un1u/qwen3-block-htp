# EXP-0325 — Frozen R3 software/hardware alignment

## Outcome
Major software contract mismatches were corrected in an independent deployment reference. Full8192-target bit-exact alignment was NOT achieved. This is a valid diagnostic result, not a whole-model bit-exact pass, numerical gate relaxation, quality acceptance or baseline promotion.

Frozen model: Qwen3-1.7B W4A8, uniform INT16 Down, FP32 residual, actual dense FP16 HMX R3. Weights, scales, QK calibration, prefix, tokens, masks and device arithmetic remain EXP0324. The only source change1fa5fca adds an optional host-only capture-position filter. DSP shared-library hashes are identical. No speed campaign or quality tuning occurred.

## Matched WikiText2 raw test subset
8192 target tokens, positions64..8255, five windows1984/1984/1984/1984/256 targets,64 warmup each; local contexts64..2047. This is not full-corpus WikiText2.

| Reference | PPL | Meaning |
|---|---:|---|
| Original BF16 teacher |17.874598|Frozen matched teacher|
| Previous hardware without R3 |28.467048|Historical same scoring protocol|
| Ideal R3 fakequant |21.780755|Exact reproduction of EXP0323, not HMX arithmetic|
| Corrected independent SDK/HVX/HMX reference |23.059651|Residual long-sequence mismatch remains|
| Actual HMX R3 hardware |23.319592|Sealed EXP0324; diagnostic scoring regression exact|
| Two measured QK codes injected into otherwise independent reference |23.367734928|Conditional attribution only; NOT independent accuracy result|

Independent reference remains1.114688% below hardware PPL; per-token max absolute NLL difference7.78378677. Three of five windows agree within float32 score-reduction error; two do not. Similar aggregate PPL is not evidence of per-token equality.

## Main contract mismatch
The ideal reference used floating projection scales and ideal nearest quantization. Hardware uses serialized bias words, half-encoded channel ratios, integer HMX output conversion and its rounding. Replacing this converter alone changes ideal21.780755 to23.665038. Backbone-only replacement gives23.683037, while head-only gives21.752390. This localizes the main discrepancy to backbone projection-output conversion, not LM head.

On real layer0 M64+3 Q/K/V outputs,53.67%/53.99%/53.61% of actual converter codes are one lower than ideal; the remaining codes match. This sample-level observation is not a universal floor formula: a simple floor with half scale still disagrees on approximately4%, so the SDK integer converter is retained.

Other matched semantics: sequential FP32 RMSNorm reduction, production HVX QK preparation, postRoPE FP16 carrier, half-encoded R3 normalization, SDK custom HMX accumulator/output conversion, FP32 residual, integer attention/LUT/Down16, and head conversion. The qfloat strict-IEEE flag configures the CPU emulator to reproduce the independently compiled V79 QK preparation; it does not change production hardware flags.

Single-factor PPLs are not additive error contributions. RMSNorm-only21.690861 and approximate R3-carrier-only21.526075 do not reproduce the dominant projection-converter effect.

## Independent correctness evidence
- Full28 M64+3:112/112 layer-step hashes exact,56/56 Q/K raw and quantized-code comparisons exact; no boundary injection.
- On5,763,072 captured FP16 rotation results, the SDK rate8 reduction differs at7 values, each within one FP16 ULP; short-sequence QK codes still all match. This does NOT prove full-input equivalence.
- Three fresh frozen-device diagnostic runs reproduce325+159+307=791 scoring rows exactly, including metadata/logits/NLL. Production DSP hashes unchanged. Selective captures are not timing measurements.
- Every original numerical failure, intermediate reference and rejected emulator hypothesis remains archived.

## Long-context divergence attribution
All layer numbers below are zero-based.
1. Window0, layer1, K head7/channel50, context370: identical raw FP16 input, adjacent FP16 R3 outputs, software K code226 versus hardware225. Full K/V history otherwise agrees at first residual divergence. The cached difference first changes the layer output/NLL at context388; it is not a KV append/layout error.
2. Window1, layer10, Q head8/channel118, context222: identical raw FP16 input, software R3 result6.5 versus hardware6.49609375; Q code230 versus229. The layer output then diverges. KV history matches.
Both differences are within the pre-existing component tolerance but cross quantization thresholds. They distinguish implementation-contract correctness from whole-model sensitivity.

Conditional diagnostic: replace only these two measured codes, never hidden states, weights, scales or other boundaries. Window0 becomes score-aligned throughout. Window1 remains aligned until context1812/corpus3796, then234 target rows differ above1e-4. Combined conditional PPL23.367734928, max absolute NLL5.530036926; this is not a full alignment pass. The later divergence has NOT been component-localized, and must not be assumed to be the same cause without capture.

## Limits and next step
The SDK arithmetic model substantially improves contract fidelity but is not a verified bit-exact surrogate of actual HMX R3 on every input. Neither the original mathematical dense dot nor changing the SDK reduction width/configuration reproduced all held-out device captures. Standalone HMX instruction-simulator attempts raised exception0x18 and produced no output; these attempts are not evidence of correctness.

No evidence supports changing weights/scales to conceal these discrepancies. Retain actual hardware PPL23.319592 as the deployment result. Further strict alignment must first resolve the HMX R3 accumulator/conversion reference or explicitly agree a bounded numerical contract and independently evaluate its sensitivity; the residual context1812 divergence remains a concrete capture target. Do not resume calibration on the assumption that the current simulator is bit-exact.

## Artifacts
- summary.json: machine-readable comparison and explicit partial-alignment classification.
- sdk-eval/: independent8192-target reference; conditional-eval/: separately marked two-code diagnostic.
- divergence-0/, divergence-1/, divergence-0-ctx370/: device protocols, original captures, score regression and per-boundary checks.
- README.md: tools and reproduction; diagnostic-attempts.md: retained failure history.
- EVIDENCE_SHA256.json: final immutable evidence ledger.

Evidence ledger SHA256: b420cdece676ef89a070d5a693b53ed32d20ea1979846d0eb1f2f4369ce04c4d.
