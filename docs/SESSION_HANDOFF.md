# L32-0023 closed: full16 numerical gate failure

Source `/home/daniuniu/work/llama32-htp`, branch `codex/llama32-no-rotation`,
closure `11807d80b16b6c5efe376226befeac56e7b0cfc8`; clean and pushed. Tested N16 build `022ded9aeeeeb13189831a259e43c0bebfa97a6e`.
Native unchanged from0022 `3fb9a4f694e36552c1d7f75fc40de59007cfc8cf`.
Active experiment: none. Next ID: L32-0024. No device work until registered preflight.
Read `docs/LLAMA32_ROTATION_FULLMODEL_VALIDATION.md` and `docs/experiments/L32-0023.md`.

User continuation authorized full16 then frontend/E2E conditional on unchanged
numerical gates. OFF full16 M64+decode1 is bit-exact. BOTH fails minrowcos>=.999:
prefill0.5051155954/NRMSE0.3883987963, decode0.7882501259/NRMSE0.6913102269.
CLI exit1 preserved, all outputs finite. Cache values also diverge, structural
and preserved-prefix checks pass. This is implementation-reference divergence,
not a PPL/quality measurement. No frontend/E2E/formal/PPL run and no promotion.

Diagnostics: first-layer R4 difference2.3841858e-7 crosses one SP2 code10/12.
Only injecting audited native layer0 into subsequent ideal software is enough
to amplify strongly by layer index3. Actual full16 prefill hidden capture confirms
all same-input local layer mincos>=.999999971; layer index3 is locally exact but
global mincos=.9768659137. Its inputNorm has five changed A8 codes, each one code.
This supports rounding-threshold/trajectory amplification, without proving every
component exact. Do not replace independent goldens with local/captured outputs.
Full16 failure must not be accepted using local cosines or A8 quality exception.

No native repair attempted0023. Suggested next direction: general accurate
refinement for ambiguous R4-to-SP2 decisions, considering both R4 stages and a
justified error bound; no fixture-specific code/LUT adjustment. Cost and success
unknown. Recalibration for robustness changes frozen quantization contract and
needs a separately aligned experiment. Keep0022 repair: reverting0020 already
reproduced three-layer failure. Existing singlelayer speed/chain3 evidence is
valid only within its measured scope. R3/R4 fullmodel remain unvalidated.

Artifacts: `/mnt/d/llm_exp/results/llama32-htp/l32-0023`; ledger `evidence-ledger-checkpoint-a01.json` SHA256 `88535e31adafd7cc192e5d283a564da5ddfeabdecc190bc87c8cb8de28939975`.
33 result files,20 model manifests,1079 model payload entries hash-verified;
prior0022 ledger249files/sixmanifests reverified. ThreeCLI/fiveRPC; twoexit0,
oneexit1. Untimed capture writes8MiB diagnosticDDR and is NOT formal physical
or speed evidence. Ordinary runs zero intermediateDDR/spill;8MiB acquired;
BOTH peak8229344. Legacy capture filename*_u8.bin contains FP32.

Models `/mnt/d/llm_exp/models/llama32-htp/l32-0023`:
`rotated-down-extension-a01` has11 fresh originalBF16 folds3-6/8-14;
`chain16-both-a01` combines those with verified0/1/2/7/15 and independent goldens.
OFF uses frozen0016/chain16-a01. Do not rebuild/overwrite sealed artifacts.
Source build currently N16 sealed at testedHEAD (closure only docs differs);
new device deployment needs clean matching build seal or explicit sealed reuse.
Other Llama rotation worktree and frozen Qwen3 branch unchanged.
