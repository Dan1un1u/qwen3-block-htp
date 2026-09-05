# qbh-lite-v1

EXP-0218 implements the approved lightweight evaluation. The committed
data/eval/qbh-lite-v1.json is the frozen, project-authored CC0 diagnostic set.
It is not a general benchmark. All task questions, answer rules, full chat
instructions and token IDs were frozen before any model inference.

- Full: 32 bilingual raw-text windows, each 64 context + 16 distinct scored
  continuation tokens; 512 targets total. Two overlapping contexts come from
  each of 16 documents; these are not 32 independent documents. The two target
  spans in each document are disjoint.
- Quick: fixed 8 windows × first 8 targets (64 scored tokens) and 8 short tasks.
- Tasks: 24 fixed questions, 8 extraction, 8 arithmetic, 8 format; half Chinese
  and half English. Greedy, non-thinking, at most 16 output tokens.
- Open text: 4 examples, 16-token prefixes; unscored and may be incomplete.
- Holdout: 8 additional document windows, not run in this initial snapshot.
  Run before a future promotion, not for tuning.

Every chat prompt contains exactly 64 genuine tokens. The preparation script
selects meaningful explicit instructions before inference to achieve this
length; it never pads or truncates a question. Raw-text windows are not chat.
Each sample starts at position zero with empty per-layer caches. Between
samples the resource session is renewed and both DDR cache state and private
VTCM tails reset; loaded and mapped weights are reused.

Teacher forcing scores the first target from the M64 last row, then feeds the
previous true target into each decode. Each actual DSP LM head contributes
every vocabulary logit exactly once. A phase-dead Down buffer stores a histogram
of 65536 FP16 bit patterns or 256 U8 codes; the latter are dequantized with the
actual output scale and zero point. Only scalar NLL/rank/tie diagnostics leave
the DSP. Formal speed does not initialize, fill or reduce these histograms.

Scoring stops textual interpretation at the first EOS, even though the device
uses the fixed 16-call sequence. Text answers use the frozen exact aliases,
numeric answers require a number alone, and JSON answers require a parsable
object with exactly the requested values and types. No post-hoc stripping of
explanations, Markdown fences or punctuation is allowed. This deliberately
measures short-answer instruction following as well as factual correctness.

## Running the retained benchmark

First follow Project Memory bootstrap/preflight and register the experiment.
Use the isolated Python environment at
/home/daniuniu/.cache/qwen3-block-htp-py/bin/python.

The retained EXP-0218 files are under
/mnt/d/llm_exp/results/qwen3-block-htp/exp0218.
Each remote runtime directory contains the compiled ABI-108 runtime, immutable
block_package_layer14_m64 symlink and the frozen suite .bin files.
Do not overwrite retained evidence. For future candidates, use a new experiment
result directory and remote root, while retaining the v1 dataset/tokenizer
hashes and the same scoring implementation.

    python scripts/test_exp0218_score.py
    python scripts/eval_exp0218.py teacher
    python scripts/measure_exp0218.py quick
    python scripts/measure_exp0218.py full
    python scripts/measure_exp0218.py repeat
    python scripts/measure_exp0218.py warmup
    python scripts/measure_exp0218.py short
    python scripts/measure_exp0218.py formal
    python scripts/summarize_exp0218.py

The teacher command creates the original BF16 cache once and refuses to replace
it. The measurement runner also refuses to overwrite a raw run. The preparation
script is for the initial freeze, not a command to regenerate the dataset each
experiment. Standalone device invocation supports a different remote root:

    EXP0218_REMOTE_ROOT=/data/local/tmp/qwen3-block-htp/new-candidate \
      bash scripts/run_exp0218.sh w4f16 quick > new-candidate-quick.jsonl

The fixed binary suite format is little-endian uint32: header magic 0x51424556,
version 1, sample count, record size 83 words. Each record has sample ID, mode
(1 teacher forcing, 2 free generation), step count, 64 input IDs and 16 target
IDs. There is no implicit token padding. Quality mode is selected by
QBH_EVAL_FILE; QBH_EVAL_QUIET suppresses the large profiling records.

Speed uses the same historical 64-token Chinese prompt for all recipes,
15 feedback decode calls and 16 returned tokens. One warmup, five short rounds
and ten formal rounds rotate recipe order. Prefill rate is 64 / complete Host
RPC wall; decode rate is 15 / summed complete Host RPC wall. Full loop wall,
device cold staging/open/map/prepare, WSL tokenizer/detokenizer and ADB launch
to first text are separately archived. Token/s describes hot token execution,
not cold request latency. No module-based extrapolation is used.

## Interpretation and first snapshot

F16F16 and W4F16 have deterministic v1 score snapshots; these do not promote a
Selected performance Baseline. W4U8's score is diagnostic only: repeated samples
show small logit-statistic differences and one initial instrumented regression
selected a different token. All failures are retained. A later measurement
repair or W4U8 investigation must preserve this evidence and rerun affected
gates; repeated success does not erase the failure.

Initial full scores are BF16 22/24 (NLL 3.641952), DSP F16F16 22/24
(3.633608), DSP W4F16 8/24 (4.231668), DSP W4U8 0/24 (18.3682,
provisional diagnostic). Independent software W4F16 is also 8/24 with
NLL 4.250193. Software and DSP W4F16 are not bitwise equivalent; their
teacher-forced top-1 agrees on 445/512 positions.

Reading a generated paragraph is insufficient evidence of model quality.
The proposed delta-NLL 0.02 or two-task drop thresholds remain alert proposals,
not hard acceptance criteria. Current W4U8 specialization remains paused.
