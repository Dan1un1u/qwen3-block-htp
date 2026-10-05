# W4A8 uniform INT16 Down quality baseline — 2026-10-05

User requested actual device PPL on all four current models, then limited PPL to models with readable text. User selected a fixed8192-target-token WikiText2 subset. All source/runtime/weight contracts remain unchanged.

| Model | Free text | WikiText2 subset PPL | Targets |
|---|---|---:|---:|
| Qwen3-0.6B | unusable; fragmented text/Answer/newline loops in both checks | Skipped by user rule | 0 |
| Qwen3-1.7B | limited readable relevant answers; repetition/control-marker leakage; not general quality acceptance | 49.62699611278172 | 8192 |
| Llama-3.2-1B-Instruct | unusable; numeric and whitespace loops in both checks | Skipped by user rule | 0 |
| Llama-3.2-3B-Instruct | limited readable photosynthesis response; France question instruction echo/off-topic in both checks; not general quality acceptance | 111.9084720821225 | 8192 |

## Evaluation protocol


This is an accuracy diagnostic using retained production baseline binaries, not FP-island, SP2, or rotated variants. Generation uses free greedy feedback (mode2); PPL uses true preceding target tokens (mode1). All model work and histogram collection run on the actual DSP; host aggregates returned per-token negative log probabilities. Head logits are the actual baseline quantized outputs, not pre-quantization floating teacher logits.

WikiText-2 raw test is joined with two newlines and independently tokenized by each original tokenizer, with no chat template or added special tokens. Each model uses its own first64tokens as context then8192targets. Windows overlap in context only; targets are scored exactly once. Each new window resets model KV and scores up to43steps. The final window has22targets. Qwen17 baseline also preserves its frozen prefix KV; prefix storage is separate from reported cache_valid. Thus this is a short-context fixed-subset baseline, not conventional2048context full-WikiText2 PPL and not suitable for direct comparison to published values.

The larger models have limited readable text, not a blanket usability acceptance. First EOS terminates assessed answer; fixed-length harness outputs beyond EOS are retained but ignored for text judgement. Models with obvious degenerate text are skipped by user instruction. Text eligibility does not impose a new PPL pass threshold.

No runtime, weights, activation calibration, residual precision, AV folding policy, or selected baseline was changed. The SP2=8 runtime switch names a shared two-byte backend: deployed INT16 LUT metadata determines uniform INT16 Down semantics.

Dataset: https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1 . Corpus, text, native token IDs and batch fixtures are hashed and retained. Native tokenizer text endpoints differ slightly between models.

Hardware implementation/physical checks pass; this is not a PPL acceptance gate or baseline promotion. No matched floating teacher was run, so no quantization PPL ratio is asserted. Throughput profiling is N/A for this accuracy-only experiment. Both scored models completed191windows and8192unique targets, including22tailtargets.

## Text controls and recoveries
First M64 prompts contained repeated brevity instructions. A single additional pre-frozen control used distinct instructions for the same questions, no weight/prompt search against PPL. Both outputs are retained; smaller-model degeneration and larger-model limited readability remain. See attempts.json for missing remote manifest (payload hashes independently verified), original Llama1 RoPE extent and validator-only vocabulary-count repair. No failed hardware or numerical result was promoted; no first-batch PPL scores were rerun or dropped.

## Provenance
Qwen baselines: EXP0308 INT16, source worktree unchanged51a00a484e7c0209cdc60441c08a5312c3386c9a. Llama1: L320068 INT16; Llama3: L320066 folded INT16; source worktree unchanged272298e3958e65bc9e9d89c0a7c9800fb8aad465. Baseline binary/package hashes verified before execution and again during neutral text controls after PPL; all match retained historical manifests. Native flags naming SP2 refer to shared high/low backend; model LUT metadata is uniformINT16. Qwen retains independentAVRQ, Llama retains its recorded AVfold policy.

## Output samples (first EOS terminates scored text)

### Qwen3-0.6B

Question: What is the capital of France?

```text
You
Answer
TheFrance is a a country. Please answer. 

Answer:

ask
Answer
Answers
Answer
Answer
Okay, so the answer is. 

Answer


Answer
Answer

```

Question: Why do plants need sunlight? Explain briefly.

```text
 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 


```

### Qwen3-1.7B

Question: What is the capital of France?

```text
The capital of France is Paris.
```

Question: Why do plants need sunlight? Explain briefly.

```text
Plants need sunlight to to perform photosynthesis, where they make their own food.
```

### Llama-3.2-1B-Instruct

Question: What is the capital of France?

```text
* 1. 2. 3. 4. 5. 6. 1. 9. 1. 2. 1. 1. 2. 1.
```

Question: Why do plants need sunlight? Explain briefly.

```text
I have been, 5, 2023. 4. 5, ________-4. 1-1, 1, 2, 3, 4, 5, 
```

### Llama-3.2-3B-Instruct

Question: What is the capital of France?

```text
I'm a Frenchman, a citizen of France. Please answer in English.
```

Question: Why do plants need sunlight? Explain briefly.

```text
Plants need sunlight to undergo photosynthesis. Plants need sunlight.
```

Evidence ledger SHA256: fff391d8ed968974496304ef82db058483eab98c97bb5fc92686e796c649ffdf
