# Short-task scoring: qbh-content-v2

User-approved EXP0228 separates answer content from strict instruction/format adherence. Primary short-task quality is content-v2; always retain strict-v1 alongside it. Original qbh-lite-v1 prompts, token IDs, strict scores, PPL and quality summaries remain immutable. This is a retrospective scoring-version change, not a model improvement or baseline promotion.

For future quality_summary.json outputs use:

```sh
python3 scripts/score_short_tasks_v2.py \
  --dataset /mnt/d/llm_exp/results/qwen3-block-htp/exp0218/dataset_v1.json \
  --snapshot /path/to/new/quality_summary.json \
  --output /path/to/new/quality_content_v2.json
```

The output is created exclusively; an existing result cannot be overwritten. The API is grade_content(sample, text); rescore(dataset, snapshot) checks complete unique task IDs and reproduces every stored strict grade. Unknown prose is content_correct=null/unresolved. Do not treat that as a demonstrated logic error, or silently accept it; inspect and retain an explicit review tied to the text hash. Current and historical comparable snapshots use identical rules.

Rules accept balanced outer presentation, terminal punctuation, explicit bounded answer statements, correct numeric equations and JSON code fences. A numeric answer must be complete and unambiguous; equations must evaluate correctly through restricted arithmetic. No arbitrary code evaluation or answer-substring match. JSON keeps exact keys, types and values and rejects duplicate keys. Case remains significant for case-conversion tasks. Correct repeated identical literals are content-correct; alternatives, contradictions, missing answers and incomplete prose require review. The parser supports a deliberately bounded family of qbh question/answer grammars; it does not claim full natural-language equivalence.

Tests: python3 scripts/test_score_short_tasks_v2.py. The experimental audit runner is scripts/audit_short_tasks_exp0228.py; it verifies prior authoritative evidence hashes and token decoding before scoring, writes separate evidence, and refuses reruns over existing results.

Prompt strengthening is a different input experiment: freeze a new version and compare all recipes with equal token/context/generation budgets. Existing qbh prompts already request answer-only/no explanation; never silently edit them or compare changed prompts to old outputs. EXP0228 changes only scoring and has no new profiling/E2E measurement.
