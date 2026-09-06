# EXP-0228: Short-task content versus formatting audit

Same frozen qbh-lite-v1 prompts, outputs and token IDs; no inference, training or timing rerun. Retrospective qbh-content-v2 scoring is separate from immutable strict-v1 evidence. Content-correct answers count as correct despite presentation; strict-v1 remains a formatting/instruction diagnostic.

| Configuration | Strict-v1 | Content-v2 | Formatting-only failures recovered | Incorrect/incomplete | PPL unchanged |
|---|---:|---:|---:|---:|---:|
| BF16 teacher | 22/24 | 23/24 | 1 | 1 | 38.1663 |
| F16A16 | 22/24 | 23/24 | 1 | 1 | 37.8491 |
| W4A16 original RTN | 8/24 | 14/24 | 6 | 10 | 68.8320 |
| W4A16 A0 / EXP0224 | 19/24 | 22/24 | 3 | 2 | 38.8736 |
| W4A16 A / EXP0227 | 18/24 | 22/24 | 4 | 2 | 38.6609 |
| W4A16 R0 / EXP0225 | 20/24 | 22/24 | 2 | 2 | 50.2245 |
| W4A16 R / EXP0227 | 19/24 | 22/24 | 3 | 2 | 43.0176 |
| W4A16 broader-sampling rotation / EXP0226 | 9/24 | 16/24 | 7 | 8 | 40.4446 |
| W4A8 frozen provisional snapshot | 0/24 | 0/24 | 0 | 24 | 94886933.9957 |

## All six former failures of current best-PPL A

| ID | Question | Expected | Actual text | Classification |
|---|---|---|---|---|
| 42 | 小林负责采购，小王负责记录，小陈负责检查。谁负责记录？只回答姓名。 | ['小王'] | 小王负责记录。 | format_only |
| 45 | 一盒有12块饼干，吃掉5块，还剩几块？只回答数字。 | 7 | 12 - 5 = 7 | format_only |
| 46 | 每袋有6个苹果，3袋共有几个苹果？只回答数字。 | 18 | 6 | answer_error |
| 56 | What is 7 plus 8? Answer with a number. | 15 | 7 plus 8 is 15. | format_only |
| 58 | Each bag has 6 apples. How many apples are in 3 bags? Answer with a number. | 18 | The answer is 18. | format_only |
| 59 | Share 20 pens equally among 4 people. How many pens does each person get? Answer with a number. | 5 | 15 | answer_error |

## Interpretation

A has four format-only failures and two actual wrong numeric answers: Chinese 6*3 -> 6, English 20/4 -> 15. F16A16/BF16 have one actual error (Chinese 12-5 -> 5); their correct 6*3=18 equation was a strict failure. A0, A, R0 and R all reach22/24 content accuracy with different remaining errors. Thus the previously reported one-task drop after block reconstruction is not a content-accuracy drop. The former18/24 versus22/24 FP16 gap becomes22/24 versus23/24;24 tasks remain a small diagnostic sample. PPL and independent validation remain unchanged, so this does not establish an incremental rotation benefit.

Existing prompts already repeat answer-only/no-explanation requirements. Improve matching first; any prompt experiment must freeze a new input version and rerun every compared recipe with equal context/generation budgets. Do not mix altered-prompt scores with this audit or increase answer correctness by accepting ambiguous substrings.

## Validation and limits

Authority-linked input hashes, tokenizer hash, EOS-aware token decoding and all 264 stored strict grades verified. Six groups of independent synthetic positive/negative scorer tests pass. All current A/R/control/teacher wrong answers were independently reviewed; legacy unsupported outputs have explicit review in manual_review.json. Content-v2 is a bounded parser, not a general semantic judge. Unsupported outputs remain unresolved until reviewed. JSON key/type/value integrity, case-conversion semantics, arithmetic consistency and ambiguity checks remain required. W4A8 is an unchanged provisional historical snapshot. No weights, prompts, original evidence or selected baselines changed.

Profiling/E2E: N/A; this is a host-only scoring change.

## Per-item audit

### BF16 teacher

| ID | Expected | Actual | Strict | Content | Reason |
|---|---|---|---|---|---|
| 40 | 5 | 5 | True | True | numeric_value |
| 41 | ['周四', '星期四', '礼拜四'] | 周四 | True | True | literal_content_equality |
| 42 | ['小王'] | 小王 | True | True | literal_content_equality |
| 43 | ['李青'] | 李青 | True | True | literal_content_equality |
| 44 | 15 | 15 | True | True | numeric_value |
| 45 | 7 | 5 | False | False | wrong_numeric_answer |
| 46 | 18 | 6×3=18 | False | True | verified_equation |
| 47 | 5 | 5 | True | True | numeric_value |
| 48 | ['苹果'] | 苹果 | True | True | literal_content_equality |
| 49 | {'ok': True} | {"ok": true} | True | True | typed_JSON_equality |
| 50 | ['CAT'] | CAT | True | True | literal_content_equality |
| 51 | {'n': 3} | {"n": 3} | True | True | typed_JSON_equality |
| 52 | 5 | 5 | True | True | numeric_value |
| 53 | ['Thursday'] | Thursday | True | True | literal_content_equality |
| 54 | ['Ben'] | Ben | True | True | literal_content_equality |
| 55 | ['Nora'] | Nora | True | True | literal_content_equality |
| 56 | 15 | 15 | True | True | numeric_value |
| 57 | 7 | 7 | True | True | numeric_value |
| 58 | 18 | 18 | True | True | numeric_value |
| 59 | 5 | 5 | True | True | numeric_value |
| 60 | ['apple'] | apple | True | True | literal_content_equality |
| 61 | {'ok': True} | {"ok": true} | True | True | typed_JSON_equality |
| 62 | ['CAT'] | CAT | True | True | literal_content_equality |
| 63 | {'n': 3} | {"n": 3} | True | True | typed_JSON_equality |

### F16A16

| ID | Expected | Actual | Strict | Content | Reason |
|---|---|---|---|---|---|
| 40 | 5 | 5 | True | True | numeric_value |
| 41 | ['周四', '星期四', '礼拜四'] | 周四 | True | True | literal_content_equality |
| 42 | ['小王'] | 小王 | True | True | literal_content_equality |
| 43 | ['李青'] | 李青 | True | True | literal_content_equality |
| 44 | 15 | 15 | True | True | numeric_value |
| 45 | 7 | 5 | False | False | wrong_numeric_answer |
| 46 | 18 | 6×3=18 | False | True | verified_equation |
| 47 | 5 | 5 | True | True | numeric_value |
| 48 | ['苹果'] | 苹果 | True | True | literal_content_equality |
| 49 | {'ok': True} | {"ok": true} | True | True | typed_JSON_equality |
| 50 | ['CAT'] | CAT | True | True | literal_content_equality |
| 51 | {'n': 3} | {"n": 3} | True | True | typed_JSON_equality |
| 52 | 5 | 5 | True | True | numeric_value |
| 53 | ['Thursday'] | Thursday | True | True | literal_content_equality |
| 54 | ['Ben'] | Ben | True | True | literal_content_equality |
| 55 | ['Nora'] | Nora | True | True | literal_content_equality |
| 56 | 15 | 15 | True | True | numeric_value |
| 57 | 7 | 7 | True | True | numeric_value |
| 58 | 18 | 18 | True | True | numeric_value |
| 59 | 5 | 5 | True | True | numeric_value |
| 60 | ['apple'] | apple | True | True | literal_content_equality |
| 61 | {'ok': True} | {"ok": true} | True | True | typed_JSON_equality |
| 62 | ['CAT'] | CAT | True | True | literal_content_equality |
| 63 | {'n': 3} | {"n": 3} | True | True | typed_JSON_equality |

### W4A16 original RTN

| ID | Expected | Actual | Strict | Content | Reason |
|---|---|---|---|---|---|
| 40 | 5 | 2 | False | False | wrong_numeric_answer |
| 41 | ['周四', '星期四', '礼拜四'] | Tuesday | False | False | wrong_literal_answer |
| 42 | ['小王'] | 答案：小王 | False | True | literal_content_equality |
| 43 | ['李青'] | 李青 | True | True | literal_content_equality |
| 44 | 15 | 5<br><br>**注意**：根据题目要求，仅输出结果，不添加 | False | False | Leading answer 5 is wrong for 7+8; explanation is incomplete. |
| 45 | 7 | 5 | False | False | wrong_numeric_answer |
| 46 | 18 | 每袋有6个苹果，3袋的话，就是6个苹果一 | False | False | Truncated restatement; no complete answer 18 is asserted. |
| 47 | 5 | 3 | False | False | wrong_numeric_answer |
| 48 | ['苹果'] | 苹果 | True | True | literal_content_equality |
| 49 | {'ok': True} | ```json<br>{"ok": true}<br>``` | False | True | typed_JSON_equality |
| 50 | ['CAT'] | **cat**  <br>→ **CAT** | False | True | explicit_answer_statement |
| 51 | {'n': 3} | ```json<br>{"n": 3}<br>``` | False | True | typed_JSON_equality |
| 52 | 5 | 3. | False | False | wrong_numeric_answer |
| 53 | ['Thursday'] | Thursday | True | True | literal_content_equality |
| 54 | ['Ben'] | Ben takes notes. | False | True | explicit_answer_statement |
| 55 | ['Nora'] | Nora | True | True | literal_content_equality |
| 56 | 15 | 15 | True | True | numeric_value |
| 57 | 7 | A box has 12 biscuits. You eat 5. How many remain | False | False | Only repeats the question and truncates; no answer 7. |
| 58 | 18 | 7<br><br>(Each bag has 6 apples, and if there are 3 | False | False | Leading answer 7 is wrong for 6*3; explanation is incomplete. |
| 59 | 5 | 10 | False | False | wrong_numeric_answer |
| 60 | ['apple'] | apple | True | True | literal_content_equality |
| 61 | {'ok': True} | {"ok": true} | True | True | typed_JSON_equality |
| 62 | ['CAT'] | the word cat converted to uppercase is "CAT" | False | True | explicit_answer_statement |
| 63 | {'n': 3} | {"n": 3} | True | True | typed_JSON_equality |

### W4A16 A0 / EXP0224

| ID | Expected | Actual | Strict | Content | Reason |
|---|---|---|---|---|---|
| 40 | 5 | 5 | True | True | numeric_value |
| 41 | ['周四', '星期四', '礼拜四'] | 周四 | True | True | literal_content_equality |
| 42 | ['小王'] | 小王 | True | True | literal_content_equality |
| 43 | ['李青'] | 李青 | True | True | literal_content_equality |
| 44 | 15 | 7+8=15 | False | True | verified_equation |
| 45 | 7 | 12 - 5 = 7 | False | True | verified_equation |
| 46 | 18 | 6 | False | False | wrong_numeric_answer |
| 47 | 5 | 5 | True | True | numeric_value |
| 48 | ['苹果'] | 苹果 | True | True | literal_content_equality |
| 49 | {'ok': True} | {"ok":true} | True | True | typed_JSON_equality |
| 50 | ['CAT'] | CAT | True | True | literal_content_equality |
| 51 | {'n': 3} | {"n":3} | True | True | typed_JSON_equality |
| 52 | 5 | 5 | True | True | numeric_value |
| 53 | ['Thursday'] | Thursday | True | True | literal_content_equality |
| 54 | ['Ben'] | Ben takes notes. | False | True | explicit_answer_statement |
| 55 | ['Nora'] | Nora | True | True | literal_content_equality |
| 56 | 15 | 15 | True | True | numeric_value |
| 57 | 7 | 5 | False | False | wrong_numeric_answer |
| 58 | 18 | 18 | True | True | numeric_value |
| 59 | 5 | 5 | True | True | numeric_value |
| 60 | ['apple'] | apple | True | True | literal_content_equality |
| 61 | {'ok': True} | {"ok": true} | True | True | typed_JSON_equality |
| 62 | ['CAT'] | CAT | True | True | literal_content_equality |
| 63 | {'n': 3} | {"n":3} | True | True | typed_JSON_equality |

### W4A16 A / EXP0227

| ID | Expected | Actual | Strict | Content | Reason |
|---|---|---|---|---|---|
| 40 | 5 | 5 | True | True | numeric_value |
| 41 | ['周四', '星期四', '礼拜四'] | 周四 | True | True | literal_content_equality |
| 42 | ['小王'] | 小王负责记录。 | False | True | explicit_answer_statement |
| 43 | ['李青'] | 李青 | True | True | literal_content_equality |
| 44 | 15 | 15 | True | True | numeric_value |
| 45 | 7 | 12 - 5 = 7 | False | True | verified_equation |
| 46 | 18 | 6 | False | False | wrong_numeric_answer |
| 47 | 5 | 5 | True | True | numeric_value |
| 48 | ['苹果'] | 苹果 | True | True | literal_content_equality |
| 49 | {'ok': True} | {"ok": true} | True | True | typed_JSON_equality |
| 50 | ['CAT'] | CAT | True | True | literal_content_equality |
| 51 | {'n': 3} | {"n":3} | True | True | typed_JSON_equality |
| 52 | 5 | 5 | True | True | numeric_value |
| 53 | ['Thursday'] | Thursday | True | True | literal_content_equality |
| 54 | ['Ben'] | Ben | True | True | literal_content_equality |
| 55 | ['Nora'] | Nora | True | True | literal_content_equality |
| 56 | 15 | 7 plus 8 is 15. | False | True | verified_equation |
| 57 | 7 | 7 | True | True | numeric_value |
| 58 | 18 | The answer is 18. | False | True | numeric_value |
| 59 | 5 | 15 | False | False | wrong_numeric_answer |
| 60 | ['apple'] | apple | True | True | literal_content_equality |
| 61 | {'ok': True} | {"ok": true} | True | True | typed_JSON_equality |
| 62 | ['CAT'] | CAT | True | True | literal_content_equality |
| 63 | {'n': 3} | {"n":3} | True | True | typed_JSON_equality |

### W4A16 R0 / EXP0225

| ID | Expected | Actual | Strict | Content | Reason |
|---|---|---|---|---|---|
| 40 | 5 | 2 | False | False | wrong_numeric_answer |
| 41 | ['周四', '星期四', '礼拜四'] | 周四 | True | True | literal_content_equality |
| 42 | ['小王'] | 小王 | True | True | literal_content_equality |
| 43 | ['李青'] | 李青 | True | True | literal_content_equality |
| 44 | 15 | 15 | True | True | numeric_value |
| 45 | 7 | 7 | True | True | numeric_value |
| 46 | 18 | 24 | False | False | wrong_numeric_answer |
| 47 | 5 | 5 | True | True | numeric_value |
| 48 | ['苹果'] | 苹果 | True | True | literal_content_equality |
| 49 | {'ok': True} | ```json<br>{"ok": true}<br>``` | False | True | typed_JSON_equality |
| 50 | ['CAT'] | CAT | True | True | literal_content_equality |
| 51 | {'n': 3} | ```json<br>{"n": 3}<br>``` | False | True | typed_JSON_equality |
| 52 | 5 | 5 | True | True | numeric_value |
| 53 | ['Thursday'] | Thursday | True | True | literal_content_equality |
| 54 | ['Ben'] | Ben | True | True | literal_content_equality |
| 55 | ['Nora'] | Nora | True | True | literal_content_equality |
| 56 | 15 | 15 | True | True | numeric_value |
| 57 | 7 | 7 | True | True | numeric_value |
| 58 | 18 | 18 | True | True | numeric_value |
| 59 | 5 | 5 | True | True | numeric_value |
| 60 | ['apple'] | apple | True | True | literal_content_equality |
| 61 | {'ok': True} | {"ok": true} | True | True | typed_JSON_equality |
| 62 | ['CAT'] | CAT | True | True | literal_content_equality |
| 63 | {'n': 3} | {"n": 3} | True | True | typed_JSON_equality |

### W4A16 R / EXP0227

| ID | Expected | Actual | Strict | Content | Reason |
|---|---|---|---|---|---|
| 40 | 5 | 2 | False | False | wrong_numeric_answer |
| 41 | ['周四', '星期四', '礼拜四'] | 周四 | True | True | literal_content_equality |
| 42 | ['小王'] | 小王 | True | True | literal_content_equality |
| 43 | ['李青'] | 李青 | True | True | literal_content_equality |
| 44 | 15 | 15 | True | True | numeric_value |
| 45 | 7 | 5 | False | False | wrong_numeric_answer |
| 46 | 18 | 18 | True | True | numeric_value |
| 47 | 5 | 5 | True | True | numeric_value |
| 48 | ['苹果'] | 苹果 | True | True | literal_content_equality |
| 49 | {'ok': True} | ```json<br>{"ok": true}<br>``` | False | True | typed_JSON_equality |
| 50 | ['CAT'] | CAT | True | True | literal_content_equality |
| 51 | {'n': 3} | ```json<br>{"n": 3}<br>``` | False | True | typed_JSON_equality |
| 52 | 5 | 5 | True | True | numeric_value |
| 53 | ['Thursday'] | Thursday. | False | True | literal_content_equality |
| 54 | ['Ben'] | Ben | True | True | literal_content_equality |
| 55 | ['Nora'] | Nora | True | True | literal_content_equality |
| 56 | 15 | 15 | True | True | numeric_value |
| 57 | 7 | 7 | True | True | numeric_value |
| 58 | 18 | 18 | True | True | numeric_value |
| 59 | 5 | 5 | True | True | numeric_value |
| 60 | ['apple'] | apple | True | True | literal_content_equality |
| 61 | {'ok': True} | {"ok": true} | True | True | typed_JSON_equality |
| 62 | ['CAT'] | CAT | True | True | literal_content_equality |
| 63 | {'n': 3} | {"n": 3} | True | True | typed_JSON_equality |

### W4A16 broader-sampling rotation / EXP0226

| ID | Expected | Actual | Strict | Content | Reason |
|---|---|---|---|---|---|
| 40 | 5 | 3 | False | False | wrong_numeric_answer |
| 41 | ['周四', '星期四', '礼拜四'] | 根据题目信息，会议最终在周四举行。 | False | True | explicit_answer_statement |
| 42 | ['小王'] | 小王 | True | True | literal_content_equality |
| 43 | ['李青'] | 李青 | True | True | literal_content_equality |
| 44 | 15 | 15 | True | True | numeric_value |
| 45 | 7 | 12 - 5 = 7 | False | True | verified_equation |
| 46 | 18 | 6<br>5<br>4<br>3<br>2<br>1 | False | False | Countdown 6,5,4,3,2,1 has no answer 18. |
| 47 | 5 | 15 | False | False | wrong_numeric_answer |
| 48 | ['苹果'] | 苹果  <br>苹果 | False | True | literal_content_equality |
| 49 | {'ok': True} | ```json<br>{<br>  "ok": true<br>}<br>``` | False | True | typed_JSON_equality |
| 50 | ['CAT'] | CAT | True | True | literal_content_equality |
| 51 | {'n': 3} | ```json<br>{<br>  "n": 3<br>}<br>``` | False | True | typed_JSON_equality |
| 52 | 5 | 4<br><br>Explanation: The blue box has 5 pens. | False | False | Leading answer 4 contradicts subsequent blue-box count 5. |
| 53 | ['Thursday'] | Thursday | True | True | literal_content_equality |
| 54 | ['Ben'] | Ben takes notes. | False | True | explicit_answer_statement |
| 55 | ['Nora'] | Nora | True | True | literal_content_equality |
| 56 | 15 | 15 | True | True | numeric_value |
| 57 | 7 | 12<br><br>The box has 12 biscuits. You eat 5. | False | False | Leading answer 12 gives initial count, not remaining 7. |
| 58 | 18 | 6<br><br>3 bags × 6 apples = 18 apples<br><br>The answer | False | False | Leading answer 6 contradicts 3*6=18; final answer is truncated. |
| 59 | 5 | 12 | False | False | wrong_numeric_answer |
| 60 | ['apple'] | apple | True | True | literal_content_equality |
| 61 | {'ok': True} | ```json<br>{<br>  "ok": true<br>}<br>``` | False | True | typed_JSON_equality |
| 62 | ['CAT'] | CATE<br></think><br><br>CATE | False | False | CATE is not uppercase CAT; repeated wrong text and stray think tag. |
| 63 | {'n': 3} | {<br>  "n": 3<br>} | True | True | typed_JSON_equality |

### W4A8 frozen provisional snapshot

| ID | Expected | Actual | Strict | Content | Reason |
|---|---|---|---|---|---|
| 40 | 5 | ảoordoordoalysisordoытаAPHcdbAPHAPHAPHAPH副主任��APHlang | False | False | Unrelated garbled text with no requested answer; frozen W4U8 snapshot remains provisional. |
| 41 | ['周四', '星期四', '礼拜四'] | ảoordoordoalysisordoытаAPHcdbAPHAPHAPHAPH副主任��APHlang | False | False | Unrelated garbled text with no requested answer; frozen W4U8 snapshot remains provisional. |
| 42 | ['小王'] | ảoordoordoalysisordoытаAPHcdbAPHAPHAPHAPH副主任��APHlang | False | False | Unrelated garbled text with no requested answer; frozen W4U8 snapshot remains provisional. |
| 43 | ['李青'] | ảoordoordoalysisordoытаAPHcdbAPHAPHAPHAPH副主任��APHlang | False | False | Unrelated garbled text with no requested answer; frozen W4U8 snapshot remains provisional. |
| 44 | 15 | ảoordoordoalysisordoытаAPHcdbAPHAPHAPHAPH副主任��APHlang | False | False | Unrelated garbled text with no requested answer; frozen W4U8 snapshot remains provisional. |
| 45 | 7 | ảoordoordoalysisordoытаAPHcdbAPHAPHAPHAPH副主任��APHlang | False | False | Unrelated garbled text with no requested answer; frozen W4U8 snapshot remains provisional. |
| 46 | 18 | ảoordoordoalysisordoытаAPHcdbAPHAPHAPHAPH副主任��APHlang | False | False | Unrelated garbled text with no requested answer; frozen W4U8 snapshot remains provisional. |
| 47 | 5 | ảoordoordoalysisordoытаAPHcdbAPHAPHAPHAPH副主任��APHlang | False | False | Unrelated garbled text with no requested answer; frozen W4U8 snapshot remains provisional. |
| 48 | ['苹果'] | ảoordoordoalysisordoытаAPHcdbAPHAPHAPHAPH副主任��APHlang | False | False | Unrelated garbled text with no requested answer; frozen W4U8 snapshot remains provisional. |
| 49 | {'ok': True} | ảoordoordoalysisordoытаAPHcdbAPHAPHAPHAPH副主任��APHlang | False | False | Unrelated garbled text with no requested answer; frozen W4U8 snapshot remains provisional. |
| 50 | ['CAT'] | ảoordoordoalysisordoытаAPHcdbAPHAPHAPHAPH副主任��APHlang | False | False | Unrelated garbled text with no requested answer; frozen W4U8 snapshot remains provisional. |
| 51 | {'n': 3} | ảoordoordoalysisordoытаAPHcdbAPHAPHAPHAPH副主任��APHlang | False | False | Unrelated garbled text with no requested answer; frozen W4U8 snapshot remains provisional. |
| 52 | 5 | ảoordoordoalysisordoытаAPHcdbAPHAPHAPHAPH副主任��APHlang | False | False | Unrelated garbled text with no requested answer; frozen W4U8 snapshot remains provisional. |
| 53 | ['Thursday'] | ảoordoordoalysisordoытаAPHcdbAPHAPHAPHAPH副主任��APHlang | False | False | Unrelated garbled text with no requested answer; frozen W4U8 snapshot remains provisional. |
| 54 | ['Ben'] | ảoordoordoalysisordoытаAPHcdbAPHAPHAPHAPH副主任��APHlang | False | False | Unrelated garbled text with no requested answer; frozen W4U8 snapshot remains provisional. |
| 55 | ['Nora'] | ảoordoordoalysisordoытаAPHcdbAPHAPHAPHAPH副主任��APHlang | False | False | Unrelated garbled text with no requested answer; frozen W4U8 snapshot remains provisional. |
| 56 | 15 | ảoordoordoalysisordoытаAPHcdbAPHAPHAPHAPH副主任��APHlang | False | False | Unrelated garbled text with no requested answer; frozen W4U8 snapshot remains provisional. |
| 57 | 7 | ảoordoordoalysisordoытаAPHcdbAPHAPHAPHAPH副主任��APHlang | False | False | Unrelated garbled text with no requested answer; frozen W4U8 snapshot remains provisional. |
| 58 | 18 | ảoordoordoalysisordoытаAPHcdbAPHAPHAPHAPH副主任��APHlang | False | False | Unrelated garbled text with no requested answer; frozen W4U8 snapshot remains provisional. |
| 59 | 5 | ảoordoordoalysisordoытаAPHcdbAPHAPHAPHAPH副主任��APHlang | False | False | Unrelated garbled text with no requested answer; frozen W4U8 snapshot remains provisional. |
| 60 | ['apple'] | ảoordoordoalysisordoытаAPHcdbAPHAPHAPHAPH副主任��APHlang | False | False | Unrelated garbled text with no requested answer; frozen W4U8 snapshot remains provisional. |
| 61 | {'ok': True} | ảoordoordoalysisordoытаAPHcdbAPHAPHAPHAPH副主任��APHlang | False | False | Unrelated garbled text with no requested answer; frozen W4U8 snapshot remains provisional. |
| 62 | ['CAT'] | ảoordoordoalysisordoытаAPHcdbAPHAPHAPHAPH副主任��APHlang | False | False | Unrelated garbled text with no requested answer; frozen W4U8 snapshot remains provisional. |
| 63 | {'n': 3} | ảoordoordoalysisordoытаAPHcdbAPHAPHAPHAPH副主任��APHlang | False | False | Unrelated garbled text with no requested answer; frozen W4U8 snapshot remains provisional. |

