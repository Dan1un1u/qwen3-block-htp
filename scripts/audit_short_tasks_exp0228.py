#!/usr/bin/env python3
"""Re-score immutable EXP218/226/227 short-task evidence, with explicit review."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import yaml
from tokenizers import Tokenizer
from score_short_tasks_v2 import rescore, sha, VERSION

SOURCE = Path(__file__).resolve().parents[1]
MEMORY = Path('/home/daniuniu/work/qwen3-block-htp-project-memory')
RESULTS = Path('/mnt/d/llm_exp/results/qwen3-block-htp')
OUT = RESULTS / 'exp0228'
EXPERIMENTS = ['exp0218', 'exp0226', 'exp0227']


def write(name, value):
    with (OUT/name).open('x', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    subprocess.run(['python3', str(MEMORY/'scripts/project_memory.py'), 'preflight',
                    '--source-worktree', str(SOURCE)], check=True)
    index = yaml.safe_load((MEMORY/'experiments/index.yaml').read_text())
    records = {e['id']: e for e in index['experiments']}
    assert records['EXP-0228']['execution_state'] == 'running'
    assert not OUT.exists(), 'Refusing to overwrite EXP0228 evidence'
    provenance = {}
    snapshots = {}
    for exp in EXPERIMENTS:
        evidence = records['EXP-'+exp[-4:]]['evidence']
        expected = evidence.get('evidence_ledger_sha256', evidence.get('evidence_sha256_ledger_sha256'))
        ledger_path = RESULTS/exp/'evidence_sha256.json'
        assert sha(ledger_path) == expected, (exp, 'ledger hash')
        ledger = json.loads(ledger_path.read_text())
        # Every input used in this audit is verified against its published ledger.
        names = ['quality_summary.json']
        if exp == 'exp0218': names += ['dataset_v1.json', 'dataset_freeze.json', 'teacher_bf16.json']
        hashes = {}
        for name in names:
            path = RESULTS/exp/name
            assert sha(path) == ledger[name], (exp, name)
            hashes[name] = sha(path)
        provenance[exp] = dict(ledger_sha256=expected, files=hashes)
        snapshots[exp] = json.loads((RESULTS/exp/'quality_summary.json').read_text())
    dataset = json.loads((RESULTS/'exp0218/dataset_v1.json').read_text())
    token_path = Path('/mnt/d/llm_exp/models/Qwen3-origin/qwen3-tokenizer.json')
    assert sha(token_path) == dataset['tokenizer_sha256']
    tokenizer = Tokenizer.from_file(str(token_path))
    outputs = {}
    decoded = 0
    for exp in EXPERIMENTS:
        snapshot = snapshots[exp]
        for variant, rows in snapshot['samples'].items():
            for row in rows:
                if row['kind'] != 'task': continue
                tokens = row['token_ids']
                if dataset['eos_token_id'] in tokens:
                    tokens = tokens[:tokens.index(dataset['eos_token_id'])]
                assert tokenizer.decode(tokens, skip_special_tokens=False) == row['text'], (exp, variant, row['id'])
                decoded += 1
        outputs[exp] = rescore(dataset, snapshot)
    # Explicit retrospective review only for unsupported historical outputs.
    # These decisions do not enter the reusable parser or allow answers by ID.
    manual = {
        ('exp0218','w4f16',44): 'Leading answer 5 is wrong for 7+8; explanation is incomplete.',
        ('exp0218','w4f16',46): 'Truncated restatement; no complete answer 18 is asserted.',
        ('exp0218','w4f16',57): 'Only repeats the question and truncates; no answer 7.',
        ('exp0218','w4f16',58): 'Leading answer 7 is wrong for 6*3; explanation is incomplete.',
        ('exp0226','selected_step100',46): 'Countdown 6,5,4,3,2,1 has no answer 18.',
        ('exp0226','selected_step100',52): 'Leading answer 4 contradicts subsequent blue-box count 5.',
        ('exp0226','selected_step100',57): 'Leading answer 12 gives initial count, not remaining 7.',
        ('exp0226','selected_step100',58): 'Leading answer 6 contradicts 3*6=18; final answer is truncated.',
        ('exp0226','selected_step100',62): 'CATE is not uppercase CAT; repeated wrong text and stray think tag.',
    }
    reviewed = []
    for exp, output in outputs.items():
        for variant, value in output['variants'].items():
            for row in value['samples']:
                if row['content_correct'] is not None: continue
                key = (exp, variant, row['id'])
                if exp == 'exp0218' and variant == 'w4u8':
                    reason = 'Unrelated garbled text with no requested answer; frozen W4U8 snapshot remains provisional.'
                else:
                    assert key in manual, ('requires explicit review', key, row['text'])
                    reason = manual.pop(key)
                row['manual_review'] = dict(content_correct=False, reason=reason,
                    reviewer='Codex explicit per-output content review',
                    text_sha256=hashlib.sha256(row['text'].encode()).hexdigest())
                reviewed.append(dict(experiment=exp, variant=variant, id=row['id'],
                                     text=row['text'], expected=row['expected'], **row['manual_review']))
            value['reviewed_content_correct'] = sum(
                r.get('manual_review', {}).get('content_correct', r['content_correct']) is True
                for r in value['samples'])
            value['manual_review_count'] = sum('manual_review' in r for r in value['samples'])
            value['reviewed_incorrect'] = value['total']-value['reviewed_content_correct']
    assert not manual, ('unused review entry', manual)
    # Independent manually enumerated current/floating error sets.
    expected_wrong = {('exp0218','BF16 teacher'):{45}, ('exp0218','f16f16'):{45},
        ('exp0227','control_A'):{46,57}, ('exp0227','control_R'):{40,46},
        ('exp0227','selected_A'):{46,59}, ('exp0227','selected_R'):{40,45}}
    for (exp, variant), ids in expected_wrong.items():
        rows = outputs[exp]['variants'][variant]['samples']
        assert {r['id'] for r in rows if r['content_correct'] is not True} == ids
        assert all(r['content_correct'] is not None for r in rows)
    tests = subprocess.run([sys.executable, str(SOURCE/'scripts/test_score_short_tasks_v2.py')],
                           text=True, capture_output=True)
    assert tests.returncode == 0, tests.stdout+tests.stderr
    # Recheck inputs after work to prove no historical output/PPL/token changes.
    for exp, info in provenance.items():
        for name, digest in info['files'].items():
            assert sha(RESULTS/exp/name) == digest
    OUT.mkdir()
    (OUT/'tests.log').write_text(tests.stdout+tests.stderr)
    write('input_verification.json', dict(files=provenance, tokenizer_sha256=sha(token_path),
         decoded_task_rows=decoded, strict_grades_reproduced=decoded, original_inputs_unchanged=True))
    for exp, output in outputs.items(): write(exp+'_rescored.json', output)
    write('manual_review.json', reviewed)
    head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=SOURCE,text=True).strip()
    write('protocol.json', dict(experiment='EXP-0228', scorer_version=VERSION,
        source_head=head, script_sha256={p.name:sha(p) for p in [Path(__file__),
          SOURCE/'scripts/score_short_tasks_v2.py',SOURCE/'scripts/test_score_short_tasks_v2.py',
          SOURCE/'scripts/eval_exp0218.py']},
        scope='retrospective_rescore_same_frozen_prompts_and_generated_tokens',
        content_metric='bounded_parsing_with_explicit_manual_review_for_unsupported_output',
        strict_metric='unchanged_strict_v1', new_inference=False, new_profiling=False,
        original_PPL_and_speed_unchanged=True, baseline_promotion=False))
    labels = [
      ('exp0218','BF16 teacher','BF16 teacher'), ('exp0218','f16f16','F16A16'),
      ('exp0218','w4f16','W4A16 original RTN'), ('exp0227','control_A','W4A16 A0 / EXP0224'),
      ('exp0227','selected_A','W4A16 A / EXP0227'), ('exp0227','control_R','W4A16 R0 / EXP0225'),
      ('exp0227','selected_R','W4A16 R / EXP0227'),
      ('exp0226','selected_step100','W4A16 broader-sampling rotation / EXP0226'),
      ('exp0218','w4u8','W4A8 frozen provisional snapshot')]
    report = ['# EXP-0228: Short-task content versus formatting audit', '',
      'Same frozen qbh-lite-v1 prompts, outputs and token IDs; no inference, training or timing rerun. '
      'Retrospective qbh-content-v2 scoring is separate from immutable strict-v1 evidence. '
      'Content-correct answers count as correct despite presentation; strict-v1 remains a formatting/instruction diagnostic.', '',
      '| Configuration | Strict-v1 | Content-v2 | Formatting-only failures recovered | Incorrect/incomplete | PPL unchanged |',
      '|---|---:|---:|---:|---:|---:|']
    totals = {}
    for exp,v,label in labels:
        value = outputs[exp]['variants'][v]
        n = value['reviewed_content_correct']; strict=value['strict_correct']
        ppl = snapshots[exp]['summary'][v]['ppl']
        report.append(f'| {label} | {strict}/24 | {n}/24 | {n-strict} | {24-n} | {ppl:.4f} |')
        totals[label] = dict(strict=strict,content=n,total=24,format_only=n-strict,incorrect=24-n,ppl=ppl)
    report += ['', '## All six former failures of current best-PPL A', '',
       '| ID | Question | Expected | Actual text | Classification |', '|---|---|---|---|---|']
    for row in outputs['exp0227']['variants']['selected_A']['samples']:
        if not row['strict_correct']:
            prompt = row['prompt'].split(' 请')[0].split(' Read the question')[0]
            report.append(f"| {row['id']} | {prompt} | {row['expected']} | {row['text']} | {row['classification']} |")
    report += ['', '## Interpretation', '',
      'A has four format-only failures and two actual wrong numeric answers: Chinese 6*3 -> 6, English 20/4 -> 15. '
      'F16A16/BF16 have one actual error (Chinese 12-5 -> 5); their correct 6*3=18 equation was a strict failure. '
      'A0, A, R0 and R all reach22/24 content accuracy with different remaining errors. '
      'Thus the previously reported one-task drop after block reconstruction is not a content-accuracy drop. '
      'The former18/24 versus22/24 FP16 gap becomes22/24 versus23/24;24 tasks remain a small diagnostic sample. '
      'PPL and independent validation remain unchanged, so this does not establish an incremental rotation benefit.', '',
      'Existing prompts already repeat answer-only/no-explanation requirements. Improve matching first; '
      'any prompt experiment must freeze a new input version and rerun every compared recipe with equal context/generation budgets. '
      'Do not mix altered-prompt scores with this audit or increase answer correctness by accepting ambiguous substrings.', '',
      '## Validation and limits', '',
      f'Authority-linked input hashes, tokenizer hash, EOS-aware token decoding and all {decoded} stored strict grades verified. '
      'Six groups of independent synthetic positive/negative scorer tests pass. '
      'All current A/R/control/teacher wrong answers were independently reviewed; legacy unsupported outputs have explicit review in manual_review.json. '
      'Content-v2 is a bounded parser, not a general semantic judge. Unsupported outputs remain unresolved until reviewed. '
      'JSON key/type/value integrity, case-conversion semantics, arithmetic consistency and ambiguity checks remain required. '
      'W4A8 is an unchanged provisional historical snapshot. No weights, prompts, original evidence or selected baselines changed.', '',
      'Profiling/E2E: N/A; this is a host-only scoring change.', '', '## Per-item audit', '']
    for exp,v,label in labels:
        report += ['### '+label, '', '| ID | Expected | Actual | Strict | Content | Reason |', '|---|---|---|---|---|---|']
        for row in outputs[exp]['variants'][v]['samples']:
            actual=row['text'].replace('\n','<br>').replace('|','\\|')
            correct=row.get('manual_review',{}).get('content_correct',row['content_correct'])
            reason=row.get('manual_review',{}).get('reason',row['reason'])
            report.append(f"| {row['id']} | {row['expected']} | {actual} | {row['strict_correct']} | {correct} | {reason} |")
        report.append('')
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
    write('summary.json', dict(experiment='EXP-0228', scorer_version=VERSION, results=totals,
        decoded_task_rows=decoded, manual_review_count=len(reviewed), test_groups=6,
        source_head=head, historical_evidence_unchanged=True, PPL_unchanged=True,
        profiling='not_applicable', selected_baseline_changed=False))
    write('evidence_sha256.json', {str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()})
    print(json.dumps(totals,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
