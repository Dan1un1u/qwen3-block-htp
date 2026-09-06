#!/usr/bin/env python3
"""Close the bounded frozen-model PPL test without promoting a baseline."""
import json,subprocess
from pathlib import Path
from data_exp0229 import RESULT,BASE,SOURCE,sha,write,preflight,verified
from measure_exp0229 import frozen,PACKAGES

def main():
    preflight();frozen()
    primary=json.loads((RESULT/'primary_summary.json').read_text())
    combined=primary['reserve_required']
    summary=json.loads((RESULT/('combined_summary.json' if combined else 'primary_summary.json')).read_text())
    for name in ['before_gate.json','after_gate.json','independent_data_audit.json']:
        assert json.loads((RESULT/name).read_text())['pass_all']
    teacher=json.loads((RESULT/'teacher_checks.json').read_text())
    assert teacher['same_batch_repeat_exact'] and teacher['future_token_causal_mask_exact']
    dataset=json.loads((RESULT/'dataset.json').read_text())
    for v,(relative,_,digest) in PACKAGES.items():
        assert sha(Path('/mnt/d/llm_exp/models/qwen3-block-htp')/relative/'manifest.json')==digest
    # Preserve authoritative references for unchanged performance scope.
    prior={}
    for exp in ['exp0218','exp0224','exp0227']:
        p=verified(exp,'full_profiling_report.md')
        prior[exp]=dict(path=str(p),sha256=sha(p))
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
    changed=subprocess.check_output(['git','diff','--name-only','365d1e2f136125462fa4f7ecbe1abc93b4309ca1','HEAD'],cwd=SOURCE,text=True).splitlines()
    assert all(p.startswith('scripts/') for p in changed),changed
    text=['# EXP-0229 independent W4A16 PPL acceptance','',
        f"Status: A0 **{summary['status']['A0']}**, A **{summary['status']['A']}**. "
        f"{summary['documents']} distinct documents; {summary['scored_targets']} conditional target tokens. "
        f"Recommended candidate: {summary['recommended_candidate']}. No baseline promoted.",'',
        'Fixed M64 raw-text contexts plus16 continuation targets. Same actual DSP FP16/W4A16 runtime, '
        'weights and masks. English/Chinese x Wikipedia/news balanced. Additional original BF16 GPU reference is named separately. '
        'This is short-context conditional PPL, not a published full benchmark or long-context acceptance.','',
        '| Stratum | DSP F16 PPL | A0 PPL | A PPL | A0 ratio vs F16 [95% CI] | A ratio vs F16 [95% CI] | Limit |',
        '|---|---:|---:|---:|---|---|---:|']
    for group,g in summary['statistics'].items():
        cells=[]
        for v in ['A0','A']:
            d=g['vs_F16'][v];lo,hi=d['ratio_ci95']
            cells.append(f"{d['ppl_ratio']:.5f} [{lo:.5f}, {hi:.5f}]")
        text.append(f"| {group} | {g['ppl']['F']:.5f} | {g['ppl']['A0']:.5f} | {g['ppl']['A']:.5f} | {cells[0]} | {cells[1]} | {g['vs_F16']['A']['limit']:.2f} |")
    whole=summary['statistics']['overall']
    text += ['',f"Additional original BF16 GPU PPL: {whole['ppl']['BF16_GPU']:.6f}. "
        f"A/A0 PPL ratio: {whole['A_vs_A0']['ppl_ratio']:.6f}, paired95% CI {whole['A_vs_A0']['ratio_ci95']}.",'',
        '## Acceptance interpretation','',
        'Gates were fixed before scores: overall<=1.05; each language/domain/cell<=1.10. '
        '5000 paired bootstrap resamples use distinct documents, stratified across four equal cells. '
        'Point failures are fail; passing points whose upper interval crosses a gate are inconclusive; '
        'pass requires point and upper-bound gates. Reserve is evaluated for all variants only if a primary interval straddles a gate. '
        f"Reserve triggered: {combined}. Short-task scores do not veto PPL acceptance.",
        'A0 is the simpler GPTQ/output-aware scale package; A additionally reconstructs block scales. '
        'Candidate recommendations use only the predeclared rule; no test-driven retraining, clipping search, checkpoint or weight changes occurred. '
        'No new W4A8 work or automatic promotion is included.','',
        '## Evidence and recovery','',
        'Independent corpus/token-offset reconstruction passed1024windows; document, text-hash and32gram separation covers '
        'all prior calibration/training/checkpoint validation and qbh full/holdout. English Wikipedia explicitly excludes all WikiText training-corpus titles. '
        'Original checkpoint, tokenizer, local/device package and runtime hashes verified. '
        'Old qbh numerical controls reproduced; balanced sentinel scores match within-run, before/full/after exactly. '
        'Every measured target has finite score diagnostics and valid8MiB/no-intermediate-DDR/cache progression. '
        'Teacher repeat and causal-mask tests passed; independent CE agrees. See per-step raw JSONL and validated outputs.','',
        'The first dataset attempt exposed a legacy WikiText pseudo-heading parser issue; its snapshot and BF16 diagnostic remain under data_attempt_v1 and never enter acceptance. '
        'A new explicit-document Wikipedia source was frozen without consulting model quality scores. '
        'A symlink-relative deployment checksum path was changed to an absolute path; model hashes did not change. '
        'The control fixture ID was corrected from an unused holdout ID to the existing English full-evaluation ID before any device evaluation. '
        'Primary/reserve test tokens were unaffected by that fixture correction. All original artifacts and recovery notes are retained.','',
        '## Performance scope','',
        'No new speed profiling: evaluation-only inputs changed, runtime and packages are frozen. '
        'Quality-suite elapsed time is not token-generation throughput. Prior full profiling references:']
    for exp,record in prior.items():text.append(f"- {exp}: {record['path']} (SHA256 {record['sha256']})")
    text += ['',f'Source HEAD: {head}. Dataset SHA256: {sha(RESULT/"dataset.json")}.','']
    (RESULT/'REPORT.md').write_text('\n'.join(text))
    (RESULT/'full_profiling_report.md').write_text('# EXP0229 profiling: N/A\n\nQuality-only frozen-model acceptance; no timing claim. Prior full profiling evidence:\n\n'+json.dumps(prior,indent=2)+'\n')
    write('closure.json',dict(experiment='EXP-0229',execution_state='completed',evidence_validity='valid',
        local_gate='pass' if any(s=='pass' for s in summary['status'].values()) else 'fail' if all(s=='fail' for s in summary['status'].values()) else 'not_applicable',
        adoption_status='pending',source_head=head,dataset_sha256=sha(RESULT/'dataset.json'),
        dataset_freeze_sha256=sha(RESULT/'dataset_freeze.json'),summary_sha256=sha(RESULT/('combined_summary.json' if combined else 'primary_summary.json')),
        report_sha256=sha(RESULT/'REPORT.md'),status=summary['status'],recommended_candidate=summary['recommended_candidate'],
        targets=summary['scored_targets'],reserve_used=combined,statistics=summary['statistics'],
        baseline_promoted=False,other_recipes='frozen',runtime_unchanged=True,profiling='N/A_quality_only',prior_profiling=prior))
    write('evidence_sha256.json',{str(p.relative_to(RESULT)):sha(p) for p in sorted(RESULT.rglob('*')) if p.is_file()})
    print('ACCEPTANCE_CLOSED',summary['status'],summary['recommended_candidate'],flush=True)
if __name__=='__main__':main()
