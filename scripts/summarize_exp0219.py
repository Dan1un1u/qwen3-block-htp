#!/usr/bin/env python3
"""Report offline rotation A/B/C without modifying frozen EXP-0218 scores."""
import json
import math
from statistics import mean,median
from transformers import AutoTokenizer
import eval_exp0218 as ev
import measure_exp0218 as old
from rotation_exp0219 import RESULT,write_json
from summarize_exp0217 import normalized
from summarize_exp0218 import table

def quality():
    data={s['id']:s for s in ev.dataset()['samples']}
    tok=AutoTokenizer.from_pretrained(ev.MODEL,local_files_only=True)
    historic=json.loads((ev.ROOT/'quality_summary.json').read_text())
    teacher={s['id']:s for s in historic['samples']['BF16 teacher']}
    entries={};summaries={}
    for variant in ['A','B','C']:
        full=json.loads((RESULT/'full'/f'{variant}.validated.json').read_text())
        quick=json.loads((RESULT/'quick'/f'{variant}.validated.json').read_text())
        repeat=json.loads((RESULT/'repeat'/f'{variant}.validated.json').read_text())
        byid={s['id']:s for s in full['samples']}
        mismatch=[]
        for sample in quick['samples']+repeat['samples']:
            if data[sample['id']]['kind']!='nll':continue
            for step,(a,b) in enumerate(zip(sample['steps'],byid[sample['id']]['steps'])):
                fields=[k for k in ['token_id','target_code','nll','rank','max_ties'] if a[k]!=b[k]]
                if fields:mismatch.append(dict(id=sample['id'],step=step,fields=fields))
        rows=[]
        for sample in full['samples']:
            s=data[sample['id']];steps=sample['steps'];r=dict(id=s['id'],kind=s['kind'],language=s['language'])
            if s['kind']=='nll':r.update(nll=[x['nll'] for x in steps],top1=[x['token_id'] for x in steps])
            else:
                ids=[x['token_id'] for x in steps];text=ev.decode(tok,ids);r.update(token_ids=ids,text=text)
                if s['kind']=='task':r['correct']=ev.grade(s,text)
            rows.append(r)
        entries[variant]=rows
        nll=mean(v for r in rows if r['kind']=='nll' for v in r['nll'])
        summaries[variant]=dict(nll=nll,ppl=math.exp(nll),
            delta_nll_teacher=nll-historic['summary']['BF16 teacher']['nll'],
            language_nll={l:mean(v for r in rows if r['kind']=='nll' and r['language']==l for v in r['nll']) for l in ['zh','en']},
            tasks_correct=sum(r['correct'] for r in rows if r['kind']=='task'),tasks_total=24,
            teacher_top1_agreement=mean(v==w for r in rows if r['kind']=='nll' for v,w in zip(r['top1'],teacher[r['id']]['top1'])),
            determinism=dict(repeat_equal=repeat['repeat_equal'],quick_repeat_full_mismatches=mismatch),
            software_diagnostic=None if variant=='A' else json.loads((RESULT/variant/'software_quality.json').read_text()))
    a,c=summaries['A'],summaries['C']
    out=dict(experiment='EXP-0219',dataset_sha256=ev.digest(ev.ROOT/'dataset_v1.json'),
        summary=summaries,samples=entries,
        effectiveness_gate=c['nll']<a['nll'] and c['tasks_correct']>a['tasks_correct'],
        heldout_executed=False,selected_baseline_changed=False)
    write_json(RESULT/'quality_summary.json',out)
    labels={'A':'W4A16 原始','B':'W4A16 仅 gamma 折叠','C':'W4A16 R1/R2'}
    rows=[]
    for name,score in [('BF16 teacher',historic['summary']['BF16 teacher']),('F16A16（冻结参照）',historic['summary']['f16f16'])]+[(labels[v],summaries[v]) for v in ['A','B','C']]:
        rows.append([name,f"{score['nll']:.4f}",f"{score['ppl']:.2f}",f"{score['tasks_correct']}/24",f"{score['teacher_top1_agreement']*100:.2f}%"])
    (RESULT/'quality_table.md').write_text(table(['实现','NLL ↓','条件 PPL ↓','短题','Teacher top-1'],rows))
    return out

def speed():
    history=json.loads((ev.ROOT/'speed_summary.json').read_text());series={};times={}
    for variant in ['A','C']:
        old.ROOT=RESULT/variant
        paths=sorted((RESULT/'formal').glob(f'round_*_{variant}.jsonl'));assert len(paths)==10
        runs=[old.speed_validate(p,'w4f16')[0] for p in paths]
        series[variant]={m:[normalized([p for p in run if p['mode']==m]) for run in runs] for m in ['prefill','decode']}
        p=median(r['host_us'] for r in series[variant]['prefill']);d=median(r['host_us'] for r in series[variant]['decode'])
        times[variant]=dict(prefill_host_us=p,decode_host_us_per_token=d,prefill_tokens_per_second=64e6/p,decode_tokens_per_second=1e6/d)
    out=dict(data=series,times=times,sessions_per_variant=10,invocation_ledgers=320,layer_ledgers=8960,
        paired_speed_percent={m:100*(median(a['host_us']/c['host_us'] for a,c in zip(series['A'][m],series['C'][m]))-1) for m in ['prefill','decode']})
    write_json(RESULT/'speed_summary.json',out)
    cols=[history['data']['f16f16']['prefill'],series['C']['prefill'],history['data']['w4u8']['prefill']]
    walls=[median(p['host_us'] for p in col) for col in cols]
    labels=['I/O、metadata','Input RMSNorm','QKV＋Q/K Norm-RoPE','QK–Softmax–AV','O projection',
        'Post-attention residual＋RMSNorm','Gate/Up＋SwiGLU','Down','Final residual','KV carrier conversion',
        'KV append DMA','Block orchestration','Layer bookkeeping','Stage-boundary bookkeeping','DSP unattributed',
        'Runtime setup/teardown','Embedding','Final model RMSNorm','LM head＋greedy，不含 final norm','Host–DSP 边界','完整 Host wall']
    rows=[]
    for label,(_,fields) in zip(labels,old.OVERVIEW):
        vals=[median(sum(p[f] if f.endswith('_us') else p[f]/19.2 for f in fields) for p in col) for col in cols]
        cells=[f'{v:.1f} ({100*v/w:.2f}%)' for v,w in zip(vals,walls)]
        rows.append([label,*cells,f'{100*(vals[1]/vals[2]-1):+.2f}%' if vals[2] else 'N/A'])
    module=table(['模块','F16A16 冻结 EXP-0218','W4A16 R1/R2 EXP-0219','W4A8 冻结 EXP-0218','W4A8 相对 W4A16 增速'],rows)
    (RESULT/'module_table.md').write_text(module)
    report=['# EXP-0219 full profiling archive','Fixed ABI108 / EXP0218 speed binary; new offline model experiment EXP0219. A/C alternate; each formal call has full 28-layer invocation and layer ledgers. No online rotation. Historical other-recipe columns are nonpaired and quantized weights differ, so no activation-only attribution.',module]
    for variant in ['A','C']:
        for mode,data in series[variant].items():
            report.extend([f'## {variant} {mode}',table(['Field','R1','R10 median'],[[k,f'{data[0][k]:.6f}',f'{median(r[k] for r in data):.6f}'] for k in sorted(data[0])])])
    (RESULT/'full_profiling_report.md').write_text('\n\n'.join(report)+'\n')
    return out

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--quality-only',action='store_true');a=p.parse_args()
    q=quality();print((RESULT/'quality_table.md').read_text())
    if not a.quality_only:print(json.dumps(speed()['times'],indent=2))
