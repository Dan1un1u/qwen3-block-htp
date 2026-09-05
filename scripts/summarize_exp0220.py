#!/usr/bin/env python3
"""Precision/position attribution and R2-only actual-device results."""
import argparse
import json
import math
from statistics import mean,median
from transformers import AutoTokenizer
import eval_exp0218 as ev
import measure_exp0218 as measure
from experiment_exp0220 import RESULT,VARIANTS
from rotation_exp0219 import write_json
from summarize_exp0217 import normalized
from summarize_exp0218 import table

def quality():
    data={s['id']:s for s in ev.dataset()['samples']};tok=AutoTokenizer.from_pretrained(ev.MODEL,local_files_only=True)
    old=json.loads((ev.ROOT.parent/'exp0219/quality_summary.json').read_text())
    fp=json.loads((RESULT/'fp16_control.json').read_text())
    entries={'host_FP16_original':fp['original'],'host_FP16_fold':fp['folded'],
             'DSP_original':old['samples']['A'],'DSP_full_fold':old['samples']['B']}
    determinism={}
    for v in VARIANTS:
        full=json.loads((RESULT/'full'/f'{v}.validated.json').read_text());byid={s['id']:s for s in full['samples']}
        results=[]
        for sample in full['samples']:
            d=data[sample['id']];steps=sample['steps'];r=dict(id=d['id'],kind=d['kind'],language=d['language'])
            if d['kind']=='nll':r.update(nll=[s['nll'] for s in steps],top1=[s['token_id'] for s in steps])
            else:
                ids=[s['token_id'] for s in steps];text=ev.decode(tok,ids);r.update(token_ids=ids,text=text)
                if d['kind']=='task':r['correct']=ev.grade(d,text)
            results.append(r)
        entries['DSP_'+v]=results;mismatch=[]
        for phase in ['quick','repeat']:
            d=json.loads((RESULT/phase/f'{v}.validated.json').read_text())
            for s in d['samples']:
                for step,(a,b) in enumerate(zip(s['steps'],byid[s['id']]['steps'])):
                    fields=[k for k in ['token_id','target_token','target_code','nll','rank','target_ties','max_ties','saturated'] if a[k]!=b[k]]
                    if fields:mismatch.append(dict(id=s['id'],step=step,fields=fields,phase=phase))
            if phase=='repeat':repeat=d['repeat_equal']
        determinism[v]=dict(repeat_equal=repeat,quick_repeat_full_mismatches=mismatch)
        assert repeat and not mismatch,(v,determinism[v])
    teacher=json.loads((ev.ROOT/'teacher_bf16.json').read_text());teacher={s['id']:s for s in teacher['samples']}
    summaries={}
    for name,rows in entries.items():
        nll=mean(x for r in rows if r['kind']=='nll' for x in r['nll'])
        summaries[name]=dict(nll=nll,ppl=math.exp(nll),tasks_correct=sum(r['correct'] for r in rows if r['kind']=='task'),
            teacher_top1_agreement=mean(a==b for r in rows if r['kind']=='nll' for a,b in zip(r['top1'],teacher[r['id']]['top1'])),
            language_nll={lang:mean(x for r in rows if r['kind']=='nll' and r['language']==lang for x in r['nll']) for lang in ['zh','en']})
    conditional={}
    for proj in ['q','k','v','gate','up','head']:
        rows=[r for r in fp['conditional_projection'] if r['projection']==proj]
        conditional[proj]={variant:dict(mean_nrmse=mean(r[variant]['nrmse'] for r in rows),max_nrmse=max(r[variant]['nrmse'] for r in rows)) for variant in ['original_W4','folded_W4','folded_FP16']}
    a=summaries['DSP_original'];c=summaries['DSP_r2_only']
    out=dict(experiment='EXP-0220',summary=summaries,samples=entries,determinism=determinism,
        conditional_projection_error=conditional,fp16_probe_logits=fp['probe_logits'],
        r2_effectiveness=c['nll']<a['nll'] and c['tasks_correct']>a['tasks_correct'],
        dataset_sha256=ev.digest(ev.ROOT/'dataset_v1.json'),holdout_used=False,baseline_promoted=False)
    software=json.loads((RESULT/'r2_only/software_quality.json').read_text())['samples']
    out['r2_software_diagnostic']=dict(nll=mean(x for r in software if r['kind']=='nll' for x in r['nll']),
        tasks_correct=sum(r['correct'] for r in software if r['kind']=='task'),role='FP16 software from packed codes/scales; not bit-exact DSP')
    write_json(RESULT/'quality_summary.json',out)
    labels=['FP16 原始（Host）','FP16 全折叠（Host）','W4A16 原始（DSP，冻结）','W4A16 全折叠（DSP，冻结）',
        'W4A16 仅 QKV 折叠','W4A16 仅 Gate/Up 折叠','W4A16 仅最终 norm/head 折叠','W4A16 R2-only，保留 gamma']
    (RESULT/'quality_table.md').write_text(table(['实现','NLL ↓','条件 PPL ↓','短题','Teacher top-1'],[[label,f"{s['nll']:.4f}",f"{s['ppl']:.2f}",f"{s['tasks_correct']}/24",f"{s['teacher_top1_agreement']*100:.2f}%"] for label,s in zip(labels,summaries.values())]))
    return out

def speed():
    history=json.loads((ev.ROOT/'speed_summary.json').read_text());data={};times={}
    for v in ['original','r2_only']:
        measure.ROOT=RESULT/v;paths=sorted((RESULT/'formal').glob(f'round_*_{v}.jsonl'));assert len(paths)==10
        runs=[measure.speed_validate(p,'w4f16')[0] for p in paths]
        data[v]={mode:[normalized([p for p in run if p['mode']==mode]) for run in runs] for mode in ['prefill','decode']}
        p=median(x['host_us'] for x in data[v]['prefill']);d=median(x['host_us'] for x in data[v]['decode'])
        times[v]=dict(prefill_tokens=64,prefill_host_us=p,prefill_tokens_per_second=64e6/p,
            decode_tokens=15,decode_total_host_us=d*15,decode_tokens_per_second=1e6/d)
    out=dict(data=data,times=times,formal_invocations=320,formal_layer_ledgers=8960,
        paired_speed_percent={mode:100*(median(a['host_us']/b['host_us'] for a,b in zip(data['original'][mode],data['r2_only'][mode]))-1) for mode in ['prefill','decode']})
    write_json(RESULT/'speed_summary.json',out)
    cols=[history['data']['f16f16']['prefill'],data['r2_only']['prefill'],history['data']['w4u8']['prefill']]
    walls=[median(p['host_us'] for p in col) for col in cols]
    labels=['I/O、metadata','Input RMSNorm','QKV＋Q/K Norm-RoPE','QK–Softmax–AV','O projection',
        'Post-attention residual＋RMSNorm','Gate/Up＋SwiGLU','Down','Final residual','KV carrier conversion',
        'KV append DMA','Block orchestration','Layer bookkeeping','Stage-boundary bookkeeping','DSP unattributed',
        'Runtime setup/teardown','Embedding','Final model RMSNorm','LM head＋greedy，不含 final norm','Host–DSP 边界','完整 Host wall']
    rows=[]
    for label,(_,fields) in zip(labels,measure.OVERVIEW):
        vals=[median(sum(p[f] if f.endswith('_us') else p[f]/19.2 for f in fields) for p in col) for col in cols]
        rows.append([label,*[f'{v:.1f} ({100*v/w:.2f}%)' for v,w in zip(vals,walls)],f'{100*(vals[1]/vals[2]-1):+.2f}%' if vals[2] else 'N/A'])
    module=table(['模块','F16A16 冻结 EXP-0218','W4A16 R2-only EXP-0220','W4A8 冻结 EXP-0218','W4A8 相对 W4A16 增速'],rows)
    (RESULT/'module_table.md').write_text(module)
    report=['# EXP-0220 complete profiling comparison','Same frozen ABI108 binary, independent original versus R2-only token sequences. One M64 plus15 feedback steps per session; five short then ten alternating formal rounds. Frozen other recipes are historical nonpaired columns; no activation-only attribution with different W4 values. Quality scoring disabled.',module]
    report.append('All numeric fields are retained below. Additive timing ledger fields are mutually exclusive; engine-work, waits and diagnostic counters overlap and must not be summed. Host-DSP boundary is computed for each record as Host wall minus DSP invocation. Relative changes below are candidate/control minus one, so positive timing changes mean slower. Zero control denominators are N/A, not omitted evidence.')
    for mode in ['prefill','decode']:
        a,b=data['original'][mode],data['r2_only'][mode];rows=[]
        for k in sorted(a[0]):
            vals=[a[0][k],b[0][k],median(x[k] for x in a),median(x[k] for x in b)]
            pct=lambda x,y:f'{100*(y/x-1):+.4f}%' if x else 'N/A: zero control'
            rows.append([k,f'{vals[0]:.6f}',f'{vals[1]:.6f}',pct(*vals[:2]),f'{vals[2]:.6f}',f'{vals[3]:.6f}',pct(*vals[2:])])
        report.extend([f'## {mode}',table(['Field','R1 original','R1 R2-only','R1 change','R10 original median','R10 R2-only median','R10 change'],rows)])
    (RESULT/'full_profiling_report.md').write_text('\n\n'.join(report)+'\n')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--quality-only',action='store_true');p.add_argument('--speed-only',action='store_true');args=p.parse_args()
    if not args.speed_only:quality();print((RESULT/'quality_table.md').read_text())
    if not args.quality_only:print(json.dumps(speed()['times'],indent=2))
