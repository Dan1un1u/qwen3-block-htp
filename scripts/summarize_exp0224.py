#!/usr/bin/env python3
"""Frozen absmax GPTQ versus transformer-output-selected GPTQ candidates, actual DSP evidence."""
import argparse
import json
import math
from statistics import mean,median
from transformers import AutoTokenizer
import eval_exp0218 as ev
import measure_exp0218 as measure
from experiment_exp0224 import RESULT,VARIANTS
from output_scale_exp0224 import PREVIOUS,CLIPPED
from measure_exp0224 import ALL
from rotation_exp0219 import write_json
from summarize_exp0217 import normalized
from summarize_exp0218 import table

def score(rows,teacher):
    nll=mean(x for r in rows if r['kind']=='nll' for x in r['nll'])
    return dict(nll=nll,ppl=math.exp(nll),tasks_correct=sum(r['correct'] for r in rows if r['kind']=='task'),tasks_total=24,
        teacher_top1_agreement=mean(a==b for r in rows if r['kind']=='nll' for a,b in zip(r['top1'],teacher[r['id']]['top1'])),
        language_nll={lang:mean(x for r in rows if r['kind']=='nll' and r['language']==lang for x in r['nll']) for lang in ['zh','en']})

def quality():
    data={s['id']:s for s in ev.dataset()['samples']};tok=AutoTokenizer.from_pretrained(ev.MODEL,local_files_only=True)
    assert ev.digest(PREVIOUS/'quality_summary.json')=='9f4396e9724098745f841f740b6c872008136ef6fa74452e9acfad0acd3c6990'
    old=json.loads((PREVIOUS/'quality_summary.json').read_text())
    assert ev.digest(CLIPPED/'quality_summary.json')=='4521c8819e97da716fb3d6712503bb08ae9bd94f5ad7eb430ed8efec532ae505'
    clipped=json.loads((CLIPPED/'quality_summary.json').read_text())
    entries={'control_A':old['samples']['GPTQ_A'],'control_C':clipped['samples']['clipped_C']};determinism={};software={}
    for v in VARIANTS:
        full=json.loads((RESULT/'full'/f'{v}.validated.json').read_text());byid={s['id']:s for s in full['samples']};rows=[]
        for s in full['samples']:
            d=data[s['id']];r=dict(id=s['id'],kind=d['kind'],language=d['language'])
            if d['kind']=='nll':r.update(nll=[x['nll'] for x in s['steps']],top1=[x['token_id'] for x in s['steps']])
            else:
                ids=[x['token_id'] for x in s['steps']];text=ev.decode(tok,ids);r.update(token_ids=ids,text=text)
                if d['kind']=='task':r['correct']=ev.grade(d,text)
            rows.append(r)
        entries['selected_'+v]=rows;mismatches=[]
        for phase in ['quick','repeat']:
            d=json.loads((RESULT/phase/f'{v}.validated.json').read_text())
            for s in d['samples']:
                for i,(a,b) in enumerate(zip(s['steps'],byid[s['id']]['steps'])):
                    fields=[k for k in ['token_id','target_token','target_code','nll','rank','target_ties','max_ties','saturated'] if a[k]!=b[k]]
                    if fields:mismatches.append(dict(id=s['id'],step=i,fields=fields,phase=phase))
            if phase=='repeat':repeat=d['repeat_equal']
        determinism[v]=dict(repeat_equal=repeat,overlap_mismatches=mismatches);assert repeat and not mismatches
        software[v]=json.loads((RESULT/v/'software_quality.json').read_text())['samples']
    teacher={s['id']:s for s in json.loads((ev.ROOT/'teacher_bf16.json').read_text())['samples']}
    summaries={k:score(rows,teacher) for k,rows in entries.items()}
    improved=lambda x,y:x['nll']<y['nll'] and x['tasks_correct']>y['tasks_correct']
    chosen=min(VARIANTS,key=lambda v:(summaries['selected_'+v]['nll'],-summaries['selected_'+v]['tasks_correct'],v))
    out=dict(experiment='EXP-0224',summary=summaries,samples=entries,software_diagnostic={k:score(rows,teacher) for k,rows in software.items()},
        determinism=determinism,effectiveness={v:improved(summaries['selected_'+v],summaries['control_'+v]) for v in VARIANTS},
        C_incremental_rotation=improved(summaries['selected_C'],summaries['selected_A']),profile_candidate=chosen,
        calibration_sha256=ev.digest(PREVIOUS/'calibration.json'),dataset_sha256=ev.digest(ev.ROOT/'dataset_v1.json'),holdout_used=False,baseline_promoted=False)
    write_json(RESULT/'quality_summary.json',out)
    labels=['A0 EXP0221 absmax（冻结）','C0 EXP0223 clipping（冻结）','output-selected GPTQ A','output-selected GPTQ C']
    (RESULT/'quality_table.md').write_text(table(['实现','NLL ↓','条件 PPL ↓','短题','Teacher top-1'],[[l,f"{s['nll']:.4f}",f"{s['ppl']:.2f}",f"{s['tasks_correct']}/24",f"{s['teacher_top1_agreement']*100:.2f}%"] for l,s in zip(labels,summaries.values())]))
    return out

def speed():
    quality=json.loads((RESULT/'quality_summary.json').read_text());chosen=quality['profile_candidate']
    history=json.loads((ev.ROOT/'speed_summary.json').read_text());data={};times={}
    for v in ALL:
        measure.ROOT=RESULT/v;paths=sorted((RESULT/'formal').glob(f'round_*_{v}.jsonl'));assert len(paths)==10
        runs=[measure.speed_validate(p,'w4f16')[0] for p in paths]
        data[v]={mode:[normalized([p for p in run if p['mode']==mode]) for run in runs] for mode in ['prefill','decode']}
        p=median(x['host_us'] for x in data[v]['prefill']);d=median(x['host_us'] for x in data[v]['decode'])
        times[v]=dict(prefill_tokens=64,prefill_host_us=p,prefill_tokens_per_second=64e6/p,decode_tokens=15,decode_total_host_us=d*15,decode_tokens_per_second=1e6/d)
    paired={v:{mode:100*(median(a['host_us']/b['host_us'] for a,b in zip(data[v+'0'][mode],data[v][mode]))-1) for mode in ['prefill','decode']} for v in VARIANTS}
    write_json(RESULT/'speed_summary.json',dict(data=data,times=times,paired_speed_percent=paired,profile_candidate=chosen,formal_invocations=640,formal_layer_ledgers=17920))
    cols=[history['data']['f16f16']['prefill'],data[chosen]['prefill'],history['data']['w4u8']['prefill']];walls=[median(p['host_us'] for p in col) for col in cols]
    labels=['I/O、metadata','Input RMSNorm','QKV＋Q/K Norm-RoPE','QK–Softmax–AV','O projection','Post-attention residual＋RMSNorm','Gate/Up＋SwiGLU','Down','Final residual','KV carrier conversion','KV append DMA','Block orchestration','Layer bookkeeping','Stage-boundary bookkeeping','DSP unattributed','Runtime setup/teardown','Embedding','Final model RMSNorm','LM head＋greedy，不含 final norm','Host–DSP 边界','完整 Host wall'];rows=[]
    for label,(_,fields) in zip(labels,measure.OVERVIEW):
        vals=[median(sum(p[f] if f.endswith('_us') else p[f]/19.2 for f in fields) for p in col) for col in cols]
        rows.append([label,*[f'{v:.1f} ({100*v/w:.2f}%)' for v,w in zip(vals,walls)],f'{100*(vals[1]/vals[2]-1):+.2f}%' if vals[2] else 'N/A'])
    module=table(['模块','F16A16 冻结 EXP-0218',f'W4A16 output-selected GPTQ {chosen} EXP-0224','W4A8 冻结 EXP-0218','W4A8 相对 W4A16 增速'],rows);(RESULT/'module_table.md').write_text(module)
    report=['# EXP-0224 complete profiling comparison','Frozen ABI108 binary, frozen A0/C0 and output-selected A/C independent software token sequences. One M64 plus15 feedback steps;5short then10four-way rotating formal rounds. Other recipe columns historical nonpaired, changed W4 prevents activation-only attribution. Quality scoring disabled. All numeric fields retained; additive ledger fields exclusive, engine/wait counters overlapping. Host-DSP boundary per record = Host wall minus DSP invocation. Percent changes below candidate/control minus one; positive timing changes are slower.',module]
    for v in VARIANTS:
        for mode in ['prefill','decode']:
            a,b=data[v+'0'][mode],data[v][mode];rows=[]
            for k in sorted(a[0]):
                vals=[a[0][k],b[0][k],median(x[k] for x in a),median(x[k] for x in b)];pct=lambda x,y:f'{100*(y/x-1):+.4f}%' if x else 'N/A: zero control'
                rows.append([k,f'{vals[0]:.6f}',f'{vals[1]:.6f}',pct(*vals[:2]),f'{vals[2]:.6f}',f'{vals[3]:.6f}',pct(*vals[2:])])
            report.extend([f'## Control {v}0 versus output-selected {v} {mode}',table(['Field','R1 original','R1 candidate','R1 change','R10 original median','R10 candidate median','R10 change'],rows)])
    report.append('## Direct E2E\n\n```json\n'+json.dumps(dict(times=times,paired_speed_percent=paired),indent=2)+'\n```')
    (RESULT/'full_profiling_report.md').write_text('\n\n'.join(report)+'\n');return times

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--quality-only',action='store_true');p.add_argument('--speed-only',action='store_true');a=p.parse_args()
    if not a.speed_only:quality();print((RESULT/'quality_table.md').read_text())
    if not a.quality_only:print(json.dumps(speed(),indent=2))
