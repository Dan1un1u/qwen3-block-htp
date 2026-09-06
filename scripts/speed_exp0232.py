#!/usr/bin/env python3
"""Five short and ten rotated full-token formal pairs, all ledgers retained."""
import argparse,json,os
from statistics import median
import numpy as np
import eval_exp0218 as ev
import measure_exp0218 as measure
from summarize_exp0217 import normalized
from summarize_exp0218 import table
from data_exp0232 import RESULT,write,sha,preflight,verified
from awq_exp0232 import frozen
from device_exp0232 import variants,deployed,REMOTE,package

def run(phase):
    preflight();frozen();vs=variants()[1:]
    assert json.loads((RESULT/'after_gate.json').read_text())['pass_all']
    for v in vs:
        deployed(v);p=RESULT/v/'regression_expected.json'
        if not p.exists():write(str(p.relative_to(RESULT)),dict(w4f16=np.fromfile(package(v)/'generation_expected_token_ids_u32.bin',dtype='<u4').tolist()))
    if phase=='formal':
        assert len(list((RESULT/'short').glob('round_*.validated.json')))==10
    for i in range(dict(warmup=1,short=5,formal=10)[phase]):
        for v in vs[i%2:]+vs[:i%2]:
            path=RESULT/phase/f'round_{i+1:02d}_{v}.jsonl'
            if path.with_suffix('.validated.json').exists():
                measure.ROOT=RESULT/v;measure.speed_validate(path,'w4f16')
                assert sha(path)==json.loads(path.with_suffix('.validated.json').read_text())['raw_sha256']
                continue
            os.environ['EXP0218_REMOTE_ROOT']=REMOTE+v;measure.ROOT=ev.ROOT
            meta=measure.run('w4f16',None,path)
            measure.ROOT=RESULT/v;profiles,final=measure.speed_validate(path,'w4f16')
            write(str(path.with_suffix('.validated.json').relative_to(RESULT)),dict(
                raw_sha256=sha(path),profiles=16,layer_ledgers=448,all_steps_pass=final['all_steps_pass'],
                remote=REMOTE+v,manifest_sha256=sha(package(v)/'manifest.json')))
            print('SPEED_COMPLETE',phase,i+1,v,round(meta['elapsed_s'],2),flush=True)

def summarize():
    preflight();frozen();vs=variants()[1:];data={};times={}
    history=json.loads(verified('exp0218','speed_summary.json').read_text())
    for v in vs:
        measure.ROOT=RESULT/v;paths=sorted((RESULT/'formal').glob(f'round_*_{v}.jsonl'));assert len(paths)==10
        runs=[measure.speed_validate(p,'w4f16')[0] for p in paths]
        data[v]={mode:[normalized([p for p in run if p['mode']==mode]) for run in runs] for mode in ['prefill','decode']}
        p=median(x['host_us'] for x in data[v]['prefill']);d=median(x['host_us'] for x in data[v]['decode'])
        times[v]=dict(prefill_tokens=64,prefill_host_us=p,prefill_tokens_per_second=64e6/p,
            decode_tokens=15,decode_total_host_us=15*d,decode_tokens_per_second=1e6/d)
    control,chosen=vs
    paired={mode:100*(median(a['host_us']/b['host_us'] for a,b in zip(data[control][mode],data[chosen][mode]))-1) for mode in ['prefill','decode']}
    write('speed_summary.json',dict(data=data,times=times,paired_speed_percent=paired,profile_candidate=chosen,
        formal_invocations=320,formal_layer_ledgers=8960,prior_speed_sha256=sha(verified('exp0218','speed_summary.json'))))
    cols=[history['data']['f16f16']['prefill'],data[chosen]['prefill'],history['data']['w4u8']['prefill']]
    walls=[median(p['host_us'] for p in col) for col in cols]
    labels=['I/O、metadata','Input RMSNorm','QKV＋Q/K Norm-RoPE','QK–Softmax–AV','O projection',
        'Post-attention residual＋RMSNorm','Gate/Up＋SwiGLU','Down','Final residual','KV carrier conversion',
        'KV append DMA','Block orchestration','Layer bookkeeping','Stage-boundary bookkeeping','DSP unattributed',
        'Runtime setup/teardown','Embedding','Final model RMSNorm','LM head＋greedy，不含 final norm','Host–DSP 边界','完整 Host wall']
    rows=[]
    for label,(_,fields) in zip(labels,measure.OVERVIEW):
        vals=[median(sum(p[f] if f.endswith('_us') else p[f]/19.2 for f in fields) for p in col) for col in cols]
        rows.append([label,*[f'{v:.1f} ({100*v/w:.2f}%)' for v,w in zip(vals,walls)],
            f'{100*(vals[1]/vals[2]-1):+.2f}%' if vals[2] else 'N/A'])
    module=table(['模块','F16A16 历史 EXP0218',f'W4A16 {chosen} EXP0232','W4A8 历史 EXP0218','W4A8 相对 W4A16 增速'],rows)
    (RESULT/'module_table.md').write_text(module)
    report=['# EXP0232 complete profiling',
        'Unchanged ABI108/runtime and per-channel format. Both weight packages have independent expected tokens. '
        'One warmup,5short,10rotated C64/candidate formal pairs;M64+15 feedback steps. '
        'Historical F16/W4U8 columns are nonpaired; changed W4 prevents activation-only attribution. '
        'Quality scoring disabled. Complete Host wall denominators, additive ledgers and overlapping counters retained.',module]
    for mode in ['prefill','decode']:
        a,b=data[control][mode],data[chosen][mode];rows=[]
        for key in sorted(a[0]):
            vals=[a[0][key],b[0][key],median(x[key] for x in a),median(x[key] for x in b)]
            pct=lambda x,y:f'{100*(y/x-1):+.4f}%' if x else 'N/A'
            rows.append([key,f'{vals[0]:.6f}',f'{vals[1]:.6f}',pct(*vals[:2]),
                f'{vals[2]:.6f}',f'{vals[3]:.6f}',pct(*vals[2:])])
        report.extend([f'## {mode}',table(['Field','R1 C64','R1 candidate','change','R10 C64 median','R10 candidate median','change'],rows)])
    report.append('## Direct full-token throughput\n\n'+json.dumps(dict(times=times,paired_speed_percent=paired),indent=2))
    (RESULT/'full_profiling_report.md').write_text('\n\n'.join(report)+'\n')
    print('PROFILING_CLOSED',json.dumps(times),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['warmup','short','formal','summarize']);a=p.parse_args()
    summarize() if a.phase=='summarize' else run(a.phase)
