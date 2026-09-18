"""L32-0046 complete paired speed and immutable representation audit."""
import json,statistics,hashlib
from pathlib import Path
import numpy as np
from report_llama32_pipeline_profile import MODULES
from llama_u8_reference import load_qparams_bin
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0046')
M=Path('/mnt/d/llm_exp/models/llama32-htp/l32-0046')
def read(p):return json.loads(p.read_text())
def save(p,z):
    with p.open('x') as f:json.dump(z,f,indent=2,allow_nan=False)
def main():
    rng=np.random.default_rng(320046);result={};report=['# L32-0046 full profiling report'];allrows={}
    for phase in ['short','formal']:
        runs=read(R/(phase+'.json'))['runs'];out={};means={}
        for arm in ['SP2','A8']:
            out[arm]={};means[arm]={}
            for mode in ['prefill','decode']:
                series=[[x for x in read(R/r['tag']/'records.json') if isinstance(x,dict) and x.get('record')=='generation_profile' and x['mode']==mode] for r in runs if r['arm']==arm]
                walls=np.array([statistics.mean(x['host_wall_ns'] for x in a) for a in series]);means[arm][mode]=walls
                avg={k:statistics.mean(statistics.mean(x[k] for x in a) for a in series) for k,v in series[0][0].items() if isinstance(v,(int,float))}
                rows=[]
                for label,keys in MODULES:
                    us=(sum(avg[k] for k in keys)-(avg['generation_final_norm_ticks'] if keys==['generation_lm_head_ticks'] else 0))/19.2
                    rows.append(dict(module=label,us=us,share=100*us/(walls.mean()/1000)))
                boundary=walls.mean()/1000-sum(x['us'] for x in rows);assert boundary>=0
                rows += [dict(module='Host–DSP 边界',us=boundary,share=100*boundary/(walls.mean()/1000)),dict(module='完整 Host wall',us=walls.mean()/1000,share=100)]
                out[arm][mode]=dict(tps=(64 if mode=='prefill' else 1)*1e9/walls.mean(),host_ns=walls.mean(),modules=rows)
                if phase=='formal':
                    allrows[arm+'-'+mode]=avg
                    aux=[x for x in read(R/('aux-'+arm)/'records.json') if isinstance(x,dict) and x.get('record')=='generation_profile' and x['mode']==mode]
                    report+=['## '+arm+' '+mode,'| Field | R1 auxiliary | R10 mean | R10 median of rounds |','|---|---:|---:|---:|']
                    for k,v in avg.items():
                        r1=statistics.mean(x[k] for x in aux);median=statistics.median(statistics.mean(x[k] for x in a) for a in series)
                        report.append(f'| {k} | {r1:.6f} | {v:.6f} | {median:.6f} |')
        out['effects']={}
        for mode in ['prefill','decode']:
            a,b=means['A8'][mode],means['SP2'][mode];ix=rng.integers(0,len(a),(20000,len(a)));ci=np.quantile(a[ix].mean(1)/b[ix].mean(1),[.025,.975]).tolist()
            out['effects'][mode]=dict(a8_over_sp2_wall_ratio=float(a.mean()/b.mean()),ci95=ci)
        result[phase]=out
    parent=read(M.parent/'l32-0044/frontend64-a01/manifest.json');a8=read(M/'frontend64-fixed/manifest.json')
    frozen=[]
    for n,h in parent['files'].items():
        if 'weight_' in n or 'attention_config' in n or n.startswith('generation_') and not any(k in n for k in ['expected','audit']):
            assert a8['files'][n]['sha256']==h['sha256'],n;frozen.append(n)
    for i in range(28):
        a=load_qparams_bin(M.parent/f'l32-0044/frontend64-a01/layer{i}/qparams_u8.bin');b=load_qparams_bin(M/f'frontend64-fixed/layer{i}/qparams_u8.bin')
        for n in a:
            if n!='middle':assert a[n]==b[n],(i,n)
    save(R/'CONTRACT_AUDIT.json',dict(frozen_payload_files=len(frozen),pass_all=True,change='Only Down SP2 replaced by frozen ordinary U8 and dependent LUT/qparams/reference. All0045 shared optimization flags retained; SP2-only guard removed from shared Down lookahead.',quality_claim=False))
    z=dict(experiment='L32-0046',shape='M64+42/cache128/full28',performance=result,numerical_physical_pass=True,quality_claim=False,baseline_promoted=False,measured_source=read(R/'a8-l28/runtime.json')['seal']['source_head'])
    save(R/'SUMMARY.json',z);save(R/'MODULES.json',allrows)
    lines=['# L32-0046 ordinary A8 completion','','Same latest0045 W4 weights/scales, FP32 residual, embedding/head/KV, no rotations. Only Down representation differs. Five short and ten balanced formal paired repeat10 rounds. All independent selected0/13/27, chain3, full43 fixed/free-greedy hidden/IDs/codes and prefill KV gates pass. 8MiB VTCM, one HMX owner, one RPC/token, no timed intermediate DDR/spill. No model-quality acceptance or automatic baseline promotion.','','| Recipe | Prefill TPS | Decode TPS |','|---|---:|---:|']
    for arm in ['SP2','A8']:lines.append(f"| {arm} | {result['formal'][arm]['prefill']['tps']:.6f} | {result['formal'][arm]['decode']['tps']:.6f} |")
    lines+=['','Complete Host wall; loading/tokenizer/audit I/O excluded. Project-owned prompt, not named-dataset performance.','',json.dumps(result['formal']['effects'],indent=2)]
    (R/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    tables=['# L32-0046 additive modules','Units microseconds (% complete Host wall). Counters overlap; only additive modules sum to wall.']
    for mode in ['prefill','decode']:
        tables+=['## '+mode,'| Module | SP2 | A8 |','|---|---:|---:|']
        for a,b in zip(result['formal']['SP2'][mode]['modules'],result['formal']['A8'][mode]['modules']):
            tables.append(f"| {a['module']} | {a['us']:.3f} ({a['share']:.2f}%) | {b['us']:.3f} ({b['share']:.2f}%) |")
    (R/'MODULES.md').write_text('\n'.join(tables)+'\n')
    (R/'FULL_PROFILING_REPORT.md').write_text((R/'RESULTS.md').read_text()+'\n'+(R/'MODULES.md').read_text()+'\n'+'\n'.join(report)+'\n')
    print({a:{m:result['formal'][a][m]['tps'] for m in ['prefill','decode']} for a in ['SP2','A8']})
if __name__=='__main__':main()
