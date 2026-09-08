#!/usr/bin/env python3
"""Independent aggregation for frozen dense-R3 and matched recalibration arms."""
import json,math,hashlib,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0246');S=Path('/home/daniuniu/work/qwen3-block-htp');M=S.parent/'qwen3-block-htp-project-memory'
CELLS=['en_wiki','zh_wiki','en_news','zh_news']
def read(n):return json.loads((R/n).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):
    with (R/n).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def run():
    subprocess.run(['python3',str(M/'scripts/project_memory.py'),'preflight','--source-worktree',str(S)],check=True)
    inputs=read('inputs.json');pf=read('parameters_freeze.json');scores={};audit={}
    assert sha(M/'docs/experiments/EXP-0246.md')==read('inputs_freeze.json')['protocol_sha256']
    assert sha(R/'inputs.json')==read('inputs_freeze.json')['inputs_sha256']==pf['inputs_sha256']
    assert read('checks/dense_numerical.json')['pass_all'] and read('checks/A16_dense_equivalence.json')['pass_all']
    for n,h in pf['files'].items():assert sha(R/f'parameters/{n}.json')==h
    for phase,names in [('development',inputs['development_names']),('final',inputs['final_names'])]:
        expected=inputs['development'] if phase=='development' else read('dataset.json')['samples']
        for n in names:
            d=read(f'scores/{phase}_{n}_sequential.json')
            assert d['parameters_sha256']==pf['files'][n] and d['dataset_sha256']==sha(R/'dataset.json')
            assert [(r['id'],r['cell']) for r in expected]==[(r['id'],r['cell']) for r in d['samples']]
            assert all(len(r['nll'])==16 and all(math.isfinite(v) for v in r['nll']) for r in d['samples'])
            means={c:math.fsum(v for r in d['samples'] if r['cell']==c for v in r['nll'])/(16*sum(r['cell']==c for r in d['samples'])) for c in CELLS}
            mean=math.fsum(means.values())/4;assert abs(mean-d['mean_nll'])<1e-12 and abs(math.exp(mean)-d['ppl'])<1e-9
            assert all(abs(means[c]-d['cell_nll'][c])<1e-12 for c in CELLS)
            checks=d['checks'];assert checks['repeat_exact'] and checks['causal_exact'] and checks['independent_CE_error']<5e-6 and d['prefix_immutable']
            if phase=='development':assert checks['float_cache_oracle_exact']
            layers=([0] if n.endswith('L0') else list(range(28))) if 'R3_' in n else []
            assert d['rotation_layers']==layers
            calls=len(layers)*len(expected)//4;assert d['rotation_counts']==({'prefix_K':calls,'body_K':calls*16,'Q':calls*16} if calls else {})
            audit[phase+'_'+n]=means;scores[phase+'_'+n]=d
    assert len(scores)==16 and len(list((R/'scores').glob('*.json')))==16
    for recipe in ['C64','F']:
        ds=[d for d in scores.values() if (d['name']=='F')==(recipe=='F')];assert len({d['prefix_digest'] for d in ds})==1
        dev=read(f'checks/{recipe}_development_weights.json');final=read(f'checks/{recipe}_final_weights.json');assert dev['unchanged'] and final['unchanged'] and dev['digest']==final['digest']
    names=inputs['final_names'];final={n:scores['final_'+n] for n in names}
    arrays={n:np.asarray([[math.fsum(r['nll'])/16 for r in d['samples'] if r['cell']==c] for c in CELLS]) for n,d in final.items()}
    assert all(v.shape==(4,64) for v in arrays.values());idx=np.random.default_rng(246).integers(0,64,(10000,4,64))
    draws={n:v[np.arange(4)[None,:,None],idx].mean((1,2)) for n,v in arrays.items()}
    def compare(n,b):
        x=arrays[n]-arrays[b];boot=draws[n]-draws[b];cells={}
        for i,c in enumerate(CELLS):
            cb=x[i][idx[:,i,:]].mean(1);cells[c]=dict(ppl_ratio=math.exp(float(x[i].mean())),ratio_ci95=np.exp(np.quantile(cb,[.025,.975])).tolist())
        return dict(delta_nll=float(x.mean()),ppl_ratio=math.exp(float(x.mean())),delta_nll_ci95=np.quantile(boot,[.025,.975]).tolist(),ratio_ci95=np.exp(np.quantile(boot,[.025,.975])).tolist(),cells=cells)
    comparisons={n:{b:compare(n,b) for b in ['F','C64','A8']} for n in names}
    rotation={scope:compare('R3_'+scope,'recal_'+scope) for scope in ['L0','ALL']}
    calibration={scope:compare('recal_'+scope,'A8') for scope in ['L0','ALL']}
    device={scope:rotation[scope]['ratio_ci95'][1]<1 and comparisons['R3_'+scope]['A8']['ratio_ci95'][1]<1 for scope in ['L0','ALL']}
    quality={n:dict(point_pass=comparisons[n]['F']['ppl_ratio']<=1.05 and all(v['ppl_ratio']<=1.1 for v in comparisons[n]['F']['cells'].values()),confident_pass=comparisons[n]['F']['ratio_ci95'][1]<=1.05 and all(v['ratio_ci95'][1]<=1.1 for v in comparisons[n]['F']['cells'].values())) for n in names if n!='F'}
    original=read('parameters/A8.json')['parameters']
    for n in inputs['development_names']:
        p=read('parameters/'+n+'.json');expected=[f'L{i:02d}.{k}' for i in ([0] if n.endswith('L0') else range(28)) for k in ['q_rope','k_cache']] if n.startswith(('recal_','R3_')) else []
        assert p['changed_sites']==sorted(expected) and all(v==original[k] for k,v in p['parameters'].items() if k not in expected)
    summary=dict(experiment='EXP-0246',method='explicit_dense_GEMM_H128_no_butterfly',final_names=names,documents=256,targets=4096,
        ppl={n:d['ppl'] for n,d in final.items()},cell_ppl={n:{c:math.exp(v) for c,v in d['cell_nll'].items()} for n,d in final.items()},
        development_ppl={n:scores['development_'+n]['ppl'] for n in inputs['development_names']},comparisons=comparisons,
        rotation_vs_matched_recalibration=rotation,recalibration_vs_original=calibration,device_eligible=device,quality_gate=quality,
        A16_equivalence=read('checks/A16_dense_equivalence.json'),bootstrap=dict(resamples=10000,seed=246,unit='paired_document_stratified_four_cells',interval='nominal_pointwise_95percent'),
        software_only=True,new_device_runs=0,weights_changed=False,baseline_promoted=False,e2e_tokens_per_second=None)
    write('summary.json',summary);write('checks/independent_score_audit.json',dict(pass_all=True,files=16,aggregates=audit,other_parameters_exact=True,prefix_and_weights_across_phases_exact=True))
    (R/'figures').mkdir(exist_ok=True);plt.rcParams.update({'figure.dpi':160,'font.size':10})
    fig,axs=plt.subplots(1,2,figsize=(12,5));y=np.arange(len(names));bars=axs[0].barh(y,[final[n]['ppl'] for n in names]);axs[0].bar_label(bars,fmt='%.2f',padding=3);axs[0].set_xlim(0,max(final[n]['ppl'] for n in names)*1.18)
    axs[0].set_yticks(y,names);axs[0].invert_yaxis();axs[0].set_xlabel('PPL (lower is better)');axs[0].set_title('Fresh final: 256 documents / 4096 targets')
    labels=['R3 L0 vs matched recal','R3 ALL vs matched recal','Recal L0 vs original A8','Recal ALL vs original A8'];rows=[rotation['L0'],rotation['ALL'],calibration['L0'],calibration['ALL']]
    values=np.asarray([v['ppl_ratio'] for v in rows]);ci=np.asarray([v['ratio_ci95'] for v in rows]);ys=np.arange(4)
    axs[1].errorbar(values,ys,xerr=np.maximum(0,np.stack([values-ci[:,0],ci[:,1]-values])),fmt='o',capsize=4);axs[1].axvline(1,color='gray');axs[1].set_yticks(ys,labels);axs[1].invert_yaxis();axs[1].set_xlabel('Paired PPL ratio (95% CI; below 1 = better)');axs[1].set_title('Separate rotation from recalibration')
    fig.suptitle('Dense post-RoPE R3; frozen per-channel W4 / EOS / other A8 parameters');fig.tight_layout();fig.savefig(R/'figures/final_ppl.png');plt.close(fig)
    cs={mode:read(f'calibration/{mode}.json') for mode in ['raw','rotated']};bins=8192;centers=np.exp2(-24+(np.arange(bins)+.5)*40/bins)
    fig,axs=plt.subplots(1,3,figsize=(14,4.5));stats={}
    for ax,i in zip(axs,[0,14,27]):
        name=f'L{i:02d}.k_cache';stats[name]={}
        for mode,color in [('raw','#c55b3c'),('rotated','#287a9c')]:
            with np.load(R/f'calibration/{mode}_arrays.npz') as z:h=z[name+'/hist'];mass=h[:bins]+h[bins+1:]
            # Sum adjacent log bins; show probability mass per log interval, not a linear PDF.
            ax.plot(centers.reshape(-1,64).mean(1),mass.reshape(-1,64).sum(1)/h.sum(),label=mode,color=color)
            rec=cs[mode]['sites'][name];p=cs[mode]['parameters'][name]['mse'];stats[name][mode]=dict(median_abs=rec['abs_quantiles']['0.5'],p999_abs=rec['abs_quantiles']['0.999'],minimum=rec['minimum'],maximum=rec['maximum'],scale=p['scale'])
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(2**-10,2**10);ax.set_ylim(1e-7,1);ax.set_title(f'Layer {i}: key before cache');ax.set_xlabel('Absolute activation');ax.set_ylabel('Mass per log interval');ax.legend()
    fig.suptitle('Frozen calibration trajectories: raw versus dense-rotated keys');fig.tight_layout();fig.savefig(R/'figures/key_distributions.png');plt.close(fig)
    # Histogram midpoint estimate, explicitly not a replayed elementwise MSE.
    mse={};signed=np.concatenate([-centers,[0.],centers])
    for mode in ['raw','rotated']:
        mse[mode]={}
        with np.load(R/f'calibration/{mode}_arrays.npz') as z:
            for name,rec in cs[mode]['sites'].items():
                p=cs[mode]['parameters'][name]['mse'];h=z[name+'/hist']
                q=np.clip(np.floor(signed.astype(np.float32)*np.float32(p['inv_scale'])+np.float32(p['zero'])+np.float32(.5)),0,255)
                dq=(q-np.float32(p['zero']))*np.float32(p['scale']);v=float(np.dot(h,(dq-signed)**2)/h.sum())
                mse[mode][name]=dict(histogram_midpoint_mse=v,relative_to_activation_energy=v/(rec['rms']**2),scale=p['scale'])
    write('quantization_mse_summary.json',dict(method='8192_logbins_each_sign_midpoint_estimate_not_exact_elementwise',sites=mse))
    write('distribution_summary.json',stats)
    lines=['# EXP0246 dense online R3 results','', 'Fixed C64 per-channel W4, EOS151645 and all non-Q/K MSE A8 parameters. Dense X@S128 then normalization before FP16 handoff; no butterfly or learned matrix. Q and K share the transform after RoPE, with prefix keys transformed once before U8 storage.','',
        '| Variant | PPL | vs A8 | vs C64 | vs F16 |','|---|---:|---:|---:|---:|']
    for n in names:lines.append(f"| {n} | {final[n]['ppl']:.4f} | "+' | '.join(f"{100*(comparisons[n][b]['ppl_ratio']-1):+.2f}%" for b in ['A8','C64','F'])+' |')
    lines+=['','| Scope | R3 / matched recal PPL (95% CI) | Recal / original A8 PPL (95% CI) | Device progression eligible |','|---|---:|---:|---|']
    for scope in ['L0','ALL']:
        vals=[f"{d['ppl_ratio']:.4f} [{d['ratio_ci95'][0]:.4f}, {d['ratio_ci95'][1]:.4f}]" for d in [rotation[scope],calibration[scope]]]
        lines.append('| '+scope+' | '+' | '.join(vals)+f' | {device[scope]} |')
    lines+=['','| Variant | en wiki | zh wiki | en news | zh news |','|---|---:|---:|---:|---:|']
    for n in names:lines.append('| '+n+' | '+' | '.join(f"{summary['cell_ppl'][n][c]:.4f}" for c in CELLS)+' |')
    lines+=['','Calibration:128 frozen documents x128 tokens, batch4 prefill64+64 sequential steps; same unquantized C64 trajectories, record raw and rotated Q/K at each call shape. Only Q/K recalibrated; K range includes prefix extrema. Development32 exposed documents,9 fixed variants; final256 new documents,7 fixed variants. No tuning or selection on final data.','',
        'Independent matrix parity/FP64 orthogonality/dense arithmetic and paired-attention checks passed; profiler observes aten::mm. Parent F/C64/A8 per-token development controls exact. A16 R3 controls satisfy abs(delta meanNLL)<=0.005. Repeated/causal/CE and independent float-QDQ cache checks, prefix-once/count/dtype/length assertions, immutable whole weights and remaining parameters passed.','',
        'Confidence intervals are nominal paired document bootstrap10000 seed246, stratified four cells. Existing5%/10% quality gates unchanged. Device progression requires R3 upper ratio CI below1 against both original A8 and matched recalibration; eligibility is not deployment quality acceptance or promotion.','',
        'No HMX implementation, hardware timings, or device accuracy claimed for this software phase. E2E token/s: N/A.','',
        '![Final PPL](figures/final_ppl.png)','', '![Key distributions](figures/key_distributions.png)','']
    with (R/'REPORT.md').open('x') as f:f.write('\n'.join(lines))
    print(json.dumps(dict(ppl=summary['ppl'],rotation=rotation,calibration=calibration,device_eligible=device,quality_gate=quality),indent=2))
if __name__=='__main__':run()
