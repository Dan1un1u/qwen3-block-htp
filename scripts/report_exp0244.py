#!/usr/bin/env python3
"""Independent PC059 score aggregation and conditional-effect plots."""
import json, math, hashlib, subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0244')
SOURCE=Path('/home/daniuniu/work/qwen3-block-htp')
MEMORY=SOURCE.parent/'qwen3-block-htp-project-memory'
CELLS=['en_wiki','zh_wiki','en_news','zh_news']
def read(n):return json.loads((ROOT/n).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):
    with (ROOT/n).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')

def run():
    subprocess.run(['python3',str(MEMORY/'scripts/project_memory.py'),'preflight','--source-worktree',str(SOURCE)],check=True)
    selected=read('final_selection.json');dataset=read('dataset.json');masks=read('masks.json')['masks']
    assert sha(MEMORY/'docs/experiments/EXP-0244.md')==read('dataset_freeze.json')['protocol_sha256']
    assert sha(ROOT/'dataset.json')==read('dataset_freeze.json')['files']['dataset.json']
    for n,h in selected['scores_sha256'].items():assert sha(ROOT/f'scores/rerank_{n}_sequential.json')==h
    all_scores={};independent={}
    for path in sorted((ROOT/'scores').glob('*.json')):
        d=json.loads(path.read_text());assert d['dataset_sha256']==sha(ROOT/'dataset.json')
        assert len({r['id'] for r in d['samples']})==len(d['samples'])
        independent[path.stem]={c:math.fsum(x for r in d['samples'] if r['cell']==c for x in r['nll'])/(16*sum(r['cell']==c for r in d['samples'])) for c in CELLS}
        mean=math.fsum(independent[path.stem].values())/4
        assert abs(mean-d['mean_nll'])<1e-12 and abs(math.exp(mean)-d['ppl'])<1e-9
        if d['phase']!='reproduction':
            expected=[r for r in dataset['samples'] if r['split']==('final' if d['phase']=='final' else 'development')]
            if d['phase']=='rerank':expected=[r for i in range(8) for c in CELLS for r in [[v for v in expected if v['cell']==c][i]]]
            assert [(r['id'],r['cell']) for r in expected]==[(r['id'],r['cell']) for r in d['samples']]
        if d['mode']=='sequential':assert d['checks']['repeat_exact'] and d['checks']['causal_exact'] and d['checks']['independent_CE_error']<5e-6
        assert d['mask']==masks.get(d['name'],[]) and d['prefix_immutable']
        all_scores[path.stem]=d
    scores={n:all_scores[f'final_{n}_sequential'] for n in selected['names']}
    arrays={n:np.asarray([[np.mean(r['nll']) for r in d['samples'] if r['cell']==c] for c in CELLS],np.float64) for n,d in scores.items()}
    assert all(x.shape==(4,64) for x in arrays.values())
    rng=np.random.default_rng(244);indices=rng.integers(0,64,(10000,4,64))
    draws={n:x[np.arange(4)[None,:,None],indices].mean((1,2)) for n,x in arrays.items()}
    denominator=(arrays['A8']-arrays['C64']).mean();draw_den=draws['A8']-draws['C64']
    def compare(n,b):
        delta=float((arrays[n]-arrays[b]).mean());boot=draws[n]-draws[b]
        cells={}
        for i,c in enumerate(CELLS):
            local=arrays[n][i]-arrays[b][i]
            localboot=local[indices[:,i,:]].mean(1)
            cells[c]=dict(delta_nll=float(local.mean()),ppl_ratio=math.exp(float(local.mean())),ratio_ci95=np.exp(np.quantile(localboot,[.025,.975])).tolist())
        return dict(delta_nll=delta,ppl_ratio=math.exp(delta),delta_nll_ci95=np.quantile(boot,[.025,.975]).tolist(),ratio_ci95=np.exp(np.quantile(boot,[.025,.975])).tolist(),cells=cells)
    comparisons={n:{b:compare(n,b) for b in ['A8','C64','F']} for n in scores}
    effects={}
    for n in scores:
        reduction=float((arrays['A8']-arrays[n]).mean());boot=draws['A8']-draws[n]
        effects[n]=dict(nll_reduction=reduction,ci95=np.quantile(boot,[.025,.975]).tolist(),fraction_excess_A8_removed=reduction/denominator,
            fraction_ci95=np.quantile(boot/draw_den,[.025,.975]).tolist())
    interaction=(arrays['A8']-arrays['union']).mean()-sum((arrays['A8']-arrays[n]).mean() for n in ['global_swiglu','global_down','global_residual'])
    iboot=(draws['A8']-draws['union'])-sum(draws['A8']-draws[n] for n in ['global_swiglu','global_down','global_residual'])
    summary=dict(experiment='EXP-0244',final_documents=256,final_targets=4096,prefix=[151645],fixed_weights='C64_per_channel',fixed_policy='EXP0243_EOS_MSE',
        ppl={n:d['ppl'] for n,d in scores.items()},cell_ppl={n:{c:math.exp(v) for c,v in d['cell_nll'].items()} for n,d in scores.items()},
        comparisons=comparisons,conditional_effects=effects,selected=selected,
        union_minus_sum_individual_NLL_reduction=dict(value=float(interaction),ci95=np.quantile(iboot,[.025,.975]).tolist()),
        bootstrap=dict(seed=244,resamples=10000,unit='paired_document_stratified_by_four_cells',denominator_ci95=np.quantile(draw_den,[.025,.975]).tolist()),
        software_only=True,new_device_runs=0,baseline_promoted=False,rotation_run=False,e2e_tokens_per_second=None,
        quality_gate={n:dict(point_pass=comparisons[n]['F']['ppl_ratio']<=1.05 and all(v['ppl_ratio']<=1.1 for v in comparisons[n]['F']['cells'].values()),
                             confident_pass=comparisons[n]['F']['ratio_ci95'][1]<=1.05 and all(v['ratio_ci95'][1]<=1.1 for v in comparisons[n]['F']['cells'].values())) for n in scores if n!='F'})
    write('summary.json',summary)
    write('checks/independent_score_audit.json',dict(pass_all=True,files=len(all_scores),math_fsum_NLL=True,all_sample_ids_and_masks_exact=True,aggregates=independent))
    (ROOT/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'figure.dpi':160,'font.size':10})
    base=all_scores['development_A8_bulk']['mean_nll'];families=['layer','swiglu','down','residual']
    heat=np.asarray([[base-all_scores[f'development_{family}_L{i:02d}_bulk']['mean_nll'] for i in range(28)] for family in families])
    fig,ax=plt.subplots(figsize=(14,3.8));lim=max(abs(heat.min()),abs(heat.max()));im=ax.imshow(heat,cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto')
    ax.set_xticks(range(28),[f'{i:02d}' for i in range(28)]);ax.set_yticks(range(4),['Whole layer','SwiGLU','Down output','Residual mid + out']);ax.set_xlabel('Layer (zero based)')
    ax.set_title('Development screening: NLL reduction when restoring A16 (positive = better)\nFixed EOS, C64 W4, frozen MSE A8; bulk screening only')
    fig.colorbar(im,ax=ax,label='A8 NLL minus restored NLL');fig.tight_layout();fig.savefig(ROOT/'figures/layer_boundary_heatmap.png');plt.close(fig)
    names=list(scores);fig,axs=plt.subplots(1,2,figsize=(14,5.5));y=np.arange(len(names))
    axs[0].barh(y,[scores[n]['ppl'] for n in names],color=['#8595a5' if n in ['F','C64'] else '#d7654a' if n=='A8' else '#267c94' for n in names]);axs[0].set_yticks(y,names);axs[0].invert_yaxis();axs[0].set_xlabel('PPL (lower is better)');axs[0].set_title('Fresh final: 256 documents / 4096 targets')
    values=np.asarray([effects[n]['nll_reduction'] for n in names]);ci=np.asarray([effects[n]['ci95'] for n in names]);axs[1].errorbar(values,y,xerr=np.maximum(0,np.stack([values-ci[:,0],ci[:,1]-values])),fmt='o',color='#267c94',capsize=3);axs[1].axvline(0,color='gray',linewidth=1);axs[1].set_yticks(y,names);axs[1].invert_yaxis();axs[1].set_xlabel('NLL reduction vs corrected A8 (95% paired CI)');axs[1].set_title('Sequential prefill64 + 15 cached steps')
    fig.suptitle('Conditional restoration diagnostics; weights and quantizer parameters unchanged');fig.tight_layout();fig.savefig(ROOT/'figures/final_ppl_effects.png');plt.close(fig)
    lines=['# EXP0244 fixed-prefix A8 layer/boundary attribution','',
        'Frozen EOS=151645, C64 per-channel W4 and EXP0243 MSE A8 parameters. All PPL rows use the same prefix and fresh final body tokens. Restorations are software diagnostics; weights never change.','',
        '| Variant | PPL | vs C64 | vs F16 | Excess A8 NLL removed |','|---|---:|---:|---:|---:|']
    for n in scores:lines.append(f"| {n} | {scores[n]['ppl']:.4f} | {(comparisons[n]['C64']['ppl_ratio']-1)*100:+.2f}% | {(comparisons[n]['F']['ppl_ratio']-1)*100:+.2f}% | {effects[n]['fraction_excess_A8_removed']*100:.1f}% |")
    lines+=['','Ratios are paired, same-prefix references. The fraction is a conditional intervention result, not additive attribution or deployable mixed precision. Full 95% intervals and four cells are in summary.json.','',
        f"Fresh final: 256 documents, 4096 scored targets, prefill64 then15 cached steps. Development128 documents; sequential rerank32 documents. Final list frozen before final scoring: {selected['chosen']}.",'',
        '## Proxy correction','',
        'Historical EXP0242/243 residual_mid pre-hook affected the norm input only. EXP0244 quantizes once before the midpoint fanout, feeding norm and skip the same tensor. The archived EXP0243 development panel is reproduced exactly using the legacy mode; historical evidence is unchanged.','',
        '| Development execution | C64 A16 PPL | Legacy A8 PPL | Shared-midpoint A8 PPL |','|---|---:|---:|---:|']
    for phase,mode in [('development','bulk'),('rerank','sequential')]:lines.append('| '+phase+' '+mode+' | '+' | '.join(f"{all_scores[f'{phase}_{n}_{mode}']['ppl']:.4f}" for n in ['C64','legacy','A8'])+' |')
    lines+=['','Bulk and sequential use different FP16 GEMM shapes; bulk is screening only and sequential rerank determines the selected masks. The rerank subset is smaller than bulk and final; their PPL values are not a matched comparison.','',
        f'Union benefit minus sum of individual benefits = {float(interaction):+.6f} NLL; paired95%CI {summary["union_minus_sum_individual_NLL_reduction"]["ci95"]}. This documents interactions, not an additive partition.','',
        'Checks: disabled/all-restored exact; shared-fanout sentinel; legacy128-document exact reproduction; independent same-shape U8 versus float-QDQ cache; per-layer K/V dtype/length/appends; repeated and causal sequential logits; independent CE; immutable prefix, fixed parameters and whole-state digest; document/text/32-token exclusions and reconstruction audit.','',
        'Existing overall5% / per-cell10% versus F16 thresholds unchanged. No hardware run, speed estimate, weight artifact, rotation or baseline promotion. E2E token/s: N/A.','',
        '![Layer/boundary screening](figures/layer_boundary_heatmap.png)','',
        '![Final PPL and paired effects](figures/final_ppl_effects.png)','']
    with (ROOT/'REPORT.md').open('x') as f:f.write('\n'.join(lines))
    print(json.dumps(dict(ppl=summary['ppl'],effects=effects,gate=summary['quality_gate']),indent=2),flush=True)

if __name__=='__main__':run()
