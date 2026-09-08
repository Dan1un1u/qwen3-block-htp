#!/usr/bin/env python3
"""Independent fixed-list EXP0245 aggregation; no candidate selection."""
import json, math, hashlib, subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0245')
SOURCE=Path('/home/daniuniu/work/qwen3-block-htp')
MEMORY=SOURCE.parent/'qwen3-block-htp-project-memory'
CELLS=['en_wiki','zh_wiki','en_news','zh_news']
def read(n):return json.loads((ROOT/n).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):
    with (ROOT/n).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')

def run():
    subprocess.run(['python3',str(MEMORY/'scripts/project_memory.py'),'preflight','--source-worktree',str(SOURCE)],check=True)
    frozen=read('masks.json');masks=frozen['masks']
    assert frozen['final_list_frozen_before_inference']
    assert sha(MEMORY/'docs/experiments/EXP-0245.md')==frozen['protocol_sha256']==read('dataset_freeze.json')['protocol_sha256']
    assert sha(ROOT/'dataset.json')==frozen['dataset_sha256']==read('dataset_freeze.json')['files']['dataset.json']
    assert sha(ROOT/'development.json')==frozen['development_sha256']
    all_scores={};independent={}
    for phase in ['development','final']:
        expected=read('development.json' if phase=='development' else 'dataset.json')['samples']
        for name in frozen[phase+'_names']:
            stem=f'{phase}_{name}_sequential';d=read('scores/'+stem+'.json')
            assert d['phase']==phase and d['name']==name and d['mode']=='sequential'
            assert d['dataset_sha256']==frozen['dataset_sha256'] and d['parameters_sha256']==frozen['parameters_sha256']
            assert d['source_head']==frozen['source_head']
            assert [(r['id'],r['cell']) for r in expected]==[(r['id'],r['cell']) for r in d['samples']]
            assert len({r['id'] for r in d['samples']})==len(expected)
            assert all(len(r['nll'])==16 and all(math.isfinite(x) for x in r['nll']) for r in d['samples'])
            per_cell={c:math.fsum(x for r in d['samples'] if r['cell']==c for x in r['nll'])/(16*sum(r['cell']==c for r in d['samples'])) for c in CELLS}
            mean=math.fsum(per_cell.values())/4
            assert abs(mean-d['mean_nll'])<1e-12 and abs(math.exp(mean)-d['ppl'])<1e-9
            assert all(abs(per_cell[c]-d['cell_nll'][c])<1e-12 for c in CELLS)
            assert d['checks']['repeat_exact'] and d['checks']['causal_exact'] and d['checks']['independent_CE_error']<5e-6
            assert d['mask']==masks.get(name,[]) and d['prefix_immutable']
            independent[stem]=per_cell;all_scores[stem]=d
    assert len(all_scores)==35 and len(list((ROOT/'scores').glob('*.json')))==35
    for recipe in ['C64','F']:
        group=[d for d in all_scores.values() if (d['name']=='F')==(recipe=='F')]
        assert len({d['prefix_digest'] for d in group})==1
        dev=read(f'checks/{recipe}_development_weights.json');final=read(f'checks/{recipe}_final_weights.json')
        assert dev['unchanged'] and final['unchanged'] and dev['digest']==final['digest'] and dev['manifest_sha256']==final['manifest_sha256']
    for n in ['F','C64','A8','L0_FULL','ALL_QKV']:
        proof=read(f'checks/reproduction_{n}.json');assert proof['exact'] and proof['current_score_sha256']==sha(ROOT/f'scores/development_{n}_sequential.json')
    assert read('checks/mask_oracles.json')['pass_all'] and read('development_complete.json')['pass_all']
    scores={n:all_scores[f'final_{n}_sequential'] for n in frozen['final_names']}
    arrays={n:np.asarray([[math.fsum(r['nll'])/16 for r in d['samples'] if r['cell']==c] for c in CELLS],np.float64) for n,d in scores.items()}
    assert all(x.shape==(4,64) for x in arrays.values())
    indices=np.random.default_rng(245).integers(0,64,(10000,4,64))
    draws={n:x[np.arange(4)[None,:,None],indices].mean((1,2)) for n,x in arrays.items()}
    denominator=float((arrays['A8']-arrays['C64']).mean());draw_den=draws['A8']-draws['C64']
    def interval(value,boot):return dict(value=float(value),ci95=np.quantile(boot,[.025,.975]).tolist())
    def compare(n,b):
        delta=float((arrays[n]-arrays[b]).mean());boot=draws[n]-draws[b];cells={}
        for i,c in enumerate(CELLS):
            local=arrays[n][i]-arrays[b][i];lb=local[indices[:,i,:]].mean(1)
            cells[c]=dict(delta_nll=float(local.mean()),ppl_ratio=math.exp(float(local.mean())),ratio_ci95=np.exp(np.quantile(lb,[.025,.975])).tolist())
        return dict(delta_nll=delta,ppl_ratio=math.exp(delta),delta_nll_ci95=np.quantile(boot,[.025,.975]).tolist(),ratio_ci95=np.exp(np.quantile(boot,[.025,.975])).tolist(),cells=cells)
    comparisons={n:{b:compare(n,b) for b in ['A8','C64','F']} for n in scores}
    effects={}
    for n in scores:
        reduction=float((arrays['A8']-arrays[n]).mean());boot=draws['A8']-draws[n]
        effects[n]=dict(nll_reduction=reduction,ci95=np.quantile(boot,[.025,.975]).tolist(),fraction_excess_A8_removed=reduction/denominator,fraction_ci95=np.quantile(boot/draw_den,[.025,.975]).tolist())
    interactions={}
    for scope in ['L0','ALL']:
        q,k,qk,qkv=[scope+'_'+n for n in ['Q','K','QK','QKV']]
        value=(arrays[q]+arrays[k]-arrays[qk]-arrays['A8']).mean()
        boot=draws[q]+draws[k]-draws[qk]-draws['A8']
        interactions[scope]=dict(QK_minus_Q_minus_K=interval(value,boot),added_V_over_QK=interval((arrays[qk]-arrays[qkv]).mean(),draws[qk]-draws[qkv]))
    layer0_vs_all={kind:compare('L0_'+kind,'ALL_'+kind) for kind in ['Q','K','V','QK','QKV']}
    extra_pairwise={scope:compare(scope+'_QK',scope+'_K') for scope in ['L0','ALL']}
    summary=dict(experiment='EXP-0245',final_documents=256,final_targets=4096,development_documents=32,development_targets=512,prefix=[151645],
        fixed_weights='C64_per_channel',fixed_policy='EXP0243_EOS_MSE',final_names=frozen['final_names'],
        ppl={n:d['ppl'] for n,d in scores.items()},cell_ppl={n:{c:math.exp(v) for c,v in d['cell_nll'].items()} for n,d in scores.items()},
        development_ppl={n:all_scores[f'development_{n}_sequential']['ppl'] for n in frozen['development_names']},
        comparisons=comparisons,conditional_effects=effects,interactions=interactions,layer0_vs_all=layer0_vs_all,
        additional_QK_vs_K=dict(role='exploratory_contrasts_of_fixed_final_variants_no_reselection',comparisons=extra_pairwise),
        bootstrap=dict(seed=245,resamples=10000,unit='paired_document_stratified_by_four_cells',denominator_A8_minus_C64=denominator,denominator_ci95=np.quantile(draw_den,[.025,.975]).tolist(),intervals='nominal_pointwise_not_multiplicity_adjusted'),
        software_only=True,new_device_runs=0,baseline_promoted=False,rotation_run=False,e2e_tokens_per_second=None,
        quality_gate={n:dict(point_pass=comparisons[n]['F']['ppl_ratio']<=1.05 and all(v['ppl_ratio']<=1.1 for v in comparisons[n]['F']['cells'].values()),
            confident_pass=comparisons[n]['F']['ratio_ci95'][1]<=1.05 and all(v['ratio_ci95'][1]<=1.1 for v in comparisons[n]['F']['cells'].values())) for n in scores if n!='F'})
    write('summary.json',summary)
    write('checks/independent_score_audit.json',dict(pass_all=True,files=len(all_scores),math_fsum_NLL=True,all_sample_ids_and_masks_exact=True,identical_prefix_digest_per_weight_recipe=True,identical_weight_digest_across_phases=True,parent_reproductions=5,aggregates=independent))
    (ROOT/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'figure.dpi':160,'font.size':10})
    scopes=['L0','EARLY','LATE','ALL'];kinds=['Q','K','V','QK'];a8=all_scores['development_A8_sequential']['mean_nll']
    heat=np.asarray([[a8-all_scores[f'development_{scope}_{kind}_sequential']['mean_nll'] for kind in kinds] for scope in scopes])
    fig,ax=plt.subplots(figsize=(8.5,5));lim=max(abs(heat.min()),abs(heat.max()));im=ax.imshow(heat,cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto')
    ax.set_xticks(range(4),['Q after RoPE','K before cache','V out + cache','Q + K']);ax.set_yticks(range(4),['Layer 0','Layers 1-6','Layers 7-27','All layers 0-27'])
    for i in range(4):
        for j in range(4):ax.text(j,i,f'{heat[i,j]:+.4f}',ha='center',va='center',color='white' if abs(heat[i,j])>.65*lim else 'black')
    ax.set_title('Exposed development: 32 documents / 512 targets\nConditional NLL reduction vs A8 (positive = better)')
    fig.colorbar(im,ax=ax,label='NLL reduction');fig.tight_layout();fig.savefig(ROOT/'figures/scope_boundary_heatmap.png');plt.close(fig)
    names=list(scores);fig,axs=plt.subplots(1,2,figsize=(14,7));y=np.arange(len(names))
    bars=axs[0].barh(y,[scores[n]['ppl'] for n in names],color=['#8595a5' if n in ['F','C64'] else '#d7654a' if n=='A8' else '#267c94' for n in names])
    axs[0].bar_label(bars,fmt='%.2f',padding=3);axs[0].set_xlim(0,max(scores[n]['ppl'] for n in names)*1.16)
    axs[0].set_yticks(y,names);axs[0].invert_yaxis();axs[0].set_xlabel('PPL (lower is better)');axs[0].set_title('Fresh final: 256 documents / 4096 targets')
    values=np.asarray([effects[n]['nll_reduction'] for n in names]);ci=np.asarray([effects[n]['ci95'] for n in names])
    axs[1].errorbar(values,y,xerr=np.maximum(0,np.stack([values-ci[:,0],ci[:,1]-values])),fmt='o',color='#267c94',capsize=3);axs[1].axvline(0,color='gray',linewidth=1)
    axs[1].set_yticks(y,names);axs[1].invert_yaxis();axs[1].set_xlabel('NLL reduction vs A8 (95% paired CI)');axs[1].set_title('Sequential prefill64 + 15 cached steps')
    fig.suptitle('Fixed EOS / C64 W4 / MSE A8; diagnostic A16 restoration only');fig.tight_layout();fig.savefig(ROOT/'figures/final_ppl_effects.png');plt.close(fig)
    lines=['# EXP0245 early Q/K/V attribution','',
        'Fixed EOS151645, C64 per-channel W4, EXP0243 MSE A8 parameters and EXP0244 shared midpoint fanout. Q is post-RoPE; K is post-RoPE before cache; V projection output/cache are tied. Restored cache includes prefix and body.','',
        '| Variant | PPL | vs C64 | vs F16 | Excess A8 NLL removed (95% CI) |','|---|---:|---:|---:|---:|']
    for n in scores:
        e=effects[n];fraction='N/A' if n in ['F','C64'] else f"{100*e['fraction_excess_A8_removed']:.1f}% [{100*e['fraction_ci95'][0]:.1f}, {100*e['fraction_ci95'][1]:.1f}]"
        lines.append(f"| {n} | {scores[n]['ppl']:.4f} | {(comparisons[n]['C64']['ppl_ratio']-1)*100:+.2f}% | {(comparisons[n]['F']['ppl_ratio']-1)*100:+.2f}% | {fraction} |")
    lines+=['','## Four-cell PPL','', '| Variant | English wiki | Chinese wiki | English news | Chinese news |','|---|---:|---:|---:|---:|']
    for n in scores:lines.append('| '+n+' | '+' | '.join(f"{summary['cell_ppl'][n][c]:.4f}" for c in CELLS)+' |')
    lines+=['','## Predeclared interactions','', '| Scope | QK benefit minus Q and K benefits (95% CI) | Added V benefit over QK (95% CI) |','|---|---:|---:|']
    for scope,d in interactions.items():lines.append('| '+scope+' | '+' | '.join(f"{v['value']:+.6f} [{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}]" for v in d.values())+' |')
    lines+=['','## Layer0 versus all layers','', '| Boundary | PPL(layer0 restore) / PPL(all restore), 95% CI |','|---|---:|']
    for k,d in layer0_vs_all.items():lines.append(f"| {k} | {d['ppl_ratio']:.4f} [{d['ratio_ci95'][0]:.4f}, {d['ratio_ci95'][1]:.4f}] |")
    lines+=['','Exploratory conditional Q addition to K-only restoration (fixed final arms; not a new preregistered hypothesis or a selection rule):','']
    for scope,d in extra_pairwise.items():lines.append(f"- {scope}: PPL(QK)/PPL(K) = {d['ppl_ratio']:.4f}, nominal95% paired CI [{d['ratio_ci95'][0]:.4f}, {d['ratio_ci95'][1]:.4f}].")
    lines+=['','Final13 configurations and development22 were frozen before inference; no final-set selection. Development is exposed historical32 documents; final256 documents/4096 targets is new, with document/text/32-token exclusions and independent reconstruction. Identical batch4 sequential prefill64+15 path throughout.','',
        'Five historical controls reproduced exactly per token. Independent same-shape float-QDQ cache oracles for Q/K/V/QK/QKV masks, cache dtype/length/appends, repeated/causal logits, independent CE and math.fsum scoring, immutable prefix and full weight digests all checked.','',
        'Intervals: paired stratified document bootstrap10000, seed245; nominal pointwise95%, no multiplicity adjustment. Conditional restoration effects are not additive causal shares, deployed mixed precision, or bounds on rotation gains. Prefix/body contributions remain combined.','',
        'Original5% overall/10% cell thresholds remain descriptive; diagnostic gate does not accept a deployable A8 model. No hardware, new weights, rotations, calibration or baseline promotion. E2E token/s: N/A.','',
        '![Development scope and boundary](figures/scope_boundary_heatmap.png)','', '![Final PPL and paired effects](figures/final_ppl_effects.png)','']
    with (ROOT/'REPORT.md').open('x') as f:f.write('\n'.join(lines))
    print(json.dumps(dict(ppl=summary['ppl'],effects=effects,interactions=interactions,layer0_vs_all=layer0_vs_all,gate=summary['quality_gate']),indent=2),flush=True)

if __name__=='__main__':run()
