#!/usr/bin/env python3
"""Analyze PC057 retained scores and render standalone scientific figures."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0242')
CELLS=['en_wiki','zh_wiki','en_news','zh_news']
FAMILIES=['norm_input','qkv_output','projection_output','swiglu','qk_kv','attention','residual','head_input']
LABELS=['Norm / linear input','QKV projection output','O / Gate / Up / Down output','SwiGLU / Down input','QK RoPE / KV carrier','Attention prob / context','Residual stream','LM-head input']
COLORS={'minmax':'#b95044','mse':'#287b98','percentile':'#947237'}


def read(name):return json.loads((ROOT/name).read_text())


def save(name,obj):
    with (ROOT/name).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')


def paired(candidate,reference):
    assert candidate['ids']==reference['ids']
    rng=np.random.default_rng(242)
    boot=np.zeros(10000)
    cell={}
    for c in CELLS:
        a=np.array([np.mean(r['nll']) for r in candidate['samples'] if r['cell']==c])
        b=np.array([np.mean(r['nll']) for r in reference['samples'] if r['cell']==c])
        d=a-b
        boot+=d[rng.integers(0,len(d),(10000,len(d)))].mean(-1)/4
        cell[c]=dict(ppl_ratio=float(np.exp(d.mean())),nll_delta=float(d.mean()))
    return dict(ppl_ratio=float(np.exp(candidate['mean_nll']-reference['mean_nll'])),
                ppl_ratio_ci95=np.exp(np.quantile(boot,[.025,.975])).tolist(),cells=cell)


def figure(fig,name):
    fig.savefig(ROOT/'figures'/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(ROOT/'figures'/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)


def main():
    (ROOT/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white',
                         'axes.labelcolor':'#27323d','text.color':'#27323d'})
    cal=read('calibration.json');selection=read('selection.json');local=read('local_tensor_error.json');drift=read('residual_drift.json')
    arrays=np.load(ROOT/'activation_arrays.npz');centers=arrays['centers']
    dev={p.stem[len('development_'):]:json.loads(p.read_text()) for p in (ROOT/'scores').glob('development_*.json')}
    primary={p.stem[len('primary_'):]:json.loads(p.read_text()) for p in (ROOT/'scores').glob('primary_*.json')}
    summary=dict(selected=selection['selected'],primary={},development={},local_worst={},distribution_examples=[],
                 scope='static affine U8 QDQ, fixed C64 W4, FP16 software, no DSP measurements',
                 data='128 calibration documents / 16384 tokens; 128 development documents / 2048 targets; exposed independent primary 512 documents / 8192 targets')
    for n,r in primary.items():
        summary['primary'][n]=dict(ppl=r['ppl'],cell_ppl={c:math.exp(v) for c,v in r['cell_nll'].items()},
                                  vs_F=paired(r,primary['F']),vs_C64=paired(r,primary['C64']))
    summary['development']={n:dict(ppl=r['ppl'],delta_nll_vs_C64=r['mean_nll']-dev['C64']['mean_nll']) for n,r in dev.items()}
    for method in ['minmax','mse','percentile']:
        metrics=[]
        for n,v in local['sites'].items():
            q=v[method];metrics.append(dict(site=n,family=cal['sites'][n]['family'],
              nmse=q['sse']/max(q['energy'],1e-30),sqnr_db=10*math.log10(max(q['energy'],1e-30)/max(q['sse'],1e-30)),
              clipped_fraction=q['clipped']/q['count']))
        summary['local_worst'][method]=sorted(metrics,key=lambda x:x['nmse'],reverse=True)[:20]
    # Representative extreme-tailed site in each family, selected from calibration only.
    examples=[]
    for family in ['norm_input','qkv_output','projection_output','swiglu','qk_kv','residual']:
        name=max((n for n,s in cal['sites'].items() if s['family']==family),key=lambda n:cal['sites'][n]['maximum_over_p999'])
        examples.append(name)
        summary['distribution_examples'].append(dict(site=name,**cal['sites'][name],parameters=cal['parameters'][name]))
    fig,axs=plt.subplots(2,3,figsize=(16,9))
    fig.suptitle('Where the 8-bit range goes: fixed-W4 activation distributions',fontsize=17,y=1.02)
    for ax,n in zip(axs.flat,examples):
        s=cal['sites'][n];h=arrays[n+'/hist'];bins=(len(centers)-1)//2
        # Rebin full signed log histogram; ordinate is mass per logarithmic bin.
        neg=h[:bins].reshape(-1,32).sum(-1);pos=h[bins+1:].reshape(-1,32).sum(-1)
        cx=centers[bins+1:].reshape(-1,32).mean(-1)
        xs=np.concatenate([-cx[::-1],cx]);ys=np.concatenate([neg[::-1],pos])/s['count']
        ax.plot(xs,np.maximum(ys,1e-12),color='#596b7d',lw=1.25)
        ax.set_xscale('symlog',linthresh=max(s['abs_quantiles']['0.5']/10,1e-5));ax.set_yscale('log')
        ax.set_ylim(max(1/s['count'],1e-9),1)
        for method in ['minmax','mse']:
            p=cal['parameters'][n][method]
            for j,v in enumerate([p['lo'],p['hi']]):
                if v:ax.axvline(v,color=COLORS[method],ls='--',lw=1,label=method+' range' if j==1 else None)
        ax.set_title(n+'  |  '+s['family'],fontsize=11)
        ax.text(.04,.95,f"max |x| = {max(-s['minimum'],s['maximum']):.3g}\np99.9 |x| = {s['abs_quantiles']['0.999']:.3g}\nratio = {s['maximum_over_p999']:.1f}x",transform=ax.transAxes,va='top',fontsize=9,
                bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
        ax.set_xlabel('Activation value (symmetric-log scale)');ax.set_ylabel('Fraction / log bin');ax.grid(alpha=.15)
    axs[0,0].legend(loc='lower right',fontsize=8)
    fig.text(.01,-.015,'All 16,384 calibration tokens; extrema and histograms use every value. Exact zeros have a separate bin (omitted from log-axis curves).\nEach panel is the largest max/p99.9 site in its family, selected on calibration only. Dashed lines show representable static U8 endpoints.',fontsize=10)
    fig.tight_layout();figure(fig,'activation_distributions')

    fig,axs=plt.subplots(2,3,figsize=(16,8.5))
    fig.suptitle('Activation outliers across token positions and channel regions',fontsize=17,y=1.02)
    for ax,n in zip(axs.flat,examples):
        a=arrays[n+'/heatmap'];limit=max(float(np.max(a)),1e-6)
        im=ax.imshow(np.maximum(a,limit*1e-5),aspect='auto',origin='lower',cmap='magma',norm=LogNorm(vmin=limit*1e-4,vmax=limit))
        ax.set_title(n,fontsize=11);ax.set_xlabel('Channel region (max pooled to <=128 bins)');ax.set_ylabel('Token position')
        fig.colorbar(im,ax=ax,pad=.02,fraction=.045,label='max |activation|')
    fig.text(.01,-.01,'One fixed 128-token English-Wikipedia calibration window; color scales differ by panel. Each pixel is a channel-region maximum, not an individual channel.\nPlots show a representative window; full-corpus tail statistics are in activation_distributions and calibration.json.',fontsize=10)
    fig.tight_layout();figure(fig,'activation_heatmaps')

    fig,axs=plt.subplots(1,3,figsize=(19,6))
    ax=axs[0];y=np.arange(len(FAMILIES))
    for offset,method in [(-.18,'minmax'),(.18,'mse')]:
        vals=[dev[f'{f}_{method}']['ppl'] for f in FAMILIES]
        ax.barh(y+offset,vals,height=.34,color=COLORS[method],label=method)
        for yy,v in zip(y+offset,vals):ax.text(v,yy,f' {v:.1f}',va='center',fontsize=8)
    ax.set_yticks(y,LABELS);ax.invert_yaxis();ax.set_xscale('log');ax.axvline(dev['C64']['ppl'],color='#353d45',ls=':',label='C64 A16')
    ax.set_xlabel('Development PPL (log scale)');ax.set_title('One activation family at a time');ax.legend(fontsize=8)
    ax=axs[1];depth=[0,1,2,4,8,14,21,28]
    values=[dev['C64']['ppl']]+[dev[f'prefix{d}_minmax']['ppl'] for d in depth[1:-1]]+[dev['all_minmax']['ppl']]
    ax.plot(depth,values,'o-',color=COLORS['minmax']);ax.set_yscale('log');ax.set_xticks(depth)
    ax.set_xlabel('First N transformer layers quantized');ax.set_ylabel('Development PPL (log scale)');ax.set_title('Cumulative static-minmax damage')
    ax.text(.02,.03,'Depth 28 also includes LM-head input',transform=ax.transAxes,fontsize=8)
    ax=axs[2]
    for method in ['minmax','mse','percentile']:
        vals=[drift['policies'][method][f'L{i:02d}.residual_out']['relative_rmse'] for i in range(28)]
        ax.plot(range(1,29),vals,label=method,color=COLORS[method],lw=2)
    ax.set_xlabel('Transformer layer');ax.set_ylabel('Residual relative RMSE vs C64 A16');ax.set_title('All-boundary error accumulation');ax.legend()
    for ax in axs[1:]:ax.grid(alpha=.2)
    fig.suptitle('A8 localization: sensitivity, earliest damage, and propagation',fontsize=17,y=1.02)
    fig.text(.01,-.015,'Software QDQ diagnostic with unchanged per-output-channel W4 weights. Residual drift uses 16 disjoint development documents.\nMinmax and MSE thresholds come only from calibration. Isolated-family results are attribution controls, not proposed mixed-precision deployments.',fontsize=10)
    fig.tight_layout();figure(fig,'a8_error_localization')

    names=['F','C64','all_minmax']
    if 'all_'+selection['selected'] not in names:names.append('all_'+selection['selected'])
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    vals=[primary[n]['ppl'] for n in names]
    axs[0].bar(names,vals,color=['#8fa3b4','#405d72','#b95044','#287b98'][:len(names)])
    axs[0].set_yscale('log');axs[0].set_ylabel('PPL (log scale)');axs[0].set_title('Primary regression panel: 8,192 targets')
    for i,v in enumerate(vals):axs[0].text(i,v,f'{v:.3f}',ha='center',va='bottom')
    x=np.arange(4);width=.8/len(names)
    for j,n in enumerate(names):
        axs[1].bar(x+(j-(len(names)-1)/2)*width,[math.exp(primary[n]['cell_nll'][c]) for c in CELLS],width=width,label=n)
    axs[1].set_xticks(x,CELLS);axs[1].set_yscale('log');axs[1].set_ylabel('PPL (log scale)');axs[1].set_title('Language / domain breakdown');axs[1].legend(fontsize=8)
    fig.text(.01,-.02,'Canonical FP16 and C64 controls re-evaluated on identical tokens/masks. Historical panel is exposed; it did not select clipping in EXP0242.\nNo new device profiling; E2E tokens/s is N/A.',fontsize=10)
    fig.tight_layout();figure(fig,'primary_ppl')
    save('summary.json',summary)
    lines=['# EXP-0242 results','',summary['scope'],'',summary['data'],'','## Primary PPL','',
           '| Variant | Overall | English Wiki | Chinese Wiki | English news | Chinese news | vs F16 | vs C64 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for n in names:
        r=summary['primary'][n]
        lines.append('| '+n+' | '+' | '.join(f'{v:.6f}' for v in [r['ppl']]+[r['cell_ppl'][c] for c in CELLS])+f" | {(r['vs_F']['ppl_ratio']-1)*100:+.2f}% | {(r['vs_C64']['ppl_ratio']-1)*100:+.2f}% |")
    lines+=['','Selected complete-scope policy: '+selection['selected']+'. No weight changes or device deployment.','',
            '## Development attribution','', '| Family | Minmax PPL | MSE PPL |','|---|---:|---:|']
    for f in FAMILIES:lines.append(f"| {f} | {dev[f+'_minmax']['ppl']:.6f} | {dev[f+'_mse']['ppl']:.6f} |")
    lines+=['','## Files','', 'Calibration thresholds/full moments: calibration.json; full histograms/channel/token traces: activation_arrays.npz; independent tensor errors: local_tensor_error.json; residual propagation: residual_drift.json; per-document scores: scores/; paired 95% intervals: summary.json.','',
            'Histogram MSE is an approximate local reconstruction criterion, not a guarantee of lower model NLL. Raw tensor error is independently measured on development. Quantized softmax probabilities retain zero and are not renormalized. QKV V and KV carrier share a boundary; combined paths quantize it once.','',
            '## Scope and limits','', 'This diagnostic injects static U8 QDQ into eager FP16 execution of fixed C64 effective weights. It is not bit-exact HMX conversion, DSP nonlinear LUTs or KV append validation, and cannot certify final W4A8 device quality. The old EXP0218 W4U8 weights differ from C64; this is not a matched reconstruction of that historical collapse. All activation sites remain enabled in final A8 candidates. Short answers were not used for selection.','',
            'No device execution: complete Host wall, module profiling, physical VTCM/DMA counters and E2E token/s are N/A. Existing >10% single-layer latency stop rule remains for future device work.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:summary[k] for k in ['selected','primary']},indent=2))


if __name__=='__main__':main()
