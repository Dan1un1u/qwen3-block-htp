"""Frozen EXP0243 evidence aggregation; no inference or candidate selection."""
from pathlib import Path
import json, hashlib, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0243')
def read(p):return json.loads((R/p).read_text())
def write(p,d):
 with (R/p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def series(row,site,field='token_max'):
 return np.concatenate([np.asarray(t[field],float)[0] for t in row['trace'][site]])
def paired(x,y):
 assert [(r['id'],r['cell']) for r in x['samples']]==[(r['id'],r['cell']) for r in y['samples']]
 cells=sorted(x['cell_nll']);rng=np.random.default_rng(243);boots=np.zeros(10000);out={}
 for cell in cells:
  d=np.array([np.mean(a['nll'])-np.mean(b['nll']) for a,b in zip(x['samples'],y['samples']) if a['cell']==cell])
  cell_boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1);boots+=cell_boot/len(cells)
  out[cell]=dict(delta_nll=float(d.mean()),ppl_ratio=float(np.exp(d.mean())),ppl_ratio_ci95=np.exp(np.quantile(cell_boot,[.025,.975])).tolist())
 delta=x['mean_nll']-y['mean_nll'];ci=np.quantile(boots,[.025,.975])
 return dict(delta_nll=delta,ppl_ratio=math.exp(delta),delta_nll_ci95=ci.tolist(),ppl_ratio_ci95=np.exp(ci).tolist(),cells=out)

def main():
 inp=read('inputs.json');sel=read('selection.json');chosen=sel['selected'];prefix=chosen['prefix']['name'];policy=chosen['policy']
 diags={v:read(f'diagnostics/{v}.json') for v in ['F','C64']}
 attribution={}
 for v,d in diags.items():
  attribution[v]={}
  for case in sorted(set(r['case'] for r in d['cases'])):
   rr=[r for r in d['cases'] if r['case']==case];ss=[series(r,'L02.swiglu') for r in rr]
   attribution[v][case]=dict(windows=len(rr),first_token_gt1000=sum(int(s[0]>1000) for s in ss),later_tokens_gt1000=sum(int((s[1:]>1000).sum()) for s in ss),first_median=float(np.median([s[0] for s in ss])),later_max=float(max(s[1:].max() for s in ss)),global_max=float(max(s.max() for s in ss)))
 final={p.stem:json.loads(p.read_text()) for p in sorted((R/'scores').glob('final_*.json'))}
 def score(v,p,pol):return final[f'final_{v}_{p}_{pol}_sequential']
 selected=score('C64',prefix,policy);same=score('C64',prefix,'a16');floating=score('F',prefix,'a16')
 comparisons={'selected_vs_same_prefix_C64':paired(selected,same),'selected_vs_same_prefix_F16':paired(selected,floating),'selected_vs_empty_F16':paired(selected,score('F','empty','a16')),'C64_prefix_conditioning':paired(same,score('C64','empty','a16')),'F16_prefix_conditioning':paired(floating,score('F','empty','a16')),'empty_C64_vs_F16':paired(score('C64','empty','a16'),score('F','empty','a16'))}
 cal={p['name']:read(f'calibration/{p["name"]}.json') for p in inp['prefixes'] if p['eligible']}
 dist={p:{s:{'max_abs':max(abs(c['sites'][s]['minimum']),abs(c['sites'][s]['maximum'])),'p999_abs':c['sites'][s]['abs_quantiles']['0.999'],'minmax_step':c['parameters'][s]['minmax']['scale']} for s in ['L02.swiglu','L02.down_out','L02.residual_out']} for p,c in cal.items()}
 cache={p:read(f'checks/{p}_cache_paths.json') for p in cal}
 summary=dict(experiment='EXP-0243',selected=chosen,empty_policy=sel['empty'],attribution=attribution,calibration=dist,
  remaining_selected_sites={n:cal[prefix]['sites'][n] for n in ['L27.swiglu','L27.down_out','L27.residual_out','head_input']},
  final={k:{j:v[j] for j in ['ppl','mean_nll','cell_nll','cache_storage','cache_length','cache_append_calls']} for k,v in final.items()},comparisons=comparisons,cache_checks=cache,
  quality_gate=dict(overall_limit=1.05,cell_limit=1.10,point_pass=comparisons['selected_vs_same_prefix_F16']['ppl_ratio']<=1.05 and all(v['ppl_ratio']<=1.10 for v in comparisons['selected_vs_same_prefix_F16']['cells'].values()),confident_pass=comparisons['selected_vs_same_prefix_F16']['ppl_ratio_ci95'][1]<=1.05 and all(v['ppl_ratio_ci95'][1]<=1.10 for v in comparisons['selected_vs_same_prefix_F16']['cells'].values())),
  bootstrap=dict(replicates=10000,seed=243,unit='paired_document_stratified_by_four_cells'),new_device_runs=0,baseline_promoted=False)
 write('summary.json',summary)
 figdir=R/'figures';figdir.mkdir(exist_ok=True)
 plt.rcParams.update({'font.size':10,'figure.dpi':150})
 fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
 for col,v in enumerate(['F','C64']):
  for case,label in [('bare128','Fresh cache'),('cached64_then64steps','64 prefill + 64 steps'),('restart_second64','Restart on second half')]:
   ss=np.array([series(r,'L02.swiglu') for r in diags[v]['cases'] if r['case']==case]);axes[0,col].plot(np.median(ss,0),label=label)
  row=next(r for r in diags[v]['cases'] if r['case']=='chat');ss=series(row,'L02.swiglu');axes[1,col].plot(ss,label='Actual full chat template')
  for ax in axes[:,col]:ax.set_yscale('log');ax.set_xlabel('Position in input/cache');ax.set_ylabel('L02 SwiGLU max |activation|');ax.grid(alpha=.2);ax.legend()
  axes[0,col].set_title(f'{v}: median across 16 windows');axes[1,col].set_title(f'{v}: first chat example')
 fig.savefig(figdir/'positions.png');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
 for p in cal:
  with np.load(R/f'calibration/{p}_arrays.npz') as z:
   for ax,s in zip(axes,['L02.swiglu','L02.residual_out']):
    values=np.sort(z[s+'/token_absmax']);ax.plot(np.maximum(values,1e-9),np.arange(1,len(values)+1)/len(values),label=p)
 for ax,s in zip(axes,['L02 SwiGLU','L02 residual']):ax.set_xscale('log');ax.set_xlabel('Per-token max |activation| (online body only)');ax.set_ylabel('Empirical CDF');ax.set_title(s);ax.grid(alpha=.2);ax.legend()
 fig.savefig(figdir/'prefix_distributions.png');plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
 for field,label in [('max','Maximum'),('0.999','p99.9 |element|'),('0.5','Median |element|')]:
  vals=[max(abs(cal[prefix]['sites'][f'L{i:02d}.swiglu']['minimum']),abs(cal[prefix]['sites'][f'L{i:02d}.swiglu']['maximum'])) if field=='max' else cal[prefix]['sites'][f'L{i:02d}.swiglu']['abs_quantiles'][field] for i in range(28)]
  ax.plot(range(28),vals,marker='.',label=label)
 ax.set_yscale('log');ax.set_xlabel('Transformer layer (zero based)');ax.set_ylabel('SwiGLU absolute magnitude');ax.set_title(f'{prefix}: remaining online-body SwiGLU tails');ax.legend();ax.grid(alpha=.2);fig.savefig(figdir/'remaining_layer_tails.png');plt.close(fig)
 labels=[];vals=[]
 for k,v in final.items():labels.append(k.removeprefix('final_').removesuffix('_sequential').replace('_',' '));vals.append(v['ppl'])
 fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True);bars=ax.barh(labels,vals,color=['#7d9bbb' if 'a16' in s else '#d99658' for s in labels]);ax.set_xscale('log');ax.set_xlabel('PPL (lower is better; log scale)');ax.set_title('Fresh 512 documents / 8192 targets; real cache append/read')
 for bar,v in zip(bars,vals):ax.text(v*1.07,bar.get_y()+bar.get_height()/2,f'{v:,.2f}',va='center')
 ax.set_xlim(min(vals)*.7,max(vals)*2.3);fig.savefig(figdir/'final_ppl.png');plt.close(fig)
 lines=['# EXP-0243 results','',f'Selected on development only: `{prefix}` {chosen["prefix"]["ids"]}, `{policy}`. Frozen per-output-channel C64 W4 weights; static U8 online boundaries and actual U8 prefix/body KV storage. Software FP16 arithmetic is not DSP/HMX bit equivalence.','', '## Fresh final PPL','', '| Configuration | PPL | en wiki | en news | zh wiki | zh news |','|---|---:|---:|---:|---:|---:|']
 for label,v in zip(labels,final.values()):lines.append('| '+label+' | '+' | '.join(f'{n:.4f}' for n in [v['ppl']]+[math.exp(v['cell_nll'][c]) for c in ['en_wiki','en_news','zh_wiki','zh_news']])+' |')
 lines+=['','## Paired attribution','', '| Comparison | PPL ratio | 95% paired document CI |','|---|---:|---|']
 for k,v in comparisons.items():lines.append(f'| {k} | {v["ppl_ratio"]:.6f} | {v["ppl_ratio_ci95"]} |')
 lines+=['','## Development selection (exposed, not final)','', '| Prefix | F16 | C64 A16 | A8 minmax | A8 MSE | A8 p99.99 |','|---|---:|---:|---:|---:|---:|']
 for p in cal:
  names=[f'development_F_{p}_a16_bulk',f'development_C64_{p}_a16_bulk']+[f'development_C64_{p}_{method}_bulk' for method in ['minmax','mse','percentile']]
  lines.append('| '+p+' | '+' | '.join(f'{read("scores/"+n+".json")["ppl"]:.4f}' for n in names)+' |')
 lines+=['','## First-position diagnostics','', '| Model/case | First >1000 /16 | Later >1000 count | First median | Later maximum |','|---|---:|---:|---:|---:|']
 for v,cases in attribution.items():
  for c,d in cases.items():lines.append(f'| {v}/{c} | {d["first_token_gt1000"]} | {d["later_tokens_gt1000"]} | {d["first_median"]:.4f} | {d["later_max"]:.4f} |')
 lines+=['','## Evidence and limits','', 'All final rows score the same 64 context + 16 target body tokens, using prefill64 and 15 teacher-forced cache append/read steps. Prefix is unscored and never replaces body context. All models use matching prefix controls. Fresh data were frozen before inference and audited for document/text/32-token and Wikitext-title overlap. Development alone chose the policy. Paired bootstrap: 10000, seed243, document-stratified by four cells.','', 'Repeated logits/NLL and causal future-token controls are exact; independent cross entropy error <5e-6. F16/C64 model weights remain unchanged. FP32 cached/uncached attention oracle tolerance1e-5; measured error recorded in checks. Bulk/sequential FP16 and full-prefix/seeded-prefix differences are reported in cache checks, not hidden by a false byte-exact claim. Prefix U8 NPZ bytes roundtrip exactly; runtime stores prefix and body in uint8.','', 'PPL acceptance remains overall <=5% and every language/domain cell <=10% versus the paired floating reference; matching the imperfect W4A16 alone is insufficient. Short questions not used. No device throughput is inferred from software evaluator duration. New prefill/decode E2E token/s: N/A. Device feasibility, if justified, requires a separate experiment and the existing >10% single-layer latency stop rule.','', 'Figures: figures/positions.png, figures/prefix_distributions.png, figures/final_ppl.png. Full numerical results: summary.json and scores/.']
 (R/'REPORT_AUTOGENERATED.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({k:summary[k] for k in ['selected','final','comparisons','quality_gate']},indent=2))
if __name__=='__main__':main()
