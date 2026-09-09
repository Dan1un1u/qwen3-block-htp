#!/usr/bin/env python3
"""Paired PPL reporting and independent arithmetic/evidence audit for PC069."""
import json,math,hashlib,subprocess
from pathlib import Path
import numpy as np
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0255')
S=Path('/home/daniuniu/work/qwen3-block-htp');M=S.parent/'qwen3-block-htp-project-memory'
CELLS=['en_wiki','zh_wiki','en_news','zh_news'];MODES=['float_core','wide_nr64']
def read(n):return json.loads((R/n).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(n,x):
 with (R/n).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def audit():
 for n in ['reproduction_complete.json','development_complete.json','final_complete.json','independent_data_audit.json','checks/numerical.json']:assert read(n)['pass_all']
 for f in ['dataset_freeze.json','freeze.json','parameters_freeze.json']:
  d=read(f)
  for n,h in d['files'].items():assert sha(R/n)==h
  for n,h in d.get('references',{}).items():assert sha(n)==h
  if 'protocol_sha256' in d:assert sha(M/'docs/experiments/EXP-0255.md')==d['protocol_sha256']
 ds=read('dataset.json')['samples'];cal=read('inputs.json')['calibration'];devold=read('inputs.json')['development']
 allscores=list((R/'scores').glob('*.json'));recomputed={}
 for p in allscores:
  d=json.loads(p.read_text());rows=devold if d['phase']=='reproduction' else [x for x in ds if x['split']==d['phase']]
  assert [(r['id'],r['cell']) for r in rows]==[(r['id'],r['cell']) for r in d['samples']]
  cm={c:math.fsum(v for r in d['samples'] if r['cell']==c for v in r['nll'])/(16*sum(r['cell']==c for r in rows)) for c in CELLS}
  mean=math.fsum(cm.values())/4
  assert all(len(r['nll'])==16 and all(math.isfinite(v) for v in r['nll']) for r in d['samples'])
  assert abs(mean-d['mean_nll'])<1e-12 and abs(math.exp(mean)-d['ppl'])<1e-10
  assert all(abs(cm[c]-d['cell_nll'][c])<1e-12 for c in CELLS)
  assert d['checks']['repeat_exact'] and d['checks']['causal_exact'] and d['checks']['prefix_immutable'] and d['checks']['CE_max_abs']<5e-6
  assert d['freeze_sha256']==sha(R/'freeze.json')
  b=read(f"checks/{d['phase']}_{d['name']}_bindings.json");assert b['parameters_sha256']==sha(R/'parameters_freeze.json') and b['shared_fanout_exact']
  if d['phase']=='final':assert b['selection_sha256']==sha(R/'selection.json') and p.stat().st_mtime>(R/'selection.json').stat().st_mtime
  recomputed[p.name]=math.exp(mean)
 sel=read('selection.json');eligible=['B','R_MSE','R_FAN'];scores={c:{m:read(f'scores/development_{c}__{m}.json')['mean_nll'] for m in MODES} for c in eligible}
 worst={c:max(scores[c][m]-scores['B'][m] for m in MODES) for c in eligible};choice=min(eligible,key=lambda c:(round(worst[c],6),eligible.index(c)))
 assert choice==sel['selected'] and worst==sel['worst_mode_deltaNLL'] and sel['final_not_used']
 original=read('inputs.json')['parameters']['R3_ALL'];changes={}
 for c in ['R_MSE','R_FAN','S_MSE']:
  q=read(f'parameters/{c}.json');actual=[n for n in original if original[n]!=q['parameters'][n]]
  assert actual==q['changed_sites'];assert set(actual)<=set(q['allowed_sites']);assert all(n.endswith('swiglu')==(c=='S_MSE') for n in actual)
  changes[c]=len(actual)
 for p in (R/'checks').glob('*_weights.json'):assert json.loads(p.read_text())['unchanged']
 for n in ['F','C64','B__float_core','B__wide_nr64']:assert read(f'checks/reproduction_{n}.json')['exact']
 result=dict(pass_all=True,independent_math_fsum_PPL=recomputed,score_count=len(allscores),selection_reconstructed=choice,parameter_change_counts=changes,allowed_sites_and_frozen_rest_exact=True,calibration_documents=len(cal),frozen_dataset_references_rechecked=True,source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip())
 write('independent_integrity_checks.json',result);return result

def main():
 subprocess.run(['python3',str(M/'scripts/project_memory.py'),'preflight','--source-worktree',str(S)],check=True)
 checked=audit();sel=read('selection.json');selected=sel['selected'];names=read('final_complete.json')['names'];scores={n:read(f'scores/final_{n}.json') for n in names}
 ar={n:np.array([[np.mean(r['nll']) for r in d['samples'] if r['cell']==c] for c in CELLS]) for n,d in scores.items()}
 assert all(v.shape==(4,32) for v in ar.values());rng=np.random.default_rng(255);ix=rng.integers(0,32,(10000,4,32))
 def compare(a,b):
  v=ar[a]-ar[b];boot=np.take_along_axis(np.broadcast_to(v,(10000,4,32)),ix,2).mean(2);ci=np.quantile(boot.mean(1),[.025,.975])
  return dict(delta_nll=float(v.mean()),ppl_ratio=float(np.exp(v.mean())),ratio_ci95=np.exp(ci).tolist(),cells={c:dict(ppl_ratio=float(np.exp(v[j].mean())),ratio_ci95=np.exp(np.quantile(boot[:,j],[.025,.975])).tolist()) for j,c in enumerate(CELLS)})
 vsf={n:compare(n,'F') for n in names if n!='F'};gates={n:dict(point_pass=d['ppl_ratio']<=1.05 and all(v['ppl_ratio']<=1.10 for v in d['cells'].values()),confident_pass=d['ratio_ci95'][1]<=1.05 and all(v['ratio_ci95'][1]<=1.10 for v in d['cells'].values())) for n,d in vsf.items()}
 effects={c:{m:compare(c+'__'+m,'B__'+m) for m in MODES} for c in dict.fromkeys([selected,'S_MSE'])}
 summary=dict(experiment='EXP-0255',selected=selected,selection=sel,final_ppl={n:d['ppl'] for n,d in scores.items()},cell_ppl={n:{c:math.exp(d['cell_nll'][c]) for c in CELLS} for n,d in scores.items()},vs_F=vsf,vs_C64={n:compare(n,'C64') for n in names if n not in ['F','C64']},effects=effects,quality_gate=gates,parameter_change_counts=checked['parameter_change_counts'],samples=128,targets=2048,bootstrap=dict(repetitions=10000,seed=255,strata=CELLS,unit='document',intervals='nominal_pointwise_not_multiplicity_adjusted'),device_runs=0,device_PPL=None,E2E_tokens_per_second=None,baseline_promoted=False,source_head=checked['source_head'],inference_source_heads=sorted(set(d['source_head'] for d in scores.values())))
 write('summary.json',summary)
 lines=['# EXP-0255 静态 A8 残差校准','',f'固定原始 per-channel C64 W4、EOS 与 dense R3，实际整数 attention 轨迹校准；开发集预声明规则选中 **{selected}**。确认集128文档、2048目标token，未用于调参。','',f"F16 PPL={scores['F']['ppl']:.6f}；W4A16 C64={scores['C64']['ppl']:.6f}。C64表示65536校准token的per-channel GPTQ对照，不是group64。",'', '| Attention 路径 | 原 A8 PPL | 选中残差候选 PPL | 候选/原值（95% CI） | SwiGLU-only PPL |','|---|---:|---:|---:|---:|']
 for m in MODES:
  d=effects[selected][m];ci=d['ratio_ci95'];lines.append(f"| {m} | {scores['B__'+m]['ppl']:.6f} | {scores[selected+'__'+m]['ppl']:.6f} | {d['ppl_ratio']:.6f} [{ci[0]:.6f}, {ci[1]:.6f}] | {scores['S_MSE__'+m]['ppl']:.6f} |")
 lines+=['','参数范围/零点仍为每层边界一个静态U8量化器；仅残差或SwiGLU指定边界改变，所有其它参数、权重、R3实现及设备二进制冻结。没有恢复FP16激活。校准使用128篇既有校准文档、10112个实际执行token；每边界固定1024个完整通道向量用于拟合，非全token拟合。范围极值统计覆盖全部执行token。','', '| 候选 | 参数改变边界数 | 开发集 float NLL | 开发集 integer NLL |','|---|---:|---:|---:|']
 for c in ['B','R_MSE','R_FAN','S_MSE']:
  ds=sel['development_scores'][c];lines.append(f"| {c} | {checked['parameter_change_counts'].get(c,0)} | {ds['float_core']:.6f} | {ds['wide_nr64']:.6f} |")
 lines+=['','R_MSE优化残差张量MSE；R_FAN优化归一化后的残差MSE与后续RMSNorm输出误差之和；S_MSE仅为预声明次要对照，不参与残差候选选择。范围候选16组，整数零点偏移-2/0/+2，加原参数对照；所有拟合后冻结，不做迭代重拟合。','', '| 配置 | en_wiki PPL | zh_wiki PPL | en_news PPL | zh_news PPL | PPL/F16 | 点门槛 | 置信门槛 |','|---|---:|---:|---:|---:|---:|---|---|']
 for n in names:
  g=gates.get(n,{});vals=' | '.join(f"{summary['cell_ppl'][n][c]:.6f}" for c in CELLS);ratio=vsf[n]['ppl_ratio'] if n!='F' else 1.;lines.append(f"| {n} | {vals} | {ratio:.6f} | {g.get('point_pass','reference')} | {g.get('confident_pass','reference')} |")
 lines+=['','原门槛保持：总体PPL相对F16不超过5%，每个语言×领域不超过10%；点通过与置信通过分开。配对文档分层bootstrap10000次，名义95%区间，不作多重比较校正。SwiGLU-only作为次要对照，不能根据其确认结果回头选择残差参数。','',f"全部{checked['score_count']}份评分已用math.fsum独立重算；数据重分词/重叠、参数允许集、范围/零点舍入oracle、校准目标、残差共享norm输入、重复/因果性/CE、权重和前缀不变及四个历史控制精确复现检查通过。",'', 'float_core是保留U8边界的FP32 TwoSum标准attention参考；wide_nr64是已有硬件算术的软件代理。两者均不代表全设备PPL；尚未解决的dense R3设备整层数值gate仍有效。不能将静态参数不增加算子解释为已测得零速度开销。','', '完整profiling见FULL_PROFILE.md，全部新硬件测量N/A。E2E prefill/decode token/s：N/A。']
 with (R/'REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
 parent=R.parent/'exp0254/FULL_PROFILE.md';profile=parent.read_text().replace('EXP-0254','EXP-0255').replace('EXP0254','EXP0255').replace('Ordered software precision attribution only; FP16 restorations are diagnostic and no new device work was performed.','Static A8 calibration software validation only; no new device work was performed.')
 with (R/'FULL_PROFILE.md').open('x') as f:f.write(profile)
 print(json.dumps(summary['final_ppl'],indent=2),flush=True)
if __name__=='__main__':main()
