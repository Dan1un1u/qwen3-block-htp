#!/usr/bin/env python3
"""Independent reconstruction and paired fixed-candidate confirmation report."""
from pathlib import Path
import json,math,hashlib,subprocess
import numpy as np
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0256');S=Path('/home/daniuniu/work/qwen3-block-htp');M=Path(str(S)+'-project-memory')
CELLS=['en_wiki','zh_wiki','en_news','zh_news'];NAMES=['F','C64','B__wide_nr64','R_MSE__wide_nr64']
def read(n):return json.loads((R/n).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(4194304),b''):h.update(x)
 return h.hexdigest()
def write(n,d):
 with (R/n).open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def audit():
 for n in ['independent_data_audit.json','checks/numerical.json','reproduction_complete.json','final_complete.json']:assert read(n)['pass_all']
 for n in ['dataset_freeze.json','inputs_freeze.json','parameters_freeze.json','freeze.json']:
  f=read(n)
  for p,h in f['files'].items():assert sha(R/p)==h
  for p,h in f.get('references',{}).items():assert sha(p)==h
  if 'protocol_sha256' in f:assert f['protocol_sha256']==sha(M/'docs/experiments/EXP-0256.md')
 assert (R/'inputs_freeze.json').stat().st_mtime<(R/'dataset.json').stat().st_mtime
 assert read('selection.json')['rule']=='fixed_by_user_before_new_dataset' and read('selection.json')['no_new_selection']
 base=read('inputs.json')['parameters']['R3_ALL'];candidate=read('parameters/R_MSE.json');actual=[n for n in base if base[n]!=candidate['parameters'][n]]
 assert actual==candidate['changed_sites'] and len(actual)==55 and all(n.endswith(('residual_mid','residual_out')) for n in actual)
 records={};scores=sorted((R/'scores').glob('*.json'));assert {p.name for p in scores}=={f'{phase}_{n}.json' for phase in ['reproduction','final'] for n in NAMES}
 for p in scores:
  d=json.loads(p.read_text());n=d['name'];phase=d['phase'];rows=read('regression_inputs.json')[n]['rows'] if phase=='reproduction' else read('dataset.json')['samples']
  assert [(x['id'],x['cell']) for x in rows]==[(x['id'],x['cell']) for x in d['samples']]
  cm={c:math.fsum(v for x in d['samples'] if x['cell']==c for v in x['nll'])/(16*sum(x['cell']==c for x in rows)) for c in CELLS};mean=math.fsum(cm.values())/4
  assert all(len(x['nll'])==16 and all(math.isfinite(v) for v in x['nll']) for x in d['samples'])
  assert abs(mean-d['mean_nll'])<1e-12 and abs(math.exp(mean)-d['ppl'])<1e-10 and all(abs(cm[c]-d['cell_nll'][c])<1e-12 for c in CELLS)
  ck=d['checks'];assert ck['repeat_exact'] and ck['causal_exact'] and ck['prefix_immutable'] and ck['CE_max_abs']<5e-6
  assert d['freeze_sha256']==sha(R/'freeze.json');b=read(f'checks/{phase}_{n}_bindings.json');assert b['parameters_sha256']==sha(R/'parameters_freeze.json') and b['shared_fanout_exact']
  if phase=='final':assert b['selection_sha256']==sha(R/'selection.json')
  if n not in ['F','C64']:
   batches=len(rows)//4;expect=(batches+3)*16
   assert len(b['site_calls'])==84 and set(b['site_calls'].values())=={expect} and b['shared_norm_calls']==expect*28
   assert ck['rotation_counts']==dict(prefix_K=28*batches,body_K=28*batches*16,Q=28*batches*16)
  if phase=='reproduction':assert d['samples']==read('regression_inputs.json')[n]['reference'] and read(f'checks/reproduction_{n}.json')['exact']
  records[p.name]=math.exp(mean)
 for p in (R/'checks').glob('*_weights.json'):assert json.loads(p.read_text())['unchanged']
 result=dict(pass_all=True,independent_math_fsum_PPL=records,score_count=len(scores),parameters_frozen_before_data=True,changed_sites=actual,all_invariants_checked=True)
 write('independent_integrity_checks.json',result);return result

def main():
 subprocess.run(['python3',str(M/'scripts/project_memory.py'),'preflight','--source-worktree',str(S)],check=True)
 checked=audit();scores={n:read(f'scores/final_{n}.json') for n in NAMES};arrays={n:np.array([[np.mean(x['nll']) for x in d['samples'] if x['cell']==c] for c in CELLS]) for n,d in scores.items()};assert all(x.shape==(4,64) for x in arrays.values())
 rng=np.random.default_rng(256);ix=rng.integers(0,64,(10000,4,64))
 def compare(a,b):
  delta=arrays[a]-arrays[b];boot=np.take_along_axis(np.broadcast_to(delta,(10000,4,64)),ix,2).mean(2);ci=np.quantile(boot.mean(1),[.025,.975])
  return dict(delta_nll=float(delta.mean()),delta_nll_ci95=ci.tolist(),ppl_ratio=float(np.exp(delta.mean())),ratio_ci95=np.exp(ci).tolist(),cells={c:dict(delta_nll=float(delta[j].mean()),ppl_ratio=float(np.exp(delta[j].mean())),ratio_ci95=np.exp(np.quantile(boot[:,j],[.025,.975])).tolist()) for j,c in enumerate(CELLS)})
 effect=compare('R_MSE__wide_nr64','B__wide_nr64');ci=effect['ratio_ci95'];status='confirmed_improvement' if ci[1]<1 else 'no_improvement_or_worsening_supported' if ci[0]>=1 else 'inconclusive'
 vf={n:compare(n,'F') for n in NAMES if n!='F'};quality={n:dict(point_pass=d['ppl_ratio']<=1.05 and all(v['ppl_ratio']<=1.1 for v in d['cells'].values()),confident_pass=d['ratio_ci95'][1]<=1.05 and all(v['ratio_ci95'][1]<=1.1 for v in d['cells'].values())) for n,d in vf.items()}
 summary=dict(experiment='EXP-0256',scope='fixed R_MSE independent integer-attention software confirmation, not full DSP PPL',candidate='R_MSE',effect_status=status,effect=effect,final_ppl={n:d['ppl'] for n,d in scores.items()},cell_ppl={n:{c:math.exp(v) for c,v in d['cell_nll'].items()} for n,d in scores.items()},vs_F=vf,vs_C64={n:compare(n,'C64') for n in NAMES[2:]},quality_gate=quality,documents=256,targets=4096,bootstrap=dict(repetitions=10000,seed=256,unit='document',strata=CELLS,intervals='nominal95percent'),independent_score_count=checked['score_count'],numerical_gate='pass',device_runs=0,short_rounds=0,formal_rounds=0,device_ppl=None,e2e_tokens_per_second=None,baseline_promoted=False,source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip(),inference_source_heads=sorted(set(d['source_head'] for d in scores.values())),freeze_sha256=sha(R/'freeze.json'),candidate_sha256=sha(R/'parameters/R_MSE.json'))
 write('summary.json',summary)
 labels={'F':'F16','C64':'W4A16 C64','B__wide_nr64':'原整数 A8','R_MSE__wide_nr64':'整数 A8 + 固定 R_MSE'}
 lines=['# EXP0256 固定 R_MSE 独立确认','',f'独立256篇文档、4096个目标token，64篇/语言×领域；候选在生成新数据前已冻结，未重新校准。主效果结论：**{status}**。','',f"R_MSE/原整数A8 PPL比值 **{effect['ppl_ratio']:.6f}**，配对95%CI **[{ci[0]:.6f}, {ci[1]:.6f}]**；PPL变化 {(effect['ppl_ratio']-1)*100:+.2f}%。",'', '| 配置 | PPL | PPL/F16 | 模型点门槛 | 模型置信门槛 |','|---|---:|---:|---|---|']
 for n in NAMES:
  ratio=vf[n]['ppl_ratio'] if n!='F' else 1.;g=quality.get(n,{});lines.append(f"| {labels[n]} | {scores[n]['ppl']:.6f} | {ratio:.6f} | {g.get('point_pass','reference')} | {g.get('confident_pass','reference')} |")
 lines+=['','| 分层 | 原整数 A8 PPL | R_MSE PPL | 候选/原值（95% CI） |','|---|---:|---:|---:|']
 for c in CELLS:
  v=effect['cells'][c];q=v['ratio_ci95'];lines.append(f"| {c} | {summary['cell_ppl']['B__wide_nr64'][c]:.6f} | {summary['cell_ppl']['R_MSE__wide_nr64'][c]:.6f} | {v['ppl_ratio']:.6f} [{q[0]:.6f}, {q[1]:.6f}] |")
 lines+=['','研究效果与模型验收分开：候选/原值整体置信上界<1确认改善，置信下界>=1支持无改善/退化，其余为证据不足。原模型门槛保持总体PPL/F16<=1.05、每语言×领域<=1.10，点与置信通过分别列出。所有区间为文档分层配对bootstrap10000次、seed256的名义95%区间。','', '固定原始per-channel C64 W4、EOS151645、dense R3、wide_nr64及所有其它A8参数。C64表示65536校准token，不是group64。仅复用EXP0255已冻结55处残差参数改变；没有新拟合、权重变更、浮点路径否决、额外候选或中途加样。','', '112组残差参数/舍入检查通过；4组历史回归各首批4文档逐token NLL/top1精确复现。8份新评分已独立math.fsum重算，数据重分词/去重、原参数哈希、真实U8缓存、rotation/shared residual计数、重复/因果性/CE、完整权重及前缀不变检查通过。','', '该实验是全模型软件诊断，只有attention使用已验证的整数算术代理，不能称为全DSP PPL。dense R3硬件整层数值门槛仍待解决。没有新设备执行、基线晋升或进一步校准；其它recipes/原生二进制冻结。','', 'E2E prefill/decode token/s：N/A。完整profiling空值记录见FULL_PROFILE.md。']
 with (R/'REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
 profile=(R.parent/'exp0255/FULL_PROFILE.md').read_text().replace('EXP-0255','EXP-0256').replace('EXP0255','EXP0256').replace('Static A8 calibration software validation only','Fixed-parameter integer A8 independent software confirmation only')
 with (R/'FULL_PROFILE.md').open('x') as f:f.write(profile)
 print(json.dumps({'ppl':summary['final_ppl'],'effect':effect,'status':status,'quality':quality},indent=2),flush=True)
if __name__=='__main__':main()
