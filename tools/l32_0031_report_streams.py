import sys,json,statistics
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/daniuniu/work/llama32-htp/tools')
from run_llama32_frontend import records as parse_records
def records(p):return parse_records(Path(p).read_text())
def normalized(vals):
 x=dict(vals[0]);x['generation_lm_head_ticks']-=x['generation_final_norm_ticks'];return x
from report_llama32_pipeline_profile import MODULES as GROUPS
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0031')
ARMS=['ALL','SPLIT']
rng=np.random.default_rng(30031);idx=rng.integers(0,10,(20000,10));out={'experiment':'L32-0031','scopes':{},'precision':'FP32 residual, nativeW4/SP2mode8, no rotation; quality unassessed'}
for scope in [16]:
 z=json.loads((R/f'l{scope}_formal.json').read_text());assert z['pass_all'] and len(z['runs'])==20;arms={}
 for a in ARMS:
  ps=[]
  for c in range(10):ps += [v for v in records(R/f'l{scope}-formal-a01/{c:02d}-{a}/stdout.txt') if v.get('record')==('generation_profile' if scope==16 else 'exp0240_profile')]
  assert len(ps)==(1600 if scope==16 else 200);arms[a]={}
  for mode in ['prefill','decode']:
   vals=[v for v in ps if v['mode']==mode];ns=[normalized([v]) for v in vals];host=statistics.mean(v['host_wall_ns']/1000 for v in vals)
   rows=[dict(module=n,us=statistics.mean(sum(v[k] for k in keys)/19.2 for v in ns)) for n,keys in GROUPS]
   rows += [dict(module='Host–DSP 边界',us=host-sum(v['us'] for v in rows)),dict(module='完整 Host wall',us=host)]
   for v in rows:v['host_share_pct']=100*v['us']/host
   assert all(v['us']>=0 for v in rows)
   counters={k:statistics.mean(float(v[k]) for v in vals) for k in vals[0] if any(w in k for w in ['wait_ticks','overlap','prefetch_count','command_count','dma_read_bytes']) and isinstance(vals[0][k],(int,float))}
   arms[a][mode]=dict(host_us=host,modules=rows,counters=counters)
   if scope==16:arms[a][mode]['e2e_tps']=(64 if mode=='prefill' else 1)*1e6/host
 effects={}
 for mode in ['prefill','decode']:
  t={a:np.array([next(v[mode+'_ns'] for v in z['runs'] if v['arm']==a and v['cycle']==c) for c in range(10)]) for a in ARMS}
  effects[mode]={a:dict(ratio=float(t[a].mean()/t['ALL'].mean()),ci95=np.quantile(t[a][idx].mean(1)/t['ALL'][idx].mean(1),[.025,.975]).tolist()) for a in ARMS[1:]}
 out['scopes'][str(scope)]=dict(arms=arms,effects=effects)
with (R/'SUMMARY.json').open('x') as f:json.dump(out,f,indent=2,ensure_ascii=False)
lines=['# L32-0031：SP2 decode 共装与两独立流','','固定原生W4/SP2mode8、FP32残差1、无旋转，权重码/尺度、LUT和所有非目标优化不变。SPLIT由SwiGLU直接输出独立high平面，Down对同一已搬入权重执行两次矩阵累积；ALL用空闲物理行共装一次矩阵流。Prefill合同不变。','','全模5short+10formal，交替顺序，repeat10主计时，repeat1辅助。95%CI按十个配对轮次bootstrap20000seed30031，保留全部样本。']
for mode in ['prefill','decode']:
 lines += ['',f'## 完整16层 {mode} 模块表','','| 模块 | 共装 μs（Host占比） | 两独立流 μs（Host占比） |','|---|---:|---:|']
 for i,row in enumerate(out['scopes']['16']['arms']['ALL'][mode]['modules']):
  cells=[out['scopes']['16']['arms'][a][mode]['modules'][i] for a in ARMS];lines.append('| '+row['module']+' | '+' | '.join(f"{v['us']:.1f} ({v['host_share_pct']:.2f}%)" for v in cells)+' |')
 v=out['scopes']['16']['effects'][mode]['SPLIT'];lines += ['',f"两流/共装 Host wall：{v['ratio']:.6f}，95%CI [{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}]。"]
lines += ['','## 完整E2E','','| 配置 | Prefill token/s | Decode token/s |','|---|---:|---:|']
for a in ARMS:
 v=out['scopes']['16']['arms'][a];lines.append(f"| {a} | {v['prefill']['e2e_tps']:.3f} | {v['decode']['e2e_tps']:.3f} |")
lines += ['','M64+15、cache80、原始提示词。包含embedding/16层/finalnorm/head/greedy/FastRPC，排除tokenizer/冷加载/ADB。HMXcommand计数是worker提交次数；真正额外工作看tilepair计数，decode每层增加16384，16层增加262144。权重搬运不增加。结果只解释固定SP2合同的执行方式，不是SP2相对普通A8的成本（A5另测），也不代表模型质量。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(out['scopes']['16']['effects'],indent=2))
