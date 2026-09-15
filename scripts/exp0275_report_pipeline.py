import sys,json,statistics
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/daniuniu/work/qwen3-block-htp/scripts')
from device_exp0275 import records
from summarize_exp0217 import normalized
from report_exp0273 import GROUPS
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0275')
ARMS=['ALL','QKV','FFN','EPILOGUE','LOOKAHEAD']
rng=np.random.default_rng(275);idx=rng.integers(0,10,(20000,10));out={'experiment':'EXP-0275','scopes':{},'precision':'FP32 residual, nativeW4/SP2mode8, no rotation; quality unassessed'}
for scope in [1,3,28]:
 z=json.loads((R/f'l{scope}_formal.json').read_text());assert z['pass_all'] and len(z['runs'])==50;arms={}
 for a in ARMS:
  ps=[]
  for c in range(10):ps += [v for v in records(R/f'l{scope}-formal/{c:02d}-{a}/stdout.jsonl') if v.get('record')==('generation_profile' if scope==28 else 'exp0240_profile')]
  assert len(ps)==(1600 if scope==28 else 200);arms[a]={}
  for mode in ['prefill','decode']:
   vals=[v for v in ps if v['mode']==mode];ns=[normalized([v]) for v in vals];host=statistics.mean(v['host_wall_ns']/1000 for v in vals)
   rows=[dict(module=n,us=statistics.mean(sum(v[k] for k in keys)/19.2 for v in ns)) for n,keys in GROUPS]
   rows += [dict(module='Host–DSP 边界',us=host-sum(v['us'] for v in rows)),dict(module='完整 Host wall',us=host)]
   for v in rows:v['host_share_pct']=100*v['us']/host
   assert all(v['us']>=0 for v in rows)
   counters={k:statistics.mean(float(v[k]) for v in vals) for k in vals[0] if any(w in k for w in ['wait_ticks','overlap','prefetch_count','command_count','dma_read_bytes']) and isinstance(vals[0][k],(int,float))}
   arms[a][mode]=dict(host_us=host,modules=rows,counters=counters)
   if scope==28:arms[a][mode]['e2e_tps']=(64 if mode=='prefill' else 1)*1e6/host
 effects={}
 for mode in ['prefill','decode']:
  t={a:np.array([next(v[mode+'_ns'] for v in z['runs'] if v['arm']==a and v['cycle']==c) for c in range(10)]) for a in ARMS}
  effects[mode]={a:dict(ratio=float(t[a].mean()/t['ALL'].mean()),ci95=np.quantile(t[a][idx].mean(1)/t['ALL'][idx].mean(1),[.025,.975]).tolist()) for a in ARMS[1:]}
 out['scopes'][str(scope)]=dict(arms=arms,effects=effects)
with (R/'SUMMARY.json').open('x') as f:json.dump(out,f,indent=2,ensure_ascii=False)
lines=['# EXP0275：逐项流水消融及层数扩展','','固定原生接口、SP2、FP32残差、无旋转。ALL为当前流水；QKV关闭QKV与后处理重叠；FFN关闭提前Gate/Up/SwiGLU就绪消费；EPILOGUE关闭O和Down输出处理重叠；LOOKAHEAD关闭O到Gate跨算子预取。每组保持其他优化和GEMM内部DMA双缓冲。','','全模5short+10formal，1/3层各10formal，五臂循环换序，repeat10主计时。置信区间按十个配对轮次bootstrap20000seed275。全部样本保留，包括Host波动；不以局部外推全模吞吐。']
for mode in ['prefill','decode']:
 lines += ['',f'## 完整28层 {mode} 模块表','','| 模块 | ALL μs（Host占比） | QKV | FFN | EPILOGUE | LOOKAHEAD |','|---|---:|---:|---:|---:|---:|']
 for i,row in enumerate(out['scopes']['28']['arms']['ALL'][mode]['modules']):
  cells=[out['scopes']['28']['arms'][a][mode]['modules'][i] for a in ARMS];lines.append('| '+row['module']+' | '+' | '.join(f"{v['us']:.1f} ({v['host_share_pct']:.2f}%)" for v in cells)+' |')
 lines += ['',f'## {mode} 关闭机制后的墙钟变化','','| 关闭机制 | 单层14 | 连续3层 | 完整28层含首尾 |','|---|---:|---:|---:|']
 for a in ARMS[1:]:
  cells=[out['scopes'][str(n)]['effects'][mode][a] for n in [1,3,28]]
  lines.append('| '+a+' | '+' | '.join(f"{100*(v['ratio']-1):+.2f}% [{100*(v['ci95'][0]-1):+.2f}, {100*(v['ci95'][1]-1):+.2f}]" for v in cells)+' |')
lines += ['','## 实测完整E2E','','| 配置 | Prefill token/s | Decode token/s |','|---|---:|---:|']
for a in ARMS:
 v=out['scopes']['28']['arms'][a];lines.append(f"| {a} | {v['prefill']['e2e_tps']:.3f} | {v['decode']['e2e_tps']:.3f} |")
lines += ['','M64+15，cache128，固定EOS前缀。计入embedding/28层/finalnorm/head/greedy/FastRPC，排除tokenizer、冷加载和ADB。单层与3层为各自冻结输入replay，只用于实际局部效果，不能作为同一轨迹的线性外推。关闭机制的效应有条件性，不相加成总收益。诊断追踪仅为软件观测，worker区间包含等待；DMA完成观测是上界，不是引擎利用率。无质量、外部竞争baseline或自动晋升结论。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({s:v['effects'] for s,v in out['scopes'].items()},indent=2))
