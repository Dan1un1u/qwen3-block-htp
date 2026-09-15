import sys,json,statistics
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/daniuniu/work/llama32-htp/tools')
from run_llama32_frontend import records as parse_records
from report_llama32_pipeline_profile import MODULES as GROUPS
def records(p):return parse_records(Path(p).read_text())
def normalized(vals):
 x=dict(vals[0]);x['generation_lm_head_ticks']-=x['generation_final_norm_ticks'];return x
R=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0030')
ARMS=['ALL','QKV','FFN','EPILOGUE','LOOKAHEAD']
rng=np.random.default_rng(30030);idx=rng.integers(0,10,(20000,10));out={'experiment':'L32-0030','scopes':{},'precision':'FP32 residual, nativeW4/SP2mode8, no rotation; quality unassessed'}
for scope in [1,3,16]:
 z=json.loads((R/f'l{scope}_formal.json').read_text());assert z['pass_all'] and len(z['runs'])==50;arms={}
 for a in ARMS:
  ps=[]
  for c in range(10):ps += [v for v in records(R/f'l{scope}-formal-a02/{c:02d}-{a}/stdout.txt') if v.get('record')==('generation_profile' if scope==16 else 'replay_profile')]
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
out['mechanism_coverage']={'QKV': 'N/A: Qwen ring scheduling site is inactive in the measured Llama no-rotation path; QKV arm is a no-op control, not evidence against QKV overlap', 'FFN': 'active', 'EPILOGUE': 'prefill active; decode current implementation inline', 'LOOKAHEAD': 'decode active; no prefill cross-operator lookahead path'}
with (R/'SUMMARY.json').open('x') as f:json.dump(out,f,indent=2,ensure_ascii=False)
lines=['# L32-0030：逐项流水消融及层数扩展','','固定原生接口、SP2、FP32残差、无旋转。ALL为当前流水；QKV关闭QKV与后处理重叠；FFN关闭提前Gate/Up/SwiGLU就绪消费；EPILOGUE关闭O和Down输出处理重叠；LOOKAHEAD关闭O到Gate跨算子预取。每组保持其他优化和GEMM内部DMA双缓冲。','','全模5short+10formal，1/3层各10formal，五臂循环换序，repeat10主计时。置信区间按十个配对轮次bootstrap20000seed30030。全部样本保留，包括Host波动；不以局部外推全模吞吐。']
for mode in ['prefill','decode']:
 lines += ['',f'## 完整16层 {mode} 模块表','','| 模块 | ALL μs（Host占比） | QKV | FFN | EPILOGUE | LOOKAHEAD |','|---|---:|---:|---:|---:|---:|']
 for i,row in enumerate(out['scopes']['16']['arms']['ALL'][mode]['modules']):
  cells=[out['scopes']['16']['arms'][a][mode]['modules'][i] for a in ARMS];lines.append('| '+row['module']+' | '+' | '.join(f"{v['us']:.1f} ({v['host_share_pct']:.2f}%)" for v in cells)+' |')
 lines += ['',f'## {mode} 关闭机制后的墙钟变化','','| 关闭机制 | 单层7 | 连续3层 | 完整16层含首尾 |','|---|---:|---:|---:|']
 for a in ARMS[1:]:
  cells=[out['scopes'][str(n)]['effects'][mode][a] for n in [1,3,16]]
  lines.append('| '+('QKV（无操作对照）' if a=='QKV' else a)+' | '+' | '.join(f"{100*(v['ratio']-1):+.2f}% [{100*(v['ci95'][0]-1):+.2f}, {100*(v['ci95'][1]-1):+.2f}]" for v in cells)+' |')
lines += ['','## 实测完整E2E','','| 配置 | Prefill token/s | Decode token/s |','|---|---:|---:|']
for a in ARMS:
 v=out['scopes']['16']['arms'][a];lines.append(f"| {a} | {v['prefill']['e2e_tps']:.3f} | {v['decode']['e2e_tps']:.3f} |")
lines += ['','M64+15，cache80，无额外前缀。计入embedding/16层/finalnorm/head/greedy/FastRPC，排除tokenizer、冷加载和ADB。单层与3层为各自冻结输入replay，只用于实际局部效果，不能作为同一轨迹的线性外推。关闭机制的效应有条件性，不相加成总收益。诊断追踪仅为软件观测，worker区间包含等待；DMA完成观测是上界，不是引擎利用率。无质量、外部竞争baseline或自动晋升结论。']
lines += ['', 'QKV bit1仅控制未启用的Qwen风格ring，测量Llama路径dispatch全部0，该臂是无操作对照，因果结论N/A。EPILOGUE仅prefill有效，LOOKAHEAD仅decode有效。trace完整性另见FINAL_AUDIT及TRACE_DISPOSITION。局部repeat入口修复前记录不进入正式计时；修复后固定十次，所有迭代保留。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({s:v['effects'] for s,v in out['scopes'].items()},indent=2))
