import sys,json,statistics
sys.path.insert(0,'/home/daniuniu/work/qwen3-block-htp/scripts')
from common_exp0274 import *
from device_exp0274 import records
from report_exp0273 import GROUPS
from summarize_exp0217 import normalized
import numpy as np
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0274/consumer-row1-a02')
ARMS=['B00','B10','B01','B11']
z=read(R/'formal.json');assert z['pass_all'] and len(z['runs'])==40
rng=np.random.default_rng(274);idx=rng.integers(0,10,(20000,10));out={'experiment':'EXP-0274','fullmodel':True,'fp32_residual':2,'rotation':False,'sp2':8,'PPL':None,'arms':{}}
for arm in ARMS:
 ps=[]
 for c in range(10):ps += [x for x in records(R/f'formal/{c:02d}-{arm}/stdout.jsonl') if x.get('record')=='generation_profile']
 assert len(ps)==1600
 out['arms'][arm]={}
 for mode in ['prefill','decode']:
  vals=[v for v in ps if v['mode']==mode];norms=[normalized([v]) for v in vals];host=statistics.mean(v['host_wall_ns']/1000 for v in vals)
  rows=[dict(module=name,us=statistics.mean(sum(v[k] for k in keys)/19.2 for v in norms)) for name,keys in GROUPS]
  rows += [dict(module='Host–DSP 边界',us=host-sum(v['us'] for v in rows)),dict(module='完整 Host wall',us=host)]
  for v in rows:v['host_share_pct']=100*v['us']/host
  assert abs(sum(v['us'] for v in rows[:-1])-host)<1e-6
  counters={k:statistics.mean(float(v[k]) for v in vals) for k in vals[0] if any(w in k for w in ['wait_ticks','overlap','lifetime_ticks','prefetch_count','command_count','dma_read_bytes']) and isinstance(vals[0][k],(int,float))}
  out['arms'][arm][mode]=dict(host_us=host,tps=(64 if mode=='prefill' else 1)*1e6/host,modules=rows,diagnostic_counters=counters)
out['effects']={}
for mode in ['prefill','decode']:
 t={a:np.array([next(v[mode+'_ns'] for v in z['runs'] if v['arm']==a and v['cycle']==c) for c in range(10)]) for a in ARMS}
 means={a:x.mean() for a,x in t.items()};bs={a:x[idx].mean(1) for a,x in t.items()}
 funcs={'B11_vs_B00_latency':lambda v:v['B11']/v['B00'],'format_given_conventional':lambda v:v['B10']/v['B00'],'format_given_pipeline':lambda v:v['B11']/v['B01'],'pipeline_given_modular':lambda v:v['B01']/v['B00'],'pipeline_given_native':lambda v:v['B11']/v['B10'],'interaction':lambda v:v['B11']*v['B00']/(v['B10']*v['B01'])}
 out['effects'][mode]={name:dict(ratio=float(f(means)),ci95=np.quantile(f(bs),[.025,.975]).tolist()) for name,f in funcs.items()}
write(R/'SUMMARY.json',out)
lines=['# EXP0274：全模格式 × 流水消融','', '固定 Qwen 28 层、M64+15、cache128、离线 EOS 前缀、无旋转、SP2mode8、FP32 residual mode2。全程同一组权重、量化参数与整数算法。B00 模块接口/常规调度；B10 原生格式/常规调度；B01 模块接口/协同流水；B11 原生格式/协同流水。保留所有组的 GEMM 内部双缓冲。','', '格式范围为每层两处 Norm 输出和 SwiGLU→Down 接口；流水范围为 QKV 后处理、Gate/Up/SwiGLU、O/Down epilogue 与 O→Gate 预取。首尾实现相同且计入端到端时间。该消融不代表移除全部硬件优化。','', '各组通过独立 layers0/14/27、连续3层、全28层88个输出边界与封存基线比对及独立 finalnorm/full-vocabulary head 检查。FP32残差固定；模型质量未验收，本次不测PPL。5轮short后10轮formal，四组循环换序；repeat10主结果，repeat1辅助。所有轮次保留。', '']
for mode in ['prefill','decode']:
 lines += [f'## {mode} 完整模块表','', '| 模块 | B00 μs（占比） | B10 | B01 | B11 |','|---|---:|---:|---:|---:|']
 for i,row in enumerate(out['arms']['B00'][mode]['modules']):
  cells=[out['arms'][a][mode]['modules'][i] for a in ARMS];lines.append('| '+row['module']+' | '+' | '.join(f"{v['us']:.1f} ({v['host_share_pct']:.2f}%)" for v in cells)+' |')
 lines += ['', '| 效应（后/前墙钟比） | 比值 | 配对95% CI |','|---|---:|---:|']
 for k,v in out['effects'][mode].items():lines.append(f"| {k} | {v['ratio']:.5f} | [{v['ci95'][0]:.5f}, {v['ci95'][1]:.5f}] |")
lines += ['', '## E2E','', '| 配置 | prefill token/s | decode token/s |','|---|---:|---:|']
for a in ARMS:lines.append(f"| {a} | {out['arms'][a]['prefill']['tps']:.3f} | {out['arms'][a]['decode']['tps']:.3f} |")
lines += ['', '计时含 warm embedding、28层、finalnorm、head、greedy 与 FastRPC；不含CPU tokenizer、冷加载、ADB。Decode分母为15个连续token的平均Host wall。重叠计数不是可相加的模块时间；交互比小于1表示在墙钟比尺度上的互补，需结合CI解释。没有基线自动晋升。其余Llama、独立流水/SP2/softmax等批准阶段尚未完成。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({a:{m:out['arms'][a][m]['tps'] for m in ['prefill','decode']} for a in ARMS},indent=2));print(json.dumps(out['effects'],indent=2))
