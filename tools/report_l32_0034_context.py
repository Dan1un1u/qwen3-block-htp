"""Audit every paired full-model profile and report supported longer-KV scope."""
import sys,json,hashlib,statistics
from pathlib import Path
import numpy as np
MODEL=sys.argv[1];L=MODEL=='llama';ID='L32-0034' if L else 'EXP-0279';S=Path('/home/daniuniu/work/llama32-htp' if L else '/home/daniuniu/work/qwen3-block-htp');R=Path('/mnt/d/llm_exp/results')/('llama32-htp/l32-0034' if L else 'qwen3-block-htp/exp0279');SEED=30034 if L else 279;LAYERS=16 if L else 28;STEPS=34
if L:
 sys.path.insert(0,str(S/'tools'));from run_llama32_frontend import records as parse;from report_llama32_pipeline_profile import MODULES as GROUPS
 def records(p):return parse(p.read_text())
 def normalized(v):
  x=dict(v);x['generation_lm_head_ticks']-=x['generation_final_norm_ticks'];return x
else:
 sys.path.insert(0,str(S/'scripts'));from device_exp0279 import records;from summarize_exp0217 import normalized as _normalized;from report_exp0273 import GROUPS
 def normalized(v):return _normalized([v])
STDOUT='stdout.txt' if L else 'stdout.jsonl';ARMS=['ALL','FFN']
def read(p):return json.loads(p.read_text())
def write(p,z):
 with p.open('x') as f:json.dump(z,f,indent=2,ensure_ascii=False);f.write('\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert read(R/'full_gate.json')['pass_all'];profiles=0;measured=set();binaries=set();data={a:[] for a in ARMS};physical=[]
for phase,n in [('short',5),('formal',10)]:
 assert read(R/(phase+'.json'))['pass_all']
 for cycle in range(n):
  pair=[]
  for a in ARMS:
   d=R/f'{phase}/{cycle:02d}-{a}';cfg=read(d/'protocol.json');seal=cfg['runtime']['seal'];assert not seal['paper_trace'];measured.add(seal['source_head']);assert read(d/'validated.json')['pass_all']
   for name,h in seal['files'].items():
    p=Path(cfg['runtime']['archive'])/Path(name).name
    if str(p) not in binaries:assert sha(p)==h;binaries.add(str(p))
   ps=[v for v in records(d/STDOUT) if v.get('record')=='generation_profile'];assert len(ps)==STEPS*10
   for i,v in enumerate(ps):
    step=i%STEPS;assert v['generation_step']==step and v['valid_length']==64+step and v['logical_m']==(64 if step==0 else 1)
    assert v['paper_format_disable']==0 and v['paper_pipeline_disable']==(0 if a=='ALL' else 2)
    assert v['block_invocation_count']==LAYERS and v['vtcm_requested_bytes']==v['vtcm_acquired_bytes']==8388608 and v['vtcm_peak_plan_bytes']<=8388608
    assert all(v[k]==0 for k in ['intermediate_ddr_read_bytes','intermediate_ddr_write_bytes','intermediate_spill_fill_count','ledger_unattributed_ticks','dense_r3_mode','dense_r4_mode','scan_cache_append_mismatch_count'])
   pair.append(ps);profiles+=len(ps)
   if phase=='formal':data[a].extend(ps)
  for x,y in zip(*pair):
   for k in ['valid_length','weight_ddr_read_bytes','hmx_command_count','hmx_fp16_tile_pair_count','hmx_u8s8_tile_pair_count','w4u8_decode_direct_n_weight_ddr_read_bytes']:assert x[k]==y[k],k
   if phase=='formal' and cycle==0 and x['generation_step'] in [0,1,31,32,33]:physical.append({k:x[k] for k in ['generation_step','valid_length','weight_ddr_read_bytes','hmx_command_count','hmx_u8s8_tile_pair_count','cache_ddr_read_bytes','cache_ddr_write_bytes','vtcm_peak_plan_bytes'] if k in x})
assert len(measured)==1 and profiles==10200
arms={};scopes={'prefill':lambda v:v['mode']=='prefill','decode':lambda v:v['mode']=='decode','decode_KV65_96':lambda v:65<=v['valid_length']<=96,'decode_KV97':lambda v:v['valid_length']==97}
for a in ARMS:
 arms[a]={}
 for mode,pred in scopes.items():
  vs=[v for v in data[a] if pred(v)];host=statistics.mean(v['host_wall_ns']/1000 for v in vs);ns=[normalized(v) for v in vs]
  rows=[dict(module=name,us=statistics.mean(sum(v[k] for k in keys)/19.2 for v in ns)) for name,keys in GROUPS];rows += [dict(module='Host–DSP 边界',us=host-sum(v['us'] for v in rows)),dict(module='完整 Host wall',us=host)]
  for v in rows:v['host_share_pct']=100*v['us']/host
  assert all(v['us']>=0 for v in rows)
  arms[a][mode]=dict(host_us=host,e2e_tps=(64 if mode=='prefill' else 1)*1e6/host,profiles=len(vs),modules=rows)
rng=np.random.default_rng(SEED);idx=rng.integers(0,10,(20000,10));effects={}
for mode,pred in scopes.items():
 vals={a:np.array([statistics.mean(v['host_wall_ns'] for v in data[a][c*STEPS*10:(c+1)*STEPS*10] if pred(v)) for c in range(10)]) for a in ARMS}
 effects[mode]=dict(ffn_disabled_over_all_wall=float(vals['FFN'].mean()/vals['ALL'].mean()),ci95=np.quantile(vals['FFN'][idx].mean(1)/vals['ALL'][idx].mean(1),[.025,.975]).tolist())
out=dict(experiment=ID,model=MODEL,measured_source=next(iter(measured)),shape='M64+33 cache128, 34 outputs',layers=LAYERS,timed_profiles=profiles,arms=arms,effects=effects,physical_gate=True,weight_and_hmx_work_identical=True,physical_examples=physical,peak=max(v['vtcm_peak_plan_bytes'] for a in ARMS for v in data[a]),quality_claim=False,baseline_promoted=False,numerical_scope='independent full16 CPU greedy selected token/logit code exact, first16 matches old frozen model' if L else 'same frozen LOG2 DSP arithmetic; own selected/chain3 gates inherited from EXP0278; full28 paired outputs/hidden bytes exact and all34 actual finalNorm/full-vocabulary head audited; no full28 CPU transformer equivalence claim')
write(R/'SUMMARY.json',out)
lines=[f'# {ID}：更长 KV 的完整模型流水消融','','原生 W4/SP2 + FP32 残差，无旋转，保留原 log2 softmax；all-on 对比仅关闭 Gate/Up 提前交错（FFN mask2）。计算核、权重、尺度、HMX 工作量、正常核内双缓冲不变。M64+33，cache128，KV 实际到97。','','5 short + 10 formal，交替配对、repeat10 主计时、repeat1 辅助；真实 greedy 全模执行，两臂数值合同相同。完整 Host wall 包含 embedding、所有层、final norm、LM head、greedy、FastRPC；排除 tokenizer、冷加载、ADB。',f'{profiles} 个计时 profile 全部通过物理检查；配对 bootstrap20000 seed{SEED}。']
for mode in ['prefill','decode']:
 lines += ['',f'## {mode}','','| 模块 | all-on μs（Host占比） | FFN 关闭 μs（Host占比） |','|---|---:|---:|']
 for x,y in zip(arms['ALL'][mode]['modules'],arms['FFN'][mode]['modules']):lines.append(f"| {x['module']} | {x['us']:.1f} ({x['host_share_pct']:.2f}%) | {y['us']:.1f} ({y['host_share_pct']:.2f}%) |")
lines += ['','## 配对效应','','| 范围 | FFN关闭 / all-on 墙钟 | 95%CI |','|---|---:|---|']
for mode,e in effects.items():lines.append(f"| {mode} | {e['ffn_disabled_over_all_wall']:.7f} | [{e['ci95'][0]:.7f}, {e['ci95'][1]:.7f}] |")
lines += ['','KV97 分段每条轨迹仅末尾一个 token，仅作边界诊断；不能把不同 token 的差异全部归因于长度。跨实验 M64+15 与本实验绝对速度不作配对加速分母。',out['numerical_scope'],'无模型质量验收，不晋升基线。','','## 端到端 token/s','','| 配置 | Prefill | Decode |','|---|---:|---:|']
for a in ARMS:lines.append(f"| {a} | {arms[a]['prefill']['e2e_tps']:.4f} | {arms[a]['decode']['e2e_tps']:.4f} |")
with (R/'REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps(effects,indent=2))
