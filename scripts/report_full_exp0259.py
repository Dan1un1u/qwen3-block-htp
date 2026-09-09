#!/usr/bin/env python3
"""Audit every EXP0259 fullmodel timed invocation and report actual fullmodel E2E."""
import sys,json,math,statistics,subprocess,hashlib
from pathlib import Path
sys.path.insert(0,'/home/daniuniu/work/qwen3-block-htp/scripts')
from device_full_exp0259 import *
from device_exp0258 import LEDGER,normalized
from measure_exp0218 import OVERVIEW
preflight();assert read(R/'full_formal_gate.json')['integrity_pass'];data={};loops={};calls=0
for phase,n in [('short',5),('formal',10)]:
 g=read(R/('full_'+phase+'_gate.json'));rows=[];rng=np.random.default_rng(259)
 for i in range(1,n+1):
  for rep in [1,10]:
   for arm in [0,1]:
    tag=f'full_{phase}/round{i:02d}_r{rep}_a{arm}';rs=records(R/tag/'stdout.jsonl');ps=[q for q in rs if q.get('record')=='generation_profile'];fs=[q for q in rs if q.get('generation_sequence_complete')];assert len(ps)==rep*16 and len(fs)==rep;physical(ps,28,arm)
    gold=read(R/f'smoke_a{arm}/validated.json')['token_ids']
    for j,f in enumerate(fs):
     assert f['all_steps_pass'] and f['token_ids']==gold and len(gold)==16
     assert f['total_host_wall_ns']==sum(p['host_wall_ns'] for p in ps[j*16:j*16+16]);assert f['generation_loop_wall_ns']>=f['total_host_wall_ns']
    for p in ps:
     assert p['repeat_count']==1 and p['backend']=='standalone_fastrpc_dsp' and p['qnn']=='none'
     assert p['u8_attention_audit_ddr_write_bytes']==0
     assert p['w4u8_decode_direct_n_projection_count']==(196 if p['mode']=='prefill' else 197)
     assert p['dense_r3_total_hmx_calls']==28*arm and p['dense_r3_total_refined_values']==0
    calls+=len(ps);z=read(R/tag/'validated.json');assert abs(z['prefill_ns']-statistics.mean(p['host_wall_ns'] for p in ps if p['mode']=='prefill'))<1e-5;assert abs(z['decode_ns']-statistics.mean(p['host_wall_ns'] for p in ps if p['mode']=='decode')*15)<1e-5;rows.append(dict(round=i,**z))
    if phase=='formal':
     key=f'a{arm}_r{rep}';data.setdefault(key,{'prefill':[],'decode':[]})
     for mode in ['prefill','decode']:data[key][mode].append(normalized([p for p in ps if p['mode']==mode]))
     u=loops.setdefault(key,{'model_ns':[],'loop_ns':[],'startup':[]});u['model_ns'].append(statistics.mean(f['total_host_wall_ns'] for f in fs));u['loop_ns'].append(statistics.mean(f['generation_loop_wall_ns'] for f in fs));u['startup'].append(next(x for x in rs if x.get('record')=='generation_startup'))
 for rep in [1,10]:
  for mode in ['prefill_ns','decode_ns']:
   pairs=np.array([[next(r[mode] for r in rows if r['round']==i+1 and r['repeat']==rep and r['arm']==a) for a in [0,1]] for i in range(n)]);ratios=pairs[:,1]/pairs[:,0];ci=np.quantile(np.median(ratios[rng.integers(0,n,(10000,n))],axis=1),[.025,.975]);v=g['performance'][f'r{rep}_{mode}'];assert abs(np.median(ratios)-v['paired_ratio'])<1e-12 and np.max(abs(ci-v['ci95']))<1e-12
assert calls==5280
# Verify historical raw through previously authority-sealed EXP0257 provenance.
parent=R.parent/'exp0257';assert sha(parent/'EVIDENCE_SHA256.json')=='80b1ac2415364ef7898e118b111215279f558748035f74d26d7979ee845a3f4f';seal=read(parent/'EVIDENCE_SHA256.json');assert sha(parent/'historical_provenance.json')==seal['files']['historical_provenance.json']['sha256'];hist=read(parent/'historical_provenance.json');fseries=[];wseries=[]
for name,h in sorted(hist.items()):
 assert sha(name)==h
 s=normalized([q for q in records(name) if q.get('record')=='generation_profile' and q['mode']=='prefill'])
 (fseries if 'exp0218' in name else wseries).append(s)
assert len(fseries)==len(wseries)==10
write(R/'historical_provenance.json',hist)
def val(series,fields):return statistics.median(sum(p[k] if k.endswith('_us') else p[k]/19.2 for k in fields) for p in series)
def table(h,rs):return '\n'.join(['| '+' | '.join(h)+' |','|'+'|'.join(['---']*len(h))+'|']+['| '+' | '.join(str(v) for v in r)+' |' for r in rs])
labels=['I/O、metadata','Input RMSNorm','QKV＋Q/K Norm-RoPE（含R3）','QK–Softmax–AV','O projection','Post-attention residual＋RMSNorm','Gate/Up＋SwiGLU','Down','Final residual','KV carrier conversion','KV append DMA','Block orchestration','Layer bookkeeping','Stage-boundary bookkeeping','DSP unattributed','Runtime setup/teardown','Embedding','Final model RMSNorm','LM head＋greedy，不含 final norm','Host–DSP 边界','完整 Host wall'];series=[fseries,wseries,data['a1_r10']['prefill']];mr=[];modules=[]
for label,(_,fields) in zip(labels,OVERVIEW):
 vs=[val(s,fields) for s in series];cells=[f'{v:.1f} ({100*v/val(s,["host_us"]):.2f}%)' for v,s in zip(vs,series)];mr.append([label,*cells,f'{100*(vs[1]/vs[2]-1):+.2f}%' if vs[2] else 'N/A']);modules.append(dict(module=label,F16A16_us=vs[0],W4A16_us=vs[1],W4A8_R3_us=vs[2]))
compact=table(['模块','F16A16 EXP0218','W4A16 EXP0166','W4A8 R3 EXP0259','A8相对W4A16增速'],mr);(R/'module_table.md').write_text(compact+'\n');through={};r3cost={}
for key,d in data.items():
 pre=val(d['prefill'],['host_us']);dec=val(d['decode'],['host_us']);model=statistics.median(loops[key]['model_ns']);loop=statistics.median(loops[key]['loop_ns'])
 through[key]=dict(prefill_tokens=64,prefill_host_us=pre,prefill_tokens_per_second=64e6/pre,decode_tokens=15,decode_host_us=dec*15,decode_host_us_per_token=dec,decode_tokens_per_second=1e6/dec,selected_output_tokens=16,complete_model_host_us=model/1000,complete_model_effective_tokens_per_second=16e9/model,generation_loop_us=loop/1000,generation_loop_tokens_per_second=16e9/loop,cold_startup_median_ns={k:statistics.median(t[k] for t in loops[key]['startup']) for k in loops[key]['startup'][0] if k!='record'})
 if key.startswith('a1'):
  r3cost[key]={mode:{k:val(d[mode],[f'dense_r3_total_{k}_ticks']) for k in ['prepare','matmul','finish']} for mode in ['prefill','decode']}
summary=dict(experiment='EXP-0259',execution_state='completed',evidence_validity='valid',local_gate='pass',gate_role='diagnostic_measurement_integrity_only_not_numerical_or_model_acceptance',numerical_eligible=False,numerical_gate='known_ideal_dense_R3_whole_layer_fail_unchanged_exact_implementation_optimization_PC073',speed_eligible=read(R/'full_formal_gate.json')['speed_eligible'],model_quality='not_assessed',baseline_promoted=False,short_rounds=5,formal_rounds=10,timed_full_model_RPCs=calls,formal_RPCs=3520,layer_ledgers_checked=calls*28,actual_layer_count=28,scope='nativepackedW4 28layers embedding norm W4head greedy; M64+15continuousdecode; no audit DDR',recipe='frozen C64 per-output-channel W4; R3_ALL QK; rotatedofflineEOSprefix; denseHMX mode1 OPT2 constant/vector/fusedK/stream; wideNR64',device_PPL=None,throughput=through,performance=read(R/'full_formal_gate.json')['performance'],R3_breakdown_us=r3cost,modules=modules,capture_source_head=read(read(R/'runtime_l28.json')['manifest'])['source_head'],head_lifetime_fix='EXP0258 retained147guardedjoins eachdecode')
write(R/'summary.json',summary);write(R/'numeric_diagnostics.json',dict(data=data,loops=loops));write(R/'independent_integrity_checks.json',dict(pass_all=True,timed_RPCs=calls,layer_ledgers=calls*28,all_selected_sequences_exact=True,R3_calls_28_per_step=True,R3_live_rows_prefill=43008,R3_live_rows_decode=672,all_native_W4_projection_counts_prefill196_decode197=True,all_decode_head_slot_joins147=True,head_mathematical_reference_unchanged=True,physical_and_ledgers=True,independent_statistics_CI_recomputed=True,historical_hashes_verified=20))
intro=['# EXP-0259 optimized dense R3 full-model E2E','','本轮优化当前R3的数据搬运与流水，保持原有denseHMX矩阵乘和所有数值运算。固定C64 per-output-channelW4、R3_ALL Q/K参数、wideNR64、正确旋转的固定EOS U8前缀，其他recipe冻结。既有idealFloat64整层gate仍失败，本轮不作PPL或模型质量验收。','','实现：32KiB显式FP16符号矩阵预打包，逐层HVX复制并计入constant_read_bytes；四行向量转置替代小块标量搬运；R3量化同时形成Koperand；五个worker随Q/K就绪准备FP16输入，QKV结束后join再使用共享HMX。QKV偏置双缓冲移至当时空闲的attention_concat，使gate仅由R3准备worker写入。没有蝶形、重新校准或近似rsqrt替换。','','单层先完成四臂5short10formal；初始repeat1decode置信区间上沿10.96%尚不能确认10%门槛，随后固定补充10轮双组配对，未调参、未删样本，合并20轮满足扩展门槛。原始10轮与补充轮次分别保留；不是事先计划的20轮。单层原始统计见LAYER_REPORT.md，补充见confirmation_gate.json。','','单层108份完整捕获/输出逐字节比较通过，并复现EXP0252参考；独立denseFloat64组件和矩阵布局检查通过。正确前缀的连续三层九步，优化R3及重复输出与EXP0258相应完整捕获逐字节一致；全模型反馈token复现EXP0258。','','以下为新运行时、无R3对照与OPT2 R3候选的真实28层M64+15连续decode、5short10formal配对测量。5280完整模型RPC和147840逐层账本闭合；每次28次R3HMX/672个并行准备heads，8MiB VTCM、零计时中间DDR/spill/audit导出，每token一个RPC。原生W4线性路径和EXP0258 LMhead同步修复保留。','','并行准备work_ticks是worker耗时之和，可能与QKV的DMA/HMX重叠，不能再加到Host或R3串行prepare上。主线程prepare下降同时包含工作被移入流水和冗余被消除；完整QKV与Host wall才是总成本。配对比值与各臂耗时中位数是不同统计量。']
pr=[]
for k,v in summary['performance'].items():pr.append([k,f'{v["control_ns"]/1000:.3f}',f'{v["R3_ns"]/1000:.3f}',f'{100*(v["paired_ratio"]-1):+.2f}%',f'[{100*(v["ci95"][0]-1):+.2f}%,{100*(v["ci95"][1]-1):+.2f}%]'])
intro+=['',table(['阶段（decode为15步合计）','无R3 Host μs','R3 Host μs','配对耗时变化','95%区间'],pr),'','三recipe M64概览：μs（占Host wall）；F16A16 EXP0218和Selected W4A16 EXP0166为非配对历史参照，当前R3仅性能诊断。逐行中位数不必精确相加，原始每次排他账本已检查。','',compact]
costrows=[]
for rep in [1,10]:
 for mode in ['prefill','decode']:
  d=r3cost[f'a1_r{rep}'][mode];costrows.append([f'r{rep} {mode}',*[f'{d[k]:.3f}' for k in ['prepare','matmul','finish']]])
intro+=['','R3主线程子项已包含在QKV阶段；并行prepare工作另在完整字段表中。不能再加到Host总表，decode子项为单token平均。','',table(['阶段','prepare μs','matmul μs','finish μs'],costrows)]
end=['','## 实测E2E','',table(['arm/repeat','prefill64 tok/s','decode15 tok/s','完整模型16输出/Host tok/s','16输出/generation loop tok/s'],[[k,*[f'{v[f]:.3f}' for f in ['prefill_tokens_per_second','decode_tokens_per_second','complete_model_effective_tokens_per_second','generation_loop_tokens_per_second']]] for k,v in through.items()]),'','prefill64包含1个固定EOS；首个输出在prefill内产生，后续15步持续反馈并维护KV。Host包含完整embedding/28层/finalnorm/W4head/greedy/FastRPC，generation loop另含设备Host循环和计时信息序列化。热态速度不含模型加载、ADB和外部tokenizer/detokenizer；冷启动单列保存在summary.json。未做逐层外推、PPL或文本质量验收。']
(R/'REPORT.md').write_text('\n'.join(intro+end)+'\n');full=intro+['','## 完整导出字段','','字段ticks为19.2ticks/μs，其余保留原单位。重叠引擎计数不能相加；缺失字段不补零。']
for rep in [1,10]:
 for mode in ['prefill','decode']:
  ds=[data[f'a{a}_r{rep}'][mode] for a in [0,1]];keys=set.intersection(*(set(x) for d in ds for x in d));rr=[]
  for k in sorted(keys):
   vs=[statistics.median(x[k] for x in d) for d in ds];rr.append([k,*[f'{v:.6f}' for v in vs],f'{vs[1]-vs[0]:+.6f}'])
  full+=['',f'### r{rep} {mode}','',table(['field','noR3','R3','delta'],rr)]
(R/'FULL_PROFILING_REPORT.md').write_text('\n'.join(full+end)+'\n')
print(json.dumps(dict(throughput=through,performance=summary['performance'],R3_breakdown_us=r3cost,speed_eligible=summary['speed_eligible']),indent=2))
