#!/usr/bin/env python3
"""Read-only EXP0257 raw-evidence audit and report. Run after formal capture."""
import sys,json,hashlib,subprocess,statistics,time
from pathlib import Path
S=Path('/home/daniuniu/work/qwen3-block-htp');sys.path.insert(0,str(S/'scripts'))
from measure_exp0218 import LEDGER,OVERVIEW
from summarize_exp0217 import normalized
import numpy as np
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0257');B=R.parent;M=Path(str(S)+'-project-memory')
def read(p):return json.loads(Path(p).read_text())
def write(n,z):
 with (R/n).open('x') as f:json.dump(z,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def records(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.startswith('{')]
def table(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','|'+'|'.join(['---']*len(h))+'|']+['| '+' | '.join(str(v) for v in row)+' |' for row in rows])
def val(series,fields):return statistics.median(sum(p[k] if k.endswith('_us') else p[k]/19.2 for k in fields) for p in series)
subprocess.run(['python3',str(M/'scripts/project_memory.py'),'preflight','--source-worktree',str(S)],check=True)
assert read(R/'full_model_gate.json')['pass_all'] and read(R/'seed_slice_gate.json')['pass_all']
layer_fields=['metadata_stage_ticks','input_stage_ticks','input_norm_ticks','qkv_projection_ticks','qk_norm_rope_ticks','attention_ticks','o_projection_ticks','post_attention_residual_ticks','post_attention_norm_ticks','gate_up_ticks','activation_ticks','down_ticks','final_residual_ticks','cache_append_pack_ticks','cache_append_dma_ticks','block_orchestration_ticks','layer_bookkeeping_ticks','layer_unattributed_ticks']
data={};timings={};calls=0;token_failures=[];partial_perf={}
for phase,n in [('short',5),('formal',7)]:
 gate=read(R/'short_gate.json') if phase=='short' else None
 if gate:assert gate['pass_all']
 rows=[];rng=np.random.default_rng(257)
 for i in range(1,n+1):
  for repeat in [1,10]:
   for wide in [0,1,4]:
    p=R/f'{phase}/round{i:02d}_r{repeat}_w{wide}'
    if not (p/'stdout.jsonl').exists():continue
    rs=records(p/'stdout.jsonl');ps=[x for x in rs if x.get('record')=='generation_profile'];fs=[x for x in rs if x.get('generation_sequence_complete')]
    assert len(ps)==16*repeat and len(fs)==repeat
    gold=read(R/({0:'speed_anchor',1:'seed_full_sole',4:'seed_full_nr64'}[wide])/'validated.json')['token_ids'][:16]
    for j,f in enumerate(fs):
     assert f['all_steps_pass']
     if f['token_ids']!=gold:token_failures.append(dict(path=str(p),sequence_index=j,expected=gold,actual=f['token_ids'],mismatching_steps=[k for k,(a,b) in enumerate(zip(gold,f['token_ids'])) if a!=b]))
     assert f['total_host_wall_ns']==sum(q['host_wall_ns'] for q in ps[j*16:(j+1)*16])
     assert f['generation_loop_wall_ns']>=f['total_host_wall_ns']
    for j,q in enumerate(ps):
     step=j%16;assert q['generation_step']==step and q['wide_score_mode']==wide and q['prefix_kv_mode']==(wide!=0)
     assert q['prefix_group_patch_count']==(224 if wide and step==0 else 0)
     assert q['block_invocation_count']==28 and q['vtcm_acquired_bytes']==q['vtcm_requested_bytes']==8388608
     assert q['boundary_ddr_write_bytes']==q['intermediate_ddr_read_bytes']==q['intermediate_ddr_write_bytes']==q['intermediate_spill_fill_count']==q['ledger_unattributed_ticks']==0
     assert sum(normalized([q])[k] for _,k in LEDGER)==q['invocation_ticks']
     for layer in range(28):
      t=q[f'slice_layer_{layer}'];assert t['status']==3 and t['layer_index']==layer
      assert sum(t[k] for k in layer_fields)==t['layer_ticks']
      assert t['hidden_ddr_read_bytes']==t['hidden_ddr_write_bytes']==t['layer_unattributed_ticks']==0
      assert t['cache_valid_before']==(0 if step==0 else 63+step) and t['cache_valid_after']==64+step
    calls+=len(ps)
    d=dict(round=i,repeat=repeat,wide=wide,prefill_ns=sum(q['host_wall_ns'] for q in ps[::16])/repeat,decode_ns=sum(q['host_wall_ns'] for j,q in enumerate(ps) if j%16)/repeat)
    if (p/'validated.json').exists():
     v=read(p/'validated.json');assert d['prefill_ns']==v['prefill_ns'] and d['decode_ns']==v['decode_ns']
    rows.append(d)
    if phase=='formal' and i<=6:
     key=f'w{wide}_r{repeat}';data.setdefault(key,{'prefill':[],'decode':[]})
     for mode in ['prefill','decode']:data[key][mode].append(normalized([q for q in ps if q['mode']==mode]))
     ts=timings.setdefault(key,{'loop_ns':[],'model_ns':[],'cold_startup':[]})
     ts['loop_ns'].append(statistics.mean(f['generation_loop_wall_ns'] for f in fs));ts['model_ns'].append(statistics.mean(f['total_host_wall_ns'] for f in fs));ts['cold_startup'].append(next(q for q in rs if q.get('record')=='generation_startup'))
 count=5 if phase=='short' else 6
 for control in [0,1]:
  for repeat in [1,10]:
   for mode in ['prefill_ns','decode_ns']:
    pairs=np.array([[next(r[mode] for r in rows if r['round']==i+1 and r['repeat']==repeat and r['wide']==w) for w in [control,4]] for i in range(count)])
    ratios=pairs[:,1]/pairs[:,0];ci=np.quantile(np.median(ratios[rng.integers(0,count,(10000,count))],axis=1),[.025,.975])
    if phase=='short':
     g=gate['performance'][f'c{control}_r{repeat}_{mode}'];assert abs(float(np.median(ratios))-g['paired_ratio'])<1e-12 and np.max(abs(ci-g['ci95']))<1e-12
    else:partial_perf[f'c{control}_r{repeat}_{mode}']=dict(control_ns=float(np.median(pairs[:,0])),candidate_ns=float(np.median(pairs[:,1])),paired_ratio=float(np.median(ratios)),ci95=ci.tolist(),diagnostic_only=True)
assert calls==6016 and len(token_failures)==1 and token_failures[0]['sequence_index']==7
write('formal_interruption.json',dict(reason='original-speed anchor transient output nondeterminism; not parser/hash mismatch',formal_gate='fail',complete_rounds=6,partial_round=7,token_failures=token_failures,performance_first6_diagnostic=partial_perf,failed_round_retained=True,formal_resumption_not_authorized_without_attributable_repair=True))
# Historical raw measurements are verified through authority-pinned closure ledgers.
p=B/'exp0218/evidence_sha256.json';assert sha(p)=='1d8f8e5eeeac7ea40a747698d71f8b78f58a9aa501512a4721926d262d2259e7';h18=read(p)
p=B/'exp0217/closure_evidence_sha256.json';assert sha(p)=='c1ac7ac45924c34dd3b37181689182366eb959c33fa83e73aa715385e3d37fcf';h17=read(p)
p=B/'exp0217/formal/historical_evidence_sha256.json';assert sha(p)==h17['formal/historical_evidence_sha256.json']['sha256'];h16=read(p)
historical={};fseries=[];wseries=[]
for i in range(1,11):
 p=B/f'exp0218/formal/round_{i:02d}_f16f16.jsonl';assert sha(p)==h18[str(p.relative_to(B/'exp0218'))];historical[str(p)]=sha(p);fseries.append(normalized([q for q in records(p) if q.get('record')=='generation_profile' and q['mode']=='prefill']))
 p=B/f'exp0166/20260903T_exp0166_8e0dcf8_formal/raw/round_{i:02d}_candidate.jsonl';assert sha(p)==h16[str(p)];historical[str(p)]=sha(p);wseries.append(normalized([q for q in records(p) if q.get('record')=='generation_profile' and q['mode']=='prefill']))
write('historical_provenance.json',historical)
chinese=['I/O、metadata','Input RMSNorm','QKV＋Q/K Norm-RoPE','QK–Softmax–AV','O projection','Post-attention residual＋RMSNorm','Gate/Up＋SwiGLU','Down','Final residual','KV carrier conversion','KV append DMA','Block orchestration','Layer bookkeeping','Stage-boundary bookkeeping','DSP unattributed','Runtime setup/teardown','Embedding','Final model RMSNorm','LM head＋greedy，不含 final norm','Host–DSP 边界','完整 Host wall']
series=[fseries,wseries,data['w4_r10']['prefill']];module_rows=[];modules=[]
for label,(_,fields) in zip(chinese,OVERVIEW):
 values=[val(s,fields) for s in series];cells=[f'{v:.1f} ({100*v/val(s,["host_us"]):.2f}%)' for v,s in zip(values,series)];delta=f'{100*(values[1]/values[2]-1):+.2f}%' if values[2] else 'N/A'
 module_rows.append([label,*cells,delta]);modules.append(dict(module=label,F16A16_us=values[0],W4A16_us=values[1],W4A8_us=values[2]))
compact=table(['模块','F16A16 EXP0218','W4A16 EXP0166','W4A8 EXP0257','A8相对W4A16增速'],module_rows);(R/'module_table.md').write_text(compact+'\n')
# Decode frozen semantic traces; stop at first EOS, raw fixed64 budget remains intact.
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained('/mnt/d/llm_exp/models/Qwen3-origin',local_files_only=True)
prompts=read(R/'prompts.json');assert sha(R/'prompts.json')=='cbc3e656a49041620cb7ca7e0d3cd1c9805b5cdcd9e5ef9c36d427869a3ce073'
control=read(R/'W4A16_prompt_control.json');assert control['weights_unchanged'] and control['prompts_sha256']==sha(R/'prompts.json')
reasons=['Repeated deliberation/question, no Paris answer','Repetitive unrelated Chinese, no21','Question echo and repetition, no Beijing answer','Off-topic divisibility discussion, no21','Repeated numbering/formatting, no explanation','Question echo before first EOS; later continuation excluded']
texts=[]
for i,p in enumerate(prompts['samples']):
 tag='seed_full_nr64' if i==0 else f'seed_text_sample{i}';ids=read(R/tag/'validated.json')['token_ids'];assert len(ids)==64
 end=next((j for j,x in enumerate(ids) if x in [151645,151643]),len(ids));text=tokenizer.decode(ids[:end],skip_special_tokens=False)
 c=control['samples'][i];assert c['id']==p['id']
 texts.append(dict(id=p['id'],prompt=p['prompt'],rendered=p['rendered'],prompt_token_ids=p['token_ids'],raw_token_ids=ids,raw_text=tokenizer.decode(ids,skip_special_tokens=False),semantic_token_count=end,eos_index=end if end<len(ids) else None,semantic_text=text,usable=False,reason=reasons[i],W4A16_software_control=c,W4A16_content_usable=True))
write('text_outputs.json',dict(samples=texts,A8_usable=0,W4A16_software_usable=6,total=6,metric='manual content-aware auxiliary usability; formatting-only issues ignored; not PPL acceptance'))
frontend=[]
for i in range(51):
 t=time.perf_counter_ns();ids=tokenizer.encode(prompts['samples'][0]['rendered'],add_special_tokens=False);u=time.perf_counter_ns();tokenizer.decode(texts[0]['raw_token_ids'],skip_special_tokens=False);v=time.perf_counter_ns();assert ids==prompts['samples'][0]['token_ids'][1:]
 if i:frontend.append([u-t,v-u])
throughput={}
for key,d in data.items():
 pre=val(d['prefill'],['host_us']);dec=val(d['decode'],['host_us']);loop=statistics.median(timings[key]['loop_ns']);model=statistics.median(timings[key]['model_ns'])
 throughput[key]=dict(prefill_tokens=64,prefill_host_us=pre,prefill_tokens_per_second=64e6/pre,decode_tokens=15,decode_host_us_per_token=dec,decode_tokens_per_second=1e6/dec,complete_selected_tokens=16,model_16pass_wall_us=model/1000,model_effective_output_tokens_per_second=16e9/model,generation_loop_us=loop/1000,generation_loop_output_tokens_per_second=16e9/loop,cold_startup_median_ns={k:statistics.median(t[k] for t in timings[key]['cold_startup']) for k in timings[key]['cold_startup'][0] if k!='record'})
summary=dict(experiment='EXP-0257',execution_state='aborted',evidence_validity='valid',local_gate='fail',gate_role='text_unusable_and_formal_anchor_nondeterminism; candidate_component_checks_pass',implementation_gate='pass',physical_gate='pass',speed_gate='incomplete_formal_control_nondeterminism',text_usability_gate='fail',A8_usable=0,W4A16_software_usable=6,text_samples=6,baseline_promoted=False,recipe='C64 per-output-channel W4 / original OFF A8 / offline fixed EOS W4A16-to-U8 prefix / wide NR64 / no R3',device='PJZ110 SM8750 HTP V79',scope='actual28layers embedding finalnorm nativeW4 head greedy, M64+15 speed, M64+63 text/cache',short_rounds=5,formal_rounds=6,formal_partial_round=7,timed_full_model_RPCs=calls,formal_RPCs=3376,layer_ledgers_checked=calls*28,performance=partial_perf,throughput=throughput,modules=modules,frontend=dict(platform='WSL CPU host; separate from Android; tokenizer loaded and50warm repetitions',encode_median_ns=statistics.median(x[0] for x in frontend),decode64_median_ns=statistics.median(x[1] for x in frontend)),device_PPL=None,source_head_at_capture='65915ede3ec216214047a3d78b40ab6552d3a639',logits_grid=dict(scale=.5,zero_point=128,software_PPL_reference='floating logits; not same head arithmetic'),next_action='discuss_device_software_boundary_attribution_including_native_U8_logits_before_new_algorithm')
write('summary.json',summary);write('numeric_diagnostics.json',dict(data=data,timings=timings));write('independent_integrity_checks.json',dict(pass_all=True,timed_RPCs=calls,formal_RPCs=3376,layer_ledgers=calls*28,statistics_CI_independently_recomputed=True,formal_speed_not_accepted=True,all_timed_token_sequences_exact=False,token_failures=token_failures,all_layer_cache_lengths_and_ledgers_exact=True,historical_raw_hashes_verified=20))
intro=['# EXP-0257：完整模型真机文本完成，正式速度验证中断','', '完整28层真机文本已经执行；候选组件/切片/独立head检查与五轮短测通过，但十轮正式profiling在第七轮因原始A8对照不确定输出停止。修复后的A8文本仍不可用：六条固定样本0/6，同提示词冻结W4A16软件对照6/6（按内容判定，忽略格式差异）。这不是新的PPL验收，不晋升基线。', '', summary['recipe'], '', '本次补齐此前硬件移植遗漏的固定EOS前缀：离线用原始C64 W4A16计算一次K/V，存成56KiB U8种子，设备预填充覆盖row0。无在线FP16运行、权重更新、分组量化或R3。初次无种子的坏输出保留为移植未完成的诊断，不混入正式候选。', '', '独立证据：单层20文件精确复现、三层连续执行和缓存切换、种子单列快补与完整K重打包精确一致、实际QK/概率/AV对独立整数参考逐字节一致。64步完整模型NR64标量/HVX token完全一致；三步独立LMhead算术/greedy对照通过。这证明已检查范围的实现一致性，不证明全模型每个张量等价于teacher或软件浮点模型。', '', '5轮短测完成；正式测试完成6轮，第7轮repeat10原始对照的第8条序列在step10/11/14生成token不一致后停止。全部已采集6016个完整模型RPC和168448个逐层计时账本均闭合，但token确定性gate失败，8MiB VTCM，零计时中间DDR/spill，每次完整模型一个RPC。repeat10是复用已加载权重后重建10个独立会话；每会话均为M64 prefill加15步真实token反馈decode。', '', '速度主对照C0：原始整数attention、无种子；B：宽差值SOLE加正确种子；A：宽差值NR64加正确种子。A/C0计入全部修复开销，A/B用于归一化归因。下表仅为失败发生前6个完整轮次的诊断统计，不能替代预定10轮正式验收；失败原始文件未丢弃。五轮短测速度门槛通过，正式速度结论不予通过。']
perfrows=[]
for k,v in summary['performance'].items():perfrows.append([k,f'{100*(v["paired_ratio"]-1):+.3f}%',f'[{100*(v["ci95"][0]-1):+.3f}%, {100*(v["ci95"][1]-1):+.3f}%]'])
intro+=['',table(['比较与阶段','Host wall变化','配对95%区间'],perfrows),'','M64模块单位μs，括号占各自Host wall。F16A16 EXP0218和Selected W4A16 EXP0166为非配对历史参照；当前A8取失败前6个完整轮次repeat10中位数，仅作已测速度展示。各行分别取中位数，因此舍入前也不必恰好相加；原始调用账本均逐条闭合。','',compact]
report=intro+['','## 文本检查','', '六题包含中英文首都、12+9及植物光合作用。为固定总M64，提示词含重复的简短回答指令；提示词在输出前冻结，同样提示词W4A16对照可用。不给只有格式差异的内容扣分：W4A16英文解释只有一句也按内容可用。A8问题为语义缺失/循环。原始64token固定预算用于缓存测试；文本仅取首次EOS之前，EOS之后不能作为正确答案。']
for x in texts:report+=['', '### '+x['id'],'',x['prompt'],'','A8：','```text',x['semantic_text'],'```','判定：'+x['reason'],'','W4A16软件对照：','```text',x['W4A16_software_control']['text'],'```']
report+=['','## 解释边界与后续讨论','','此结果已是完整模型设备文本验证，不再是单block速度外推。正式测试先因原始A8速度对照的偶发不确定输出中断；短测和已完成候选运行一致不代表该共用运行时隐患已排除。须定位并修复后重新做完整正式测试。当前不能仅凭组件通过，将全模退化完全归于模型量化本身。设备head使用U8 logits(scale0.5/zero128)，此前软件PPL保留浮点logits；还存在完整设备HMX与软件仿真边界差异。下一步先定位原始速度对照的瞬态数值不一致，再做同token teacher-forcing设备/软件逐层对齐和输出head消融，区分transformer累积误差、归一化/转换及logit网格影响；本轮不擅自改变算法。无新设备PPL。']
through_rows=[[key,f'{v["prefill_tokens_per_second"]:.3f}',f'{v["decode_tokens_per_second"]:.3f}',f'{v["model_effective_output_tokens_per_second"]:.3f}',f'{v["generation_loop_output_tokens_per_second"]:.3f}'] for key,v in throughput.items()]
end=['','## 实测E2E速度','',table(['arm/repeat','prefill64 tok/s','decode15 tok/s','16输出/模型总Host tok/s','16输出/完整generation loop tok/s'],through_rows),'','prefill分子64包含1个固定EOS种子位置；首个输出已在prefill产生，decode分子15。模型Host包含embedding/28层/finalnorm/head/greedy及Host-DSP边界；generation loop另含设备Host循环和profiling序列化。两者均为实际整模型计时，无逐层外推。上述热态速度不含模型加载、ADB、WSL tokenizer/detokenizer；这些不能混合平台后伪称设备推理耗时。cold startup和独立WSL前后处理实测值保存在summary.json。']
(R/'REPORT.md').write_text('\n'.join(report+end)+'\n')
profile=intro+['','## 全量导出诊断','','字段*_ticks为19.2ticks/μs；其它保留原单位，均为每模型调用平均后跨轮取中位数。底层计数器可能重叠，不能相加作Host；排他账本单独已逐调用校验。缺失字段不补零。']
for repeat in [1,10]:
 for mode in ['prefill','decode']:
  sets=[data[f'w{w}_r{repeat}'][mode] for w in [0,1,4]];keys=set.intersection(*(set(x) for s in sets for x in s));rows=[]
  for k in sorted(keys):
   vs=[statistics.median(x[k] for x in s) for s in sets];rows.append([k,*[f'{v:.6f}' for v in vs],f'{vs[2]-vs[0]:+.6f}',f'{vs[2]-vs[1]:+.6f}'])
  profile+=['',f'### repeat{repeat} {mode}','',table(['field','C0','B','A','A-C0','A-B'],rows)]
(R/'FULL_PROFILING_REPORT.md').write_text('\n'.join(profile+end)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k in ['A8_usable','speed_gate','throughput','performance']},indent=2))
