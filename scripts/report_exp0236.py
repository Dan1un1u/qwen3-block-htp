#!/usr/bin/env python3
"""Verify stage provenance and close the fixed W4-only head experiment."""
import hashlib,json,math,subprocess,tarfile
from pathlib import Path
import numpy as np
from data_exp0236 import RESULT,OUTPUT,SOURCE,MEMORY,GROOT,CHECKPOINT,write,sha,verified,preflight,frozen
from head_exp0236 import head_read

def streamsha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def main():
 preflight();frozen();assert not (RESULT/'closure.json').exists()
 assert json.loads((RESULT/'independent_data_audit.json').read_text())['pass_all']
 assert json.loads((RESULT/'quantizer_oracles.json').read_text())['pass_all']
 audit=json.loads((RESULT/'head_state_audits.json').read_text());assert audit['pass_all'] and audit['only_head_changed'] and audit['final_G64_sentinel_exact'] and len(audit['audits'])==4
 baseline=json.loads((RESULT/'G64_state_manifest.json').read_text())['state_hashes'];packages={}
 for v in ['P64','H64']:
  p=json.loads((RESULT/v/'package.json').read_text());root=OUTPUT/v;assert sha(root/'manifest.json')==p['manifest_sha256'];packages[v]=p
  assert p['W4'] and not p['mixed_precision'] and p['only_changed_tensor']=='lm_head.weight' and p['independent_numpy_all_rows_exact']
  for n,r in p['files'].items():assert streamsha(root/n)==r['sha256'] and (root/n).stat().st_size==r['bytes']
  q=head_read(root,v=='H64');assert hashlib.sha256(q.tobytes()).hexdigest()==p['dequant_FP16_sha256'];del q
  r=np.load(root/'selection.npz');assert r['choice'].shape==(151936,)
  assert np.array_equal(r['choice'],r['candidate_output_sse'].argmin(1));assert np.array_equal(r['selected_output_sse'],r['candidate_output_sse'][np.arange(151936),r['choice']])
  assert np.isfinite(r['scale']).all() and (r['scale']>0).all()
 for a in audit['audits']:
  expected=dict(baseline)
  if a['variant']!='G64':expected['lm_head.weight']=packages[a['variant']]['dequant_FP16_sha256']
  assert a['state_hashes']==expected and a['all_parameter_and_buffer_hashes_match']
 sentinel=json.loads((RESULT/'G64_final_sentinel.json').read_text());assert sentinel['samples']==json.loads((RESULT/'software/development_G64.json').read_text())['samples']
 replay=json.loads((RESULT/'head_input_replay.json').read_text());checks=[replay['independent_numpy_final_norm'],*replay['canonical_CPU_replay']]
 assert all(c['finite'] and c['nrmse']<=.003 and c['cosine']>=.99999 for c in checks)
 inp=json.loads((RESULT/'head_inputs.json').read_text());assert streamsha(inp['path'])==inp['sha256']
 summaries={phase:json.loads((RESULT/f'summary_{phase}.json').read_text()) for phase in ['development','final','PC052']}
 for sm in summaries.values():
  assert sm['independent_raw_PPL_reductions']==36
  for n,h in sm['inputs'].items():assert streamsha(RESULT/n)==h
 # Verify all immutable G64 files again, including every inherited non-transformer file.
 gm=json.loads((GROOT/'manifest.json').read_text())
 for n,e in gm['files'].items():assert streamsha(GROOT/n)==e['sha256']
 for n,h in gm['inherited_files'].items():assert streamsha(Path(gm['frozen_base_package'])/n)==h
 assert streamsha(CHECKPOINT)==replay['checkpoint_sha256']
 commands=[]
 for p in sorted((RESULT/'commands').glob('*.json')):
  c=json.loads(p.read_text());assert streamsha(p.with_suffix('.log'))==c['log_sha256'];archive=OUTPUT/'artifacts'/c['source_head']/'source.tar';assert streamsha(archive)==c['source_archive_sha256']
  assert c['returncode']==0,c
  with tarfile.open(archive) as a:
   script=next(Path(x).name for x in c['command'] if x.endswith('.py'));entry=hashlib.sha256(a.extractfile('scripts/'+script).read()).hexdigest()
   for v,m in packages.items():
    if script=='head_exp0236.py' and c['command'][-1]==v:
     assert c['source_head']==m['source_head']
     for n,h in m['quantizer_source'].items():assert hashlib.sha256(a.extractfile('scripts/'+n).read()).hexdigest()==h
   if script=='evaluate_exp0236.py':
    assert c['source_head']==json.loads((RESULT/'environment.json').read_text())['source_head']
  commands.append(dict(command=c['command'],source_head=c['source_head'],entry_script_sha256=entry,returncode=c['returncode'],elapsed_s=c['elapsed_s']))
 assert len(commands)==len(list((RESULT/'commands').glob('*.log')))
 write('stage_execution_provenance.json',dict(commands=commands,all_source_archives_verified=True))
 import platform,sys,torch,transformers
 write('environment_at_closure.json',dict(cpu_python=sys.version,cpu_torch=torch.__version__,cpu_transformers=transformers.__version__,cpu_numpy=np.__version__,platform=platform.platform(),gpu= json.loads((RESULT/'environment.json').read_text()),CPU_export_threads=16,CPU_environment_role='same_unchanged_environment_path_as_retained_export_commands_at_closure'))
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip();archive=OUTPUT/'artifacts'/head/'source.tar';archive.parent.mkdir(parents=True,exist_ok=True)
 if not archive.exists():subprocess.run(['git','archive','--format=tar','-o',str(archive),head],cwd=SOURCE,check=True)
 final=summaries['final'];old=summaries['PC052'];dev=summaries['development']
 report=['# EXP0236：G64 上的 W4 LM head 精度修复','', '只改变 LM head；G64 的196个 transformer 投影、embedding、所有norm与RoPE保持原值。P64/H64均为W4，没有FP16/W8恢复或混合精度。F16F16/W4U8和DSP运行时冻结，无基线晋升。', '', '## 新独立测试（1024文档，16384目标token）','', '|方案|LM head|PPL|相对F16增幅|验收|','|---|---|---:|---:|---|']
 labels={'F':'FP16 teacher','G64':'旧per-channel W4 / EXP221 8K','P64':'per-channel W4 / G64输入64K + output-aware GPTQ','H64':'group128 W4 / G64输入64K + output-aware GPTQ'}
 for v in ['F','G64','P64','H64']:
  g=final['statistics']['overall'];ratio=1 if v=='F' else g['vs_F16'][v]['ppl_ratio'];report.append(f"|{v}|{labels[v]}|{g['ppl'][v]:.6f}|{100*(ratio-1):+.6f}%|{final['status'].get(v,'reference')}|")
 report+=['', 'G64命名中的64指65536校准token；其transformer为group128。旧head未随其扩展校准。本次P64/H64是重新量化head的两项固定干预；P64同时更新校准分布/预算及output-aware范围规则，不能把差异只归因于校准量。P64与H64匹配输入及求解规则，格式分组不同。', '', '|分层|F|G64|P64|H64|','|---|---:|---:|---:|---:|']
 for name,g in final['statistics'].items():report.append('|'+name+'|'+ '|'.join(f"{g['ppl'][v]:.6f}" for v in ['F','G64','P64','H64'])+'|')
 for v in ['P64','H64']:
  g=final['statistics']['overall'];ratio=g['vs_G64'][v];f=g['vs_F16'][v]
  report+=['',f"{v}/G64 PPL ratio={ratio['ppl_ratio']:.8f}, paired95%CI={ratio['ratio_ci95']}; {v}/F={f['ppl_ratio']:.8f}, CI={f['ratio_ci95']}. 全部分层状态：{final['status'][v]}。"]
 report+=['', '固定总体5%、每语言/领域/交叉分层10%门槛；点估计失败即fail，全部点估计及上置信界通过才pass，其余inconclusive。配对分层文档bootstrap5000，seed236；区间为点区间，非两候选同时多重比较保证。新测试独立于所有此前训练/校准/开发/评测文档、文本哈希及32-token片段，通过逐窗口原文重建。1024文档一次固定预算，不追加样本或按PPL再调参。这是M64+16短上下文条件PPL的软件验证，不是长上下文公开benchmark或DSP验收。', '', '## 同一历史PC052面板的配对回归','', '此面板已暴露，只作披露的回归对比；不能称为本轮新的独立测试。所有四项使用相同token/FP16后端。', '', '|方案|PPL|相对F16增幅|','|---|---:|---:|']
 for v in ['F','G64','P64','H64']:
  g=old['statistics']['overall'];ratio=1 if v=='F' else g['vs_F16'][v]['ppl_ratio'];report.append(f"|{v}|{g['ppl'][v]:.6f}|{100*(ratio-1):+.6f}%|")
 report+=['', 'EXP230开发集PPL（仅回归）：'+str(dev['statistics']['overall']['ppl']), '', '## 实现和证据', '', '原始FP32 head重新加载；复用经EXP234账本验证的末层hidden，并应用实际FP16 final RMSNorm，65536输入均用于Hessian和三候选投影SSE。四窗口完整canonical CPU回放NRMSE:'+str([c['nrmse'] for c in replay['canonical_CPU_replay']])+ '。这是允许误差内的FP16缓存路径一致性，非逐bit相同。', '', '稳定全局act-order，damping.01，FP64分解后FP32；signed[-7,7]。absmax/midpoint/weight-L2.4三次完整GPTQ，按每输出行的最终FP16投影SSE选择，tie取第一项。H64分组固定于原输入列，每128列一个FP32 scale；未使用LPBQ、二次scale量化、旋转、gamma折叠或学习权重。']
 for v,m in packages.items():report+=['',f"{v}：选择直方图{m['choice_histogram']}；head编码字节{m['files']['codes.bin']['bytes']}，scale字节{m['files']['scales.bin']['bytes']}；纯CPU导出{m['elapsed_s']:.1f}s（不是推理速度）。源码{m['source_head']}，manifest SHA256 {m['manifest_sha256']}。"]
 report+=['', '两个151936x2048 head全部通过独立NumPy nibble重排/scale展开，与导出FP16逐元素相等；量化Schur消元、clipping与SSE选择独立检查通过。完整模型参数/缓冲区hash验证仅head改变；所有新score repeat/mask/finite/CE<5e-6通过，F/G64开发集逐token NLL/top1与旧证据完全一致，恢复旧head后的G64 sentinel也完全一致。108个分层PPL独立raw-token math.fsum重算；源码、命令、产物及结果账本均保留。', '', '## 边界与下一步','', '本轮完成后讨论结果；没有自动继续MLP、混合精度、W4A8或DSP部署。新group/head格式的物理成本和端侧速度尚未测量。E2E token/s：N/A；完整profiling记录保留EXP230历史引用，不能套用于本轮新模型。']
 with (RESULT/'REPORT.md').open('x') as f:f.write('\n'.join(report)+'\n')
 oldprofile=verified('exp0230','full_profiling_report.md');table=verified('exp0230','module_table.md')
 profile=['# EXP0236 complete profiling record — host-only W4 head diagnostic','',f'Source {head}; evidence {RESULT}; models {OUTPUT}. Controls F/G64; candidates P64/H64. Repeat1/repeat10 and rotated5short/10formal rounds N/A: no new device execution.','', '|Required section|Repeat1 control/candidate/delta|Repeat10 control/candidate/delta|Reason|','|---|---|---|---|']
 for section in ['Complete Host wall / prefill / continuous decode','DSP invocation / setup / teardown / additive ledger / unattributed','QKV / RMSNorm / RoPE / QK / Softmax / AV / O','Residual / PostNorm / GateUp / SwiGLU / Down / final residual','Embedding / final RMSNorm / LM head / greedy','DMA bytes / descriptors / waits / HMX / HVX / overlap / workers','VTCM requested / granted / peak / spill / lifetime / FastRPC','Device hashes / mismatches / maximum LSB / physical gates']:
  profile.append(f'|{section}|N/A|N/A|No grouped-head DSP implementation or measurement|')
 profile+=['','No missing value is a measured zero. Overlapping counters would not be additive. Software correctness and quality are documented in REPORT.md; PyTorch elapsed time is not device throughput.','',f'## Historical stable three-recipe table\n\nVerified EXP230 historical report {oldprofile}, SHA256 {sha(oldprofile)}. The following exact table is historical; C64 is not G64/P64/H64 and not automatically promoted. F16F16/W4U8 are nonpairedEXP218 references. W4 bytes differ; no activation-only attribution.', '',table.read_text(), '', '## End-to-end throughput','', 'New G64/P64/H64 prefill/decode: N/A. Historical EXP230 C64 only:64tokens/63262.995us=1011.649859tok/s;15continuousdecode tokens/1389448.9595us=10.795647tok/s. No extrapolation.']
 with (RESULT/'full_profiling_report.md').open('x') as f:f.write('\n'.join(profile)+'\n')
 artifacts={str(p.relative_to(OUTPUT)):dict(sha256=streamsha(p),bytes=p.stat().st_size) for p in sorted(OUTPUT.rglob('*')) if p.is_file()}
 write('artifacts_sha256.json',dict(root=str(OUTPUT),files=artifacts))
 closure=dict(experiment='EXP-0236',execution_state='completed',evidence_validity='valid',source_head=head,quality_status=final['status'],final_ppl=final['statistics']['overall']['ppl'],PC052_ppl=old['statistics']['overall']['ppl'],dataset_freeze_sha256=sha(RESULT/'dataset_freeze.json'),manifest_sha256={v:m['manifest_sha256'] for v,m in packages.items()},report_sha256=sha(RESULT/'REPORT.md'),full_profiling_report_sha256=sha(RESULT/'full_profiling_report.md'),artifact_ledger_sha256=sha(RESULT/'artifacts_sha256.json'),artifact_files=len(artifacts),independent_PPL_reductions=108,only_head_changed=True,mixed_precision=False,baseline_promoted=False,device_speed='N/A',next_direction='discussion_only')
 write('closure.json',closure)
 write('evidence_sha256.json',{str(p.relative_to(RESULT)):streamsha(p) for p in sorted(RESULT.rglob('*')) if p.is_file() and p.name!='evidence_sha256.json'})
 print('EXP236_REPORT_CLOSED',json.dumps(closure),flush=True)
if __name__=='__main__':main()
