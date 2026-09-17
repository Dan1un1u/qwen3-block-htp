"""Summarize EXP0289 provisional speed evidence without promoting numerical validity."""
import sys,json,statistics
from pathlib import Path
sys.path.insert(0,'/home/daniuniu/work/qwen3-block-htp/scripts')
from common_exp0289 import *
from measure_exp0218 import LEDGER,OVERVIEW
from summarize_exp0217 import normalized,TICKS
from device_exp0289 import records
LABELS=['I/O、metadata','Input RMSNorm','QKV＋Q/K Norm-RoPE','QK–Softmax–AV',
'O projection','Post-attention residual＋RMSNorm','Gate/Up＋SwiGLU','Down',
'Final residual','KV carrier conversion','KV append DMA','Block orchestration',
'Layer bookkeeping','Stage-boundary bookkeeping','DSP unattributed','Runtime setup/teardown',
'Embedding','Final model RMSNorm','LM head＋greedy，不含 final norm','Host–DSP 边界','完整 Host wall']
def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
  ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'
def main():
 f=read(R/'speed-formal.json');assert f['pass_all'] and len(f['runs'])==20
 short=read(R/'speed-short.json');assert short['pass_all']
 series={}
 for item in f['runs']:
  ps=[v for v in records(R/item['tag']/'stdout.jsonl') if v.get('record')=='generation_profile']
  assert len(ps)==430
  for mode in ['prefill','decode']:
   series.setdefault(item['recipe']+':'+mode,[]).append(normalized([p for p in ps if p['mode']==mode]))
 means={k:{n:statistics.mean(v[n] for v in vs) for n in vs[0]} for k,vs in series.items()}
 hist=read(R.parent/'exp0288/MODULES.json')['means']
 modules={}
 for mode in ['prefill','decode']:
  vs=[means['f16f16:'+mode],means['w4f16:'+mode],hist['OPT1:'+mode]]
  rows=[]
  for label,(_,fields) in zip(LABELS,OVERVIEW):
   values=[sum(v[k] if k.endswith('_us') else v[k]/TICKS for k in fields) for v in vs]
   cells=[f'{a:.2f} ({100*a/v["host_us"]:.2f}%)' for a,v in zip(values,vs)]
   rows.append([label,*cells,f'{100*(values[1]/values[2]-1):+.2f}%' if values[2] else 'N/A'])
  modules[mode]=table(['模块','W16A16','W4A16','W4A8-SP2（历史）','SP2 相对 W4A16'],rows)
 summary=dict(experiment='EXP-0289',shape='M64+42, cache128, batch1, full28',
  measured_source=read(R/'full-f16-a01/protocol.json')['runtime']['seal']['source_head'],
  short_rounds=5,formal_rounds=10,repeat=10,formal_profiles=8600,
  numerical_alignment_pass=False,hardware_invariants_pass=True,head_boundary_pass=True,
  timed_token_logit_repeat_exact=True,baseline_promoted=False,quality_claim=False,recipes={})
 e2e=[]
 for recipe in ['f16f16','w4f16']:
  a=means[recipe+':prefill'];b=means[recipe+':decode']
  summary['recipes'][recipe]=dict(prefill_tps=64e6/a['host_us'],decode_tps=1e6/b['host_us'],
   prefill_host_ms=a['host_us']/1000,decode_42_host_ms=42*b['host_us']/1000,
   prefill_round_tps_range=[min(64e6/z['host_us'] for z in series[recipe+':prefill']),max(64e6/z['host_us'] for z in series[recipe+':prefill'])],
   decode_round_tps_range=[min(1e6/z['host_us'] for z in series[recipe+':decode']),max(1e6/z['host_us'] for z in series[recipe+':decode'])])
  v=summary['recipes'][recipe]
  e2e.append([recipe.upper(),f'{v["prefill_tps"]:.2f}',f'{v["decode_tps"]:.2f}',f'{v["prefill_host_ms"]:.3f}',f'{v["decode_42_host_ms"]:.3f}'])
 hs=read(R.parent/'exp0288/SUMMARY.json')['formal']
 e2e.append(['W4A8-SP2 EXP0288 历史',f'{hs["prefill_ns"]["candidate_tps"]:.2f}',f'{hs["decode_ns"]["candidate_tps"]:.2f}',f'{hs["prefill_ns"]["candidate_ns"]/1e6:.3f}',f'{hs["decode_ns"]["candidate_ns"]*42/1e6:.3f}'])
 et=table(['配置','Prefill token/s','Decode token/s','64-token Host ms','42-step Host ms'],e2e)
 notes="""Qwen3-0.6B, full28, M64+42/cache128/batch1. 五轮短测、十轮轮换正式测量，每轮每配置repeat10，以下按十轮均值统计。W16A16/W4A16为本次原快速向量实现；SP2列复用EXP0288正式结果，非配对历史参照。模块单位μs（各自Host wall占比），互斥账本可加，worker/engine重叠计数不可相加。
数值声明：原快速版本全模浮点参考门槛仍失败（W16A16最大NRMSE0.0065323461、W4A160.0054536203）。用户明确授权先测速，不代表精度/完整数值对齐验收；硬件缓存结构、头部独立复核、重复输出和物理账本通过。没有PPL、质量或Selected自动晋升。
"""
 (R/'MODULES.md').write_text('# EXP0289 模块总表\n\n'+notes+'\n## Prefill M64\n\n'+modules['prefill']+'\n## Decode 单步均值\n\n'+modules['decode']+'\n## 完整 E2E\n\n'+et)
 out=['# EXP0289 complete provisional speed report',notes,
  'Primary source/binary seal: '+summary['measured_source']+
  '; source branch codex/exp-0289-qwen3-06b-a16. Source/package SHA256 checked against retained manifests. No cold model loading, host tokenization, ADB or tensor audit in measured Host wall. Each timed pass covers embedding, all28 layers, final norm, streaming head, greedy and FastRPC. Prompt/fixed token IDs identical across recipes; actual greedy diagnostics separate.',
  'No previous same-model A16 baseline exists. W16A16 versus W4A16 is a rotated recipe comparison, not a same-recipe optimization gate. Auxiliary FP32-intermediate candidate repeat1 is not a formally ranked alternative. Original gate failures and unsuccessful GQA-control attempt retained; the latter is not an accepted control.',
  '## Prefill overview\n\n'+modules['prefill'],'## Decode overview\n\n'+modules['decode'],'## E2E\n\n'+et,
  '## Auxiliary repeat1\n\n'+json.dumps(read(R/'speed-repeat1.json'),indent=2),
  '## FP32-intermediate diagnostic repeat1\n\n'+json.dumps(read(R/'fp32-intermediate-repeat1.json'),indent=2),
  '## Hardware and numerical scope\n\n'+json.dumps(read(R/'hardware-invariants.json'),indent=2)+
  '\nEach recipe has2408 cache snapshots checked (28layers x2 x43). Past prefix unchanged, unused capacity zero, finite payloads, append extends once perstep. Independent final norm max NRMSE2.8795e-5/2.2191e-5; head argmax43/43 correct per recipe, selected logit differences0/1FP16 ULP. This validates these boundaries on audited hidden inputs; not whole-stack numerical equivalence. Every timed output token/logit matches its own audited run exactly. 8MiB grant, peaks5260032/6342144B; explicit timed intermediateDDR/spill/boundary writes0. Compiler stack traffic is not claimed absent.',
  '## Additive ledger and overlapping diagnostics\nAll numeric exported fields follow. *_ticks are19.2ticks/us; other values retain raw units. Only named additive ledger plus Host boundary sums to complete wall. HMX/HVX/DMA/worker counters overlap and must not be summed. Nested28layer ledgers were verified for every profile and remain in raw files. The initial numerical defaults in performance records with compared_elements0 are placeholders, not evidence of independent numerical pass.']
 aux={}
 for item in read(R/'speed-repeat1.json'):
  ps=[v for v in records(R/item['tag']/'stdout.jsonl') if v.get('record')=='generation_profile']
  for mode in ['prefill','decode']:aux[item['recipe']+':'+mode]=normalized([v for v in ps if v['mode']==mode])
 for mode in ['prefill','decode']:
  a,b=means['f16f16:'+mode],means['w4f16:'+mode]
  x,y=aux['f16f16:'+mode],aux['w4f16:'+mode]
  rows=[]
  for k in sorted(set(a)&set(b)&set(x)&set(y)):
   # Also retain median per-round diagnostics as required; throughput primary uses means.
   am=statistics.median(z[k] for z in series['f16f16:'+mode]);bm=statistics.median(z[k] for z in series['w4f16:'+mode])
   rows.append([k,f'{x[k]:.6f}',f'{y[k]:.6f}',f'{100*(y[k]/x[k]-1):+.3f}%' if x[k] else 'N/A zero denominator',
    f'{a[k]:.6f}',f'{b[k]:.6f}',f'{am:.6f}',f'{bm:.6f}',f'{100*(bm/am-1):+.3f}%' if am else 'N/A zero denominator'])
  out.append('### '+mode+'\n\n'+table(['Field','W16 R1','W4 R1','R1 delta','W16 R10 mean','W4 R10 mean','W16 R10 median','W4 R10 median','median delta'],rows))
 (R/'FULL_PROFILING_REPORT.md').write_text('\n\n'.join(out)+'\n')
 write(R/'SUMMARY.json',summary);write(R/'MODULES.json',dict(means=means,series=series,historical_source=str(R.parent/'exp0288/MODULES.json')))
 print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
