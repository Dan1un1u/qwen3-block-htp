#!/usr/bin/env python3
"""Close the fixed per-channel suite without changing any candidate or score."""
import json,subprocess,math
from pathlib import Path
from reference_exp0238 import S,M,sha,preflight
from data_exp0229 import verified,CELLS
R=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0239')
O=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0239')
def save(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
def main():
 preflight();d=json.loads((R/'closure.json').read_text());pkg=json.loads((R/'OmniQuant/package.json').read_text());up=json.loads((R/'upstream.json').read_text())
 assert len(pkg['blocks'])==28 and pkg['whole_frozen_state_exact']
 assert all(b['packed_replay_exact'] and b['disabled_adapter_exact'] and len(b['epochs'])==20 and b['successful_steps']>0 for b in pkg['blocks'])
 assert all(len(b['epochs'])==20 and [x['epoch'] for x in b['epochs']]==list(range(1,21)) for b in pkg['blocks'])
 assert sha(O/'OmniQuant/manifest.json')==pkg['manifest_sha256']
 manifest=json.loads((O/'OmniQuant/manifest.json').read_text());assert len(manifest['projections'])==196
 refs=[verified('exp0237','closure.json'),verified('exp0238','dataset.json'),verified('exp0238','dataset_freeze.json'),verified('exp0238','final_scoring_seal.json')]
 refs += [verified('exp0238',f'software/final_{v}.json') for v in ['F','C64','AR-P','Qronos']]
 save(R/'shared_reference_hashes.json',{str(p):sha(p) for p in refs})
 cleanup=json.loads(refs[0].read_text());head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip();z=d['results']['overall'];variants=list(z['ppl']);best=min([v for v in variants if v!='F'],key=lambda v:z['ppl'][v])
 text=['# Per-output-channel W4A16 reference suite: EXP0238 / EXP0239','',f"Storage cleanup EXP0237 deleted {cleanup['files_deleted']} payloads, releasing {cleanup['actual_free_increase_bytes']/1024**3:.3f} GiB. Original checkpoint, protected baseline packages and all compact scientific evidence retained;24012 protected file records matched their original hashes.",'', 'All quantized rows below use signed[-7,7] per-output-channel W4, positive FP32 scales, FP16 reconstruction/activations, identical frozen C64 W4 LM head, original gamma, no grouping and no online transform. Qwen3-1.7B;512x128 calibration tokens;independent PC0541024-document/16384-target balanced English/Chinese Wiki/news panel;M64+16 scoring. No model selection or tuning on this final panel.','', '|Method|PPL|Change vs F16|95% PPL-ratio CI vs F16|Quality gate|','|---|---:|---:|---|---|']
 for v in variants:
  if v=='F':text.append(f"|F16 reference|{z['ppl'][v]:.6f}|reference|N/A|reference|");continue
  x=z['vs_F16'][v];text.append(f"|{v}|{z['ppl'][v]:.6f}|{(x['ppl_ratio']-1)*100:+.3f}%|{x['ratio_ci95'][0]:.6f}..{x['ratio_ci95'][1]:.6f}|{d['quality_status'][v]}|")
 text+=['', '|Cell|'+'|'.join(v+' PPL' for v in variants)+'|','|---|'+'---:|'*len(variants)]
 for cell in CELLS:text.append('|'+cell+'|'+'|'.join(f"{d['results'][cell]['ppl'][v]:.6f}" for v in variants)+'|')
 text+=['', '|Method|Paired PPL ratio vs C64|95% CI|','|---|---:|---|']
 for v,x in z['vs_C64'].items():text.append(f"|{v}|{x['ppl_ratio']:.6f}|{x['ratio_ci95'][0]:.6f}..{x['ratio_ci95'][1]:.6f}|")
 text+=['',f'Lowest quantized point estimate on this fixed panel: {best}. Acceptance remains overall5% and every language/domain cell10%, with upper paired95% interval required within the limit for a confident pass. The full per-cell intervals and all raw NLLs are retained in closure.json/software files. Short answers are auxiliary and were not used for ranking.','', 'Interpretation is limited to the declared grid, shared head, corpus, and short-context workload. C64 is the existing project-enhanced GPTQ recipe; AR-P is the retained200-iteration official AutoRound adapter; Qronos uses the official fixed-scale mismatch solver; OmniQuant-LWC learns only clipping bounds for20epochs with the predeclared AdamW settings. Different algorithms have different offline optimization budgets. This is not an unconstrained reproduction or ranking of paper-native grouped/asymmetric formats.','', 'Qronos: all196 independent nibble/FP16 roundtrips and28 full-block replays pass; dense mismatch/matched/diagonal solver oracle passes; no numerical fallback. OmniQuant: official quantizer vs NumPy and gradient-flow oracle pass; all28 unquantized native-adapter comparisons and packed block replays are exact; all196 projections pass independent packing; only clipping parameters trained and frozen full-model state is unchanged. All five score files pass finite-value, exact-repeat, causal-mask and independent CE checks; raw-token reductions independently rechecked.','',f"OmniQuant optimizer updates: {sum(b['successful_steps'] for b in pkg['blocks'])}; gradient-scaler overflow skips: {sum(b['overflow_skipped_steps'] for b in pkg['blocks'])}. Export/calibration elapsed: {pkg['elapsed_s']:.3f}s, not device throughput.",'',f"OmniQuant training source: {pkg['source_head']}; reporting/evaluation branch head: {head}; package manifest: {pkg['manifest_sha256']}; shared dataset: {d['dataset_sha256']}.",'', '[Official Qronos implementation](https://github.com/Xilinx/brevitas/blob/f6c8e2e60649249187c0fa94adec6262aeb029cb/src/brevitas/graph/qronos.py). '+f"[Pinned official OmniQuant quantizer](https://github.com/OpenGVLab/OmniQuant/blob/{up['commit']}/quantize/quantizer.py). Exact source archives and environment freezes are retained.",'', 'New DSP/end-to-end token/s: N/A, software-only accuracy references. No runtime changes, no automatic baseline promotion, and no additional experiment authorized.']
 report='\n'.join(text)+'\n'
 for p in [R/'REPORT.md',M/'docs/experiments/EXP-0239-RESULTS.md']:
  with p.open('x') as f:f.write(report)
 prof=(M/'docs/experiments/EXP-0238-PROFILE.md').read_text().replace('EXP0238','EXP0239').replace('per-channel Qronos reference','per-channel OmniQuant-LWC reference').replace('/exp0238','/exp0239').replace('8980930c4d38fddfe9b72866dba70b015726f620',head).replace('Controls F/C64/AR-P; candidate Qronos.','Controls F/C64/AR-P/Qronos; candidate OmniQuant.').replace('New Qronos prefill/decode','New OmniQuant prefill/decode')
 for p in [R/'full_profiling_report.md',M/'docs/experiments/EXP-0239-PROFILE.md']:
  with p.open('x') as f:f.write(prof)
 save(R/'artifacts_sha256.json',dict(root=str(O),files={str(p.relative_to(O)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(O.rglob('*')) if p.is_file()}))
 save(R/'evidence_sha256.json',{str(p.relative_to(R)):sha(p) for p in sorted(R.rglob('*')) if p.is_file() and p.name!='evidence_sha256.json'})
 print('FINAL_REPORT_COMPLETE',best,dict(PPL=z['ppl'],quality=d['quality_status']),flush=True)
if __name__=='__main__':main()
