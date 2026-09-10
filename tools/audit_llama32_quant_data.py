#!/usr/bin/env python3
"""Independent direct tuple/window reconstruction of frozen Llama calibration/heldout."""
import json,hashlib
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from llama_reference import sha256
P=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0002/data')
def main():
 d=json.loads((P/'dataset.json').read_text())
 for n,h in json.loads((P/'freeze.json').read_text()).items():assert sha256(P/n)==h
 tok=AutoTokenizer.from_pretrained(d['original']['original_root'],local_files_only=True)
 sources={}
 for s in d['sources']:
  assert sha256(s['path'])==s['sha256'];sources[s['path']]={r['row_idx']:r['row'] for r in json.loads(Path(s['path']).read_text())['rows']}
 docs=set();grams=set();cal=[];held=[]
 for s in d['samples']:
  text=sources[s['source_path']][s['row_index']]['text'];h=hashlib.sha256(text.encode()).hexdigest();assert h==s['text_sha256'] and h not in docs;docs.add(h)
  ids=tok.encode(text,add_special_tokens=False);ids=ids[s['token_offset']:s['token_offset']+128];assert ids==s['token_ids']
  g={tuple(ids[i:i+32]) for i in range(97)};assert not g & grams;grams.update(g)
  if s['split']=='calibration':cal.append(ids)
  else:held.append(s)
 assert np.array_equal(np.fromfile(P/'calibration.bin',dtype='<u4').reshape(-1,128),np.array(cal))
 words=np.fromfile(P/'heldout.bin',dtype='<u4');assert words[:4].tolist()==[0x51424556,1,128,83]
 for r,s in zip(words[4:].reshape(128,83),held):assert r.tolist()==[s['id'],1,16]+s['token_ids'][:80]
 report={'pass':True,'documents':len(docs),'calibration_tokens':len(cal)*128,'heldout_targets':len(held)*16,'direct_tuple32_disjoint':True,'source_window_reconstruction':True,'freeze_sha256':sha256(P/'freeze.json')}
 with (P/'independent_audit.json').open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps(report))
if __name__=='__main__':main()
