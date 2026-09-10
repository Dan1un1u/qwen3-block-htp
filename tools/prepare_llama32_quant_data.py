#!/usr/bin/env python3
"""Freeze fresh Llama tokens from attributed raw corpora, never prior model tokens."""
import json,hashlib,struct,subprocess
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from llama_reference import sha256,provenance
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/mnt/d/llm_exp/results/llama32-htp/l32-0002/data')
MODEL=Path('/mnt/d/llm_exp/models/llama3.2-1B-Instruct-origin')
def main():
 subprocess.run(['python3','/home/daniuniu/work/llama32-htp-project-memory/scripts/project_memory.py','preflight','--source-worktree',str(ROOT)],check=True)
 assert not OUT.exists();prov=provenance(MODEL);tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 # Reusable public raw corpus cache only. Historical calibration/token IDs are never read.
 root=Path('/mnt/d/llm_exp/results/qwen3-block-htp/exp0230/raw')
 rng=np.random.default_rng(320002);samples=[];used=set();grams=set();sources=[]
 for lang in ['en','zh']:
  rows=[]
  for path in sorted(root.glob(lang+'_*.json')):
   payload=json.loads(path.read_text());sources.append({'path':str(path),'sha256':sha256(path),'source':'https://datasets-server.huggingface.co/rows','dataset':'wikimedia/wikipedia','config':'20231101.'+lang,'split':'train','license':'CC-BY-SA/GFDL'})
   for item in payload['rows']:
    if 'text' not in item.get('truncated_cells',[]):rows.append((path,item))
  count=0
  for j in rng.permutation(len(rows)):
   path,item=rows[int(j)];r=item['row'];text=r['text'];h=hashlib.sha256(text.encode()).hexdigest()
   if h in used:continue
   ids=tok.encode(text,add_special_tokens=False)
   if len(ids)<160:continue
   offset=int(rng.integers(0,len(ids)-128+1));window=ids[offset:offset+128]
   gs={tuple(window[i:i+32]) for i in range(97)}
   if gs & grams:continue
   split='calibration' if count<256 else 'heldout'
   row={'id':len(samples),'language':lang,'domain':'wiki','split':split,'document':str(r.get('id',item['row_idx'])),'url':r.get('url'),'source_path':str(path),'row_index':item['row_idx'],'text_sha256':h,'token_offset':offset,'token_ids':window}
   if split=='heldout':row.update(prompt_ids=window[:64],target_ids=window[64:80])
   samples.append(row);grams.update(gs);used.add(h);count+=1
   if count==320:break
  assert count==320,(lang,count)
 OUT.mkdir(parents=True)
 data={'id':'llama32-wiki-quant-v1','seed':320002,'samples':samples,'sources':sources,'original':prov,'calibration_tokens':65536,'heldout_target_tokens':2048,'document_and_32gram_disjoint':True,'quality_scope':'EN/ZH Wikipedia, M64 plus 16 targets; not long-context or broad-domain acceptance','thresholds':{'overall_ppl_ratio_max':1.05,'each_language_domain_ppl_ratio_max':1.10},'frozen_before_scoring':True}
 (OUT/'dataset.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
 cal=np.array([s['token_ids'] for s in samples if s['split']=='calibration'],dtype='<u4');cal.tofile(OUT/'calibration.bin')
 held=[s for s in samples if s['split']=='heldout'];words=[0x51424556,1,len(held),83]
 for s in held:words += [s['id'],1,16]+s['prompt_ids']+s['target_ids']
 (OUT/'heldout.bin').write_bytes(struct.pack('<'+'I'*len(words),*words))
 (OUT/'freeze.json').write_text(json.dumps({n:sha256(OUT/n) for n in ['dataset.json','calibration.bin','heldout.bin']},indent=2)+'\n')
 print('FROZEN',len(samples),sha256(OUT/'freeze.json'),flush=True)
if __name__=='__main__':main()
