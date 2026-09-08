#!/usr/bin/env python3
"""Frozen head calibration and independent final panel; no evaluation selection."""
import json,hashlib,re
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from data_exp0229 import BASE,SOURCE,MEMORY,MODEL,CELLS,sha,preflight,hashed_grams
from data_exp0245 import prior_data as old_prior, verified as previous_verified
from data_exp0233 import ids_of
from gptq_exp0221 import fetch
RESULT=BASE/'exp0246'
def verified(exp,name):
 if exp!='exp0245':return previous_verified(exp,name)
 ledger=BASE/exp/'EVIDENCE_SHA256.json'
 assert sha(ledger)=='8a59e580907adbf35a440460f49d4753f73e2273a18704ab5022a10099fc5bff'
 p=BASE/exp/name;assert sha(p)==json.loads(ledger.read_text())['files'][name]['sha256'];return p
def write(n,x):
 p=RESULT/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False);f.write('\n')
def prior_data():
 paths,ds,rows=old_prior();p=verified('exp0245','dataset.json');d=json.loads(p.read_text())
 return paths+[p],ds+[d],rows+d['samples']
def frozen():
 f=json.loads((RESULT/'dataset_freeze.json').read_text())
 assert sha(MEMORY/'docs/experiments/EXP-0246.md')==f['protocol_sha256']
 for n,h in f['files'].items():assert sha(RESULT/n)==h,n
 for n,h in f['references'].items():assert sha(n)==h,n
 return json.loads((RESULT/'dataset.json').read_text())
def prepare():
 preflight();assert not (RESULT/'dataset_freeze.json').exists();RESULT.mkdir(parents=True,exist_ok=True)
 paths,ds,prior=prior_data();tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 assert sha(MODEL/'qwen3-tokenizer.json')==ds[0]['tokenizer_sha256']
 forbidden=set();docs=set();texts=set()
 for r in prior:
  forbidden.update(hashed_grams(ids_of(r)))
  if 'document' in r:docs.add((r.get('cell',r['language']+'_'+r.get('domain','wiki')),str(r['document'])))
  if 'text_sha256' in r:texts.add(r['text_sha256'])
 wiki=verified('exp0226','learning_raw/wiki_en.parquet')
 def title_key(t):return re.sub(r'\W+','',t.replace('@-@','-').replace('@,@',',').replace('@.@','.')).casefold()
 titles={title_key(r['text'].strip().strip('= ')) for r in pq.read_table(wiki).to_pylist() if re.fullmatch(r'= [^=].*[^=] =',r['text'].strip())}
 calzh={r['row_index'] for r in ds[1]['samples'] if r['language']=='zh'}
 for p in sorted((BASE/'exp0225/learning_raw').glob('zh_*.json')):
  verified('exp0225','learning_raw/'+p.name)
  for r in json.loads(p.read_text())['rows']:
   if r['row_idx'] in calzh:docs.add(('zh_wiki',str(r['row'].get('id',r['row_idx']))))
 rng=np.random.default_rng(246);chosen={c:[] for c in CELLS};used=set();grams=set();hashes=set();sources=[]
 def select(cell,rows,path):
  if len(chosen[cell])==64:return
  for ix in rng.permutation(len(rows)):
   r=rows[int(ix)];key=(cell,r['document']);h=hashlib.sha256(r['text'].encode()).hexdigest()
   if key in docs|used or h in texts|hashes:continue
   ids=tok.encode(r['text'],add_special_tokens=False)
   if len(ids)<129:continue
   for off in rng.permutation(np.arange(1,len(ids)-80+1))[:128]:
    off=int(off);window=ids[off:off+80];g=hashed_grams(window)
    if g&(forbidden|grams):continue
    chosen[cell].append(dict(language=cell[:2],domain=cell[3:],cell=cell,document=r['document'],row_index=r['row_index'],source_path=str(path),text_sha256=h,document_tokens=len(ids),token_offset=off,token_ids=window))
    used.add(key);hashes.add(h);grams.update(g);break
   if len(chosen[cell])==64:break
 for lang in ['en','zh']:
  cell=lang+'_news';p=verified('exp0226','learning_raw/news_'+lang+'.parquet');sources.append(dict(path=str(p),sha256=sha(p),role='verified_public_news_corpus'))
  rows=[dict(document=str(r.get('url',r.get('id',r.get('gem_id',i)))),text=r.get('text',r.get('inputs','')).strip(),row_index=i) for i,r in enumerate(pq.read_table(p).to_pylist())]
  select(cell,rows,p);assert len(chosen[cell])==64;print('DATA_CELL_READY',cell,64,flush=True)
 for lang in ['en','zh']:
  cell=lang+'_wiki';pool={}
  for d in ds:
   for r in d.get('samples',[]):
    p=Path(r.get('source_path',''))
    if p.suffix=='.json' and p.name.startswith(lang+'_') and p.parent.name=='raw':pool[str(p)]=p
  def consume(p):
   raw=p.read_bytes();sources.append(dict(path=str(p),sha256=sha(p),role='verified_public_Wikipedia_corpus'))
   rows=[dict(document=str(r['row'].get('id',r['row_idx'])),text=r['row']['text'].strip(),row_index=r['row_idx']) for r in json.loads(raw)['rows'] if 'text' not in r.get('truncated_cells',[]) and not(lang=='en' and title_key(r['row']['title']) in titles)]
   select(cell,rows,p)
  for p in sorted(pool.values()):
   exp=p.relative_to(BASE).parts[0];verified(exp,str(p.relative_to(BASE/exp)));consume(p)
   if len(chosen[cell])==64:break
  if len(chosen[cell])<64:
   from urllib.parse import urlencode
   for off in range(94000,104000,100):
    url='https://datasets-server.huggingface.co/rows?'+urlencode(dict(dataset='wikimedia/wikipedia',config='20231101.'+lang,split='train',offset=off,length=100))
    p=RESULT/'raw'/f'{lang}_{off}.json';fetch(url,p);consume(p);sources[-1]['url']=url
    if len(chosen[cell])==64:break
  assert len(chosen[cell])==64,(cell,len(chosen[cell]));print('DATA_CELL_READY',cell,64,flush=True)
 samples=[]
 for i in range(64):
  for c in CELLS:
   r=chosen[c][i];r.update(id=len(samples),split='final',prompt_ids=r['token_ids'][:64],target_ids=r['token_ids'][64:]);samples.append(r)
 write('dataset.json',dict(version='qbh-EXP246-dense-R3-final-v1',seed=246,cells=CELLS,samples=samples,provenance=sources,tokenizer_sha256=sha(MODEL/'qwen3-tokenizer.json'),calibration_reference='EXP230_C64_65536',final_targets=4096,prior_paths=[str(p) for p in paths]))
 refs=paths+[wiki,verified('exp0230','inputs/C64_calibration_u32.bin'),verified('exp0218','original_checkpoint_sha256.json'),MEMORY/'docs/experiments/EXP-0239.md']
 write('dataset_freeze.json',dict(protocol_sha256=sha(MEMORY/'docs/experiments/EXP-0246.md'),files={'dataset.json':sha(RESULT/'dataset.json')},references={str(p):sha(p) for p in refs},frozen_before_export_and_scoring=True))
 print('DATA_FROZEN',sha(RESULT/'dataset_freeze.json'),flush=True)
def audit():
 preflight();d=frozen();_,_,prior=prior_data();tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 pdocs={(r.get('cell',r['language']+'_'+r.get('domain','wiki')),str(r['document'])) for r in prior if 'document' in r};pt={r['text_sha256'] for r in prior if 'text_sha256' in r};pg={tuple(ids_of(r)[i:i+32]) for r in prior for i in range(len(ids_of(r))-31)}
 corpora={};used=set();text=set();grams=set()
 for rec in d['provenance']:assert sha(rec['path'])==rec['sha256']
 for r in d['samples']:
  p=Path(r['source_path'])
  if p not in corpora:corpora[p]=pq.read_table(p).to_pylist() if p.suffix=='.parquet' else {q['row_idx']:q['row'] for q in json.loads(p.read_text())['rows']}
  raw=corpora[p][r['row_index']];t=raw.get('text',raw.get('inputs','')).strip();assert hashlib.sha256(t.encode()).hexdigest()==r['text_sha256']
  ids=tok.encode(t,add_special_tokens=False);assert ids[r['token_offset']:r['token_offset']+80]==r['token_ids'];assert len(ids)==r['document_tokens']
  key=(r['cell'],r['document']);assert key not in used|pdocs and r['text_sha256'] not in text|pt
  g={tuple(r['token_ids'][i:i+32]) for i in range(49)};assert not g&(grams|pg)
  used.add(key);text.add(r['text_sha256']);grams.update(g)
 assert len(used)==256
 write('independent_data_audit.json',dict(pass_all=True,documents=256,final_targets=4096,direct_tuple32_disjoint=True,all_windows_reconstructed=True,dataset_sha256=sha(RESULT/'dataset.json')));print('DATA_AUDIT_PASS',flush=True)
if __name__=='__main__':
 import sys
 prepare() if sys.argv[1]=='prepare' else audit()
