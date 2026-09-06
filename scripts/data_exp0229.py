#!/usr/bin/env python3
"""Freeze independent acceptance and reserve documents before any model score."""
import hashlib,json,re,struct,subprocess
from pathlib import Path
from urllib.parse import urlencode
import numpy as np
import pyarrow.parquet as pq
import yaml
from transformers import AutoTokenizer
from gptq_exp0221 import fetch

SOURCE=Path(__file__).resolve().parents[1]
MEMORY=Path('/home/daniuniu/work/qwen3-block-htp-project-memory')
BASE=Path('/mnt/d/llm_exp/results/qwen3-block-htp')
RESULT=BASE/'exp0229'
MODEL=Path('/mnt/d/llm_exp/models/Qwen3-origin')
CELLS=['en_wiki','zh_wiki','en_news','zh_news']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def hashed_grams(ids):
    return {hashlib.sha256(struct.pack('<32I',*ids[i:i+32])).digest() for i in range(len(ids)-31)}
def write(name,value):
    path=RESULT/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def preflight():
    subprocess.run(['python3',str(MEMORY/'scripts/project_memory.py'),'preflight','--source-worktree',str(SOURCE)],check=True)
def verified(exp,name):
    index=yaml.safe_load((MEMORY/'experiments/index.yaml').read_text())
    ev=next(e['evidence'] for e in index['experiments'] if e['id']=='EXP-'+exp[-4:])
    ledger=BASE/exp/'evidence_sha256.json'
    expected=ev.get('evidence_ledger_sha256',ev.get('evidence_sha256_ledger_sha256'))
    assert expected and sha(ledger)==expected,(exp,'ledger')
    hashes=json.loads(ledger.read_text());path=BASE/exp/name
    assert sha(path)==hashes[name],(exp,name)
    return path

def binary(path,samples):
    words=[0x51424556,1,len(samples),83]
    assert 0<len(samples)<=128
    for s in samples:
        assert len(s['prompt_ids'])==64 and len(s['target_ids'])==16
        words += [s['id'],1,16]+s['prompt_ids']+s['target_ids']
    with path.open('xb') as f:f.write(struct.pack('<'+'I'*len(words),*words))
    raw=path.read_bytes();assert len(raw)==16+332*len(samples)
    for i,s in enumerate(samples):
        assert struct.unpack_from('<64I',raw,16+332*i+12)==tuple(s['prompt_ids'])
        assert struct.unpack_from('<16I',raw,16+332*i+268)==tuple(s['target_ids'])

def prepare():
    preflight();assert not (RESULT/'dataset_freeze.json').exists()
    RESULT.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    roles={};provenance=[];forbidden=set();banned={c:set() for c in CELLS};hashes=set()
    for exp,name in [('exp0218','dataset_v1.json'),('exp0221','calibration.json'),('exp0225','learning_data.json'),('exp0226','learning_data.json')]:
        path=verified(exp,name);d=json.loads(path.read_text());roles[exp]=d
        provenance.append(dict(path=str(path),sha256=sha(path),role='prior_excluded_data'))
        rows=d['samples'] if 'samples' in d else d['train']+d['validation']
        for row in rows:
            ids=row.get('token_ids',row.get('prompt_ids',[])+row.get('target_ids',[]));forbidden.update(hashed_grams(ids))
            if 'text_sha256' in row:hashes.add(row['text_sha256'])
            if 'document' in row:
                domain=row.get('domain','wiki');banned[row['language']+'_'+domain].add(row['document'])
    assert sha(MODEL/'qwen3-tokenizer.json')==roles['exp0218']['tokenizer_sha256']
    tables={}
    for cell,filename in [('en_wiki','wiki_en.parquet'),('en_news','news_en.parquet'),('zh_news','news_zh.parquet')]:
        p=verified('exp0226','learning_raw/'+filename);tables[cell]=pq.read_table(p).to_pylist()
        provenance.append(dict(path=str(p),sha256=sha(p),role='candidate_corpus',source=next(v for v in roles['exp0226']['sources'] if v.get('name')==filename[:-8])))
    docs={};heading='preamble';calrows={x['row_index'] for x in roles['exp0221']['samples'] if x['language']=='en'}
    for i,row in enumerate(tables['en_wiki']):
        text=row['text'].strip()
        if re.fullmatch(r'= [^=].*[^=] =',text):heading=text
        if i in calrows:banned['en_wiki'].add(heading)
        docs.setdefault(heading,[]).append((i,text))
    candidates={'en_wiki':[dict(document=k,text='\n'.join(t for _,t in rows[1:] if t),row_index=rows[0][0]) for k,rows in docs.items() if k!='preamble']}
    calzh={x['row_index'] for x in roles['exp0221']['samples'] if x['language']=='zh'}
    for p in sorted((BASE/'exp0225/learning_raw').glob('zh_*.json')):
        verified('exp0225','learning_raw/'+p.name)
        for row in json.loads(p.read_text())['rows']:
            if row['row_idx'] in calzh:banned['zh_wiki'].add(str(row['row'].get('id',row['row_idx'])))
    for cell in ['en_news','zh_news']:
        candidates[cell]=[dict(document=str(row.get('url',row.get('id',row.get('gem_id',i)))),text=row.get('text',row.get('inputs','')).strip(),row_index=i) for i,row in enumerate(tables[cell])]
    rng=np.random.default_rng(229);chosen={c:[] for c in CELLS};used_grams=set();used_hashes=set();used_docs=set();rejects={}
    def select(cell,rows):
        for index in rng.permutation(len(rows)):
            row=rows[int(index)];doc=row['document'];text=row['text'];h=hashlib.sha256(text.encode()).hexdigest()
            if doc in banned[cell] or (cell,doc) in used_docs or h in hashes|used_hashes:continue
            ids=tok.encode(text,add_special_tokens=False)
            if len(ids)<81:continue
            window=None
            for start in rng.permutation(np.arange(1,len(ids)-79))[:128]:
                start=int(start);trial=ids[start:start+80];g=hashed_grams(trial)
                if g & (forbidden|used_grams):continue
                window=(start,trial,g);break
            if window is None:rejects[cell]=rejects.get(cell,0)+1;continue
            start,trial,g=window;chosen[cell].append(dict(language=cell[:2],domain=cell[3:],cell=cell,document=doc,row_index=row['row_index'],text_sha256=h,document_tokens=len(ids),token_offset=start,prompt_ids=trial[:64],target_ids=trial[64:],kind='nll',steps=16))
            used_docs.add((cell,doc));used_hashes.add(h);used_grams.update(g)
            if len(chosen[cell])==256:break
    for cell in ['en_wiki','en_news','zh_news']:
        select(cell,candidates[cell]);assert len(chosen[cell])==256,(cell,len(chosen[cell]))
        print('DATA_CELL_READY',cell,256,flush=True)
    for offset in range(10000,14000,100):
        url='https://datasets-server.huggingface.co/rows?'+urlencode(dict(dataset='wikimedia/wikipedia',config='20231101.zh',split='train',offset=offset,length=100))
        path=RESULT/'raw'/f'zh_{offset}.json';raw=fetch(url,path);payload=json.loads(raw)
        provenance.append(dict(path=str(path),sha256=sha(path),url=url,role='candidate_corpus',config='20231101.zh',license='Wikipedia source CC-BY-SA/GFDL; retain source attribution in rows'))
        rows=[dict(document=str(r['row'].get('id',r['row_idx'])),text=r['row']['text'].strip(),row_index=r['row_idx']) for r in payload['rows'] if 'text' not in r.get('truncated_cells',[])]
        select('zh_wiki',rows);print('DATA_CELL_PROGRESS','zh_wiki',len(chosen['zh_wiki']),flush=True)
        if len(chosen['zh_wiki'])==256:break
    assert len(chosen['zh_wiki'])==256
    samples=[]
    for split,start in [('primary',0),('reserve',128)]:
        for index in range(128):
            for cell in CELLS:
                row=chosen[cell][start+index];row.update(id=len(samples),split=split);samples.append(row)
    assert len(samples)==len(used_docs)==1024 and not forbidden & used_grams
    write('dataset.json',dict(version='qbh-ppl-accept-v1',seed=229,context_tokens=64,target_tokens_per_window=16,
        primary_targets=8192,reserve_targets=8192,cells=CELLS,samples=samples,tokenizer_sha256=sha(MODEL/'qwen3-tokenizer.json'),
        provenance=provenance,exclusion_roles=['GPTQ calibration','rotation training','checkpoint validation','qbh full and holdout'],
        document_disjoint=True,text_hash_disjoint=True,all_roles_32gram_disjoint=True,rejections=rejects,
        limitations='Independent project short-context conditional PPL, not published full benchmark or long-context acceptance'))
    inputdir=RESULT/'inputs';inputdir.mkdir(exist_ok=True)
    for split in ['primary','reserve']:
        rows=[s for s in samples if s['split']==split]
        for i in range(16):binary(inputdir/f'{split}_{i:02d}.bin',rows[i*32:(i+1)*32])
    sentinels=samples[:8];binary(inputdir/'sentinel.bin',sentinels+sentinels)
    old=[s for s in roles['exp0218']['samples'] if s['id'] in [0,16]];binary(inputdir/'regression.bin',old)
    write('dataset_freeze.json',dict(files={str(p.relative_to(RESULT)):sha(p) for p in [RESULT/'dataset.json',*sorted(inputdir.glob('*.bin'))]},frozen_before_inference=True,primary_ids=[s['id'] for s in samples[:512]],reserve_ids=[s['id'] for s in samples[512:]],quick_ids=[s['id'] for s in samples[:128]]))
    print('DATA_FROZEN',sha(RESULT/'dataset_freeze.json'),flush=True)

if __name__=='__main__':prepare()
