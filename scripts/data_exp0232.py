#!/usr/bin/env python3
"""Pre-freeze calibration/development/untouched testing for PC051 AWQ equalization."""
import hashlib,json,re,struct
from pathlib import Path
from urllib.parse import urlencode
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from data_exp0229 import BASE,SOURCE,MEMORY,MODEL,CELLS,sha,verified,preflight,binary,hashed_grams
from gptq_exp0221 import fetch

RESULT=BASE/'exp0232'
OUTPUT=Path('/mnt/d/llm_exp/models/qwen3-block-htp/exp0232')

def write(name,value):
    p=RESULT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')

def prior_data():
    paths=[verified(e,n) for e,n in [
        ('exp0218','dataset_v1.json'),('exp0221','calibration.json'),
        ('exp0225','learning_data.json'),('exp0226','learning_data.json'),
        ('exp0229','dataset.json'),('exp0230','dataset.json'),('exp0231','dataset.json')]]
    datasets=[json.loads(p.read_text()) for p in paths]
    rows=[r for d in datasets for r in (d['samples'] if 'samples' in d else d['train']+d['validation'])]
    return paths,datasets,rows

def ids_of(row):
    return row.get('token_ids',row.get('prompt_ids',[])+row.get('target_ids',[]))

def prepare():
    preflight();assert not (RESULT/'dataset_freeze.json').exists()
    RESULT.mkdir(parents=True,exist_ok=True);OUTPUT.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    paths,datasets,prior=prior_data()
    assert sha(MODEL/'qwen3-tokenizer.json')==datasets[0]['tokenizer_sha256']
    forbidden=set();text_hashes=set();docs=set()
    for row in prior:
        forbidden.update(hashed_grams(ids_of(row)))
        if 'text_sha256' in row:text_hashes.add(row['text_sha256'])
        if 'document' in row:docs.add((row['language']+'_'+row.get('domain','wiki'),str(row['document'])))
    provenance=[dict(path=str(p),sha256=sha(p),role='prior_excluded_data') for p in paths]
    wiki=verified('exp0226','learning_raw/wiki_en.parquet')
    def title_key(t):return re.sub(r'\W+','',t.replace('@-@','-').replace('@,@',',').replace('@.@','.')).casefold()
    titles={title_key(r['text'].strip().strip('= ')) for r in pq.read_table(wiki).to_pylist()
            if re.fullmatch(r'= [^=].*[^=] =',r['text'].strip())}
    provenance.append(dict(path=str(wiki),sha256=sha(wiki),role='exclude_all_wikitext_train_titles'))
    # Recover explicit IDs of the original Chinese calibration documents.
    calzh={r['row_index'] for r in datasets[1]['samples'] if r['language']=='zh'}
    for p in sorted((BASE/'exp0225/learning_raw').glob('zh_*.json')):
        verified('exp0225','learning_raw/'+p.name)
        for row in json.loads(p.read_text())['rows']:
            if row['row_idx'] in calzh:docs.add(('zh_wiki',str(row['row'].get('id',row['row_idx']))))
    rng=np.random.default_rng(232);chosen={c:[] for c in CELLS}
    used=set();seen_hashes=set();used_grams=set()
    def select(cell,rows,path):
        for index in rng.permutation(len(rows)):
            row=rows[int(index)];key=(cell,row['document']);h=hashlib.sha256(row['text'].encode()).hexdigest()
            if key in docs|used or h in text_hashes|seen_hashes:continue
            ids=tok.encode(row['text'],add_special_tokens=False)
            # Use the same document eligibility across roles, before choosing spans.
            if len(ids)<129:continue
            width=80
            for offset in rng.permutation(np.arange(1,len(ids)-width+1))[:128]:
                offset=int(offset);tokens=ids[offset:offset+width];grams=hashed_grams(tokens)
                if grams & (forbidden|used_grams):continue
                chosen[cell].append(dict(language=cell[:2],domain=cell[3:],cell=cell,
                    document=row['document'],row_index=row['row_index'],source_path=str(path),
                    text_sha256=h,document_tokens=len(ids),token_offset=offset,token_ids=tokens))
                used.add(key);seen_hashes.add(h);used_grams.update(grams);break
            if len(chosen[cell])==256:break
    for lang in ['en','zh']:
        cell=lang+'_news';p=verified('exp0226','learning_raw/news_'+lang+'.parquet')
        provenance.append(dict(path=str(p),sha256=sha(p),role='candidate_corpus',
            source=next(v for v in datasets[3]['sources'] if v.get('name')=='news_'+lang)))
        rows=[dict(document=str(r.get('url',r.get('id',r.get('gem_id',i)))),
                   text=r.get('text',r.get('inputs','')).strip(),row_index=i)
              for i,r in enumerate(pq.read_table(p).to_pylist())]
        select(cell,rows,p);assert len(chosen[cell])==256,(cell,len(chosen[cell]))
        print('DATA_CELL_READY',cell,256,flush=True)
    for lang in ['en','zh']:
        cell=lang+'_wiki'
        for offset in range(32000,38000,100):
            url='https://datasets-server.huggingface.co/rows?'+urlencode(dict(
                dataset='wikimedia/wikipedia',config='20231101.'+lang,split='train',offset=offset,length=100))
            p=RESULT/'raw'/f'{lang}_{offset}.json';raw=fetch(url,p)
            provenance.append(dict(path=str(p),sha256=sha(p),url=url,role='candidate_corpus',
                config='20231101.'+lang,license='Wikipedia CC-BY-SA/GFDL; source attribution retained'))
            rows=[dict(document=str(r['row'].get('id',r['row_idx'])),text=r['row']['text'].strip(),row_index=r['row_idx'])
                  for r in json.loads(raw)['rows'] if 'text' not in r.get('truncated_cells',[])
                  and not(lang=='en' and title_key(r['row']['title']) in titles)]
            select(cell,rows,p);print('DATA_CELL_PROGRESS',cell,len(chosen[cell]),flush=True)
            if len(chosen[cell])==256:break
        assert len(chosen[cell])==256
    samples=[]
    for split,start,count in [('primary',0,128),('reserve',128,128)]:
        for i in range(count):
            for cell in CELLS:
                row=chosen[cell][start+i];row.update(id=len(samples),split=split)
                if split=='calibration':row['in_C8']=i<16
                else:row.update(prompt_ids=row['token_ids'][:64],target_ids=row['token_ids'][64:],kind='nll',steps=16)
                samples.append(row)
    assert len(samples)==len(used)==1024 and not forbidden & used_grams
    write('dataset.json',dict(version='qbh-awq-independent-v1',seed=232,cells=CELLS,samples=samples,
        provenance=provenance,tokenizer_sha256=sha(MODEL/'qwen3-tokenizer.json'),
        calibration_context=128,eval_context=64,eval_targets=16,
        calibration_reference='EXP0230_C64_65536',scale_search_reference='first64_C64_calibration_windows_8192_tokens',development_reference='EXP0230_128docs_descriptive_only',primary_targets=8192,reserve_targets=8192,
        english_excluded_wikitext_titles=len(titles),document_disjoint=True,all_roles_32gram_disjoint=True))
    inputs=RESULT/'inputs';inputs.mkdir(exist_ok=True)
    for split in ['primary','reserve']:
        rows=[s for s in samples if s['split']==split]
        for i in range(len(rows)//32):binary(inputs/f'{split}_{i:02d}.bin',rows[i*32:(i+1)*32])
    primary=[r for r in samples if r['split']=='primary']
    binary(inputs/'sentinel.bin',primary[:8]+primary[:8])
    old=[r for r in datasets[0]['samples'] if r['id'] in [0,20]]
    binary(inputs/'regression.bin',old)
    write('dataset_freeze.json',dict(files={str(p.relative_to(RESULT)):sha(p) for p in [RESULT/'dataset.json',*sorted(inputs.glob('*.bin'))]},
        frozen_before_scoring=True,source_roles='new final primary/final reserve; prior230calibration and development references',protocol_sha256=sha(MEMORY/'docs/experiments/EXP-0232.md')))
    print('DATA_FROZEN',sha(RESULT/'dataset_freeze.json'),flush=True)

def audit():
    """Reconstruct from corpus and use direct tuple32 equality independently."""
    preflight();d=json.loads((RESULT/'dataset.json').read_text());freeze=json.loads((RESULT/'dataset_freeze.json').read_text())
    for n,h in freeze['files'].items():assert sha(RESULT/n)==h
    for item in d['provenance']:assert sha(item['path'])==item['sha256']
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    _,_,prior=prior_data();prior_docs={(r.get('cell',r['language']+'_'+r.get('domain','wiki')),str(r['document'])) for r in prior if 'document' in r};prior_text={r['text_sha256'] for r in prior if 'text_sha256' in r};forbidden={tuple(ids_of(r)[i:i+32]) for r in prior for i in range(len(ids_of(r))-31)}
    corpora={};used=set();hashes=set();grams=set()
    for row in d['samples']:
        path=row['source_path']
        if path not in corpora:
            p=Path(path)
            corpora[path]=pq.read_table(p).to_pylist() if p.suffix=='.parquet' else {r['row_idx']:r['row'] for r in json.loads(p.read_text())['rows']}
        raw=corpora[path][row['row_index']];text=raw.get('text',raw.get('inputs','')).strip()
        assert hashlib.sha256(text.encode()).hexdigest()==row['text_sha256']
        ids=tok.encode(text,add_special_tokens=False);off=row['token_offset'];n=128 if row['split']=='calibration' else 80
        assert ids[off:off+n]==row['token_ids'] and len(ids)==row['document_tokens']
        key=(row['cell'],row['document']);assert key not in used|prior_docs and row['text_sha256'] not in hashes|prior_text
        used.add(key);hashes.add(row['text_sha256'])
        new={tuple(row['token_ids'][i:i+32]) for i in range(n-31)}
        assert not new & (forbidden|grams);grams.update(new)
    assert len(used)==1024
    write('independent_data_audit.json',dict(pass_all=True,source_windows=1024,
        direct_tuple32_disjoint=True,source_token_offset_reconstruction=True,dataset_sha256=sha(RESULT/'dataset.json')))
    print('INDEPENDENT_DATA_AUDIT_PASS',1024,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','audit']);a=p.parse_args()
    prepare() if a.phase=='prepare' else audit()
