#!/usr/bin/env python3
"""Freeze sampling-only bilingual data and independent Wiki/news validation."""
import json,re,collections
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
import eval_exp0218 as ev
from gptq_exp0221 import fetch,sha
from rotation_exp0219 import write_json
from learned_rotation_exp0226 import RESULT,PLAN
from data_exp0225 import grams
OLD=RESULT.parent/'exp0225'
URLS={
 'wiki_en':'https://huggingface.co/datasets/Salesforce/wikitext/resolve/d2aad4e8c2e761345360926487e081c41c50d1e1/wikitext-2-raw-v1/train/0000.parquet',
 'news_en':'https://huggingface.co/datasets/GEM/xlsum/resolve/00487191b2aa0461a3bb4783f5aba00fb432dbd2/english/xlsum-validation.parquet',
 'news_zh':'https://huggingface.co/datasets/GEM/xlsum/resolve/00487191b2aa0461a3bb4783f5aba00fb432dbd2/chinese_simplified/xlsum-validation.parquet'}
def prepare():
    assert not (RESULT/'learning_data.json').exists()
    assert ev.digest(OLD/'learning_data.json')=='bcb67126349e8d086f29619807e9ab7ebba06253792b7fce4e5468f9d39e8280'
    old=json.loads((OLD/'learning_data.json').read_text());tok=AutoTokenizer.from_pretrained(ev.MODEL,local_files_only=True)
    assert ev.digest(RESULT.parent/'exp0221/calibration.json')=='e65edb14cb774956df92b27b5dc728b976f7ba00e9435145004180c6538a1669'
    cal=json.loads((RESULT.parent/'exp0221/calibration.json').read_text())
    forbidden=set()
    for r in ev.dataset()['samples']:forbidden.update(grams(r['prompt_ids']+r.get('target_ids',[])))
    for r in cal['samples']:forbidden.update(grams(r['token_ids']))
    oldtrain=set().union(*(grams(r['token_ids']) for r in old['train']))
    validation=[dict(r,domain='wiki',source='EXP0225_independent_validation') for r in old['validation']]
    validgrams=set().union(*(grams(r['token_ids']) for r in validation))
    sources=[];tables={};rng=np.random.default_rng(226)
    for name,url in URLS.items():
        p=RESULT/'learning_raw'/(name+'.parquet');fetch(url,p)
        tables[name]=pq.read_table(p).to_pylist();sources.append(dict(name=name,url=url,sha256=ev.digest(p),rows=len(tables[name])))
        print('RAW_READY',name,len(tables[name]),list(tables[name][0]),flush=True)
    for lang in ['en','zh']:
        count=0;seen=set();table=tables['news_'+lang]
        for index in rng.permutation(len(table)):
            r=table[int(index)];text=r.get('text',r.get('inputs','')).strip();assert text
            doc=str(r.get('url',r.get('id',r.get('gem_id',index))))
            if doc in seen:continue
            ids=tok.encode(text,add_special_tokens=False)
            if len(ids)<129:continue
            start=int(rng.integers(1,len(ids)-127));window=ids[start:start+128];g=grams(window)
            if g&(forbidden|oldtrain|validgrams):continue
            validation.append(dict(language=lang,domain='news',row_index=int(index),document=doc,text_sha256=sha(text.encode()),token_ids=window,token_offset=start,document_tokens=len(ids),source='GEM/xlsum_validation'))
            validgrams.update(g);seen.add(doc);count+=1
            if count==16:break
        assert count==16
    docs={};doc='preamble';calrows={r['row_index'] for r in cal['samples'] if r['language']=='en'};banned={r['document'] for r in validation if r['language']=='en' and r['domain']=='wiki'}
    enrows=tables['wiki_en']
    for index,r in enumerate(enrows):
        text=r['text'].strip()
        if re.match(r'^= [^=].*[^=] =$',text):doc=text
        if index in calrows:banned.add(doc)
        docs.setdefault(doc,[]).append((index,text))
    for p in sorted((OLD/'learning_raw').glob('en_*.json')):
        for r in json.loads(p.read_text())['rows']:assert enrows[r['row_idx']]['text']==r['row']['text']
    train=[];usedgrams=set();chosen_docs=[]
    for doc in rng.permutation(sorted(docs)):
        if doc in banned or doc=='preamble':continue
        rows=docs[doc];text='\n'.join(t for _,t in rows[1:] if t);ids=tok.encode(text,add_special_tokens=False)
        if len(ids)<257:continue
        starts=[];windows=[]
        for start in rng.permutation(np.arange(1,len(ids)-127)):
            start=int(start)
            if any(abs(start-x)<128 for x in starts):continue
            window=ids[start:start+128];g=grams(window)
            if g&(forbidden|validgrams|usedgrams):continue
            starts.append(start);windows.append(window)
            if len(starts)==2:break
        if len(starts)<2:continue
        for start,window in zip(starts,windows):
            train.append(dict(language='en',row_index=rows[0][0],document=str(doc),text_sha256=sha(text.encode()),token_ids=window,token_offset=start,document_tokens=len(ids)))
            usedgrams.update(grams(window))
        chosen_docs.append(str(doc))
        if len(chosen_docs)==200:break
    assert len(chosen_docs)==200,len(chosen_docs)
    zhrows={}
    for p in sorted((OLD/'learning_raw').glob('zh_*.json')):
        payload=json.loads(p.read_text());sources.append(dict(name=p.name,path=str(p),sha256=ev.digest(p)))
        for r in payload['rows']:zhrows[r['row_idx']]=r
    for oldrow in old['train'][400:]:
        r=zhrows[oldrow['row_index']];assert 'text' not in r.get('truncated_cells',[])
        text=r['row']['text'].strip();assert sha(text.encode())==oldrow['text_sha256']
        ids=tok.encode(text,add_special_tokens=False);selected=None
        for start in rng.permutation(np.arange(1,len(ids)-127)):
            start=int(start);window=ids[start:start+128];g=grams(window)
            if not g&(forbidden|validgrams|usedgrams):selected=(start,window);break
        assert selected is not None,oldrow['document']
        start,window=selected;train.append(dict(oldrow,token_ids=window,token_offset=start,document_tokens=len(ids)));usedgrams.update(grams(window))
    assert len(train)==800 and len(validation)==64
    assert not usedgrams&validgrams and not (usedgrams|validgrams)&forbidden
    assert not {(r['language'],r['document']) for r in train}&{(r['language'],r['document']) for r in validation}
    stats={lang:dict(samples=400,documents=len(set(r['document'] for r in train if r['language']==lang)),max_windows_per_document=max(collections.Counter(r['document'] for r in train if r['language']==lang).values()),mean_relative_start=float(np.mean([r['token_offset']/r['document_tokens'] for r in train if r['language']==lang]))) for lang in ['en','zh']}
    write_json(RESULT/'learning_data.json',dict(plan=PLAN,train=train,validation=validation,sources=sources,coverage=stats,calibration_ids_sha256=cal['ids_sha256'],tokenizer_sha256=ev.digest(ev.MODEL/'qwen3-tokenizer.json'),old_data_sha256=ev.digest(OLD/'learning_data.json'),train_validation_document_disjoint=True,all_roles_32gram_disjoint=True,new_validation_disjoint_old_training=True,selection_role='equal64x127target Wiki/news en/zh; frozen before any training; qbh never selects'))
    print('DATA_FROZEN',ev.digest(RESULT/'learning_data.json'),stats,flush=True)
if __name__=='__main__':prepare()
