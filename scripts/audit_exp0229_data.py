#!/usr/bin/env python3
"""Independently reconstruct every frozen window from its corpus and offset."""
import hashlib,json,re
from collections import Counter
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from data_exp0229 import RESULT,BASE,MODEL,sha,write,preflight
from measure_exp0229 import frozen

def main():
    preflight();frozen()
    data=json.loads((RESULT/'dataset.json').read_text());tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    texts={};heading=None;parts=[]
    table=pq.read_table(BASE/'exp0226/learning_raw/wiki_en.parquet').to_pylist()
    for row in table:
        text=row['text'].strip()
        if re.fullmatch(r'= [^=].*[^=] =',text):
            if heading is not None:texts[('en_wiki',heading)]='\n'.join(parts)
            heading=text;parts=[]
        elif heading is not None and text:parts.append(text)
    if heading is not None:texts[('en_wiki',heading)]='\n'.join(parts)
    for lang in ['en','zh']:
        table=pq.read_table(BASE/f'exp0226/learning_raw/news_{lang}.parquet').to_pylist()
        for i,row in enumerate(table):
            doc=str(row.get('url',row.get('id',row.get('gem_id',i))))
            texts[(lang+'_news',doc)]=row.get('text',row.get('inputs','')).strip()
    for p in (RESULT/'raw').glob('*.json'):
        for r in json.loads(p.read_text())['rows']:
            if 'text' not in r.get('truncated_cells',[]):
                texts[(p.name[:2]+'_wiki',str(r['row'].get('id',r['row_idx'])))]=r['row']['text'].strip()
    # Direct tuple equality is independent of the sampler's SHA-256 ngram sets.
    def grams(tokens):return {tuple(tokens[i:i+32]) for i in range(len(tokens)-31)}
    forbidden=set()
    for exp,name in [('exp0218','dataset_v1.json'),('exp0221','calibration.json'),('exp0225','learning_data.json'),('exp0226','learning_data.json')]:
        d=json.loads((BASE/exp/name).read_text())
        rows=d['samples'] if 'samples' in d else d['train']+d['validation']
        for r in rows:forbidden.update(grams(r.get('token_ids',r.get('prompt_ids',[])+r.get('target_ids',[]))))
    seen=set();seen_docs=set();counts=Counter()
    for row in data['samples']:
        key=(row['cell'],row['document']);assert key not in seen_docs;seen_docs.add(key)
        text=texts[key];assert hashlib.sha256(text.encode()).hexdigest()==row['text_sha256']
        ids=tok.encode(text,add_special_tokens=False);offset=row['token_offset']
        assert ids[offset:offset+80]==row['prompt_ids']+row['target_ids']
        assert len(ids)==row['document_tokens']
        g=grams(ids[offset:offset+80]);assert not g & (forbidden|seen)
        seen.update(g);counts[(row['split'],row['cell'])]+=1
    assert set(counts.values())=={128} and len(counts)==8
    write('independent_data_audit.json',dict(pass_all=True,documents=len(seen_docs),
        exact_corpus_token_windows=1024,primary_targets=8192,reserve_targets=8192,
        direct_tuple_32gram_disjoint=True,counts={str(k):v for k,v in counts.items()},
        dataset_sha256=sha(RESULT/'dataset.json'),audit_source_sha256=sha(__file__)))
    print('INDEPENDENT_DATA_AUDIT_PASS',len(seen_docs),flush=True)
if __name__=='__main__':main()
