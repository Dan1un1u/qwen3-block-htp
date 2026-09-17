#!/usr/bin/env python3
"""L32-0045 fixed final comparison; no performance stopping rule."""
import os,shlex
from pathlib import Path
from execute_llama32_3b_nogate import execute,read,save,R,adb,sha
CONTROL='control-l28'
CANDIDATE='final-l28'
def verify():
    for tag in [CONTROL,CANDIDATE]:
        rt=read(R/tag/'runtime.json')
        for src,digest in rt['seal']['files'].items():
            assert sha(R/tag/Path(src).name)==digest
            assert adb('shell','sha256sum '+shlex.quote(rt['remote']+'/'+Path(src).name)).stdout.split()[0]==digest
    cfg=read(R/'package-frontend64-a01.json');p=Path(cfg['package'])
    assert sha(p/'manifest.json')==cfg['manifest_sha256']
    mf=read(p/'manifest.json');names=[n for n in mf['files'] if not n.endswith('.npy')]
    for i in range(0,len(names),32):
        ns=names[i:i+32]
        for n in ns:assert sha(p/n)==mf['files'][n]['sha256'],n
        lines=adb('shell','sha256sum '+' '.join(shlex.quote(cfg['remote']+'/'+n) for n in ns)).stdout.splitlines()
        assert len(lines)==len(ns)
        for line in lines:
            h,n=line.split(None,1);assert h==mf['files'][n.removeprefix(cfg['remote']+'/')]['sha256'],n
    save(R/'final-inputs-verified.json',dict(pass_all=True,files=len(names),package=cfg))
def main():
    verify()
    os.environ.update(QBH_3B_DECODE_COUNT='42',QBH_3B_HEAD_TILES='32')
    for key in ['QBH_3B_AUDIT','QBH_3B_GREEDY']:os.environ.pop(key,None)
    arms=[('CONTROL',CONTROL),('OPT',CANDIDATE)]
    for arm,rt in arms:execute(rt,'frontend64-a01','warmup-'+arm,1,True)
    for phase,n in [('short',5),('formal',10)]:
        out=[]
        for i in range(n):
            row={}
            for arm,rt in (arms if i%2==0 else list(reversed(arms))):
                row[arm]=execute(rt,'frontend64-a01',f'{phase}/{i:02d}-{arm}',10,True)
            out.append(row);print('PAIRED_COMPLETE',phase,i,flush=True)
        save(R/(phase+'-paired.json'),dict(rounds=n,repeat=10,runs=out))
if __name__=='__main__':main()
