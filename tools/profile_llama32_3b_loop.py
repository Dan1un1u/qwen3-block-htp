#!/usr/bin/env python3
"""L32-0043 fixed five short/ten paired formal cycles, frozen 3B trajectory."""
from execute_llama32_3b_loop import execute,read,save,R,BASE,adb,sha
from pathlib import Path
import shlex,sys
CONTROL='control-l28'
CANDIDATE='opt3-l28'
def verify_remote():
    evidence={}
    for tag in [CONTROL,CANDIDATE]:
        rt=read(R/tag/'runtime.json')
        for source,digest in rt['seal']['files'].items():
            path=rt['remote']+'/'+Path(source).name
            assert adb('shell','sha256sum '+shlex.quote(path)).stdout.split()[0]==digest,path
        evidence[tag]=rt
    cfg=read(BASE/'package-frontend-a01.json');p=Path(cfg['package'])
    assert sha(p/'manifest.json')==cfg['manifest_sha256']
    mf=read(p/'manifest.json');names=[n for n in mf['files'] if not n.endswith('.npy')]
    for i in range(0,len(names),32):
        chosen=names[i:i+32]
        for n in chosen:assert sha(p/n)==mf['files'][n]['sha256'],n
        lines=adb('shell','sha256sum '+' '.join(shlex.quote(cfg['remote']+'/'+n) for n in chosen)).stdout.splitlines()
        assert len(lines)==len(chosen)
        for line in lines:
            h,n=line.split(None,1);assert h==mf['files'][n.removeprefix(cfg['remote']+'/')]['sha256'],n
    save(R/'paired-inputs-verified.json',dict(runtimes=evidence,package=cfg,payload_files=len(names),pass_all=True))
def main():
    verify_remote()
    for arm,rt in [('CONTROL',CONTROL),('OPT3',CANDIDATE)]:
        execute(rt,'frontend-a01','warmup-'+arm,1,True)
    for phase,n in [('short',5),('formal',10)]:
        results=[]
        for cycle in range(n):
            row={}
            order=[('CONTROL',CONTROL),('OPT3',CANDIDATE)]
            if cycle%2:order.reverse()
            for arm,rt in order:
                row[arm]=execute(rt,'frontend-a01',f'{phase}/{cycle:02d}-{arm}',10,True)
            results.append(row)
            print('PAIRED_CYCLE_COMPLETE',phase,cycle,flush=True)
        save(R/(phase+'-paired.json'),dict(pass_all=True,rounds=n,repeat=10,runs=results))
if __name__=='__main__':main()
