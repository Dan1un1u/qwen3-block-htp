"""L32-0046 matched paired campaign after exact full-model checks."""
import os,time,sys
import execute_llama32_3b_a8 as e
def wait_manifest(name):
    p=e.M.parent/'l32-0046'/name/'manifest.json'
    while not p.exists():time.sleep(5)
def main():
    e.preflight()
    os.environ.update(QBH_3B_DECODE_COUNT='42',QBH_3B_HEAD_TILES='32')
    assert e.read(e.R/'audit-SP2/hidden-audit.json')['pass_all']
    for name,tag,greedy in [('frontend64-fixed','audit-A8',False),('frontend64-greedy','greedy-A8',True)]:
        wait_manifest(name);e.deploy(name)
        os.environ['QBH_3B_AUDIT']='1';os.environ['QBH_3B_GREEDY']='1' if greedy else '0'
        e.execute('a8-l28',name,tag,1,True)
    os.environ['QBH_3B_AUDIT']='0';os.environ['QBH_3B_GREEDY']='0'
    for arm in ['SP2','A8']:e.execute('a8-l28','frontend64-sp2' if arm=='SP2' else 'frontend64-fixed','aux-'+arm,1,True)
    for phase,n in [('short',5),('formal',10)]:
        runs=[]
        for i in range(n):
            for arm in (['SP2','A8'] if i%2==0 else ['A8','SP2']):
                tag=f'{phase}/{i:02d}-{arm}'
                z=e.execute('a8-l28','frontend64-sp2' if arm=='SP2' else 'frontend64-fixed',tag,10,True)
                runs.append(dict(z,cycle=i,arm=arm,tag=tag))
        e.save(e.R/(phase+'.json'),dict(runs=runs))
if __name__=='__main__':main()
