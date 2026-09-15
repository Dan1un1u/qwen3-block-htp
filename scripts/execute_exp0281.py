"""Bounded prompt/length generalization, exact native row1 vs retained original."""
import os,sys,shlex,subprocess
from common_exp0281 import *
from device_exp0281 import adb,stage
import full_exp0281 as ff
from audit_exp0281 import head
ARMS={'LATEST':0,'ORIGINAL':8}
CASES={'A16':('A',16),'A64':('A',64),'B64':('B',64)}
def setcase(c,a):
 p,n=CASES[c];os.environ.update(QBH_SP2='8',QBH_WIDE_SCORE='4',QBH_U8_PREFILL_OPT='0',QBH_PAPER_FORMAT_DISABLE=str(ARMS[a]),QBH_PAPER_PIPELINE_DISABLE='0',QBH_PAPER_FIXED_TOKENS='0',QBH_GENERAL_PROMPT=p);ff.STEPS=n
def prepare():
 preflight();R.mkdir(exist_ok=False)
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained('/mnt/d/llm_exp/models/Qwen3-origin',local_files_only=True)
 texts={'A':'Modern processors combine several types of execution units. Matrix instructions handle dense arithmetic, vector instructions perform normalization and data conversion, and direct memory access engines transfer tensors. Efficient inference requires matching the physical layout produced by one operation to the layout consumed by the next. Explain how buffering, dependencies, and scheduling affect latency when these units operate concurrently. Consider a language model with a small batch size and a growing attention cache. Describe the main constraints carefully.', 'B':'现代处理器往往同时包含矩阵计算单元、向量计算单元和直接内存访问引擎。矩阵单元适合执行密集的乘加运算，向量单元负责归一化、非线性以及数据格式转换。为了降低语言模型推理延迟，运行时需要协调这些单元之间的数据依赖，合理安排缓冲区与异步搬运。请分析在批量较小、注意力缓存逐渐增长的场景中，哪些因素会限制吞吐量，以及如何判断优化是否真正缩短了关键路径。'}
 prompts={}
 for k,v in texts.items():
  ids=tok.encode(v,add_special_tokens=False);assert len(ids)>=64;ids=ids[:64];prompts[k]=dict(text=v,ids=ids,decoded=tok.decode(ids),token_sha256=__import__('hashlib').sha256(__import__('struct').pack('<64I',*ids)).hexdigest())
 write(R/'prompts.json',prompts)
 prior=R.parent/'exp0280/opt-a2';assert read(prior/'full_gate.json')['pass_all'];cfg=read(prior/'deployment-sp2-fp32.json');m=read(O/'sp2-fp32/manifest.json');assert sha(O/'sp2-fp32/manifest.json')==cfg['manifest_sha256']
 for n,v in m['files'].items():assert sha(O/'sp2-fp32'/n)==v['sha256'],n
 for i in range(63):
  for k in ['cos','sin']:assert f'generation_decode_rope_{k}_{i:02d}_f16.bin' in m['files']
 write(R/'deployment-sp2-fp32.json',cfg);write(R/'fixed_tokens.json',read(R.parent/'exp0279/fixed_tokens.json'));write(R/'inherited_gates.json',dict(pass_all=True,source='EXP0280',scope='same native DSP and models; independent selected0/14/27 andchain3, full28 pairedhidden, actualfinalnormhead'))
 write(R/'protocol_freeze.json',dict(cases=CASES,arms=ARMS,repeat=5,short=5,formal=10,capacity=128,max_kv=127,quality_claim=False))
 print('PREPARED_CPU_ONLY',flush=True)
def verify_device():
 preflight();cfg=read(R/'deployment-sp2-fp32.json');m=read(O/'sp2-fp32/manifest.json');names=list(m['files'])
 for i in range(0,len(names),32):
  lines=adb('shell','sha256sum '+' '.join(shlex.quote(cfg['remote']+'/'+n) for n in names[i:i+32])).stdout.splitlines();assert len(lines)==len(names[i:i+32])
  for line in lines:
   h,n=line.split(None,1);assert h==m['files'][n.removeprefix(cfg['remote']+'/')]['sha256'],n
 write(R/'device_model_gate.json',dict(pass_all=True))
def gates():
 assert read(R/'device_model_gate.json')['pass_all'];out=[]
 for c in CASES:
  runs=[]
  for a in ARMS:
   setcase(c,a);tag=f'gate/{c}-{a}';z=ff.full(2,1,tag,audit=True);head(tag);runs.append(z)
  assert runs[0]['selected_codes']==runs[1]['selected_codes']
  for step in range(CASES[c][1]):
   for name in [f'generation_hidden_step{step:02d}_f32.bin',f'generation_norm_step{step:02d}_u8_native.bin']:assert (R/f'gate/{c}-LATEST'/name).read_bytes()==(R/f'gate/{c}-ORIGINAL'/name).read_bytes(),(c,step,name)
  out.append(dict(case=c,pass_all=True,runs=runs))
 write(R/'full_gate.json',dict(pass_all=True,runs=out))
def timing():
 assert read(R/'full_gate.json')['pass_all']
 for c in CASES:
  expected=read(R/f'gate/{c}-LATEST/validated.json')['selected_codes']
  for phase,n in [('short',5),('formal',10)]:
   runs=[]
   for cycle in range(n):
    for a in (list(ARMS) if cycle%2==0 else list(ARMS)[::-1]):
     setcase(c,a);z=ff.full(2,5,f'{c}/{phase}/{cycle:02d}-{a}');assert z['selected_codes']==expected;runs.append(dict(case=c,cycle=cycle,arm=a,**z))
   write(R/f'{c}-{phase}.json',dict(pass_all=True,runs=runs))
if __name__=='__main__':
 a=sys.argv[1]
 if a=='stage':stage(28)
 else:{'prepare':prepare,'verify_device':verify_device,'gate':gates,'timing':timing}[a]()
