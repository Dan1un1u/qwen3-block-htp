#!/usr/bin/env python3
"""Resolve recent Qwen17 rotation/context switches without touching the device."""
import argparse, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def resolve(r3=False,r4='off',rounding=2,profile='speed64',client_cap='auto',targets=1984,prefill_tokens=None,decode_steps=None):
 cfg=json.loads((ROOT/'config/qwen17-research-presets.json').read_text())
 if r4 not in cfg['rotation']['r4_backends'] or profile not in cfg['profiles']:raise ValueError('unknown backend/profile')
 if rounding not in cfg['rotation']['projection_rounding_options']:raise ValueError('unretained rounding contract')
 if profile=='speed64' and client_cap!='auto':raise ValueError('client-cap diagnostic applies only to long frontend')
 if client_cap not in ('auto','2','3') or not 1<=targets<=1984:raise ValueError('unsupported cap/target count')
 backend=cfg['rotation']['r4_backends'][r4]
 key=('rotated' if r4!='off' else 'original')+('-on' if r3 else '-off')
 binding=cfg['packages'][key];assert binding['r3']==r3 and binding['r4']==(r4!='off')
 env=dict(cfg['common_env']);env.update(cfg['profiles'][profile]['env'])
 env.update(QBH_DENSE_R3='1' if r3 else '0',QBH_R3_OPT='2' if r3 else '0',QBH_DENSE_R4=str(backend['mode']),QBH_R4_OPT=str(backend['opt']),QBH_PROJECTION_ROUNDING=str(rounding))
 if profile in ('quality2048','long_session'):
  mask=cfg['schedule_controls']['client_cap_mask'];flag={'auto':0,'2':524288,'3':1048576}[client_cap]
  env['QBH_LONG_OPT']=str((int(env['QBH_LONG_OPT'])&~mask)|flag)
  if profile=='quality2048':
   if prefill_tokens not in (None,64) or decode_steps not in (None,targets-1):raise ValueError('quality window warmup/targets are frozen;use long_session for inference length experiments')
   env.update(QBH_LONG_PREFILL_TOKENS='64',QBH_LONG_DECODE_STEPS=str(targets-1),QBH_LONG_REPEATS='1',QBH_LONG_TARGET_FILE='${TARGET_FILE}')
  else:
   pref=2048 if prefill_tokens is None else prefill_tokens;dec=64 if decode_steps is None else decode_steps
   if pref not in cfg['profiles']['long_session']['prefill_tokens_supported'] or not 1<=dec<=2112-pref:raise ValueError('unretained prefill length or insufficient KV capacity')
   env.update(QBH_LONG_PREFILL_TOKENS=str(pref),QBH_LONG_DECODE_STEPS=str(dec),QBH_LONG_REPEATS='1')
 else:
  if prefill_tokens not in (None,64):raise ValueError('speed64 prefill is fixed;use long_session')
  dec=42 if decode_steps is None else decode_steps
  if not 1<=dec<=64:raise ValueError('speed64 decode exceeds128-token KV capacity')
  env['QBH_GENERATION_STEPS']=str(dec+1)
 env['QBH_PREFIX_FILE']=binding['device_prefix']
 return dict(model=cfg['model'],recipe=cfg['recipe'],profile=profile,rotation=dict(r3=r3,r4=r4,arithmetic=backend['arithmetic']),package_key=key,package_binding=binding,env=env,argv_after_package=cfg['argv_after_package'],quality_protocol=cfg['quality_protocol'],quality_status=cfg['quality_status'],warning='This resolves a configuration only. Model package/layout inputs,compiled layer count and runtime seal must match;active central device owner required for execution.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--r3',choices=['off','on'],default='off');p.add_argument('--r4',choices=['off','hmx_dense','hvx_butterfly'],default='off');p.add_argument('--rounding',type=int,choices=[0,2],default=2);p.add_argument('--profile',choices=['speed64','quality2048','long_session'],default='speed64');p.add_argument('--client-cap',choices=['auto','2','3'],default='auto');p.add_argument('--targets',type=int,default=1984);p.add_argument('--prefill-tokens',type=int);p.add_argument('--decode-steps',type=int)
 a=p.parse_args();print(json.dumps(resolve(a.r3=='on',a.r4,a.rounding,a.profile,a.client_cap,a.targets,a.prefill_tokens,a.decode_steps),indent=2))
