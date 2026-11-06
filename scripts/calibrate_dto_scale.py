#!/usr/bin/env python3
import argparse,csv,json,statistics
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--gradients',required=True);p.add_argument('--output',required=True);p.add_argument('--base-task-alpha',type=float,default=1.0);p.add_argument('--max-step',type=int,default=500);p.add_argument('--config');a=p.parse_args(); rows=[]
 with Path(a.gradients).open() as f:
  for r in csv.DictReader(f):
   if int(float(r['step']))<=a.max_step:
    d,t=float(r['distill_norm']),float(r['task_norm'])
    if d>0 and t>0: rows.append(a.base_task_alpha*t/d)
 if not rows: raise SystemExit('no finite early Meta-Train gradient rows')
 alpha=float(statistics.median(rows)); payload={'source':str(Path(a.gradients).resolve()),'max_step':a.max_step,'n_rows':len(rows),'base_task_alpha':a.base_task_alpha,'static_kl_alpha':alpha,'selection_split':'Meta-Train only'}
 Path(a.output).write_text(json.dumps(payload,indent=2)+'\n')
 if a.config:
  import yaml; p=Path(a.config); c=yaml.safe_load(p.read_text()); c.setdefault('DTO_KD',{})['STATIC_KL_ALPHA']=alpha; p.write_text(yaml.safe_dump(c,sort_keys=False))
 print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
