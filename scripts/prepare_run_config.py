#!/usr/bin/env python3
import argparse
from pathlib import Path
import yaml

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int);p.add_argument('--output-dir');p.add_argument('--mode',choices=('train','eval'),default='train');p.add_argument('--checkpoint');a=p.parse_args(); c=yaml.safe_load(Path(a.config).read_text())
 if a.seed is not None:c['TRAIN']['RANDOM_SEED']=a.seed
 if a.output_dir:c['OUTPUT_DIR']=a.output_dir;c['OUTPUT_DIR_SAVE']=a.output_dir
 if a.mode=='eval':
  c['TRAIN']['ENABLE']=False;c['TEST']['ENABLE']=True
  if not a.checkpoint: raise SystemExit('--checkpoint is required for --mode eval')
  c['TEST']['LOAD_PRETRAIN']=True;c['TEST']['PRETRAIN_MODEL_PATH']=str(Path(a.checkpoint).resolve())
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(yaml.safe_dump(c,sort_keys=False))
if __name__=='__main__':main()
