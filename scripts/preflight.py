#!/usr/bin/env python3
import argparse,importlib,os
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',default=os.environ.get('TSFSAR_SSV2_ROOT','/data/fsar_workspace/datasets/SSv2-Small'));p.add_argument('--skip-data',action='store_true');a=p.parse_args()
 import torch,transformers
 print('python',__import__('sys').version.split()[0]);print('torch',torch.__version__,'cuda',torch.version.cuda,'available',torch.cuda.is_available());print('transformers',transformers.__version__)
 for m in ('decord','einops','yaml','sklearn'):importlib.import_module(m);print('import',m,'ok')
 model=Path('model/vit-b-16'); req=('config.json','pytorch_model.bin','preprocessor_config.json','tokenizer.json','vocab.json','merges.txt'); missing=[str(model/x) for x in req if not (model/x).is_file()]
 if missing:raise SystemExit('missing CLIP files: '+', '.join(missing))
 if not a.skip_data:
  root=Path(a.data_root);ann=root/'annotations'
  for f in ('train_few_shot.txt','val_few_shot.txt','test_few_shot.txt','dataset_info.json'):
   if not (ann/f).is_file():raise SystemExit(f'missing dataset metadata: {ann/f}')
  videos=sum(1 for x in root.iterdir() if x.suffix.lower() in ('.mp4','.mkv','.webm'));print('dataset',root,'videos',videos)
  if videos==0:raise SystemExit('no SSv2 videos found')
 print('TSFSAR_PREFLIGHT_OK')
if __name__=='__main__':main()
