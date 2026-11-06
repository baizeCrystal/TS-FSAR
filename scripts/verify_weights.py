#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--model-dir',default='model/vit-b-16');p.add_argument('--manifest',default='assets/CLIP_VIT_B16_SHA256SUMS.json');a=p.parse_args(); root=Path(a.model_dir); expected=json.loads(Path(a.manifest).read_text())['files']; bad=[]
 for rel,want in expected.items():
  q=root/rel;got=sha(q) if q.is_file() else 'MISSING';print(rel,got)
  if got!=want:bad.append(f'{rel}: {got} != {want}')
 if bad:raise SystemExit('weight verification failed:\n'+'\n'.join(bad))
 print('CLIP_WEIGHT_VERIFIED')
if __name__=='__main__':main()
