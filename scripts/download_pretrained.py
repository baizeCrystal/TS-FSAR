#!/usr/bin/env python3
import argparse

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo-id',default='openai/clip-vit-base-patch16');p.add_argument('--output',default='model/vit-b-16');p.add_argument('--revision');a=p.parse_args()
 try:from huggingface_hub import snapshot_download
 except ImportError as e:raise SystemExit('install huggingface_hub first') from e
 kw={'repo_id':a.repo_id,'local_dir':a.output,'local_dir_use_symlinks':False,'resume_download':True}
 if a.revision:kw['revision']=a.revision
 snapshot_download(**kw);print('Downloaded. Run scripts/verify_weights.py; hash mismatch is a hard error.')
if __name__=='__main__':main()
