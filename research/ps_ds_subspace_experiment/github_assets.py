"""Fetch verified private Release assets after cloning this experiment branch.

No tokens in scripts, no model inference, no training or queue operations.
Existing verified files are reused. Failed downloads stay in their temporary
directory; existing unverified files are never overwritten or removed.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from .protocol import HERE, ROOT, sha_file

REPOSITORY = 'baizeCrystal/TS-FSAR-Dataset-Transfer'
TAG = 'psds-assets-20261108'
MANIFEST_NAME = 'PSDS_FIXED_EPISODES_20261108.tar.gz'
MANIFEST_SHA = 'b324a5d6e1fcb35275e8a04bc60f73cbe7dc540ee73e38e025209e1966e390d9'


def download(tag, name, expected, directory, minimum_free=64*1024**2):
    target=directory/name
    if target.exists():
        if sha_file(target)!=expected:
            raise RuntimeError('existing file failed SHA256; preserved for inspection: '+str(target))
        print('REUSE_VERIFIED',target,flush=True)
        return target
    if shutil.disk_usage(directory).free < minimum_free:
        raise RuntimeError('insufficient disk space in '+str(directory))
    temporary=Path(tempfile.mkdtemp(prefix='.gh-download-',dir=directory))
    subprocess.run(['gh','release','download',tag,'-R',REPOSITORY,'--pattern',name,'--dir',str(temporary)],check=True)
    candidate=temporary/name
    if sha_file(candidate)!=expected:
        raise RuntimeError('download SHA256 failed; file preserved: '+str(candidate))
    if target.exists():
        raise RuntimeError('destination appeared during download; refusing overwrite')
    candidate.rename(target)
    temporary.rmdir()
    print('DOWNLOADED_VERIFIED',target,flush=True)
    return target


def extract_checked(archive, destination, allowed):
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        files={m.name for m in members if m.isfile()}
        if files!=set(allowed) or len(files)!=sum(m.isfile() for m in members):
            raise RuntimeError('archive file list differs from expected inventory')
        if any(not (m.isfile() or m.isdir()) or '..' in Path(m.name).parts or Path(m.name).is_absolute() for m in members):
            raise RuntimeError('unsafe archive paths/types')
        for member in members:
            if not member.isfile():
                continue
            target=destination/member.name
            expected=allowed[member.name]
            if target.exists():
                if expected is None:
                    import hashlib
                    h=hashlib.sha256()
                    with tar.extractfile(member) as stream:
                        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
                    expected=h.hexdigest()
                if sha_file(target)!=expected:
                    raise RuntimeError('existing extracted file differs; preserved: '+str(target))
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as source,target.open('xb') as out:
                shutil.copyfileobj(source,out,length=1024*1024)
            if expected is not None and sha_file(target)!=expected:
                raise RuntimeError('extracted file failed SHA256: '+str(target))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets-dir',type=Path,required=True,help='use /data, not a nearly-full container root')
    p.add_argument('--e0-1shot',type=Path,help='reuse an existing file with the locked SHA256')
    p.add_argument('--e0-5shot',type=Path)
    p.add_argument('--clip-source',type=Path,help='reuse an existing verified vit-b-16 directory')
    args=p.parse_args()
    if shutil.which('gh') is None:
        raise RuntimeError('GitHub CLI is required; install gh and run gh auth login --hostname github.com --web')
    subprocess.run(['gh','auth','status'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
    args.assets_dir=args.assets_dir.resolve()
    args.assets_dir.mkdir(parents=True,exist_ok=True)
    lock=json.loads((HERE/'configs/locked.json').read_text())
    manifest=download(TAG,MANIFEST_NAME,MANIFEST_SHA,args.assets_dir,128*1024**2)
    allowed={f'research/ps_ds_subspace_experiment/results/shot{s}/{name}':None
             for s in (1,5) for name in ['episodes.jsonl.gz','manifest_meta.json','video_metadata.json','sampler_pixel_audit.json']}
    extract_checked(manifest,ROOT,allowed)
    checkpoints={}
    for shot,provided,tag,name in [
            (1,args.e0_1shot,'dto-kd-assets-20261107','E0_global_KL_model_best.pth'),
            (5,args.e0_5shot,TAG,'E0_5shot_model_best.pth')]:
        spec=lock['checkpoints'][str(shot)]
        path=provided.resolve() if provided else download(tag,name,spec['sha256'],args.assets_dir,spec['bytes']+64*1024**2)
        if sha_file(path)!=spec['sha256']:
            raise RuntimeError('wrong per-shot checkpoint: '+str(path))
        checkpoints[str(shot)]=str(path)
    clip=args.clip_source.resolve() if args.clip_source else None
    if clip is None:
        pack=download('dto-kd-assets-20261107','CLIP-vit-b-16-v2.tar.gz',
            '14b472af1611f39b771d7d9d329c154b23c9261af0de442c4ab6785a21d40bd6',args.assets_dir,1100*1024**2)
        extract_checked(pack,args.assets_dir/'clip',{f'vit-b-16/{n}':h for n,h in lock['clip_files'].items()})
        clip=args.assets_dir/'clip/vit-b-16'
    for name,expected in lock['clip_files'].items():
        if sha_file(clip/name)!=expected:
            raise RuntimeError('CLIP file failed SHA256: '+str(clip/name))
    link=ROOT/'model/vit-b-16'
    link.parent.mkdir(exist_ok=True)
    if link.exists() or link.is_symlink():
        for name,expected in lock['clip_files'].items():
            if sha_file(link/name)!=expected:raise RuntimeError('existing model link/files differ')
    else:
        link.symlink_to(clip,target_is_directory=True)
    from .evaluate import verify_manifest_bundle
    for shot in (1,5):verify_manifest_bundle(HERE/f'results/shot{shot}/episodes.jsonl.gz',shot,0,4)
    result=dict(repository=REPOSITORY,tag=TAG,checkpoint_paths=checkpoints,clip_directory=str(clip),
                verified=True,model_inference_performed=False)
    state=args.assets_dir/'PSDS_READY.json'
    if state.exists() and json.loads(state.read_text())!=result:
        raise RuntimeError('existing readiness record refers to other paths; choose another assets-dir')
    if not state.exists():state.write_text(json.dumps(result,indent=2)+'\n')
    print('PSDS_GITHUB_ASSETS_READY',state,flush=True)


if __name__=='__main__':main()
