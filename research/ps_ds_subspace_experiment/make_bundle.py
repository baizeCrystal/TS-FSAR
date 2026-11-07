"""Build a small standalone evaluation workspace, without weights/data/secrets.

Explicit source allowlist. Unmodified original runtime files are copied into a
NEW archive root, so campus users never have to overwrite training worktrees.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
from .protocol import ROOT,HERE,sha_file,canonical


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=HERE/'psds_evaluation_code.tar.gz')
    args=p.parse_args()
    selected=set()
    for directory in ['trainers','datasets','utils']:
        selected.update((ROOT/directory).glob('*.py'))
    lock=json.loads((HERE/'configs/locked.json').read_text())
    selected.update(ROOT/rel for rel in lock['official_files'])
    selected.add(ROOT/'requirements-campus.txt')
    for path in HERE.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        rel=path.relative_to(HERE)
        if rel.parts[0]=='results':
            if path.name not in ['episodes.jsonl.gz','manifest_meta.json','video_metadata.json',
                                 'sampler_pixel_audit.json','cpu_tests_final.log',
                                 'cpu_numerical_diagnostics.json','integrity_audit.json',
                                 'accuracy_table.md','paired_statistics_pending.json']:
                continue
        elif path.suffix not in ['.py','.md','.json','.sh','.txt']:
            continue
        if path.name.endswith('.sha256'):
            continue
        selected.add(path)
    hashes={str(path.relative_to(ROOT)):sha_file(path) for path in sorted(selected)}
    manifest=(canonical(dict(files=hashes,includes_weights=False,includes_data=False))+'\n').encode()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as f,gzip.GzipFile(fileobj=f,mode='wb',compresslevel=1,mtime=0) as gz:
        with tarfile.open(fileobj=gz,mode='w|') as archive:
            for path in sorted(selected):
                if path.is_symlink():
                    raise ValueError('unexpected source symlink: '+str(path))
                archive.add(path,arcname='TS-FSAR-PSDS-eval/'+str(path.relative_to(ROOT)),recursive=False)
            info=tarfile.TarInfo('TS-FSAR-PSDS-eval/BUNDLE_FILES_SHA256.json')
            info.size=len(manifest)
            archive.addfile(info,io.BytesIO(manifest))
    # Stream through compressed archive and validate every payload hash.
    seen=set()
    with tarfile.open(args.output,'r|gz') as archive:
        for member in archive:
            rel=Path(member.name).relative_to('TS-FSAR-PSDS-eval').as_posix()
            if rel=='BUNDLE_FILES_SHA256.json':
                continue
            stream=archive.extractfile(member)
            h=hashlib.sha256()
            for chunk in iter(lambda:stream.read(1024*1024),b''):
                h.update(chunk)
            if h.hexdigest()!=hashes[rel]:
                raise ValueError('archive payload mismatch: '+rel)
            seen.add(rel)
    assert seen==set(hashes)
    with Path(str(args.output)+'.sha256').open('x') as out:
        out.write(sha_file(args.output)+'  '+args.output.name+'\n')
    print('BUNDLE_VERIFIED',args.output,args.output.stat().st_size,'bytes',len(hashes),'files')


if __name__=='__main__':
    main()
