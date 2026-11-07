#!/usr/bin/env python3
"""CPU metadata-only presence/readability checks for all canonical SSv2 files."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(data_root):
    result, failures = {}, []
    for split in ('train', 'val', 'test'):
        rows = (ROOT / f'configs/ssv2_small/{split}_few_shot.txt').read_text().splitlines()
        count = 0
        for row in rows:
            if not row.strip():
                continue
            video = Path(data_root) / (row.split('/')[1] + '.mp4')
            try:
                valid = video.is_file() and video.stat().st_size > 0 and os.access(video, os.R_OK)
            except OSError:
                valid = False
            if valid:
                count += 1
            else:
                failures.append(str(video))
        result[split] = {'required': len(rows), 'readable_nonempty': count}
    if failures:
        raise RuntimeError(f'{len(failures)} missing/empty/unreadable videos; first 20: {failures[:20]}')
    return {'data_root': str(Path(data_root).resolve()), 'splits': result,
            'scope': 'file metadata/read permission only; not full video decode or content hash'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=os.environ.get('TSFSAR_SSV2_ROOT'))
    args = parser.parse_args()
    if not args.data_root:
        parser.error('set TSFSAR_SSV2_ROOT or --data-root')
    print(json.dumps(check(args.data_root), indent=2))


if __name__ == '__main__':
    main()
