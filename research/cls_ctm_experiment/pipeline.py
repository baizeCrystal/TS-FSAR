"""One-shot pipeline: disposable GPU preflight, fresh module train, paired test."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from .common import ROOT, output_lock


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shot', type=int, choices=(1,5), required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    os.chdir(ROOT)
    args.output = args.output.resolve()
    common = ['--shot', str(args.shot), '--data-root', str(args.data_root.resolve()),
              '--checkpoint', str(args.checkpoint.resolve())]
    with output_lock(args.output):
        for name, smoke in [('preflight_train',True),('train',False)]:
            train_out = args.output/name
            cmd = [sys.executable,'-u','-m','research.cls_ctm_experiment.train',
                   *common,'--output',str(train_out)]
            if smoke:
                cmd += ['--smoke']
            if (train_out/'resume_latest.pth').exists():
                cmd += ['--resume']
            print('CLSCTM_PHASE', name, flush=True)
            subprocess.run(cmd, check=True)
            if smoke:
                # GPU engineering check only. The formal head is initialized
                # afresh; neither Base nor Test preflight changes formal state.
                command = [sys.executable,'-u','-m','research.cls_ctm_experiment.evaluate',
                    *common,'--stage','smoke','--allow-smoke-module',
                    '--module-checkpoint',str(train_out/'module_final.pth'),
                    '--manifest',str(args.manifest.resolve()),
                    '--output',str(args.output/'preflight_evaluation')]
                subprocess.run(command,check=True)
        for stage in ('smoke','explore','formal'):
            print('CLSCTM_PHASE', stage, flush=True)
            command = [sys.executable,'-u','-m','research.cls_ctm_experiment.evaluate',
                *common,'--stage',stage,'--module-checkpoint',str(args.output/'train/module_final.pth'),
                '--manifest',str(args.manifest.resolve()),'--output',str(args.output/'evaluation')]
            subprocess.run(command,check=True)
        print('CLSCTM_PIPELINE_COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
