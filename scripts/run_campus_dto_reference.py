#!/usr/bin/env python3
"""Fresh DTO reference acceptance; formal training requires explicit --train.

Campus only with --apply. Defaults to a file-only plan, never stops other jobs,
never overwrites output, never reuses an old certificate's pass status. The
default formal selection is four E1 exact/amortized jobs; controls can be named
explicitly with --experiments. No automatic E3 calibration from another mode.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.restart_campus_official_test import gpu_idle, install_interrupt_handlers
from scripts.launch_campus_suite import build_jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-acceptance-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--gpus', default='0,2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train', action='store_true', help='only after BOTH new acceptances pass')
    parser.add_argument('--experiments', default='E1_exact,E1_amortized')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    gpus = [int(x) for x in args.gpus.split(',')]
    allowed = {'E0', 'E0_detach', 'E1_exact', 'E1_amortized', 'E1_task_anchored', 'E2'}
    experiments = set(args.experiments.split(','))
    if len(gpus) < 2 or len(set(gpus)) != len(gpus) or min(gpus) < 0 or 7 in gpus:
        parser.error('at least two distinct allocated GPUs; GPU7 excluded after reported ECC')
    if not experiments or not experiments <= allowed:
        parser.error('select explicit reference/control variants: ' + ','.join(sorted(allowed)))
    out = args.output_root.resolve()
    if out.exists():
        parser.error('output already exists; no automatic overwrite or retry')
    previous = args.previous_acceptance_root.resolve()
    for shot in (1, 5):
        path = previous / f'shot{shot}/acceptance.json'
        report = json.loads(path.read_text())
        if report.get('passed') is not True or report['shot'] != shot or report['seed'] != args.seed:
            parser.error('incompatible initial-model report: ' + str(path))
        bank = path.parent / report['initialization_file']
        h = hashlib.sha256()
        with bank.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b''):
                h.update(block)
        if h.hexdigest() != report['initialization_file_sha256']:
            parser.error('initial-model bank checksum failed')
    ancestor = out.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < 30 * 1024**3:
        parser.error('need at least 30 GiB free on the output filesystem')
    jobs = [j['name'] for j in build_jobs([1, 5], args.seed, out / 'formal', set(), 'v2', True, True)
            if j['experiment'] in experiments]
    print(json.dumps({'output': str(out), 'gpus': gpus, 'initialization_reports': str(previous),
                      'cuda_acceptance': ['E0-equivalence', 'E0', 'E0_detach', 'E1_amortized', 'E1_exact', 'E1_task_anchored'],
                      'selection': 'official 24-class Test', 'formal_jobs': jobs if args.train else [],
                      'automatic_E3': False, 'stop_existing_jobs': False}, indent=2), flush=True)
    if not args.apply:
        print('DRY_RUN; --apply runs fresh CUDA acceptance. Add --train ONLY for formal jobs after acceptance.')
        return
    for gpu in gpus:
        ecc = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-gpu=ecc.errors.uncorrected.volatile.total', '--format=csv,noheader,nounits'], text=True).strip()
        if ecc != '0' or not gpu_idle(gpu):
            parser.error(f'GPU {gpu} busy or ECC not zero; no existing process was stopped')
    out.mkdir(parents=True)
    (out / 'launcher.pid').write_text(str(os.getpid()) + '\n')
    (ROOT / 'runs/campus').mkdir(parents=True, exist_ok=True)
    (ROOT / 'runs/campus/latest_dto_reference_root.txt').write_text(str(out) + '\n')
    children = []
    install_interrupt_handlers(children)
    try:
        for shot, gpu in zip((1, 5), gpus[:2]):
            command = ['bash', str(ROOT / 'scripts/run_campus_acceptance.sh'), '--shot', str(shot),
                       '--gpu', str(gpu), '--seed', str(args.seed), '--output', str(out / f'acceptance/shot{shot}'),
                       '--selection-protocol', 'official-test',
                       '--reuse-initialization-report', str(previous / f'shot{shot}/acceptance.json')]
            if 'E1_task_anchored' in experiments:
                command.append('--task-anchored-checks')
            if experiments & {'E1_exact', 'E1_amortized'}:
                command.append('--dto-reference-checks')
            children.append(subprocess.Popen(command, cwd=ROOT, start_new_session=True))
        returncodes = [p.wait() for p in children]
        children.clear()
        if any(returncodes):
            raise RuntimeError('new DTO reference acceptance failed; formal training NOT started')
        print('BOTH_DTO_REFERENCE_ACCEPTANCES_PASSED', flush=True)
        if args.train:
            command = [sys.executable, '-u', str(ROOT / 'scripts/launch_campus_suite.py'),
                       '--protocol', 'v2', '--selection-protocol', 'official-test',
                       '--experiments', args.experiments, '--gpus', args.gpus, '--seed', str(args.seed),
                       '--acceptance-1shot', str(out / 'acceptance/shot1/acceptance.json'),
                       '--acceptance-5shot', str(out / 'acceptance/shot5/acceptance.json'),
                       '--activation-offload', 'diagnostics', '--output-root', str(out / 'formal')]
            proc = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
            children.append(proc)
            rc = proc.wait()
            children.clear()
            if rc:
                raise RuntimeError('reference/control formal queue failed; no automatic retry')
    finally:
        for proc in children:
            if proc.poll() is None:
                import signal
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


if __name__ == '__main__':
    main()
