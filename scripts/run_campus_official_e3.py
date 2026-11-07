#!/usr/bin/env python3
"""After the eight official-Test jobs finish, freeze E3 from E1_safe train probes.

Dry-run unless --apply. Never reads Test accuracy/predictions for calibration.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.launch_campus_suite import run_queue
from scripts.restart_campus_official_test import gpu_idle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', required=True, type=Path, help='new official-Test restart root, above formal/')
    p.add_argument('--gpus', default='2,3')
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    root = a.run_root.resolve()
    gpus = [int(x) for x in a.gpus.split(',')]
    if len(gpus) != 2 or len(set(gpus)) != 2 or min(gpus) < 0 or 7 in gpus:
        p.error('choose two distinct allocated healthy GPUs; GPU 7 excluded')
    status = json.loads((root / 'formal/status.json').read_text())
    if len(status['jobs']) != 8 or not all(j['status'] == 'completed' for j in status['jobs']):
        p.error('the eight new jobs must complete successfully before the E3 stage')
    target = root / 'E3'
    if target.exists():
        p.error('E3 output already exists; inspect it instead of overwriting/retrying')
    jobs = []
    for shot in (1, 5):
        source = root / f'formal/E1_safe_{shot}shot'
        protocol = json.loads((source / 'protocol_v2.json').read_text())
        if (protocol.get('selection_protocol') != 'official-test' or protocol.get('dto_mode') != 'safe_dynamic'
                or protocol.get('detach_teacher') is not True):
            p.error('E3 source is not new official-Test E1_safe')
        complete = json.loads((source / 'training_complete.json').read_text())
        if complete.get('completed_episodes') != 10000:
            p.error('source did not finish the full 10000-episode budget')
        jobs.append({'name': f'E3_{shot}shot', 'experiment': 'E3', 'shot': shot,
                     'seed': protocol['seed'], 'config': str(target / f'E3_{shot}shot.yaml'),
                     'output': str(target / f'E3_{shot}shot'), 'status': 'queued', 'gpu': None,
                     'pid': None, 'returncode': None, 'console': str(target / f'E3_{shot}shot/console.log'),
                     'protocol': 'v2', 'selection_protocol': 'official-test',
                     'acceptance_report': str(root / f'acceptance/shot{shot}/acceptance.json'),
                     'activation_offload': 'diagnostics'})
    print('E3: calibrate each shot from its new E1_safe first 500 train episodes; Test values are not read.')
    print(json.dumps(jobs, indent=2), flush=True)
    if not a.apply:
        print('DRY_RUN_ONLY; use --apply when the two allocated GPUs are idle.')
        return
    if not all(gpu_idle(gpu) for gpu in gpus):
        p.error('GPU occupied; no jobs started')
    target.mkdir()
    (target / 'launcher.pid').write_text(str(os.getpid()) + '\n')
    for job in jobs:
        shot = job['shot']
        subprocess.run([sys.executable, str(ROOT / 'scripts/calibrate_rigorous_scale.py'),
                        '--gradients', str(root / f'formal/E1_safe_{shot}shot/gradient_groups.jsonl'),
                        '--config', str(ROOT / f'configs/dto_kd/ssv2_small_{shot}shot_E3.yaml'),
                        '--output-config', job['config'],
                        '--output-report', str(target / f'calibration_{shot}shot.json')], check=True)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(run_queue(jobs, gpus, target / 'status.json'))


if __name__ == '__main__':
    main()
