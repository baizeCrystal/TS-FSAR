#!/usr/bin/env python3
"""Campus-only, bounded restart: stop named old queues, recertify, start 8 jobs.

Dry-run unless --apply is supplied. All prior files are retained. Never kills
unrelated GPU users, resumes a trained model, or changes a running source tree.
E3 runs as a separate train-gradient-only calibration stage after all eight jobs.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stop_campus_jobs import inspect_job
from scripts.prioritize_campus_e1 import validate_launcher


def inspect_running(job):
    try:
        return inspect_job(job)
    except ProcessLookupError:
        return None


def queue_info(root):
    status = json.loads((root / 'status.json').read_text())
    jobs = status['jobs']
    if not jobs or len({Path(j['config']).resolve().parents[2] for j in jobs}) != 1:
        raise RuntimeError('old queue project identity is ambiguous: ' + str(root))
    project = Path(jobs[0]['config']).resolve().parents[2]
    if project == ROOT:
        raise RuntimeError('new worktree must differ from all old running source trees')
    for job in jobs:
        if root not in Path(job['output']).resolve().parents:
            raise RuntimeError('job output outside selected old queue')
    pid_file = root / 'launcher.pid'
    if not pid_file.exists():
        pid_file = root.parent / 'launcher.pid'
    launcher = None
    if pid_file.is_file():
        pid = int(pid_file.read_text().strip())
        try:
            if validate_launcher(pid, project, root):
                launcher = pid
        except RuntimeError:
            # A recorded PID can be reused. Never signal a mismatched process.
            print('STALE_LAUNCHER_PID_NOT_SIGNALLED', pid, root, flush=True)
    targets = []
    for job in jobs:
        if job.get('pid') and job['status'] in ('running', 'interrupted'):
            target = inspect_running(job)
            if target:
                targets.append(target)
    return {'root': str(root), 'project': str(project), 'launcher': launcher, 'targets': targets}


def gpu_idle(gpu):
    active = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
                                      '--format=csv,noheader,nounits'], text=True).strip()
    if active:
        return False
    locks = Path(os.environ.get('TSFSAR_GPU_LOCK_DIR', '/tmp/tsfsar-gpu-locks-' + str(os.getuid())))
    path = locks / f'gpu-{gpu}.lock'
    if path.exists():
        with path.open() as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
    return True


def stop_queues(infos, timeout=60):
    # Identity validation for every queue has completed before any signal.
    for info in infos:
        if info['launcher']:
            if validate_launcher(info['launcher'], Path(info['project']), Path(info['root'])):
                try:
                    os.kill(info['launcher'], signal.SIGTERM)
                except ProcessLookupError:
                    pass
                print('STOP_OLD_LAUNCHER', info['launcher'], info['root'], flush=True)
    # Also handle jobs orphaned by an earlier exited launcher. Re-read after
    # signalling so any last job launched during preflight is included.
    for info in infos:
        status = json.loads((Path(info['root']) / 'status.json').read_text())
        for job in status['jobs']:
            if job.get('pid') and job['status'] in ('running', 'interrupted'):
                row = inspect_running(job)
                if row:
                    try:
                        os.killpg(row['pid'], signal.SIGTERM)
                        print('STOP_OLD_JOB', row['name'], row['pid'], flush=True)
                    except ProcessLookupError:
                        pass
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        live = []
        for info in infos:
            if info['launcher'] and validate_launcher(info['launcher'], Path(info['project']), Path(info['root'])):
                live.append(info['launcher'])
            status = json.loads((Path(info['root']) / 'status.json').read_text())
            for job in status['jobs']:
                if job.get('pid') and job['status'] in ('running', 'interrupted') and inspect_running(job):
                    live.append(job['pid'])
        if not live:
            return
        print('WAIT_OLD_PROCESSES', sorted(set(live)), flush=True)
        time.sleep(3)
    raise RuntimeError('old processes did not exit within 60s; new queue not started; no SIGKILL used')


def install_interrupt_handlers(children):
    def interrupt(signum, frame):
        for proc in children:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupt)
    # Respect the shell's ignored SIGINT for a background/nohup job: Ctrl+C
    # used to exit the foreground tail must not cancel the training driver.
    if signal.getsignal(signal.SIGINT) != signal.SIG_IGN:
        signal.signal(signal.SIGINT, interrupt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-run-root', action='append', required=True, type=Path,
                        help='directory containing an old queue status.json; repeat for each queue')
    parser.add_argument('--previous-acceptance-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    gpus = [int(x) for x in args.gpus.split(',')]
    if len(gpus) < 2 or len(set(gpus)) != len(gpus) or min(gpus) < 0 or 7 in gpus:
        parser.error('use at least two distinct allocated GPUs; GPU 7 excluded after reported ECC')
    out = args.output_root.resolve()
    if out.exists():
        parser.error('output-root exists; choose a new path')
    ancestor = out.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < 30 * 1024**3:
        parser.error('less than 30 GiB free for new acceptance/checkpoints/predictions; free space on /data first')
    infos = [queue_info(path.resolve()) for path in args.old_run_root]
    for shot in (1, 5):
        previous = args.previous_acceptance_root.resolve() / f'shot{shot}/acceptance.json'
        report = json.loads(previous.read_text())
        if report.get('passed') is not True or report['shot'] != shot or report['seed'] != args.seed:
            parser.error('previous initialization report incompatible: ' + str(previous))
        bank = previous.parent / report['initialization_file']
        h = hashlib.sha256()
        with bank.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != report['initialization_file_sha256']:
            parser.error('previous initial-model file changed; old queues have not been stopped')
    for gpu in gpus:
        ecc = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                                      '--query-gpu=ecc.errors.uncorrected.volatile.total',
                                      '--format=csv,noheader,nounits'], text=True).strip()
        if ecc != '0':
            parser.error(f'GPU {gpu} has unverified/nonzero ECC: {ecc}')
    print(json.dumps({'old_queues': infos, 'new_output': str(out), 'gpus': gpus,
                      'experiments': ['E0', 'E1_safe', 'E2_half', 'E2_double'],
                      'shots': [1, 5], 'selection': '24-class Novel Test, 10000 episodes every 500 steps',
                      'sampling': 'official worker RNG stream, no fixed episode bank',
                      'E3': 'separate calibration using new E1_safe training gradients only'}, indent=2), flush=True)
    if not args.apply:
        print('DRY_RUN_ONLY; --apply is required to stop old queues and launch acceptance.', flush=True)
        return
    out.mkdir(parents=True)
    (out / 'launcher.pid').write_text(str(os.getpid()) + '\n')
    (out / 'restart_plan.json').write_text(json.dumps({'old_queues': infos,
        'requested_at': datetime.now(timezone.utc).isoformat(), 'gpus': gpus,
        'files_retained': True, 'fresh_initialization_only': True}, indent=2) + '\n')
    stop_queues(infos)
    for _ in range(20):
        busy = [gpu for gpu in gpus if not gpu_idle(gpu)]
        if not busy:
            break
        print('WAIT_GPUS', busy, flush=True)
        time.sleep(3)
    else:
        raise RuntimeError('selected GPUs are still occupied; no unrelated process was stopped')
    os.chdir(ROOT)
    (ROOT / 'runs/campus').mkdir(parents=True, exist_ok=True)
    (ROOT / 'runs/campus/latest_official_test_root.txt').write_text(str(out) + '\n')
    children = []
    install_interrupt_handlers(children)
    for shot, gpu in zip((1, 5), gpus[:2]):
        target = out / f'acceptance/shot{shot}'
        target.parent.mkdir(parents=True, exist_ok=True)
        command = ['bash', str(ROOT / 'scripts/run_campus_acceptance.sh'), '--shot', str(shot),
                   '--gpu', str(gpu), '--seed', str(args.seed), '--output', str(target),
                   '--selection-protocol', 'official-test', '--adaptation-checks',
                   '--reuse-initialization-report', str(args.previous_acceptance_root.resolve() / f'shot{shot}/acceptance.json')]
        children.append(subprocess.Popen(command, cwd=str(ROOT), start_new_session=True))
    failed = any([proc.wait() != 0 for proc in children])
    children.clear()
    if failed:
        raise RuntimeError('new CUDA acceptance failed; formal queue NOT started')
    for gpu in gpus:
        if not gpu_idle(gpu):
            raise RuntimeError('GPU occupied after acceptance; formal queue not started')
    command = [sys.executable, '-u', str(ROOT / 'scripts/launch_campus_suite.py'),
               '--protocol', 'v2', '--selection-protocol', 'official-test',
               '--experiments', 'E0,E1_safe,E2', '--gpus', args.gpus, '--seed', str(args.seed),
               '--acceptance-1shot', str(out / 'acceptance/shot1/acceptance.json'),
               '--acceptance-5shot', str(out / 'acceptance/shot5/acceptance.json'),
               '--activation-offload', 'diagnostics', '--output-root', str(out / 'formal')]
    print('START_OFFICIAL_TEST_QUEUE', command, flush=True)
    formal = out / 'formal'
    formal.mkdir(exist_ok=True)
    proc = subprocess.Popen(command, cwd=str(ROOT), start_new_session=True)
    (formal / 'launcher.pid').write_text(str(proc.pid) + '\n')
    children.append(proc)
    rc = proc.wait()
    children.clear()
    if rc:
        raise RuntimeError('formal queue failed/interrupted; E3 NOT started')
    print('FIRST_EIGHT_COMPLETED; calibrating E3 from E1_safe training probes only.', flush=True)
    command = [sys.executable, '-u', str(ROOT / 'scripts/run_campus_official_e3.py'),
               '--run-root', str(out), '--gpus', ','.join(str(x) for x in gpus[:2]), '--apply']
    proc = subprocess.Popen(command, cwd=str(ROOT), start_new_session=True)
    children.append(proc)
    rc = proc.wait()
    children.clear()
    if rc:
        raise RuntimeError('E3 stage failed; inspect preserved calibration and logs')
    print('OFFICIAL_TEST_ALL_STAGES_COMPLETED', out, flush=True)


if __name__ == '__main__':
    main()
