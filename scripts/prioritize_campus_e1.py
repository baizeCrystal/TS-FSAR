#!/usr/bin/env python3
"""Campus-only transition from the detach-control queue to fresh E1_safe jobs.

Download this stdlib-only helper to /tmp; no training source/configuration needs
to change, so existing source-matched CUDA certificates remain applicable.
Dry-run unless --apply is supplied. Only the verified adaptation launcher is
signalled. Historical E0/E2 queues are outside its scope.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def validate_launcher(pid, project, formal, proc_root=Path('/proc')):
    """Return False for an exited PID; reject identity mismatches."""
    if pid <= 1:
        raise RuntimeError('invalid launcher PID')
    proc = proc_root / str(pid)
    try:
        args = proc.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
        cwd = proc.joinpath('cwd').resolve(strict=True)
        owner = proc.stat().st_uid
    except FileNotFoundError:
        return False
    if not args or args == ['']:
        return False
    expected_script = project / 'scripts/launch_campus_suite.py'
    if owner != os.geteuid() or cwd != project:
        raise RuntimeError('launcher PID ownership/cwd mismatch; no signal sent')
    if not any((cwd / arg).resolve() == expected_script for arg in args[1:]):
        raise RuntimeError('PID is not the expected queue launcher; no signal sent')
    if '--output-root' not in args or (cwd / args[args.index('--output-root') + 1]).resolve() != formal:
        raise RuntimeError('launcher output-root mismatch; no signal sent')
    return True


def plan_transition(project, root, output):
    formal = root / 'formal'
    status = json.loads((formal / 'status.json').read_text())
    jobs = status['jobs']
    if {job['name'] for job in jobs} != {
            f'{experiment}_{shot}shot' for experiment in ('E0_detach', 'E1_safe') for shot in (1, 5)}:
        raise RuntimeError('not the expected four-job adaptation queue; refusing transition')
    safe_jobs = sorted((j for j in jobs if j['experiment'] == 'E1_safe'), key=lambda j: j['shot'])
    if any(j['status'] != 'queued' for j in safe_jobs):
        raise RuntimeError('E1_safe already started or changed state; inspect status instead of restarting it')
    if any(j['status'] == 'running' and j['experiment'] != 'E0_detach' for j in jobs):
        raise RuntimeError('unexpected running job')
    if output.exists():
        raise RuntimeError('output-root already exists; nothing will be overwritten')
    seed = safe_jobs[0]['seed']
    if any(j['seed'] != seed for j in safe_jobs):
        raise RuntimeError('shot seeds differ')
    certificates = []
    for job in safe_jobs:
        expected_config = project / f'configs/dto_kd/ssv2_small_{job["shot"]}shot_E1_safe.yaml'
        if Path(job['config']).resolve() != expected_config:
            raise RuntimeError('configuration does not belong to the specified project')
        certificate = Path(job['acceptance_report']).resolve()
        report = json.loads(certificate.read_text())
        checks = report.get('smokes', [])
        if report.get('passed') is not True or report.get('shot') != job['shot'] or report.get('seed') != seed:
            raise RuntimeError('incompatible acceptance report')
        if not any(s.get('mode') == 'safe_dynamic' and s.get('detach_teacher') is True
                   and s.get('adaptation_check', {}).get('passed') is True for s in checks):
            raise RuntimeError('E1_safe CUDA certification is missing')
        certificates.append(str(certificate))
    pid = int((root / 'launcher.pid').read_text().strip())
    if not validate_launcher(pid, project, formal):
        raise RuntimeError('old launcher already exited; inspect state before switching')
    if any(j['status'] == 'running' and j.get('gpu') not in (2, 3) for j in jobs):
        raise RuntimeError('unexpected GPU allocation; this helper targets GPUs 2,3 only')
    command = [sys.executable, '-u', str(project / 'scripts/launch_campus_suite.py'),
               '--protocol', 'v2', '--experiments', 'E1_safe', '--gpus', '2,3', '--seed', str(seed),
               '--acceptance-1shot', certificates[0], '--acceptance-5shot', certificates[1],
               '--activation-offload', 'diagnostics', '--launch-delay', '20',
               '--output-root', str(output / 'formal')]
    return pid, command, status


def resources_idle():
    for gpu in (2, 3):
        active = subprocess.check_output(
            ['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
            text=True).strip()
        if active:
            return False
        lock_root = Path(os.environ.get('TSFSAR_GPU_LOCK_DIR', f'/tmp/tsfsar-gpu-locks-{os.getuid()}'))
        lock_path = lock_root / f'gpu-{gpu}.lock'
        if lock_path.exists():
            with lock_path.open('r') as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--run-root', required=True, type=Path, help='old adaptation root, ABOVE formal/')
    parser.add_argument('--output-root', required=True, type=Path, help='new, nonexistent E1-only root')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    project, root, output = (p.resolve() for p in (args.project_root, args.run_root, args.output_root))
    pid, command, status = plan_transition(project, root, output)
    print('Verified old adaptation launcher:', pid, flush=True)
    print('Will interrupt E0_detach only in that queue and preserve every existing file.', flush=True)
    print('New queue: E1_safe_1shot and E1_safe_5shot on GPUs 2,3.', flush=True)
    print('E0_detach is deferred; no automatic repeat of the interrupted controls.', flush=True)
    if not args.apply:
        print('DRY RUN; pass --apply to perform the transition.', flush=True)
        return
    # Audit the interruption before signalling; never rewrite status.json.
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    record = {'requested_at': stamp, 'old_launcher_pid': pid, 'old_status': status,
              'new_output': str(output), 'new_experiments': ['E1_safe'],
              'old_controls': 'interrupted/deferred, incomplete budget; retain outputs',
              'acceptance': 'reuse existing certificates; training code is unchanged'}
    with (root / f'e1_priority_request_{stamp}.json').open('x') as stream:
        json.dump(record, stream, indent=2)
    # Revalidate immediately before sending one signal to the queue owner.
    if validate_launcher(pid, project, root / 'formal'):
        os.kill(pid, signal.SIGTERM)
    for attempt in range(30):
        if not validate_launcher(pid, project, root / 'formal') and resources_idle():
            break
        if attempt % 5 == 0:
            print('Waiting for the old launcher and GPUs 2,3 to release...', flush=True)
        time.sleep(2)
    else:
        raise RuntimeError('resources did not release within 60s; E1 was NOT started; inspect GPU/process state')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'acceptance').symlink_to(root / 'acceptance', target_is_directory=True)
    (output / 'launcher.pid').write_text(str(os.getpid()) + '\n')
    (output / 'transition.json').write_text(json.dumps(record, indent=2) + '\n')
    (project / 'runs/campus/latest_adaptation_root.txt').write_text(str(output) + '\n')
    # Keep runner's python in the same environment even after a terminal reopen.
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    os.chdir(project)
    print('Starting fresh E1_safe queue:', output, flush=True)
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
