#!/usr/bin/env python3
"""Stop named jobs from a campus status.json after checking live process identity.

Read-only unless --apply is passed. Never signals the launcher, edits status.json,
deletes results, or sends SIGKILL. The OLD launcher treats SIGTERM as a failure:
it drains other running jobs and cancels jobs that have not started.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal


def inspect_job(job, proc_root=Path('/proc')):
    pid = int(job['pid'])
    if pid <= 1:
        raise RuntimeError('invalid job PID')
    proc = proc_root / str(pid)
    try:
        args = proc.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
        cwd = proc.joinpath('cwd').resolve(strict=True)
        owner = proc.stat().st_uid
    except FileNotFoundError:
        return None
    if not args or args == ['']:
        return None
    if owner != os.geteuid() or os.getpgid(pid) != pid or os.getsid(pid) != pid:
        raise RuntimeError(f'PID {pid}: ownership/session mismatch; refusing to signal')
    if '-m' not in args or args[args.index('-m') + 1] not in (
            'experiments.dto_kd.rigorous', 'experiments.dto_kd.entry') or '--cfg' not in args:
        raise RuntimeError(f'PID {pid}: not a verified TS-FSAR training entry')
    config = Path(args[args.index('--cfg') + 1])
    actual = (cwd / config).resolve()
    expected = (Path(job['output']) / 'config.yaml').resolve()
    if actual != expected:
        raise RuntimeError(f'PID {pid}: configuration path mismatch; refusing to signal')
    return {'name': job['name'], 'pid': pid, 'config': str(actual), 'gpu': job.get('gpu')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--jobs', default='E1_1shot,E1_5shot')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = args.run_root.resolve()
    status = json.loads((root / 'status.json').read_text())
    names = set(args.jobs.split(','))
    jobs = status['jobs']
    if not names <= {job['name'] for job in jobs}:
        parser.error('unknown job name')
    selected = [job for job in jobs if job['name'] in names]
    # Verify ALL targets before sending any signal.
    targets = []
    for job in selected:
        if job['status'] != 'running':
            print(job['name'], 'not signalled: status=' + job['status'])
            continue
        row = inspect_job(job)
        if row:
            targets.append((job, row))
            print('TARGET', json.dumps(row))
    pending = [job['name'] for job in jobs if job['status'] == 'queued']
    print('OLD launcher will cancel unstarted jobs after this interruption:', pending)
    print('Other running jobs are not targeted; existing files are retained.')
    if not args.apply:
        print('DRY RUN. Add --apply to send SIGTERM to verified target sessions.')
        return
    record = {'requested_at': datetime.now(timezone.utc).isoformat(),
              'targets': [row for _, row in targets], 'unstarted_at_request': pending,
              'signal': 'SIGTERM', 'results_retained': True,
              'scope': 'interrupted pilot, not a completed equal-budget comparison'}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    with (root / ('stop_request_' + stamp + '.json')).open('x') as stream:
        json.dump(record, stream, indent=2)
    for job, row in targets:
        if inspect_job(job) == row:
            try:
                os.killpg(row['pid'], signal.SIGTERM)
                print('SIGTERM_SENT', row['name'], row['pid'])
            except ProcessLookupError:
                print('ALREADY_EXITED', row['name'])


if __name__ == '__main__':
    main()
