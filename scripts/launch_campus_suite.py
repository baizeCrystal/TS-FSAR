#!/usr/bin/env python3
"""Queue the E0/E1/E2 controls, one independent process per allocated GPU.

This module uses only the Python standard library and never starts training on
import or with --dry-run. E3 needs Meta-Train calibration and is not auto-launched.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def build_jobs(shots, seed, output_root, skip, protocol='legacy', include_adaptations=False,
               include_references=False):
    jobs = []
    variants = [("E0", None), ("E1", None), ("E2", "0.05"), ("E2", "0.1"), ("E2", "0.2")]
    if include_adaptations:
        variants += [('E0_detach', None), ('E1_detach', None), ('E1_safe', None)]
    if include_references:
        variants += [('E1_exact', None), ('E1_amortized', None)]
    for experiment, alpha in variants:
        for shot in shots:
            if protocol == 'v2' and experiment == 'E2':
                # Center = E0 (identical objective certified separately), so
                # only half/double the released KL coefficient needs training.
                if alpha == '0.1':
                    continue
                if shot == 5:
                    alpha_for_shot = {'0.05': '0.005', '0.2': '0.02'}[alpha]
                else:
                    alpha_for_shot = alpha
            else:
                alpha_for_shot = alpha
            name = f"{experiment}_{shot}shot" + (f"_kl{alpha_for_shot}" if alpha_for_shot else "")
            if name in skip:
                continue
            suffix = f"_static_kl_{alpha_for_shot}" if alpha_for_shot else ""
            cfg = ROOT / f"configs/dto_kd/ssv2_small_{shot}shot_{experiment}{suffix}.yaml"
            if not cfg.is_file():
                raise ValueError(f"missing config: {cfg}")
            out = output_root / name
            jobs.append({"name": name, "experiment": experiment, "shot": shot,
                         "seed": seed, "config": str(cfg), "output": str(out),
                         "status": "queued", "gpu": None, "pid": None,
                         "returncode": None, "console": str(out / "console.log")})
    return jobs


def job_command(job, gpu):
    command = ["bash", str(ROOT / "scripts/run_campus_experiment.sh"),
            "--experiment", job["experiment"], "--shot", str(job["shot"]),
            "--seed", str(job["seed"]), "--gpu", str(gpu),
            "--config", job["config"], "--output", job["output"],
            "--activation-offload", job.get("activation_offload", "diagnostics")]
    command += ['--protocol', job.get('protocol', 'legacy')]
    command += ['--selection-protocol', job.get('selection_protocol', 'meta-val-fixed')]
    if job.get('protocol') == 'v2':
        command += ['--acceptance-report', job['acceptance_report']]
    return command


def save_status(path, jobs):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"updated_at": now(), "jobs": jobs}, indent=2) + "\n")
    tmp.replace(path)


def run_queue(jobs, gpus, status_path, launch_delay=20.0, cpu_threads=2, poll=1.0):
    """Run without retries; after failure, drain running jobs and cancel queued ones."""
    active = {}
    failed = False
    next_launch = 0.0
    env = os.environ.copy()
    env["TSFSAR_CPU_THREADS"] = str(cpu_threads)
    env["PYTHONUNBUFFERED"] = "1"
    save_status(status_path, jobs)
    try:
        while True:
            for gpu, (proc, job, stream) in list(active.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                stream.close()
                job.update(status="completed" if rc == 0 else "failed",
                           returncode=rc, finished_at=now())
                print(f"[{now()}] {job['status'].upper()} {job['name']} gpu={gpu} rc={rc}", flush=True)
                failed = failed or rc != 0
                del active[gpu]
                save_status(status_path, jobs)
            pending = [job for job in jobs if job["status"] == "queued"]
            if failed and pending:
                for job in pending:
                    job["status"] = "cancelled_after_failure"
                save_status(status_path, jobs)
                pending = []
            free = [gpu for gpu in gpus if gpu not in active]
            if pending and free and time.monotonic() >= next_launch:
                job, gpu = pending[0], free[0]
                out = Path(job["output"])
                out.mkdir(parents=True, exist_ok=True)
                # runner.log also captures launcher failures before console.log opens.
                stream = (out / "runner.log").open("a")
                try:
                    proc = subprocess.Popen(job_command(job, gpu), cwd=str(ROOT),
                                            env=env, stdout=stream,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    stream.close()
                    raise
                job.update(status="running", gpu=gpu, pid=proc.pid, started_at=now())
                active[gpu] = (proc, job, stream)
                print(f"[{now()}] START {job['name']} gpu={gpu} pid={proc.pid} log={job['console']}", flush=True)
                save_status(status_path, jobs)
                next_launch = time.monotonic() + launch_delay
            if not active and not any(j["status"] == "queued" for j in jobs):
                break
            time.sleep(poll)
    except BaseException:
        # Only signal processes launched by this queue, each in its own session.
        for proc, job, stream in active.values():
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            job.update(status="interrupted", finished_at=now())
            stream.close()
        for job in jobs:
            if job["status"] == "queued":
                job["status"] = "cancelled_after_interrupt"
        save_status(status_path, jobs)
        raise
    return 1 if failed else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6", help="only use GPUs allocated to this job; GPU 7 excluded after reported ECC errors")
    p.add_argument("--shots", default="1,5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-root", default="runs/campus/parallel_seed0")
    p.add_argument("--skip", default="", help="comma-separated job names, e.g. E1_1shot")
    p.add_argument("--launch-delay", type=float, default=20.0, help="seconds between launches")
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--activation-offload", choices=("off", "diagnostics", "all"), default="diagnostics")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument('--protocol', choices=('legacy', 'v2'), default='v2')
    p.add_argument('--selection-protocol', choices=('meta-val-fixed', 'official-test'), default='meta-val-fixed')
    p.add_argument('--experiments', default='E0,E1,E2', help='E0 only for a baseline-first queue')
    p.add_argument('--acceptance-1shot', type=Path)
    p.add_argument('--acceptance-5shot', type=Path)
    a = p.parse_args()
    try:
        gpus = [int(x) for x in a.gpus.split(",")]
        shots = [int(x) for x in a.shots.split(",")]
    except ValueError:
        p.error("--gpus and --shots must contain comma-separated integers")
    if len(gpus) != len(set(gpus)) or min(gpus) < 0:
        p.error("GPU indices must be distinct and nonnegative")
    if len(shots) != len(set(shots)) or not set(shots) <= {1, 5}:
        p.error("shots must be distinct values from 1,5")
    if a.cpu_threads < 1 or a.launch_delay < 0:
        p.error("cpu-threads must be positive; launch-delay must be nonnegative")
    out = Path(a.output_root).resolve()
    experiments = set(a.experiments.split(','))
    if not experiments or not experiments <= {'E0', 'E1', 'E2', 'E0_detach', 'E1_detach', 'E1_safe', 'E1_exact', 'E1_amortized'}:
        p.error('unknown --experiments selection')
    include_adaptations = bool(experiments & {'E0_detach', 'E1_detach', 'E1_safe'})
    include_references = bool(experiments & {'E1_exact', 'E1_amortized'})
    if (include_adaptations or include_references) and a.protocol != 'v2':
        p.error('new FSAR adaptations require --protocol v2')
    all_jobs = build_jobs(shots, a.seed, out, set(), a.protocol, include_adaptations, include_references)
    skip = set(filter(None, a.skip.split(",")))
    if not skip <= {j["name"] for j in all_jobs}:
        p.error("unknown skipped job: " + ", ".join(sorted(skip - {j["name"] for j in all_jobs})))
    jobs = build_jobs(shots, a.seed, out, skip, a.protocol, include_adaptations, include_references)
    jobs = [job for job in jobs if job['experiment'] in experiments]
    for job in jobs:
        job["activation_offload"] = a.activation_offload
        job['protocol'] = a.protocol
        job['selection_protocol'] = a.selection_protocol
        if a.protocol == 'v2':
            certificate = getattr(a, f'acceptance_{job["shot"]}shot')
            if certificate is None or not certificate.is_file():
                p.error(f'v2 requires --acceptance-{job["shot"]}shot pointing to a passed campus report')
            job.update(protocol='v2', acceptance_report=str(certificate.resolve()))
    if not jobs:
        p.error("no jobs remain")
    print(f"{len(jobs)} independent jobs, at most {len(gpus)} simultaneous; E3 requires separate calibration.", flush=True)
    for i, job in enumerate(jobs):
        print(job["name"], shlex.join(job_command(job, gpus[i % len(gpus)])), flush=True)
    if a.dry_run:
        print("DRY RUN: no files created, no processes launched. GPU assignment above is illustrative.")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error("another queue is using this output-root")
        if (out / "status.json").exists() or any(Path(j["output"]).exists() for j in jobs):
            p.error("existing queue/results found; use a new --output-root; no automatic overwrite/retry")
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        return run_queue(jobs, gpus, out / "status.json", a.launch_delay, a.cpu_threads)


if __name__ == "__main__":
    sys.exit(main())
