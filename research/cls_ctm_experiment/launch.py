"""Launch only two independent CLS-CTM jobs on explicitly idle allocated GPUs."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def atomic_json(path, value):
    tmp = path.with_suffix('.json.partial')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(tmp,path)


def check_gpu(gpu):
    value = subprocess.check_output(['nvidia-smi','-i',str(gpu),
        '--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if value:
        raise RuntimeError(f'GPU {gpu} has processes ({value}); no jobs stopped or started')
    ecc = subprocess.check_output(['nvidia-smi','-i',str(gpu),
        '--query-gpu=ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).strip()
    if ecc not in ('0','N/A','[N/A]','Not Supported','[Not Supported]'):
        raise RuntimeError(f'GPU {gpu} ECC is {ecc}; choose a healthy allocated GPU')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',default='4,5')
    p.add_argument('--assets-dir',type=Path,default=Path('/data/fsar_workspace/psds_assets'))
    p.add_argument('--data-root',type=Path,default=Path('/data/fsar_workspace/datasets/SSv2-Small'))
    p.add_argument('--output-root',type=Path,required=True)
    args = p.parse_args()
    gpus = [int(x) for x in args.gpus.split(',')]
    if len(gpus) != 2 or len(set(gpus)) != 2 or min(gpus)<0:
        p.error('provide exactly two distinct allocated GPUs, e.g. 4,5')
    if args.output_root.exists():
        p.error('output root exists; new launch refused (individual pipelines can resume)')
    ready = json.loads((args.assets_dir/'PSDS_READY.json').read_text())
    if ready.get('verified') is not True:
        p.error('assets not prepared/verified')
    if not (args.data_root/'30109.mp4').is_file():
        p.error('data-root must directly contain the SSv2-Small .mp4 files')
    for gpu in gpus:
        check_gpu(gpu)
    out = args.output_root.resolve()
    out.mkdir(parents=True)
    (out/'launcher.pid').write_text(str(os.getpid())+'\n')
    out.parent.mkdir(parents=True,exist_ok=True)
    (out.parent/'latest_root.txt').write_text(str(out)+'\n')
    if str(out).startswith('/data/fsar_workspace/'):
        Path('/data/fsar_workspace/latest_tsfsar_clsctm_root.txt').write_text(str(out)+'\n')
        Path('/data/fsar_workspace/latest_tsfsar_clsctm_tree.txt').write_text(str(ROOT)+'\n')
    jobs, processes, streams = [], [], []
    def interrupt(sig, frame):
        for proc in processes:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
        raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,interrupt)
    signal.signal(signal.SIGINT,interrupt)
    try:
        for shot,gpu in zip((1,5),gpus):
            log = out/f'shot{shot}.console.log'
            stream = log.open('x'); streams.append(stream)
            manifest = ROOT/f'research/ps_ds_subspace_experiment/results/shot{shot}/episodes.jsonl.gz'
            cmd = ['bash',str(ROOT/'research/cls_ctm_experiment/run_gpu.sh'),
                '--shot',str(shot),'--gpu',str(gpu),'--data-root',str(args.data_root.resolve()),
                '--checkpoint',ready['checkpoint_paths'][str(shot)],'--manifest',str(manifest),
                '--output',str(out/f'shot{shot}')]
            proc = subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL,start_new_session=True)
            processes.append(proc)
            jobs.append(dict(name=f'CLSCTM_{shot}shot',shot=shot,gpu=gpu,pid=proc.pid,
                             status='running',console_log=str(log),output=str(out/f'shot{shot}')))
            print('START',jobs[-1],flush=True)
        while True:
            for job,proc in zip(jobs,processes):
                rc = proc.poll()
                if rc is not None and job['status']=='running':
                    job.update(status='completed' if rc==0 else 'failed',returncode=rc)
                    print('EXIT',job,flush=True)
            atomic_json(out/'status.json',dict(jobs=jobs))
            if all(proc.poll() is not None for proc in processes):
                break
            time.sleep(2)
        if any(proc.returncode for proc in processes):
            raise RuntimeError('one or more CLS-CTM jobs failed; inspect console logs, no automatic retry')
        command = [sys.executable,'-u','-m','research.cls_ctm_experiment.analyze',
            '--shot1',str(out/'shot1/evaluation/formal'),
            '--shot5',str(out/'shot5/evaluation/formal'),
            '--bootstrap','10000','--output',str(out/'formal_analysis')]
        subprocess.run(command,cwd=ROOT,check=True)
        print('CLSCTM_ALL_COMPLETE',out/'formal_analysis/experiment_report.md',flush=True)
    finally:
        for stream in streams:
            stream.close()


if __name__=='__main__':
    main()
