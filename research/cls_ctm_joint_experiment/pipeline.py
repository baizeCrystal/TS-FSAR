"""One allocated GPU/shot: fresh bank, acceptance, disposable smokes, E1 then E0."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from .common import ROOT,output_lock,sha_file,write_new


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shot',type=int,choices=(1,5),required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();os.chdir(ROOT)
    args.output=args.output.resolve();args.data_root=args.data_root.resolve()
    manifest=ROOT/f'research/ps_ds_subspace_experiment/results/shot{args.shot}/episodes.jsonl.gz'
    bank=args.output/'initialization'
    common=['--shot',str(args.shot),'--data-root',str(args.data_root),'--manifest',str(manifest)]
    def command(module,more,console=None):
        cmd=[sys.executable,'-u','-m','research.cls_ctm_joint_experiment.'+module,*common,*more]
        print('JOINT_CTM_PHASE',module,more,flush=True)
        if console is None:subprocess.run(cmd,check=True)
        else:
            with console.open('a') as stream:
                subprocess.run(cmd,check=True,stdout=stream,stderr=subprocess.STDOUT)
    with output_lock(args.output):
        if (args.output/'pipeline_complete.json').exists():
            if not json.loads((args.output/'pipeline_complete.json').read_text())['passed']:
                raise RuntimeError('invalid pipeline completion')
            print('JOINT_CTM_ALREADY_COMPLETE',args.output,flush=True);return
        command('train',['--phase','prepare','--output',str(bank)])
        subprocess.run([sys.executable,'-m','unittest',
            'research.cls_ctm_joint_experiment.tests.test_joint.JointTests.test_cuda_offload_keeps_joint_amp_gradients_and_updates','-v'],check=True)
        acceptance=args.output/'acceptance'
        if not (acceptance/'acceptance.json').exists():
            command('acceptance',['--bank',str(bank),'--output',str(acceptance)])
        else:
            verified=json.loads((acceptance/'acceptance.json').read_text())
            if not verified['passed'] or verified['initialization_sha256']!=sha_file(bank/'initialization.pth'):
                raise RuntimeError('invalid prior acceptance')
        # Establish BOTH versions can update/infer before any full 10k training.
        for v in ('E0','E1'):
            out=args.output/f'preflight_{v}'
            if not (out/'smoke_passed.json').exists():
                more=['--variant',v,'--bank',str(bank),'--output',str(out),'--smoke']
                if (out/'resume_latest.pth').exists():more.append('--resume')
                command('train',more)
        # Prioritize the requested new joint method; matched E0 control follows.
        for v in ('E1','E0'):
            out=args.output/v
            more=['--variant',v,'--bank',str(bank),'--output',str(out)]
            if (out/'resume_latest.pth').exists():more.append('--resume')
            command('train',more,args.output/f'{v}.console.log')
        if not (args.output/'analysis/statistics.json').exists():
            subprocess.run([sys.executable,'-u','-m','research.cls_ctm_joint_experiment.analyze',
                            '--root',str(args.output)],check=True)
        write_new(args.output/'pipeline_complete.json',dict(passed=True))
        print('JOINT_CTM_PIPELINE_COMPLETE',args.output,flush=True)


if __name__=='__main__':main()
