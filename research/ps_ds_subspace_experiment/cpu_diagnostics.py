"""Synthetic shape/timing audit. Never constructs CLIP or reads checkpoints."""
import json
import platform
import time
from types import SimpleNamespace as NS
import torch
from trainers.ts_fsar import TS_DCM
from .matching import matching
from .protocol import HERE, write_new


def main():
    torch.set_num_threads(2)
    torch.manual_seed(6)
    cfg=NS(DATA=NS(NUM_INPUT_FRAMES=8),TRAIN=NS(DC_ALPHA=.4))
    dcm=TS_DCM(cfg,256).eval().requires_grad_(False)
    result=dict(kind='synthetic CPU features only; NOT model inference or accuracy',
                torch=torch.__version__,python=platform.python_version(),threads=2,shots={})
    for shot in (1,5):
        features=torch.randn((5*shot+5)*8,197,256)
        support=torch.arange(5).repeat_interleave(shot)
        with torch.no_grad():
            start=time.perf_counter()
            baseline=dcm(features,support,5,shot,1)[0]
            base_seconds=time.perf_counter()-start
            disabled,_=matching(dcm,features,support,5,shot,1,enabled=False)
            score,report=matching(dcm,features,support,5,shot,1)
        result['shots'][str(shot)]=dict(report,score_shape=list(score.shape),
            e0_seconds=base_seconds,disabled_bitwise_equal=bool(torch.equal(baseline,disabled)),
            synthetic_e0_score=baseline.tolist(),synthetic_e1_score=score.tolist())
    write_new(HERE/'results/cpu_numerical_diagnostics.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
