"""Fresh CLIP/PCA initialization shared by E0/E1; no trained E0 warm start."""
import json
import os
from pathlib import Path
import torch
from research.cls_ctm_experiment.common import (ROOT, canonical, digest, sha_file, write_new,
    seed_all, model_digest, source_audit, output_lock, identity_file, atomic_save)
from research.cls_ctm_experiment.data import training_config, train_loader
from research.cls_ctm_experiment.train import rng_state,restore_rng,cpu_state,preserve_prefix
from .core import JointModel

HERE=Path(__file__).resolve().parent


def implementation_hash():
    paths=list(HERE.glob('*.py'))
    paths += [ROOT/'research/cls_ctm_experiment'/n for n in ('module.py','data.py','common.py','train.py')]
    paths += [ROOT/'research/ps_ds_subspace_experiment'/n for n in
              ('protocol.py','episodes.py','evaluate.py','matching.py','analyze.py','configs/locked.json')]
    paths += [ROOT/n for n in ('trainers/model.py','trainers/ts_fsar.py','utils/optimizer.py','utils/functions.py')]
    return digest({str(p.relative_to(ROOT)):sha_file(p) for p in paths})


def configure(shot,data_root,output,workers,variant,skip_pca=False):
    _,raw=training_config(shot,data_root,output,workers)
    # Audit the same original functions/splits/preprocessing as the fixed replay.
    from research.ps_ds_subspace_experiment.protocol import load_config
    _,_,lock=load_config(shot,data_root,output,workers)
    raw['CTM_JOINT']={'ENABLE':variant=='E1'}
    raw['TRAIN']['RANDOM_SEED']=0
    raw['TEST']['ENABLE']=bool(skip_pca)
    raw['TRAIN']['LLM_train_json']=str(ROOT/raw['TRAIN']['LLM_train_json'])
    raw['TEST']['LLM_test_json']=str(ROOT/raw['TEST']['LLM_test_json'])
    from utils.config import Config
    return Config(load=False,cfg_dict=raw),raw,lock


def setup():
    if not torch.cuda.is_available():
        raise RuntimeError('run real-model execution on campus GPU only')
    torch.set_num_threads(int(os.environ.get('TSFSAR_CPU_THREADS','2')))
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    seed_all(0)


def prepare(args):
    setup()
    path=args.output/'initialization.pth'
    report=args.output/'initialization.json'
    if path.exists() or report.exists():
        if not(path.exists() and report.exists()) or sha_file(path)!=json.loads(report.read_text())['sha256']:
            raise RuntimeError('incomplete/changed initialization; preserved')
        return
    cfg,raw,lock=configure(args.shot,args.data_root,args.output,args.workers,'E0')
    from research.ps_ds_subspace_experiment.evaluate import verify_clip,verify_manifest_bundle
    verify_clip(lock)
    verify_manifest_bundle(args.manifest,args.shot,0)
    from trainers.model import build_model
    from utils.functions import build_embeddings
    vision,text,tokenizer=build_model(cfg)
    text.eval()
    model=JointModel(cfg,vision)  # original PCA is enabled here
    train=build_embeddings(cfg.TRAIN.CLASS_NAME,cfg.TRAIN.LLM_train_json,text,tokenizer).cpu()
    test=build_embeddings(cfg.TEST.CLASS_NAME,cfg.TEST.LLM_test_json,text,tokenizer).cpu()
    if train.shape!=(64,512) or test.shape!=(24,512):
        raise RuntimeError('missing official class text embeddings')
    identity=dict(shot=args.shot,seed=0,source='fresh original CLIP/PCA/heads; no trained checkpoint',
        implementation_sha256=implementation_hash(),source_objects=source_audit(),
        manifest_sha256=sha_file(args.manifest),model_sha256=model_digest(model),clip_files=lock['clip_files'])
    atomic_save(dict(identity=identity,model=cpu_state(model),train_embeddings=train,
                     test_embeddings=test,rng=rng_state()),path)
    write_new(report,dict(identity,sha256=sha_file(path)))
    print('JOINT_CTM_INITIALIZATION_READY',report,flush=True)


def load_model(args,variant):
    setup()
    report=json.loads((args.bank/'initialization.json').read_text())
    path=args.bank/'initialization.pth'
    if sha_file(path)!=report['sha256']:
        raise RuntimeError('shared initialization SHA256 mismatch')
    bank=torch.load(path,map_location='cpu',weights_only=True)
    expected=bank['identity']
    if (expected['shot']!=args.shot or expected['implementation_sha256']!=implementation_hash()
            or expected['manifest_sha256']!=sha_file(args.manifest)):
        raise RuntimeError('shared initialization belongs to other code/shot/manifest')
    cfg,raw,_=configure(args.shot,args.data_root,args.output,args.workers,variant,skip_pca=True)
    from trainers.model import build_model
    vision,text,tokenizer=build_model(cfg)
    model=JointModel(cfg,vision)
    del vision,text,tokenizer
    missing,unexpected=model.load_state_dict(bank['model'],strict=False)
    allowed={n for n in model.state_dict() if 'ctm_generator' in n}
    if set(missing)!=allowed or unexpected:
        raise RuntimeError('existing parameters do not load strictly from shared bank')
    cfg.TEST.ENABLE=False
    raw['TEST']['ENABLE']=False
    model.cuda()
    restore_rng(bank['rng'])
    embeddings={k:bank[k].cuda() for k in ('train_embeddings','test_embeddings')}
    return model,embeddings,bank,report,raw
