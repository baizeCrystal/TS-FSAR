"""Protocol v2: fixed auditable episodes, common loop, full state, Val selection.

Legacy entry remains available for provenance. This entry never loads a trained
historical checkpoint as a fresh initialization and requires CUDA certification.
"""
import copy
import hashlib
import json
import math
import random
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from utils.config import Config
from utils.functions import build_embeddings, cosine_lr
from utils.optimizer import construct_optimizer
from trainers.model import build_model
from scripts.audit_campus_protocol import compare_config
from scripts.check_ssv2_files import check as check_dataset
from .diagnostics import coefficients, diagnose
from .episodes import loader, unpack
from .evaluation import evaluate
from .gradient_utils import activation_context
from .model import DTOTSFSAR
from .state import (ROOT, EpisodeLedger, append_json, atomic_torch_save, cpu_state,
                    file_hash, model_hash, restore_rng, rng_state, source_hash)


def configuration_signature(cfg):
    value = copy.deepcopy(cfg.cfg_dict)
    value.get('RIGOROUS', {}).pop('RESUME', None)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def evaluation_signature(cfg):
    settings = {key: getattr(cfg.DATA, key) for key in (
        'NUM_INPUT_FRAMES', 'SAMPLING_MODE', 'SAMPLING_RATE', 'TEST_CROP_SIZE',
        'TEST_SCALE', 'TEST_CENTER_CROP', 'TARGET_FPS', 'MEAN', 'STD')}
    settings.update(shot=cfg.TRAIN.SHOT, way=cfg.TRAIN.WAY,
                    queries=cfg.TRAIN.QUERY_PER_CLASS_TEST,
                    class_names=cfg.TEST.CLASS_NAME,
                    temperatures=[cfg.TEMP_CLIP, cfg.TEMP_SIDE],
                    fusion=[cfg.TEST.FUSE_VISION, cfg.TEST.FUSE_CLIP, cfg.TEST.FUSE_SIDE],
                    prompt_sha256=file_hash(cfg.TEST.LLM_test_json),
                    list_sha256=file_hash(Path(cfg.DATA.ANNO_DIR) / 'test_few_shot.txt'))
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def train(data, val_data, model, embeddings, val_embeddings, optimizer, scheduler,
          scaler, cfg, device, metadata, resume=None):
    out = Path(cfg.OUTPUT_DIR)
    ledger = EpisodeLedger(out / 'train_episodes.jsonl')
    start, best_accuracy, best_state, best_result, successful_updates = 0, -math.inf, None, None, 0
    if resume is not None:
        if resume.get('version') != 2 or resume['configuration_signature'] != metadata['configuration_signature']:
            raise RuntimeError('resume does not match the frozen protocol/configuration')
        if resume['source_sha256'] != metadata['source_sha256']:
            raise RuntimeError('code changed since checkpoint; exact resume refused')
        model.load_state_dict(resume['model'], strict=True)
        optimizer.load_state_dict(resume['optimizer'])
        scaler.load_state_dict(resume['scaler'])
        if model.dto_controller is not None:
            model.dto_controller.load_state_dict(resume['controller'])
        model.dto_step = resume['completed_episodes']
        start = resume['completed_episodes']
        if ledger.sequence_hash(start) != resume['train_sequence_sha256']:
            raise RuntimeError('resume episode ledger changed')
        best_accuracy, best_state = resume['best_accuracy'], resume['best_state']
        best_result, successful_updates = resume['best_result'], resume['successful_updates']
        if best_state is not None:
            atomic_torch_save(best_state, out / 'model_best.pth')
            (out / 'best_result.json').write_text(json.dumps(best_result, indent=2) + '\n')
        restore_rng(resume['rng'])
    elif (out / 'train_episodes.jsonl').exists() and ledger.records:
        raise RuntimeError('existing training data; use explicit full resume or a fresh directory')
    interval = int(getattr(cfg.DTO_KD, 'GRADIENT_SAMPLE_INTERVAL', 500))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    cuda = device.type == 'cuda'
    started = time.perf_counter()
    previous_elapsed = float(resume.get('elapsed_seconds', 0)) if resume else 0.

    def checkpoint(completed):
        atomic_torch_save({
            'version': 2, 'model': cpu_state(model), 'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(), 'controller': model.dto_state_dict(),
            'rng': rng_state(cuda), 'completed_episodes': completed,
            'successful_updates': successful_updates, 'best_accuracy': best_accuracy,
            'best_state': best_state, 'best_result': best_result,
            'source_sha256': metadata['source_sha256'],
            'configuration_signature': metadata['configuration_signature'],
            'train_sequence_sha256': ledger.sequence_hash(completed),
            'elapsed_seconds': previous_elapsed + time.perf_counter() - started,
            'scheduler': 'released cosine_lr, no warmup, called after optimizer step',
        }, out / 'resume_latest.pth')

    for iteration, batch in enumerate(data, start=start):
        step = iteration + 1
        if step > cfg.TRAIN.NUM_TRAIN_TASKS:
            raise RuntimeError('loader exceeded frozen training budget')
        images, support, query, real, tensors, record = unpack(batch, device)
        ledger.observe(record)
        if step == 1 and metadata.get('expected_episode0_sha256') not in (None, record['episode_sha256']):
            raise RuntimeError('worker-loaded episode 0 differs from CUDA acceptance input')
        early_steps = int(getattr(cfg.DTO_KD, 'GRADIENT_EARLY_STEPS', 0))
        early_interval = int(getattr(cfg.DTO_KD, 'GRADIENT_EARLY_INTERVAL', 25))
        sampled = bool((step <= early_steps and step % early_interval == 0)
                       or (interval and step % interval == 0))
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        controller_before = model.dto_state_dict()
        history_length = len(model.dto_controller.history) if model.dto_controller else 0
        with activation_context(cfg.DTO_KD.ACTIVATION_OFFLOAD, sampled), torch.autocast(
                device_type=device.type, enabled=cuda):
            result = model(images, support, query, real, embeddings)
            loss = result['loss']
        with torch.autocast(device_type=device.type, enabled=cuda):
            task, distill, student, teacher, side = model._components(query)
        components = {'task': float(task.detach()), 'kl': float(distill.detach()),
                      'student_ce': float(student.detach()), 'teacher_ce': float(teacher.detach()),
                      'side_ce': float(side.detach()), 'total': float(loss.detach())}
        if not all(math.isfinite(x) for x in components.values()):
            raise FloatingPointError('non-finite loss components')
        tw, dw = coefficients(model, cfg, task, distill)
        del task, distill, student, teacher, side
        if sampled:
            diagnostic_started = time.perf_counter()
            for row in diagnose(model, cfg, query):
                append_json(out / 'gradient_groups.jsonl', {'step': step, 'mode': model.dto_mode,
                                                          'episode_sha256': record['episode_sha256'], **row})
            diagnostic_seconds = time.perf_counter() - diagnostic_started
        else:
            diagnostic_seconds = 0.
        scaler.scale(loss).backward()
        model.clear_dto_cache()
        matching_accuracy = float(result['acc'])
        # No later operation needs this training input or its completed graph.
        # Keep optimizer gradients for the released validation-before-step order.
        del batch, images, support, query, real, tensors, result, loss
        # Match the released validation-before-step timing for ALL methods.
        if step % cfg.TRAIN.VAL_FRE_ITER == 0:
            if cuda:
                # Release unused allocator blocks before FP32 validation on A10.
                torch.cuda.empty_cache()
            validation = evaluate(val_data, model, val_embeddings, cfg, 'val', step, device)
            validation['successful_updates_before_validation'] = successful_updates
            append_json(out / 'validation.jsonl', validation)
            accuracy = validation['accuracy_percent']['fusion']
            if accuracy > best_accuracy:
                best_accuracy, best_state = accuracy, cpu_state(model)
                best_result = validation
                atomic_torch_save(best_state, out / 'model_best.pth')
                (out / 'best_result.json').write_text(json.dumps(validation, indent=2) + '\n')
            print('VAL', json.dumps(validation), flush=True)
            model.train()
        before = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                  if p.requires_grad} if sampled else None
        if sampled:
            scaler.unscale_(optimizer)
            gradient_squared = sum(float(p.grad.detach().cpu().double().square().sum())
                                   for p in model.parameters() if p.grad is not None)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        effective_update = scaler.get_scale() >= old_scale
        if effective_update:
            successful_updates += 1
        elif model.dto_controller is not None:
            # An AMP overflow must not advance controller optimization alone.
            model.dto_controller.load_state_dict(controller_before)
            del model.dto_controller.history[history_length:]
        optimizer.zero_grad(set_to_none=True)
        scheduler(iteration)
        row = {'step': step, 'losses': components, 'matching_accuracy': matching_accuracy,
               'task_coefficient': tw, 'distill_coefficient': dw, 'amp_scale': scaler.get_scale(),
               'effective_update': effective_update, 'successful_updates': successful_updates,
               'learning_rates': [g['lr'] for g in optimizer.param_groups],
               'diagnostic_seconds': diagnostic_seconds, 'elapsed_seconds': previous_elapsed + time.perf_counter() - started,
               'dto': model.dto_last_metrics,
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if cuda else None}
        if model.dto_controller is not None:
            row['controller_update_applied'] = effective_update
            row['controller_weights_after_step'] = model.dto_controller.weights
        if sampled:
            delta_squared = sum(float((p.detach().cpu() - before[n]).double().square().sum())
                                for n, p in model.named_parameters() if p.requires_grad)
            row.update(unscaled_total_gradient_norm=math.sqrt(gradient_squared) if math.isfinite(gradient_squared) else None,
                       nonfinite_unscaled_gradient=not math.isfinite(gradient_squared),
                       parameter_update_norm=math.sqrt(delta_squared), clipping='none (released recipe)')
            del before
        append_json(out / 'train_trace.jsonl', row)
        if step % cfg.LOG_PERIOD == 0 or step == cfg.TRAIN.NUM_TRAIN_TASKS:
            print('TRAIN', json.dumps(row), flush=True)
        if step % cfg.TRAIN.VAL_FRE_ITER == 0 or step == cfg.TRAIN.NUM_TRAIN_TASKS:
            checkpoint(step)
    completed = cfg.TRAIN.NUM_TRAIN_TASKS
    summary = {'completed_episodes': completed, 'successful_updates': successful_updates,
               'train_sequence_sha256': ledger.sequence_hash(completed),
               'best_meta_val_percent': best_accuracy, 'best_result': best_result,
               'source_sha256': metadata['source_sha256'], 'elapsed_seconds': previous_elapsed + time.perf_counter() - started}
    (out / 'training_complete.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    cfg = Config()
    if getattr(getattr(cfg, 'RIGOROUS', None), 'VERSION', None) != 2:
        raise RuntimeError('use prepare_run_config.py --protocol v2')
    if cfg.DTO_KD.UPDATE_RULE != 'paper_pi_softmax':
        raise RuntimeError('protocol v2 requires the direct Eq.18 update')
    if not cfg.RIGOROUS.SMOKE:
        audit = compare_config(Path(cfg.args.cfg_file), generated=True)
        if not audit['configuration_checks_pass']:
            raise RuntimeError('released configuration/protocol audit failed: ' + json.dumps(audit))
    out = Path(cfg.OUTPUT_DIR)
    if not torch.cuda.is_available():
        raise RuntimeError('formal video execution requires an allocated campus CUDA GPU')
    device = torch.device('cuda:0')
    seed = cfg.TRAIN.RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    subprocess.run([__import__('sys').executable, str(ROOT / 'scripts/verify_weights.py')], check=True)
    dataset_check = check_dataset(cfg.DATA.DATA_ROOT_DIR)
    code_hash = source_hash()
    certificate = None
    if cfg.RIGOROUS.ACCEPTANCE_REPORT:
        certificate_path = Path(cfg.RIGOROUS.ACCEPTANCE_REPORT)
        certificate = json.loads(certificate_path.read_text())
        if (not certificate.get('passed') or certificate['source_sha256'] != code_hash
                or certificate['shot'] != cfg.TRAIN.SHOT or certificate['seed'] != seed
                or certificate['torch'] != torch.__version__
                or certificate['gpu'] != torch.cuda.get_device_name(0)):
            raise RuntimeError('CUDA acceptance certificate is stale or incompatible')
        bank_path = (certificate_path.parent / certificate['initialization_file']).resolve()
        if file_hash(bank_path) != certificate['initialization_file_sha256']:
            raise RuntimeError('certified initialization file changed')
        bank = torch.load(bank_path, map_location='cpu')
    elif cfg.TRAIN.ENABLE and not cfg.RIGOROUS.SMOKE:
        raise RuntimeError('formal training requires full-model CUDA acceptance')
    else:
        bank = None
    image, text, tokenizer = build_model(cfg)
    # A complete, hashed certified state replaces expensive repeated PCA. No
    # forward/update occurs before strict loading and the model hash check.
    original_test_flag = cfg.TEST.ENABLE
    if bank is not None:
        cfg.TEST.ENABLE = True
    model = DTOTSFSAR(cfg, image)
    cfg.TEST.ENABLE = original_test_flag
    if bank is not None:
        model.load_state_dict(bank['model'], strict=True)
        if model_hash(model) != certificate['initial_model_sha256']:
            raise RuntimeError('loaded initialization differs from certified baseline')
    model.to(device)
    optimizer = construct_optimizer(model, cfg)
    scheduler = cosine_lr(optimizer, 0, cfg.TRAIN.NUM_TRAIN_TASKS)
    scaler = torch.cuda.amp.GradScaler()
    embeddings = build_embeddings(cfg.TRAIN.CLASS_NAME, cfg.TRAIN.LLM_train_json, text, tokenizer)
    val_embeddings = build_embeddings(cfg.TEST.CLASS_NAME, cfg.TEST.LLM_test_json, text, tokenizer)
    del image, text, tokenizer
    metadata = {'version': 2, 'source_sha256': code_hash,
                'configuration_signature': configuration_signature(cfg),
                'initial_model_sha256': model_hash(model), 'seed': seed, 'shot': cfg.TRAIN.SHOT,
                'torch': torch.__version__, 'cuda_build': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(0), 'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
                'dataset_check': dataset_check,
                'optimizer_defaults': optimizer.defaults,
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'clip_manifest': json.loads((ROOT / 'assets/CLIP_VIT_B16_SHA256SUMS.json').read_text()),
                'controller_update': cfg.DTO_KD.UPDATE_RULE, 'selection_split': 'Meta-Val',
                'certification': cfg.RIGOROUS.ACCEPTANCE_REPORT, 'smoke': cfg.RIGOROUS.SMOKE,
                'expected_episode0_sha256': certificate['episode0_sha256'] if certificate else None}
    metadata['evaluation_protocol_sha256'] = evaluation_signature(cfg)
    metadata['dto_mode'] = model.dto_mode
    if cfg.TEST.ENABLE:
        checkpoint_path = Path(cfg.TEST.PRETRAIN_MODEL_PATH)
        metadata['trained_checkpoint_sha256'] = file_hash(checkpoint_path)
        training_metadata = checkpoint_path.parent / 'protocol_v2.json'
        metadata['training_protocol'] = json.loads(training_metadata.read_text()) if training_metadata.is_file() else None
    metadata_path = out / 'protocol_v2.json'
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if (not cfg.RIGOROUS.RESUME or old['source_sha256'] != metadata['source_sha256']
                or old['configuration_signature'] != metadata['configuration_signature']):
            raise RuntimeError('existing run metadata differs; use a fresh output directory')
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
        (out / 'frozen_config.json').write_text(json.dumps(cfg.cfg_dict, indent=2) + '\n')
    if bank is not None:
        restore_rng(bank['rng'])
    if cfg.TRAIN.ENABLE:
        resume = torch.load(cfg.RIGOROUS.RESUME, map_location='cpu') if cfg.RIGOROUS.RESUME else None
        start = resume['completed_episodes'] if resume else 0
        data = loader(cfg, 'train', start=start)
        val_data = loader(cfg, 'test', 'val')
        train(data, val_data, model, embeddings, val_embeddings, optimizer, scheduler, scaler,
              cfg, device, metadata, resume)
    else:
        if not getattr(cfg.TEST, 'LOAD_PRETRAIN', False):
            raise RuntimeError('final evaluation requires a trained checkpoint')
        model.load_state_dict(torch.load(cfg.TEST.PRETRAIN_MODEL_PATH, map_location='cpu'), strict=True)
        result = evaluate(loader(cfg, 'test', 'test'), model, val_embeddings, cfg, 'test', 0, device)
        (out / 'test_result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
