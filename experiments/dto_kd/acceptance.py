"""Bounded AMP backoff for an acceptance probe, never a training policy."""
import math
import json
from pathlib import Path

import torch


def require_adaptation_certificate(report, mode, detach_teacher):
    """A legacy E0/E1 certificate cannot admit a new gradient/objective path."""
    if not detach_teacher and mode != 'safe_dynamic':
        return
    matches = [s for s in report.get('smokes', [])
               if s.get('mode') == mode and s.get('detach_teacher') is True
               and s.get('adaptation_check', {}).get('passed') is True
               and s.get('schedule_check', {}).get('verified') is True
               and s.get('successful_updates', 0) >= 3]
    if not matches:
        raise RuntimeError('new adaptation requires current-code --adaptation-checks CUDA acceptance')


def load_certified_initialization(report_path, shot, seed, torch_version, gpu_name):
    """Reuse only a passed report's hashed initial state, never its training.

    Every equivalence and smoke check must still run under the current code.
    This avoids recomputing PCA after an acceptance/runtime-only correction.
    """
    from .state import file_hash
    path = Path(report_path).resolve()
    report = json.loads(path.read_text())
    expected = {'shot': shot, 'seed': seed, 'torch': torch_version, 'gpu': gpu_name}
    if report.get('passed') is not True or any(report.get(k) != v for k, v in expected.items()):
        raise RuntimeError('initialization report did not pass or has incompatible shot/seed/environment')
    bank_path = (path.parent / report['initialization_file']).resolve()
    bank_hash = file_hash(bank_path)
    if bank_hash != report['initialization_file_sha256']:
        raise RuntimeError('certified initialization file changed; refusing reuse')
    bank = torch.load(bank_path, map_location='cpu')
    if not {'model', 'rng'} <= bank.keys():
        raise RuntimeError('incomplete certified initialization')
    return bank, {'report': str(path), 'report_sha256': file_hash(path),
                  'source_sha256': report['source_sha256'],
                  'bank_sha256': bank_hash, 'initial_model_sha256': report['initial_model_sha256'],
                  'scope': 'reuse hashed initial model/RNG only; rerun all current-code acceptance checks'}


def verify_smoke_schedule(rows, base_lrs, steps):
    """Check the actual logged no-warmup cosine against the smoke budget."""
    if [row['step'] for row in rows] != list(range(1, steps + 1)):
        raise RuntimeError('incomplete or repeated smoke training trace')
    for row in rows:
        rates = row['learning_rates']
        factor = .5 * (1. + math.cos(math.pi * (row['step'] - 1) / steps))
        if len(rates) != len(base_lrs) or any(
                not math.isclose(actual, base * factor, rel_tol=1e-10, abs_tol=1e-12)
                for actual, base in zip(rates, base_lrs)):
            raise RuntimeError('smoke learning-rate schedule does not match its episode budget')
    return {'verified': True, 'cosine_period_episodes': steps, 'trace_rows': len(rows)}


def one_successful_update(model, initial, optimizer, scaler, forward, reset_rng,
                          record_attempt, max_attempts=17):
    """Replay one fixed input from identical weights/RNG until AMP can update.

    The caller supplies a fresh optimizer. Only the reference may back off;
    equivalent candidates must use its successful scale with max_attempts=1.
    Failed attempts must leave parameters and optimizer state untouched.
    """
    if max_attempts < 1 or optimizer.state:
        raise ValueError('acceptance needs positive attempts and a fresh optimizer')
    for attempt in range(1, max_attempts + 1):
        model.load_state_dict(initial, strict=True)
        optimizer.zero_grad(set_to_none=True)
        reset_rng()
        result = forward()
        loss = result['loss']
        if not bool(torch.isfinite(loss.detach()).all()):
            raise FloatingPointError('non-finite forward loss; AMP backoff cannot certify this probe')
        loss_value, accuracy = float(loss.detach()), float(result['acc'])
        if not math.isfinite(accuracy):
            raise FloatingPointError('non-finite acceptance accuracy')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = {name: param.grad.detach().cpu().clone() if param.grad is not None else None
                     for name, param in model.named_parameters() if param.requires_grad}
        bad = [name for name, value in gradients.items()
               if value is not None and not bool(torch.isfinite(value).all())]
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        applied = scale_after >= scale_before
        row = {'attempt': attempt, 'scale_before': scale_before, 'scale_after': scale_after,
               'loss': loss_value, 'accuracy': accuracy, 'update_applied': applied,
               'nonfinite_gradient_count': len(bad), 'nonfinite_gradient_names': bad}
        record_attempt(row)
        # In particular, never compare two unchanged models after overflow.
        if applied == bool(bad):
            raise RuntimeError('AMP update decision disagrees with finite-gradient check')
        values = {name: param.detach().cpu().clone()
                  for name, param in model.named_parameters() if param.requires_grad}
        if hasattr(model, 'clear_dto_cache'):
            model.clear_dto_cache()
        del result, loss
        if applied:
            if not any(value is not None for value in gradients.values()):
                raise RuntimeError('acceptance update had no gradients')
            if not all(bool(torch.isfinite(value).all()) for value in values.values()):
                raise FloatingPointError('acceptance produced non-finite parameters')
            if not any(not torch.equal(value, initial[name].cpu()) for name, value in values.items()):
                raise RuntimeError('acceptance optimizer did not change any parameter')
            return gradients, values, {'loss': loss_value, 'accuracy': accuracy,
                                       'attempts': attempt, 'skipped_attempts': attempt - 1,
                                       'successful_scale': scale_before, 'scale_after': scale_after}
        if optimizer.state or any(not torch.equal(value, initial[name].cpu())
                                  for name, value in values.items()):
            raise RuntimeError('overflow attempt mutated optimizer or model parameters')
        if not 0 < scale_after < scale_before:
            raise RuntimeError('AMP backoff did not reduce a positive scale')
        del gradients, values
    optimizer.zero_grad(set_to_none=True)
    model.load_state_dict(initial, strict=True)
    raise RuntimeError(f'no finite, effective AMP update after {max_attempts} fixed-input attempts; '
                       'see amp_attempts.jsonl; acceptance refused')
