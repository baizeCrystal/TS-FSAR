"""Bounded AMP backoff for an acceptance probe, never a training policy."""
import math

import torch


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
