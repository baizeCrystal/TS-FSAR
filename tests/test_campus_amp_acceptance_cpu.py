"""CPU tests of bounded AMP acceptance using a scaler that injects overflow.

The optimizer, autograd, gradients, dropout and parameter updates are real.
The simulated scaler cannot establish CUDA numerical equivalence.
"""
import copy
import unittest

import torch

from experiments.dto_kd.acceptance import one_successful_update


class SimulatedScaler:
    def __init__(self, init_scale=8., finite_at=2.):
        self.value, self.finite_at = init_scale, finite_at
        self.applied = 0

    def scale(self, loss):
        return loss * self.value

    def unscale_(self, optimizer):
        self.bad = self.value > self.finite_at
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.div_(self.value)
                    if self.bad:
                        param.grad.fill_(float('inf'))

    def step(self, optimizer):
        if not self.bad:
            optimizer.step()
            self.applied += 1

    def update(self):
        if self.bad:
            self.value *= .5

    def get_scale(self):
        return self.value


class CampusAmpAcceptanceTest(unittest.TestCase):
    def make_probe(self, finite_at=2., scale=8.):
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(.4))
        initial = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        scaler = SimulatedScaler(scale, finite_at)
        rng = torch.get_rng_state().clone()
        inputs, events = [], []

        def forward():
            output = model(torch.ones(4, 3))
            inputs.append(output.detach().clone())
            return {'loss': (output.square() + .5).mean(), 'acc': .5}

        return model, initial, optimizer, scaler, forward, lambda: torch.set_rng_state(rng), events.append, inputs, events

    def test_backoff_replays_identical_input_and_updates_once_at_common_scale(self):
        torch.manual_seed(31)
        probe = self.make_probe()
        gradients, parameters, summary = one_successful_update(*probe[:7], max_attempts=4)
        model, initial, optimizer, scaler, _, reset_rng, _, inputs, events = probe
        self.assertEqual([r['scale_before'] for r in events], [8., 4., 2.])
        self.assertEqual([r['update_applied'] for r in events], [False, False, True])
        self.assertEqual(summary['skipped_attempts'], 2)
        self.assertEqual(scaler.applied, 1)
        for value in inputs[1:]:
            torch.testing.assert_close(value, inputs[0], rtol=0, atol=0)
        candidate = copy.deepcopy(model)
        candidate.load_state_dict(initial)
        optimizer2 = torch.optim.AdamW(candidate.parameters(), lr=.001)
        events2 = []

        def forward2():
            return {'loss': (candidate(torch.ones(4, 3)).square() + .5).mean(), 'acc': .5}

        gradients2, parameters2, _ = one_successful_update(
            candidate, initial, optimizer2, SimulatedScaler(2.), forward2, reset_rng, events2.append, max_attempts=1)
        self.assertEqual(len(events2), 1)
        for name in gradients:
            torch.testing.assert_close(gradients[name], gradients2[name], rtol=0, atol=0)
            torch.testing.assert_close(parameters[name], parameters2[name], rtol=0, atol=0)
        for state in optimizer.state.values():
            self.assertEqual(float(state['step']), 1.)

    def test_persistent_overflow_is_bounded_and_leaves_weights_optimizer_unchanged(self):
        probe = self.make_probe(finite_at=0.)
        with self.assertRaisesRegex(RuntimeError, 'after 3 fixed-input attempts'):
            one_successful_update(*probe[:7], max_attempts=3)
        self.assertEqual(len(probe[-1]), 3)
        self.assertEqual(probe[2].state, {})
        for name, value in probe[0].state_dict().items():
            torch.testing.assert_close(value, probe[1][name], rtol=0, atol=0)

    def test_nonfinite_loss_and_no_actual_update_cannot_pass(self):
        probe = self.make_probe()
        invalid = lambda: {'loss': next(probe[0].parameters()).sum() * float('nan'), 'acc': .5}
        with self.assertRaisesRegex(FloatingPointError, 'non-finite forward loss'):
            one_successful_update(*probe[:4], invalid, *probe[5:7])
        probe = self.make_probe(scale=2.)
        for group in probe[2].param_groups:
            group['lr'] = group['weight_decay'] = 0.
        with self.assertRaisesRegex(RuntimeError, 'did not change any parameter'):
            one_successful_update(*probe[:7])


if __name__ == '__main__':
    unittest.main()
