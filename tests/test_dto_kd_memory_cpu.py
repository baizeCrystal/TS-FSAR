"""CPU checks for diagnostic equivalence and activation/graph lifetime."""
import types
import unittest

import torch

from experiments.dto_kd.controller import DTOController
from experiments.dto_kd.gradient_utils import activation_context, loss_gradients
from experiments.dto_kd.model import DTOTSFSAR


class DiagnosticMemoryTest(unittest.TestCase):
    def test_cpu_diagnostic_vectors_preserve_existing_grads_and_normal_backward(self):
        shared = torch.nn.Parameter(torch.tensor([0.5, 1.5]))
        unused = torch.nn.Parameter(torch.ones(3))
        shared.grad = torch.tensor([7.0, 8.0])
        existing = shared.grad.clone()
        with activation_context('diagnostics', sampled=True):
            task = shared.square().sum()
            distill = (shared - 1).square().sum()
        vectors = loss_gradients({'distill': distill, 'task': task}, [shared, unused])
        torch.testing.assert_close(vectors['task'], torch.tensor([1., 3., 0., 0., 0.]))
        torch.testing.assert_close(vectors['distill'], torch.tensor([-1., 1., 0., 0., 0.]))
        self.assertTrue(all(g.device.type == 'cpu' and not g.requires_grad for g in vectors.values()))
        torch.testing.assert_close(shared.grad, existing)
        self.assertIsNone(unused.grad)
        (task + 0.1 * distill).backward()
        torch.testing.assert_close(shared.grad, existing + torch.tensor([0.9, 3.1]))

    def test_adam_updates_equal_with_diagnostics_and_cpu_saved_tensors(self):
        def run(mode):
            param = torch.nn.Parameter(torch.tensor([[0.5, 1.5], [0.2, 0.8]]))
            optimizer = torch.optim.AdamW([param], lr=0.01)
            controller = DTOController()
            histories = []
            for step in range(3):
                sampled = step == 1
                with activation_context(mode, sampled):
                    pred = param @ torch.tensor([[1.0, 2.0], [3.0, 1.0]])
                    task = pred.square().mean()
                    distill = (pred - 1).square().mean()
                    loss = controller.weighted_log_objective(distill, task)
                    controller.update(distill, task)
                if sampled and mode != 'off':
                    loss_gradients({'distill': distill, 'task': task}, [param])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                histories.append(param.detach().clone())
            return histories, controller.weights

        reference, reference_weights = run('off')
        for mode in ('diagnostics', 'all'):
            actual, weights = run(mode)
            for got, want in zip(actual, reference):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
            self.assertEqual(weights, reference_weights)

    def test_task_loss_is_reused_and_all_episode_cache_roots_are_cleared(self):
        score = torch.tensor([[1.0, 2.0]], requires_grad=True)
        cache = types.SimpleNamespace(_dto_score=score, _dto_task_loss=None,
                                      _dto_ce_values=[score.sum()] * 3,
                                      _dto_kl_value=score.square().sum())
        first = DTOTSFSAR._components(cache, torch.tensor([1]))[0]
        second = DTOTSFSAR._components(cache, torch.tensor([1]))[0]
        self.assertIs(first, second)
        DTOTSFSAR.clear_dto_cache(cache)
        self.assertIsNone(cache._dto_score)
        self.assertIsNone(cache._dto_task_loss)
        self.assertIsNone(cache._dto_kl_value)
        self.assertEqual(cache._dto_ce_values, [])


if __name__ == '__main__':
    unittest.main()
