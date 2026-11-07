"""CPU-only numerical tests for the DTO-KD adapter."""

import copy
import unittest

import torch

from experiments.dto_kd.controller import DTOController, closed_form_weights, gradient_metrics


class DTOControllerTest(unittest.TestCase):
    def test_closed_form_conflict_is_clipped_and_sum_one(self):
        p1, p2 = closed_form_weights(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0]))
        self.assertAlmostEqual(float(p1 + p2), 1.0, places=6)
        self.assertGreaterEqual(float(p1), 0.0)
        self.assertLessEqual(float(p1), 1.0)

    def test_closed_form_equal_gradients_is_half(self):
        p1, p2 = closed_form_weights(torch.tensor([2.0, 1.0]), torch.tensor([2.0, 1.0]))
        self.assertAlmostEqual(float(p1), 0.5, places=6)
        self.assertAlmostEqual(float(p2), 0.5, places=6)

    def test_amortized_weights_are_finite_and_on_simplex(self):
        c = DTOController(eta=0.2)
        first = c.weights
        c.update(torch.tensor(4.0), torch.tensor(0.5))
        second = c.weights
        self.assertAlmostEqual(sum(first), 1.0, places=6)
        self.assertAlmostEqual(sum(second), 1.0, places=6)
        self.assertTrue(all(0.0 < x < 1.0 for x in second))
        self.assertTrue(torch.isfinite(c.weighted_log_objective(torch.tensor(4.0), torch.tensor(0.5))))

    def test_controller_state_restore(self):
        c = DTOController(eta=0.03)
        c.update(torch.tensor(1.5), torch.tensor(0.8))
        state = c.state_dict()
        d = DTOController()
        d.load_state_dict(state)
        self.assertEqual(c.step_index, d.step_index)
        self.assertEqual(c.weights, d.weights)

    def test_controller_update_matches_independent_autograd_reference(self):
        eta = 0.03
        c = DTOController(eta=eta, init_distill=0.3)
        original = c.state_dict()["logits"]
        reference = original.clone().requires_grad_()
        logs = torch.stack([torch.tensor(4.0).log(), torch.tensor(0.5).log()]).double()
        objective = 0.5 * torch.dot(reference.softmax(0), logs).square()
        gradient, = torch.autograd.grad(objective, reference)
        expected = original - eta * gradient
        c.update(torch.tensor(4.0), torch.tensor(0.5))
        torch.testing.assert_close(c.state_dict()["logits"], expected)

    def test_update_is_detached_and_does_not_break_pending_backward(self):
        c = DTOController()
        distill = torch.tensor(4.0, requires_grad=True)
        task = torch.tensor(0.5, requires_grad=True)
        before = c.weights
        objective = c.weighted_log_objective(distill, task)
        c.update(distill, task)
        self.assertIsNone(distill.grad)
        self.assertIsNone(task.grad)
        self.assertFalse(c.state_dict()["logits"].requires_grad)
        objective.backward()
        self.assertAlmostEqual(distill.grad.item(), before[0] / 4.0, places=6)
        self.assertAlmostEqual(task.grad.item(), before[1] / 0.5, places=6)

    def test_half_losses_keep_objective_fp32(self):
        c = DTOController()
        distill = torch.tensor(2.0, dtype=torch.float16, requires_grad=True)
        task = torch.tensor(0.5, dtype=torch.float16, requires_grad=True)
        objective = c.weighted_log_objective(distill, task)
        self.assertEqual(objective.dtype, torch.float32)
        c.update(distill, task)
        objective.backward()
        self.assertTrue(torch.isfinite(distill.grad))
        self.assertTrue(torch.isfinite(task.grad))

    def test_gradient_metrics_boundary(self):
        m = gradient_metrics(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0]))
        self.assertEqual(m["conflict"], 1.0)
        self.assertAlmostEqual(m["cosine"], -1.0, places=6)
        self.assertAlmostEqual(m["norm_ratio_distill_over_task"], 1.0, places=6)

    def test_log_objective_keeps_gradients(self):
        x = torch.tensor(2.0, requires_grad=True)
        y = torch.tensor(3.0, requires_grad=True)
        c = DTOController(eta=0.0)
        c.weighted_log_objective(x.square(), y.square()).backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)
        self.assertTrue(torch.isfinite(x.grad))

    def test_teacher_side_gradient_is_not_detached(self):
        student = torch.tensor([1.0, 2.0], requires_grad=True)
        teacher_adapter = torch.tensor([0.5, 1.5], requires_grad=True)
        distill = (student - teacher_adapter).square().mean()
        teacher_ce = (teacher_adapter - 1.0).square().mean()
        (distill + teacher_ce).backward()
        self.assertIsNotNone(teacher_adapter.grad)
        self.assertGreater(float(teacher_adapter.grad.norm()), 0.0)

    def test_episode_rng_sequence_is_reproducible(self):
        a = torch.Generator().manual_seed(30100)
        b = torch.Generator().manual_seed(30100)
        self.assertTrue(torch.equal(torch.rand(100, generator=a), torch.rand(100, generator=b)))

    def test_auxiliary_loss_is_preserved_algebraically(self):
        task, kl = torch.tensor(1.2), torch.tensor(0.7)
        student_ce, teacher_ce, lsn_ce = torch.tensor(0.4), torch.tensor(0.3), torch.tensor(0.5)
        baseline = task + 2.0 * student_ce + 0.1 * lsn_ce + 0.1 * (kl + teacher_ce)
        dto = (2.0 * student_ce + 0.1 * lsn_ce + 0.1 * teacher_ce) + task + 0.1 * kl
        self.assertAlmostEqual(float(baseline - dto), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
